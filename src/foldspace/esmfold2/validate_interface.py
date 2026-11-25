"""Interface validation: can the folding model be driven from a mixed single alone, and what does the single add?

Two questions, kept apart because a yes to one says nothing about the other.

``parity`` -- is injecting a mixed single the *same computation* as letting
the model derive it? Four routes into one fold of one sequence-only input:

    native          the model computes its own ESMC stack            (reference)
    stack           a stack computed outside and handed in
    single          the stack's mixed single, injected by the hook, stack still passed
    single_lite     the same single, injected into a model with no ESMC backbone,
                    given a placeholder stack -- the path a generator takes

plus the tensor-level view (the pair prior the shim makes from the stack against
the one the hook substitutes) and the cache check (a freshly computed single
against the cached one of the same chain). A route passes when it differs from
``native`` by no more than two runs of ``native`` differ from each other. The gate
is read in deterministic mode (the repeat floor is then zero by construction);
the non-deterministic reading is how the earlier probes ran and is reported beside it.

``contribution`` -- *given* that injection is exact, how much of the fold is
the single? Length-matched chains A and B, every sequence/single combination:

    (S_A, H_A) (S_A, H_B) (S_B, H_A) (S_B, H_B)     the 2 x 2
    (S_A, mean) (S_B, mean)                          the position-aware mean for H
    (S_A, none) (S_B, none)                          no language-model prior at all
    (S_A, H_A) at another seed                       the sampler floor

each read against its own reference and against the deposited structure. This
measures how sensitive the fold is to the single and how much of the whole
prior's effect a wrong single recovers. It does **not** say a generated single is
worth having: a swap that moves the fold is a sensitivity, and the value of
a generator is read from a coherent (S, H) pair (the generated one), which is the question of latent generation.

Every threshold below is fixed in advance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# --- gates, fixed in advance -----------

#: A tensor agrees when its largest absolute difference, over the largest value,
#: is within this many machine epsilons of the dtype the shim computes in.
TENSOR_EPS_FACTOR = 4.0
#: Å. A route's largest atom shift from ``native`` may not exceed this ...
STRUCTURE_ABS_TOL = 1e-3
#: ... or this many times the largest atom shift between two runs of ``native``,
#: whichever is larger.
STRUCTURE_REPEAT_FACTOR = 3.0
#: ``prior_effect_fraction`` is undefined when removing the prior moves the fold
#: less than this (1 - TM): the ratio's denominator is then noise.
MIN_PRIOR_EFFECT = 0.02

ROUTES = ("stack", "single", "single_lite")

#: The substrate's names for the folding model's own confidence, as ``fold_metrics``
#: emits them (``esmfold2_atomworks.metrics.SCALAR_METRIC_SOURCES``).
PLDDT = "esm.mean_plddt"
PTM = "esm.ptm"

# ------------------------------------------------------------------ corpus


def _residues() -> str:
    from foldspace.evaluation import residues

    return residues.RESIDUES


def sequence_of(tokens) -> str:
    letters = _residues()
    return "".join(letters[int(t)] for t in tokens)


def corpus_chains(path: Path, split: str) -> list[dict]:
    """Every chain of one split: id, sequence, Cα, cluster.

    ``xyz`` is ``[N, CA, C]`` per residue, so Cα is its middle atom.

    Each array is read **once**. An ``NpzFile`` re-reads and decodes the whole array
    on every ``held[key]``, so indexing it inside the loop reads the 150 MB ``xyz``
    once per chain -- half a terabyte for the dev split, with the job at full CPU and
    no fold started.
    """
    held = np.load(path, allow_pickle=False)
    a = {
        key: held[key]
        for key in ("splits", "starts", "lengths", "ids", "clusters", "tokens", "xyz")
    }
    out = []
    for i in np.flatnonzero(a["splits"] == split):
        s, n = int(a["starts"][i]), int(a["lengths"][i])
        out.append(
            {
                "id": str(a["ids"][i]),
                "length": n,
                "cluster": int(a["clusters"][i]),
                "sequence": sequence_of(a["tokens"][s : s + n]),
                "ca": a["xyz"][s : s + n, 1, :].astype(np.float64),
            }
        )
    return out


def spread_by_length(chains: list[dict], n: int, lo: int, hi: int) -> list[dict]:
    """``n`` chains whose lengths are evenly spread over ``[lo, hi]``, deterministically."""
    inside = sorted(
        (c for c in chains if lo <= c["length"] <= hi),
        key=lambda c: (c["length"], c["id"]),
    )
    if len(inside) < n:
        raise ValueError(f"{len(inside)} chains between {lo} and {hi}, {n} asked for")
    positions = np.linspace(0, len(inside) - 1, n).round().astype(int)
    return [inside[p] for p in positions]


def identity_at_equal_length(a: str, b: str) -> float:
    """Fraction of positions that agree, no alignment: a ceiling on similarity for unaligned chains."""
    if len(a) != len(b):
        raise ValueError("equal lengths only")
    return float(np.mean([x == y for x, y in zip(a, b, strict=True)]))


def select_pairs(
    chains: list[dict], n_pairs: int, lo: int, hi: int, seed: int
) -> list[tuple[dict, dict]]:
    """Length-matched pairs from different 30 % clusters, one per distinct length.

    Equal length is what lets A's single stand in for B's without a crop;
    different clusters is what makes it another protein and not a relative.
    """
    rng = np.random.default_rng(seed)
    by_length: dict[int, list[dict]] = {}
    for c in chains:
        if lo <= c["length"] <= hi:
            by_length.setdefault(c["length"], []).append(c)
    usable = []
    for length in sorted(by_length):
        members = by_length[length]
        if len({m["cluster"] for m in members}) >= 2:
            usable.append(length)
    if len(usable) < n_pairs:
        raise ValueError(
            f"{len(usable)} lengths admit a cross-cluster pair, {n_pairs} asked for"
        )
    picked = [
        usable[i] for i in np.linspace(0, len(usable) - 1, n_pairs).round().astype(int)
    ]
    pairs = []
    for length in picked:
        members = by_length[length]
        while True:
            i, j = rng.choice(len(members), size=2, replace=False)
            if members[i]["cluster"] != members[j]["cluster"]:
                break
        pairs.append((members[int(i)], members[int(j)]))
    return pairs


# ----------------------------------------------------------------- verdicts


def tensor_tolerance(dtype_name: str) -> float:
    import torch

    return TENSOR_EPS_FACTOR * float(torch.finfo(getattr(torch, dtype_name)).eps)


def parity_verdict(records: list[dict], *, tensor_tol: float) -> dict:
    """Whether every route matched ``native`` within what two native runs differ by.

    ``records`` carry, per chain, ``structure`` (``coord_diff`` of each route and of
    the repeat against ``native``) and ``tensor`` (``tensor_diff`` of the pair priors).
    """
    chains = []
    for r in records:
        floor = r["structure"]["repeat"]["max_atom_shift"]
        allowed = max(STRUCTURE_ABS_TOL, STRUCTURE_REPEAT_FACTOR * floor)
        routes = {
            name: {
                "max_atom_shift": r["structure"][name]["max_atom_shift"],
                "allowed": allowed,
                "within": r["structure"][name]["max_atom_shift"] <= allowed,
            }
            for name in ROUTES
        }
        tensors = {
            name: {**d, "within": d["max_abs_over_max"] <= tensor_tol}
            for name, d in r["tensor"].items()
        }
        chains.append(
            {
                "id": r["id"],
                "length": r["length"],
                "repeat_floor": floor,
                "routes": routes,
                "tensors": tensors,
                "single_roundtrip_exact": r["single_roundtrip_exact"],
                "observed_single_exact": bool(
                    r.get("observed_single", {}).get("exact", False)
                ),
            }
        )
    return {
        "tensor_tolerance": tensor_tol,
        "structure_abs_tol": STRUCTURE_ABS_TOL,
        "structure_repeat_factor": STRUCTURE_REPEAT_FACTOR,
        "chains": chains,
        "all_routes_within": all(
            v["within"] for c in chains for v in c["routes"].values()
        ),
        "all_tensors_within": all(
            v["within"] for c in chains for v in c["tensors"].values()
        ),
        "all_roundtrips_exact": all(c["single_roundtrip_exact"] for c in chains),
        "all_observed_exact": all(c["observed_single_exact"] for c in chains),
    }


def analyse_pair(
    cas: dict[str, np.ndarray], native_a: np.ndarray, native_b: np.ndarray
) -> dict:
    """Each condition against its own reference and against the deposited structure.

    ``cas`` maps a condition to its Cα. The first letter of a condition is the
    sequence's chain, the second the single's source (``A``/``B``, ``m`` the
    channel mean, ``0`` no prior); ``AA_seed`` is the reference at another seed.
    """
    from foldspace.evaluation import geometry

    reference = {"A": "AA", "B": "BB"}
    natives = {"A": native_a, "B": native_b}
    #: a swap's single belongs to the *other* chain: its reference and its deposited
    #: structure, which is where the fold goes if it follows the single and not the sequence
    donor = {"AB": ("BB", "B"), "BA": ("AA", "A")}
    out = {}
    for name, ca in cas.items():
        side = name[0]
        row = {
            "tm_to_native": geometry.tm_score(ca, natives[side]),
            "lddt_to_native": geometry.lddt_ca(ca, natives[side]),
        }
        ref = cas[reference[side]]
        row["tm_to_ref"] = geometry.tm_score(ca, ref)
        row["lddt_to_ref"] = geometry.lddt_ca(ca, ref)
        row["rmsd_to_ref"] = geometry.kabsch_rmsd(ca, ref)
        if name in donor:
            donor_ref, donor_side = donor[name]
            row["tm_to_donor_ref"] = geometry.tm_score(ca, cas[donor_ref])
            row["tm_to_donor_native"] = geometry.tm_score(ca, natives[donor_side])
        out[name] = row
    return out


_GROUPS = {
    "swap": ("AB", "BA"),
    "mean": ("Am", "Bm"),
    "none": ("A0", "B0"),
    "seed_floor": ("AA_seed",),
    "reference": ("AA", "BB"),
}


def _spread(values: list[float]) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    return {
        "n": len(v),
        "median": float(np.median(v)),
        "q25": float(np.quantile(v, 0.25)),
        "q75": float(np.quantile(v, 0.75)),
    }


def summarize_contribution(per_pair: list[dict]) -> dict:
    """Spread, over pairs, of each group's agreement with its reference."""
    out: dict = {"pairs": len(per_pair), "groups": {}}
    for group, names in _GROUPS.items():
        out["groups"][group] = {
            metric: _spread(
                [p["conditions"][n][metric] for p in per_pair for n in names]
            )
            for metric in (
                "tm_to_ref",
                "lddt_to_ref",
                "rmsd_to_ref",
                "tm_to_native",
                "lddt_to_native",
            )
        }
    fractions = []
    for p in per_pair:
        for swap, none in (("AB", "A0"), ("BA", "B0")):
            lost = 1.0 - p["conditions"][none]["tm_to_ref"]
            moved = 1.0 - p["conditions"][swap]["tm_to_ref"]
            fractions.append(moved / lost if lost >= MIN_PRIOR_EFFECT else float("nan"))
    out["prior_effect_fraction"] = {
        **_spread(fractions),
        "definition": "(1 - TM(swap, ref)) / (1 - TM(no prior, ref)), per sequence",
    }
    out["swap_follows"] = _swap_follows(per_pair)
    return out


#: TM-score above which two Cα traces are counted as one fold. The repository's TM is a
#: global-superposition lower bound, so this is a conservative reading of "same fold".
SAME_FOLD_TM = 0.5


def _swap_follows(per_pair: list[dict]) -> dict:
    """Where a swapped fold goes: to the single's protein, to the sequence's own, or neither.

    **Post hoc**: the analysis was specified on the share of the prior's effect a swap recovers
    (``prior_effect_fraction``), not where the fold goes when it does not recover it.
    A swap is counted by the best TM it reaches against the single's protein (its
    reference fold and its deposited structure) and against the sequence's own.
    """
    swaps = [p["conditions"][n] for p in per_pair for n in ("AB", "BA")]
    donor = [max(s["tm_to_donor_ref"], s["tm_to_donor_native"]) for s in swaps]
    own = [max(s["tm_to_ref"], s["tm_to_native"]) for s in swaps]
    return {
        "swaps": len(swaps),
        "same_fold_tm": SAME_FOLD_TM,
        "follow_the_single": sum(x >= SAME_FOLD_TM for x in donor),
        "keep_the_sequences_fold": sum(x >= SAME_FOLD_TM for x in own),
        "tm_to_the_singles_protein": _spread(donor),
        "tm_to_the_sequences_own": _spread(own),
    }


# ------------------------------------------------------------------- runners


def configure(deterministic: bool) -> dict:
    """Settings that have to be in force before the first CUDA call."""
    if deterministic:
        # cuBLAS needs its workspace fixed before the context exists, or its
        # reductions stay order-dependent whatever torch is asked for.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if deterministic:
        torch.use_deterministic_algorithms(True)
    return {
        "deterministic": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def lm_dropout_in_force(substrate) -> dict:
    """The language-model dropout the fold will run with, read inside the context.

    A request of ``0`` is honoured only by the pins of 2026-11-11; the esm of the earlier pins treats it as no request, and the config default (not zero) stays in force,
    which would put a random mask on the very prior this experiment measures.
    """
    from esm.models.esmfold2.processor import _lm_dropout_context

    cfg = substrate.net.config
    with _lm_dropout_context(substrate.net, 0.0):
        encoder = getattr(cfg, "lm_encoder", None)
        if encoder is not None and getattr(cfg, "type", None) != "experimental":
            return {
                "lm_dropout": float(encoder.lm_dropout),
                "per_loop": bool(encoder.per_loop_lm_dropout),
            }
        return {"lm_dropout": float(getattr(cfg, "lm_dropout", float("nan")))}


def _provenance(args, substrate, extra: dict) -> dict:
    from foldspace.evaluation.provenance import capture, checkpoint_identity

    identity = checkpoint_identity(substrate.weights or args.checkpoint)
    return capture(
        {
            "checkpoint_name": args.checkpoint,
            "checkpoint": identity,
            "allow_dirty": bool(args.allow_dirty),
            **extra,
        }
    )


def _train_sequences(path: Path) -> dict[str, str]:
    import csv

    with path.open(newline="") as handle:
        return {
            r["seq_id"]: r["sequence"] for r in csv.DictReader(handle, delimiter="\t")
        }


def _cached_rows(targets: Path, wanted: list[str]) -> dict[str, np.ndarray]:
    """Rows of the cached single for the chains in ``wanted``, one shard read at a time."""
    found: dict[str, np.ndarray] = {}
    wanted_set = set(wanted)
    for path in sorted(targets.glob("z_shard*.npz")):
        held = np.load(path, allow_pickle=False)
        # each held[key] re-reads the array: take every one once (see corpus_chains)
        ids = [str(i) for i in held["ids"]]
        hits = [(i, k) for k, i in enumerate(ids) if i in wanted_set]
        if not hits:
            continue
        starts, lengths, z = held["starts"], held["lengths"], held["z"]
        for i, k in hits:
            s, n = int(starts[k]), int(lengths[k])
            found[i] = z[s : s + n].copy()
        del z
    missing = [w for w in wanted if w not in found]
    if missing:
        raise KeyError(f"not in the cache: {missing}")
    return found


def _stamp(label: str, began: float) -> None:
    print(f"  {label}  {time.perf_counter() - began:7.1f}s", flush=True)


def run_parity(args) -> dict:
    import torch

    from foldspace.esmfold2 import interface as I
    from foldspace.esmfold2 import inject
    from foldspace.esmfold2 import substrate as sub

    settings = configure(args.deterministic)
    full = sub.open_substrate(args.checkpoint)
    # The release checkpoint bundles ESMC, so load_esmc=False leaves it resident; the
    # backbone is taken out explicitly, so that this model cannot have used it.
    lite, backbone = I.drop_backbone(
        sub.open_substrate(args.checkpoint, load_esmc=False)
    )
    dropout = lm_dropout_in_force(full)
    print(f"device {full.device}  {settings}  {dropout}  lite {backbone}", flush=True)
    if dropout["lm_dropout"] != 0.0 and not args.allow_dropout:
        raise SystemExit(
            f"lm_dropout in force is {dropout['lm_dropout']}, not 0: this esm does not "
            "honour a request of 0; run under the pins of 2026-11-11"
        )
    shim_dtype = str(inject._shim(full.net).base_z_linear[1].weight.dtype).removeprefix(
        "torch."
    )

    dev = spread_by_length(
        corpus_chains(args.chains, "dev"), args.n_chains, args.lo, args.hi
    )
    s0, s1 = args.seed, args.seed + 1
    began = time.perf_counter()
    records = []
    for chain in dev:
        prepared = sub.prepare_sequences(full, chain["sequence"])
        with torch.no_grad():
            stack = sub.lm_hidden_states(full, prepared)
            # the single as the model computes it (inside its autocast region); the one
            # made outside it, in float32, is kept as the old definition, to be read
            # beside it and not gated
            single = I.model_single(full, stack)
            single_fp32 = inject.mixed_single(full.net, stack)
            single32 = single.double().float()
        roundtrip = bool(torch.equal(single32.to(single.dtype), single))

        with I.observe_single(full.net) as seen:
            native = I.fold_native(full, prepared, seed=s0)
        observed = seen.pop("single")
        folds = {
            "native": native,
            "repeat": I.fold_native(full, prepared, seed=s0),
            "other_seed": I.fold_native(full, prepared, seed=s1),
            "stack": I.fold_stack(full, prepared, stack, seed=s0),
            "single": I.fold_single(full, prepared, single, seed=s0, stack=stack),
            "single_lite": I.fold_single(lite, prepared, single32, seed=s0),
            "single_fp32": I.fold_single(
                full, prepared, single_fp32, seed=s0, stack=stack
            ),
        }
        prior = {
            "stack": I.pair_prior(full, stack=stack),
            "single": I.pair_prior(full, single=single),
            "single_lite": I.pair_prior(lite, single=single32),
        }
        record = {
            "id": chain["id"],
            "length": chain["length"],
            "single_roundtrip_exact": roundtrip,
            "single_dtype": str(single.dtype),
            "observed_in_forward": seen,
            "observed_single": I.tensor_diff(single, observed),
            "fp32_single_vs_model_single": I.tensor_diff(single_fp32, single),
            "structure": {
                name: I.coord_diff(folds[name], folds["native"])
                for name in ("repeat", "other_seed", *ROUTES, "single_fp32")
            },
            "tensor": {
                "single_vs_stack": I.tensor_diff(prior["single"], prior["stack"]),
                "single_lite_vs_stack": I.tensor_diff(
                    prior["single_lite"], prior["stack"]
                ),
            },
            "plddt": {k: v.metrics.get(PLDDT) for k, v in folds.items()},
            "ptm": {k: v.metrics.get(PTM) for k, v in folds.items()},
        }
        records.append(record)
        _stamp(f"{chain['id']} L={chain['length']}", began)
        del stack, single, single32, single_fp32, observed, folds, prior
        torch.cuda.empty_cache()

    tensor_tol = tensor_tolerance(shim_dtype)
    try:
        cache = _cache_check(args, full, sub, inject, tensor_tol)
    except Exception as exc:  # noqa: BLE001 -- recorded and printed, see below
        # The cache check is a reading beside the gate, not part of it: a failure in it
        # is recorded and printed, and does not discard the folds already made.
        traceback.print_exc()
        cache = {"skipped": False, "error": f"{type(exc).__name__}: {exc}"}
    verdict = parity_verdict(records, tensor_tol=tensor_tol)
    return {
        "experiment": "foldspace.esmfold2.validate_interface.parity",
        "settings": settings,
        "lm_dropout": dropout,
        "lite_backbone": backbone,
        "shim_dtype": shim_dtype,
        "seeds": {"reference": s0, "other": s1},
        "verdict": verdict,
        "cache_check": cache,
        "records": records,
        "provenance": _provenance(args, full, {"mode": "parity"}),
    }


def _cache_check(args, full, sub, inject, tensor_tol: float) -> dict:
    """A freshly computed single against the cached one, chain by chain.

    The cache was made under another pin of esm; ``within`` says whether this one
    reproduces it at the tensor tolerance, and a chain that does not is the
    reason to re-extract before anything is trained on it.
    """
    if args.cache_check <= 0:
        return {"skipped": True}
    import torch

    from foldspace.esmfold2 import interface as I

    sequences = _train_sequences(args.train_sequences)
    rows = [
        {"id": i, "length": len(s), "sequence": s}
        for i, s in sequences.items()
        if args.lo <= len(s) <= args.hi
    ]
    chosen = spread_by_length(rows, args.cache_check, args.lo, args.hi)
    cached = _cached_rows(args.targets, [c["id"] for c in chosen])
    out = []
    for c in chosen:
        prepared = sub.prepare_sequences(full, c["sequence"])
        # the shim's weights require grad, so a single made outside no_grad carries a
        # graph and cannot be turned into an array
        with torch.no_grad():
            stack = sub.lm_hidden_states(full, prepared)
            fresh = inject.mixed_single(full.net, stack)[0]
        diff = I.tensor_diff(
            fresh.detach().double().cpu().numpy(), cached[c["id"]].astype(np.float64)
        )
        out.append(
            {
                "id": c["id"],
                "length": c["length"],
                **diff,
                "within": diff["max_abs_over_max"] <= tensor_tol,
            }
        )
    return {
        "skipped": False,
        "tensor_tolerance": tensor_tol,
        "chains": out,
        "all_within": all(r["within"] for r in out),
    }


def run_contribution(args) -> dict:
    import torch

    from foldspace.esmfold2 import target_audit as audit
    from foldspace.esmfold2 import interface as I
    from foldspace.esmfold2 import substrate as sub

    settings = configure(args.deterministic)
    lite, backbone = I.drop_backbone(
        sub.open_substrate(args.checkpoint, load_esmc=False)
    )
    dropout = lm_dropout_in_force(lite)
    print(f"device {lite.device}  {settings}  {dropout}  lite {backbone}", flush=True)
    if dropout["lm_dropout"] != 0.0 and not args.allow_dropout:
        raise SystemExit(f"lm_dropout in force is {dropout['lm_dropout']}, not 0")

    train = [c for c in corpus_chains(args.chains, "train")]
    pairs = select_pairs(train, args.pairs, args.lo, args.hi, args.seed)
    wanted = sorted({c["id"] for p in pairs for c in p})
    cached = _cached_rows(args.targets, wanted)
    stats = np.load(args.stats, allow_pickle=False)
    means = audit.TerminalMeans(
        n_terminus=stats["terminal_mean_n"],
        c_terminus=stats["terminal_mean_c"],
        interior=stats["interior_mean"],
    )

    device = lite.device

    def to_t(a):
        """``[L, 256]`` numpy to the ``[1, L, 256]`` float32 the hook takes."""
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None].to(
            device
        )

    s0, s1 = args.seed, args.seed + 1
    began = time.perf_counter()
    per_pair, coords = [], {}
    for k, (a, b) in enumerate(pairs):
        prep = {
            "A": sub.prepare_sequences(lite, a["sequence"]),
            "B": sub.prepare_sequences(lite, b["sequence"]),
        }
        h = {"A": to_t(cached[a["id"]]), "B": to_t(cached[b["id"]])}
        mean = to_t(audit.positional_mean(a["length"], means))
        plan = {
            "AA": ("A", h["A"]), "AB": ("A", h["B"]),
            "BA": ("B", h["A"]), "BB": ("B", h["B"]),
            "Am": ("A", mean), "Bm": ("B", mean),
        }  # fmt: skip
        folds = {
            name: I.fold_single(lite, prep[side], single, seed=s0)
            for name, (side, single) in plan.items()
        }
        folds["A0"] = I.fold_without_prior(lite, prep["A"], seed=s0)
        folds["B0"] = I.fold_without_prior(lite, prep["B"], seed=s0)
        folds["AA_seed"] = I.fold_single(lite, prep["A"], h["A"], seed=s1)
        priors = {
            "swap_A": _relative_change(I, lite, h["B"], h["A"]),
            "swap_B": _relative_change(I, lite, h["A"], h["B"]),
            "mean_A": _relative_change(I, lite, mean, h["A"]),
        }
        cas = {name: f.ca for name, f in folds.items()}
        per_pair.append(
            {
                "pair": k,
                "length": a["length"],
                "A": a["id"],
                "B": b["id"],
                "clusters": [a["cluster"], b["cluster"]],
                "sequence_identity": identity_at_equal_length(
                    a["sequence"], b["sequence"]
                ),
                "native_tm_A_vs_B": float(_tm(a["ca"], b["ca"])),
                "pair_prior_relative_change": priors,
                "conditions": analyse_pair(cas, a["ca"], b["ca"]),
                "plddt": {n: f.metrics.get(PLDDT) for n, f in folds.items()},
                "ptm": {n: f.metrics.get(PTM) for n, f in folds.items()},
            }
        )
        for name, ca in cas.items():
            coords[f"pair{k}_{name}"] = ca
        coords[f"pair{k}_native_A"], coords[f"pair{k}_native_B"] = a["ca"], b["ca"]
        _stamp(f"pair {k} L={a['length']} {a['id']}/{b['id']}", began)
        del folds, h, mean
        torch.cuda.empty_cache()

    return {
        "experiment": "foldspace.esmfold2.validate_interface.contribution",
        "settings": settings,
        "lm_dropout": dropout,
        "lite_backbone": backbone,
        "seeds": {"reference": s0, "other": s1},
        "summary": summarize_contribution(per_pair),
        "pairs": per_pair,
        "provenance": _provenance(args, lite, {"mode": "contribution"}),
        "_coords": coords,
    }


def run_diagnose(args) -> dict:
    """Where does an injected single differ from the one the model derives?

    parity found the tensors exact and the folds not. The model runs its forward inside a
    CUDA autocast region, so the single it conditions on is computed there, and the one
    ``inject.mixed_single`` makes outside it can differ at the rounding of bf16. This
    reads the model's own single during a native fold and folds from each candidate:

        ext         the single made outside the forward, in float32 (what the cache holds)
        internal    the single the model derived, taken from inside its forward
        autocast    the single made outside, under the same autocast region
        bf16        ``ext`` rounded to bfloat16

    Exploratory: there is no pass or fail here, and nothing is fixed before reading it.
    """
    import torch

    from foldspace.esmfold2 import interface as I
    from foldspace.esmfold2 import inject
    from foldspace.esmfold2 import substrate as sub

    settings = configure(True)
    full = sub.open_substrate(args.checkpoint)
    dropout = lm_dropout_in_force(full)
    print(f"device {full.device}  {settings}  {dropout}", flush=True)
    dev = spread_by_length(
        corpus_chains(args.chains, "dev"), args.n_chains, args.lo, args.hi
    )
    seed = args.seed
    began = time.perf_counter()
    records = []
    for chain in dev:
        prepared = sub.prepare_sequences(full, chain["sequence"])
        with torch.no_grad():
            stack = sub.lm_hidden_states(full, prepared)
            ext = inject.mixed_single(full.net, stack)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                auto = inject.mixed_single(full.net, stack)
        with I.observe_single(full.net) as seen:
            native = I.fold_native(full, prepared, seed=seed)
        internal = seen.pop("single")
        folds = {
            "ext": I.fold_single(full, prepared, ext, seed=seed, stack=stack),
            "internal": I.fold_single(full, prepared, internal, seed=seed, stack=stack),
            "autocast": I.fold_single(full, prepared, auto, seed=seed, stack=stack),
            "bf16": I.fold_single(
                full, prepared, ext.to(torch.bfloat16), seed=seed, stack=stack
            ),
        }
        records.append(
            {
                "id": chain["id"],
                "length": chain["length"],
                "observed_in_forward": seen,
                "dtypes": {
                    "stack": str(stack.dtype),
                    "ext": str(ext.dtype),
                    "autocast": str(auto.dtype),
                    "internal": str(internal.dtype),
                },
                "single_vs_internal": {
                    "ext": I.tensor_diff(ext, internal),
                    "autocast": I.tensor_diff(auto, internal),
                    "bf16": I.tensor_diff(ext.to(torch.bfloat16), internal),
                },
                "structure_vs_native": {
                    name: I.coord_diff(f, native) for name, f in folds.items()
                },
            }
        )
        _stamp(f"{chain['id']} L={chain['length']}", began)
        del stack, ext, auto, internal, folds
        torch.cuda.empty_cache()
    return {
        "experiment": "foldspace.esmfold2.validate_interface.diagnose",
        "settings": settings,
        "lm_dropout": dropout,
        "seed": seed,
        "records": records,
        "provenance": _provenance(args, full, {"mode": "diagnose"}),
    }


def _tm(x: np.ndarray, y: np.ndarray) -> float:
    from foldspace.evaluation import geometry

    return geometry.tm_score(x, y)


def _relative_change(I, substrate, single, reference) -> float:
    """How much the pair bias changes when ``single`` replaces ``reference``."""
    new = I.pair_prior(substrate, single=single).float()
    old = I.pair_prior(substrate, single=reference).float()
    return float((new - old).norm() / old.norm())


# ----------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub_ = parser.add_subparsers(dest="command", required=True)
    for name in ("parity", "contribution", "diagnose"):
        p = sub_.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--checkpoint", default="ESMFold2")
        p.add_argument("--chains", type=Path, default=Path("data/corpus/chains.npz"))
        p.add_argument("--targets", type=Path, default=Path("data/corpus/targets"))
        p.add_argument(
            "--stats",
            type=Path,
            default=Path("cache/esmfold2/train_normalisation.npz"),
            help="foldspace.esmfold2.target_audit's output: the terminal and interior means the 'mean' condition uses",
        )
        p.add_argument(
            "--train-sequences",
            type=Path,
            default=Path("data/corpus/train_sequences.tsv"),
        )
        p.add_argument("--seed", type=int, default=20261119)
        p.add_argument("--lo", type=int, default=70)
        p.add_argument("--hi", type=int, default=220)
        p.add_argument("--deterministic", action="store_true")
        p.add_argument("--allow-dirty", action="store_true")
        p.add_argument("--allow-dropout", action="store_true")
        p.add_argument(
            "--tag",
            default="",
            help="appended to the output file names, so a second run does not overwrite the first",
        )
        if name == "parity":
            p.add_argument("--n-chains", type=int, default=4)
            p.add_argument("--cache-check", type=int, default=4)
        elif name == "diagnose":
            p.add_argument("--n-chains", type=int, default=3)
        else:
            p.add_argument("--pairs", type=int, default=8)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from foldspace.evaluation.provenance import require_clean

    require_clean(allow=args.allow_dirty)

    runners = {
        "parity": run_parity,
        "contribution": run_contribution,
        "diagnose": run_diagnose,
    }
    if args.command == "diagnose":
        args.deterministic = True  # an exploration of a deterministic difference
    result = runners[args.command](args)
    args.out.mkdir(parents=True, exist_ok=True)
    mode = "det" if args.deterministic else "nondet"
    stem = f"{mode}_{args.tag}" if args.tag else mode
    coords = result.pop("_coords", None)
    if coords is not None:
        np.savez(args.out / f"contribution_{stem}_coords.npz", **coords)
    path = args.out / f"{args.command}_{stem}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    headline = result.get("verdict", result.get("summary"))
    print(json.dumps(headline, indent=2, sort_keys=True, default=float))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
