"""Does a sequence read back off a single still fold the way the original does?

Two commands.

``routes`` (a GPU): for each chain of the fixed dev panel (``decoder_structure_panel``) the folding model is run
deterministically on

    teacher               the sequence S, the model deriving its own single              (the reference)
    teacher_other_seed    the same at another seed                                         (its own floor)
    teacher_from_cache    S with the *cached* dev single injected, no ESMC in the model    (the cache is the model's)
    reconstructed         the decoded sequence with the single of S                        (what a generator's path does)
    reencoded             the decoded sequence, the model deriving its own single          (the sequence alone)
    twin_reconstructed    a random twin of the decoded sequence, the single of S           (as many substitutions)
    twin_reencoded        the twin, the model deriving its own single

and writes the Cα of every fold, the confidence of each, and the records AtlasFold is to refold (the original, the
decoded and the twin sequences, each against the deposited trace).

**What each route can show.** The contribution analysis found that the fold follows the single strongly, so ``reconstructed`` is nearly
insensitive to the sequence it is given and is the weak check; ``reencoded`` (the single is re-derived from the decoded
sequence, so an error in it reaches the pair prior) and AtlasFold (which never saw the single) are the discriminating ones.

``summarize`` (a CPU): reads the Cα and, when present, AtlasFold's evaluations, and reads the fold on the chains the teacher folds stably, which is where a decoder can be told from the
folding model's own indecision.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from foldspace.esmfold2 import decoder_structure_panel as panel_tools

#: fixed in advance
STABLE_TM = (
    0.9  # a teacher is stable if its fold at another seed has at least this TM with it
)
KEEP_TM = 0.9  # a route keeps the fold if its TM with the teacher is at least this
KEEP_SHARE = 0.9  # ... for at least this share of the stable chains
ATLAS_MARGIN = -0.02  # AtlasFold's scTM of the decoded sequence may fall at most this far below the original's (median)

ROUTES = (
    "teacher",
    "teacher_other_seed",
    "teacher_from_cache",
    "reconstructed",
    "reencoded",
    "twin_reconstructed",
    "twin_reencoded",
)

__all__ = ["read_panel", "summarize"]


def read_panel(path: Path) -> list[dict]:
    """``seq_id, sequence, decoded, substitutions`` as written by ``foldspace.esmfold2.evaluate_decoder``."""
    lines = Path(path).read_text().splitlines()
    if lines[0].split("\t") != ["seq_id", "sequence", "decoded", "substitutions"]:
        raise ValueError(f"{path}: not a decoded panel")
    rows = []
    for line in lines[1:]:
        if line:
            chain, sequence, decoded, k = line.split("\t")
            if len(sequence) != len(decoded):
                raise ValueError(
                    f"{chain}: {len(sequence)} residues decoded as {len(decoded)}"
                )
            rows.append(
                {
                    "id": chain,
                    "sequence": sequence,
                    "decoded": decoded,
                    "substitutions": int(k),
                }
            )
    return rows


# ------------------------------------------------------------------ routes


def run_routes(args) -> dict:
    import torch

    from foldspace.esmfold2 import validate_interface
    from foldspace.esmfold2 import interface as iface
    from foldspace.esmfold2 import substrate as sub

    settings = validate_interface.configure(True)
    # The model without a backbone is opened first and has its backbone taken out before the whole model is opened: the
    # checkpoint bundles ESMC, so opening it with ``load_esmc=False`` still brings 12.7 GiB onto the card for a moment, and
    # two of them at once do not fit a 24 GB card.
    lite, backbone = iface.drop_backbone(
        sub.open_substrate(args.checkpoint, load_esmc=False)
    )
    full = sub.open_substrate(args.checkpoint)
    dropout = validate_interface.lm_dropout_in_force(full)
    print(f"device {full.device}  {settings}  {dropout}  lite {backbone}", flush=True)
    if dropout["lm_dropout"] != 0.0:
        raise SystemExit(f"lm_dropout in force is {dropout['lm_dropout']}, not 0")

    rows = read_panel(args.panel)
    deposited = {c["id"]: c["ca"] for c in validate_interface.corpus_chains(args.chains, "dev")}
    cached = validate_interface._cached_rows(Path(args.targets), [r["id"] for r in rows])
    dev_sequences = [c for c in dict(_dev_table(args.dev_table)).values()]
    composition = np.array(
        [sum(s.count(a) for s in dev_sequences) for a in panel_tools.ALPHABET],
        dtype=np.float64,
    )
    device = lite.device

    def to_tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None].to(
            device
        )

    s0, s1 = args.seed, args.seed + 1
    began = time.perf_counter()
    out = Path(args.out)
    chain_dir = out / "chains"
    chain_dir.mkdir(parents=True, exist_ok=True)
    for j, r in enumerate(rows):
        npz, meta = chain_dir / f"{r['id']}.npz", chain_dir / f"{r['id']}.json"
        if npz.exists() and meta.exists():
            print(f"  {r['id']} already folded, not repeated", flush=True)
            continue
        seq, dec, k = r["sequence"], r["decoded"], r["substitutions"]
        twin = panel_tools.random_substitutions(
            seq, k, np.random.default_rng(args.seed + j), composition
        )
        prepared = {
            name: sub.prepare_sequences(full, s)
            for name, s in (("S", seq), ("D", dec), ("R", twin))
        }
        single = to_tensor(cached[r["id"]])
        folds = {
            "teacher": iface.fold_native(full, prepared["S"], seed=s0),
            "teacher_other_seed": iface.fold_native(full, prepared["S"], seed=s1),
            "teacher_from_cache": iface.fold_single(
                lite, prepared["S"], single, seed=s0
            ),
        }
        if k == 0:
            # a perfect read: every route that takes the decoded sequence is the teacher
            for name in (
                "reconstructed",
                "reencoded",
                "twin_reconstructed",
                "twin_reencoded",
            ):
                folds[name] = (
                    folds["teacher"]
                    if name in ("reencoded", "twin_reencoded")
                    else folds["teacher_from_cache"]
                )
        else:
            folds["reconstructed"] = iface.fold_single(
                lite, prepared["D"], single, seed=s0
            )
            folds["reencoded"] = iface.fold_native(full, prepared["D"], seed=s0)
            folds["twin_reconstructed"] = iface.fold_single(
                lite, prepared["R"], single, seed=s0
            )
            folds["twin_reencoded"] = iface.fold_native(full, prepared["R"], seed=s0)
        arrays = {f"{r['id']}__{name}": f.ca for name, f in folds.items()}
        arrays[f"{r['id']}__deposited"] = deposited[r["id"]]
        # the coordinates first and the record last: a chain is done when its record is there, so an allocation
        # that ends early is resumed and not repeated
        tmp = chain_dir / f"{r['id']}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, npz)
        meta.write_text(
            json.dumps(
                {
                    **r,
                    "length": len(seq),
                    "twin": twin,
                    "ptm": {n: f.metrics.get(validate_interface.PTM) for n, f in folds.items()},
                    "plddt": {n: f.metrics.get(validate_interface.PLDDT) for n, f in folds.items()},
                    "card": torch.cuda.get_device_name(0),
                    "slurm_job": os.environ.get("SLURM_JOB_ID"),
                },
                default=float,
            )
        )
        print(
            f"  {r['id']} L={len(seq)} k={k}  {time.perf_counter() - began:7.1f}s",
            flush=True,
        )
        del folds, prepared, single
        torch.cuda.empty_cache()

    chains: list[dict] = []
    coords: dict[str, np.ndarray] = {}
    for r in rows:
        chains.append(json.loads((chain_dir / f"{r['id']}.json").read_text()))
        arrays = np.load(chain_dir / f"{r['id']}.npz", allow_pickle=False)
        for key in arrays.files:  # each array once
            coords[key] = arrays[key]
    tmp = out / "routes_coords.tmp.npz"
    np.savez(tmp, **coords)
    os.replace(tmp, out / "routes_coords.npz")
    for label, field in (("S", "sequence"), ("D", "decoded"), ("R", "twin")):
        (out / f"records_{label}.json").write_text(
            json.dumps(
                [
                    {
                        "id": c["id"],
                        "sequence": c[field],
                        "coordinates": deposited[c["id"]].tolist(),
                    }
                    for c in chains
                ]
            )
        )
    result = {
        "experiment": "foldspace.esmfold2.decoder_structure_check.routes",
        "provenance": validate_interface._provenance(args, full, {"mode": "decoder_structure_routes"}),
        "settings": settings,
        "lm_dropout": dropout,
        "lite_backbone": backbone,
        "seeds": {"reference": s0, "other": s1},
        "chains": chains,
    }
    (out / "routes.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(f"wrote {out / 'routes.json'}")
    return result


def _dev_table(path: Path):
    from foldspace.esmfold2 import sequences

    return sequences.read_table(path)


# ----------------------------------------------------------------- summary


def _tm(geometry, mobile, target) -> float:
    return float(geometry.tm_score(mobile, target))


def summarize(routes: dict, coords: dict, atlas: dict[str, dict] | None = None) -> dict:
    """The panel read by the pre-specified criteria.

    ``routes`` is ``routes.json``; ``coords`` maps ``<id>__<route>`` to Cα; ``atlas`` maps ``S``, ``D``, ``R`` to the
    ``per_target`` lists of AtlasFold's evaluations (``refold``), when they were made.
    """
    from foldspace.evaluation import geometry

    per_chain = []
    for c in routes["chains"]:
        i = c["id"]
        teacher = coords[f"{i}__teacher"]
        deposited = coords[f"{i}__deposited"]
        row = {
            "id": i,
            "length": c["length"],
            "substitutions": c["substitutions"],
            "teacher_tm_deposited": _tm(geometry, teacher, deposited),
            "teacher_ptm": c["ptm"]["teacher"],
            "stable": _tm(geometry, coords[f"{i}__teacher_other_seed"], teacher)
            >= STABLE_TM,
            "teacher_seed_tm": _tm(
                geometry, coords[f"{i}__teacher_other_seed"], teacher
            ),
        }
        for route in ROUTES[2:]:
            fold = coords[f"{i}__{route}"]
            row[f"{route}_tm_teacher"] = _tm(geometry, fold, teacher)
            row[f"{route}_tm_deposited"] = _tm(geometry, fold, deposited)
            row[f"{route}_ptm"] = c["ptm"][route]
        row["cache_parity_rmsd"] = float(
            geometry.kabsch_rmsd(coords[f"{i}__teacher_from_cache"], teacher)
        )
        per_chain.append(row)

    stable = [r for r in per_chain if r["stable"]]
    with_errors = [r for r in stable if r["substitutions"] > 0]

    def med(rows, key):
        return float(np.median([r[key] for r in rows])) if rows else float("nan")

    keep = [r for r in stable if r["reencoded_tm_teacher"] >= KEEP_TM]
    summary = {
        "chains": len(per_chain),
        "stable_chains": len(stable),
        "exactly_recovered": sum(r["substitutions"] == 0 for r in per_chain),
        "substitutions": {
            "median": float(np.median([r["substitutions"] for r in per_chain])),
            "max": max(r["substitutions"] for r in per_chain),
        },
        "cache_parity": {
            "max_rmsd_a": max(r["cache_parity_rmsd"] for r in per_chain),
            "note": "the cached dev single injected into a model without ESMC against the teacher (0 when the cache is the model's)",
        },
        "teacher_tm_to_deposited_median": med(per_chain, "teacher_tm_deposited"),
        "reconstructed": {
            "tm_teacher_median_stable": med(stable, "reconstructed_tm_teacher"),
            "share_keeping_stable": float(
                np.mean([r["reconstructed_tm_teacher"] >= KEEP_TM for r in stable])
            )
            if stable
            else float("nan"),
        },
        "reencoded": {
            "tm_teacher_median_stable": med(stable, "reencoded_tm_teacher"),
            "share_keeping_stable": len(keep) / len(stable) if stable else float("nan"),
            "tm_teacher_median_stable_with_errors": med(
                with_errors, "reencoded_tm_teacher"
            ),
        },
        "twin": {
            "reencoded_tm_teacher_median_stable_with_errors": med(
                with_errors, "twin_reencoded_tm_teacher"
            ),
            "reconstructed_tm_teacher_median_stable_with_errors": med(
                with_errors, "twin_reconstructed_tm_teacher"
            ),
        },
        "unstable_teachers": {
            "chains": len(per_chain) - len(stable),
            "reencoded_tm_teacher_median": med(
                [r for r in per_chain if not r["stable"]], "reencoded_tm_teacher"
            ),
        },
        "per_chain": per_chain,
    }
    gates = {
        "a_reencoded_keeps_the_fold": {
            "reading": f"share of stable chains with TM(reencoded, teacher) >= {KEEP_TM}",
            "value": summary["reencoded"]["share_keeping_stable"],
            "required": KEEP_SHARE,
            "met": bool(stable)
            and summary["reencoded"]["share_keeping_stable"] >= KEEP_SHARE,
        },
        "b_decoder_errors_cost_no_more_than_random_ones": {
            "reading": "median TM(reencoded, teacher) of the decoded sequences against that of their random twins, stable chains with errors",
            "decoded": summary["reencoded"]["tm_teacher_median_stable_with_errors"],
            "twin": summary["twin"]["reencoded_tm_teacher_median_stable_with_errors"],
            "met": bool(with_errors)
            and summary["reencoded"]["tm_teacher_median_stable_with_errors"]
            >= summary["twin"]["reencoded_tm_teacher_median_stable_with_errors"],
        },
    }
    if atlas and all(k in atlas for k in ("S", "D")):
        by_id = {k: {t["id"]: t for t in v} for k, v in atlas.items()}
        ids = [
            r["id"]
            for r in per_chain
            if r["id"] in by_id["S"] and r["id"] in by_id["D"]
        ]
        delta = [by_id["D"][i]["scTM"] - by_id["S"][i]["scTM"] for i in ids]
        summary["atlasfold"] = {
            "chains": len(ids),
            "sctm_original_median": float(
                np.median([by_id["S"][i]["scTM"] for i in ids])
            ),
            "sctm_decoded_median": float(
                np.median([by_id["D"][i]["scTM"] for i in ids])
            ),
            "difference_median": float(np.median(delta)),
            "difference_worst": float(min(delta)),
        }
        if "R" in by_id:
            twin_delta = [
                by_id["R"][i]["scTM"] - by_id["S"][i]["scTM"]
                for i in ids
                if i in by_id["R"]
            ]
            summary["atlasfold"]["twin_difference_median"] = (
                float(np.median(twin_delta)) if twin_delta else None
            )
        gates["c_atlasfold_does_not_see_a_cost"] = {
            "reading": "median over the panel of scTM(decoded) - scTM(original), against the deposited traces",
            "value": summary["atlasfold"]["difference_median"],
            "required": ATLAS_MARGIN,
            "met": summary["atlasfold"]["difference_median"] >= ATLAS_MARGIN,
        }
    else:
        gates["c_atlasfold_does_not_see_a_cost"] = {
            "met": None,
            "reading": "AtlasFold was not run",
        }
    summary["gates"] = gates
    summary["passes"] = all(
        g["met"] for g in gates.values() if g["met"] is not None
    ) and all(g["met"] is not None for g in gates.values())
    return summary


def run_summarize(args) -> dict:
    routes = json.loads((Path(args.routes) / "routes.json").read_text())
    held = np.load(Path(args.routes) / "routes_coords.npz", allow_pickle=False)
    coords = {k: held[k] for k in held.files}
    atlas = {}
    for label in ("S", "D", "R"):
        path = Path(args.atlas) / f"atlas_{label}.json" if args.atlas else None
        if path and path.exists():
            atlas[label] = json.loads(path.read_text())["per_target"]
    result = summarize(routes, coords, atlas or None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "chains",
                    "stable_chains",
                    "exactly_recovered",
                    "gates",
                    "passes",
                )
            },
            indent=2,
            default=float,
        )
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub_ = parser.add_subparsers(dest="command", required=True)
    r = sub_.add_parser("routes")
    r.add_argument("--panel", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--checkpoint", default="ESMFold2")
    r.add_argument("--chains", type=Path, default=Path("data/corpus/chains.npz"))
    r.add_argument("--targets", type=Path, default=Path("cache/esmfold2/targets/dev"))
    r.add_argument(
        "--dev-table", type=Path, default=Path("cache/esmfold2/sequences/dev.tsv")
    )
    r.add_argument("--seed", type=int, default=20261120)
    r.add_argument("--allow-dirty", action="store_true")
    s = sub_.add_parser("summarize")
    s.add_argument("--routes", type=Path, required=True)
    s.add_argument("--atlas", type=Path, default=None)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "routes":
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
        from foldspace.evaluation.provenance import require_clean

        require_clean(allow=args.allow_dirty)
        run_routes(args)
    else:
        run_summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
