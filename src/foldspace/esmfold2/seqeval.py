"""The ruler for a set of sequences: what a generator that makes sequences is read by.

A generator of the single makes ``(S_hat, H_hat)``. A generation is folded four ways, and the *re-encoded* route,
``(S_hat, H(S_hat))``, is the one that says whether the sequence alone was generated: the model derives its own single from the
sequence, so nothing of what a generator made as a single is left in it. A sequence-only generator has no other route. This module
is that route for any set of sequences, and what is read off it.

``fold`` (a GPU)
    Each sequence, its shuffle (the *floor*) and the reference chains folded by the model deriving its own single, AtlasFold's refold
    to be made of each beside it; for a set that has generated singles, how far each is from the single of its own sequence; for each
    sequence ESMC's masked cross-entropy. One file per sequence, so that an allocation that ends early is resumed.
``summarize`` (a CPU)
    The readings for one set: the success share on the re-encoded route against the floor and the reference, the credited
    share (a success that is neither a copy of a train chain nor degenerate), the continuous readings, and, for the pilot's sets,
    the table of direct against re-encoded.
``classes`` (a CPU)
    The classes of the sequence-only sets against the pilot's.

The arithmetic (entropy, the shares, the exact and rank tests, the consistency of two singles, the classes) is numpy only and is what
the offline tests cover.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from foldspace.esmfold2 import latent_generation

__all__ = [
    "ROUTE",
    "atlas_repeat",
    "classify",
    "composition_entropy",
    "consistency",
    "degenerate",
    "distinct_share",
    "fisher_greater",
    "floor_sequence",
    "mannwhitney_greater",
    "margin_over_shuffles",
    "mask_positions",
    "paired_comparison",
    "paired_deltas",
    "paired_summary",
    "summarize_set",
]

ROUTE = "reencoded"
LETTERS = latent_generation.LETTERS
#: fixed in advance; the first three are those of the pilot's own reading
CONFIDENT = latent_generation.CONFIDENT
CONSISTENT = latent_generation.CONSISTENT
SUPPORTED_SHARE = latent_generation.SUPPORTED_SHARE
OVER_FLOOR = latent_generation.OVER_FLOOR
ENTROPY_MIN = 3.5  # bits: a composition poorer than this is degenerate
DISTINCT_3MER_MIN = 0.7  # distinct 3-mers per 3-mer: a repeat is degenerate
PAIRED_DRAWS = 20000  # sign flips of the paired permutation
PAIRED_RESAMPLES = 10000  # bootstrap resamples of the paired intervals
PAIRED_SEED = 20261221
MASKS = 10  # random masks per sequence for ESMC's masked cross-entropy
MASK_RATE = 0.15  # the rate ESMC was trained at
ABOVE_SHARE = 0.10
ABOVE_SHARE_P = 0.05
ABOVE_PLDDT = 5.0  # points of AtlasFold's own pLDDT
ABOVE_PLDDT_P = 0.01


# ------------------------------------------------------------- the sequences


def composition_entropy(sequence: str) -> float:
    """The Shannon entropy of the residue composition, in bits (about 4.2 for a protein, 0 for a homopolymer)."""
    counts = np.array([sequence.count(a) for a in LETTERS], dtype=np.float64)
    if counts.sum() == 0:
        return 0.0
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def distinct_share(sequence: str, k: int = 3) -> float:
    """Distinct ``k``-mers per ``k``-mer: 1 when none repeats, near 0 for a repeat."""
    n = len(sequence) - k + 1
    if n <= 0:
        return 0.0
    return len({sequence[i : i + k] for i in range(n)}) / n


def degenerate(sequence: str) -> bool:
    """A composition poorer than ``ENTROPY_MIN`` bits, or fewer than ``DISTINCT_3MER_MIN`` distinct 3-mers per 3-mer."""
    return bool(
        composition_entropy(sequence) < ENTROPY_MIN
        or distinct_share(sequence) < DISTINCT_3MER_MIN
    )


def floor_sequence(sequence: str, seed: int, seq_id: str) -> str:
    """The sequence shuffled (its composition and length kept, its order gone), deterministically for ``(seed, seq_id)``."""
    rng = np.random.default_rng(seed + latent_generation.zlib_crc("floor:" + seq_id))
    return "".join(rng.permutation(list(sequence)))


def mask_positions(
    length: int, rng: np.random.Generator, masks: int = MASKS, rate: float = MASK_RATE
) -> list[np.ndarray]:
    """``masks`` sorted sets of positions, each ``rate`` of the chain (at least one), drawn without replacement."""
    size = min(length, max(1, round(rate * length)))
    return [np.sort(rng.choice(length, size=size, replace=False)) for _ in range(masks)]


# --------------------------------------------------------------- the tests


def fisher_greater(successes_1: int, n1: int, successes_2: int, n2: int) -> float:
    """One-sided exact ``p`` that the share ``successes_1 / n1`` exceeds ``successes_2 / n2``: the hypergeometric tail ``P(X >= successes_1)``."""
    total, successes = n1 + n2, successes_1 + successes_2
    top = min(n1, successes)
    tail = sum(
        math.comb(successes, x) * math.comb(total - successes, n1 - x)
        for x in range(successes_1, top + 1)
    )
    return float(tail / math.comb(total, n1))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    lower = upper - counts + 1
    return ((lower + upper) / 2.0)[inverse]


def mannwhitney_greater(x, y) -> float:
    """One-sided ``p`` (normal approximation, ties and continuity corrected) that ``x`` is stochastically greater than ``y``."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    values = np.concatenate([x, y])
    ranks = _average_ranks(values)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    _, counts = np.unique(values, return_counts=True)
    tie = (counts**3 - counts).sum() / (n * (n - 1)) if n > 1 else 0.0
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie))
    if sigma == 0.0:
        return 0.5
    z = (u - n1 * n2 / 2.0 - 0.5) / sigma
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of a share."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------- consistency


def consistency(h_generated, h_derived, *, mean, std, spectrum) -> dict:
    """How far a generated single is from the single the model derives from the sequence read off it (``[L, D]`` each, raw).

    Everything in the standardised coordinate of the train split: the relative L2 distance, the mean cosine between corresponding
    rows, and the level of ``anoise`` that has the same error.
    """
    a = (np.asarray(h_generated, dtype=np.float64) - mean) / std
    b = (np.asarray(h_derived, dtype=np.float64) - mean) / std
    if a.shape != b.shape:
        raise ValueError(f"shapes {a.shape} and {b.shape} differ")
    diff = a - b
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cosine = (a * b).sum(1) / np.maximum(na * nb, 1e-12)
    projected = diff @ spectrum.vectors
    error = latent_generation.eigen_error((projected**2).sum(0), spectrum.values, len(diff))
    return {
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-12)),
        "cosine": float(cosine.mean()),
        "anoise_level": error["relative"],
        "relative_size": error["relative_size"],
    }


def unpaired_consistency(
    generated: dict, derived: dict, *, mean, std, spectrum
) -> dict[str, dict]:
    """Each generated single against the derived single of every *other* sample, both cropped to the shorter: the unrelated pair."""
    out = {}
    for i, a in generated.items():
        rows = []
        for j, b in derived.items():
            if j == i:
                continue
            n = min(len(a), len(b))
            rows.append(
                consistency(a[:n], b[:n], mean=mean, std=std, spectrum=spectrum)
            )
        out[i] = {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("relative_l2", "cosine", "anoise_level")
        }
    return out


# -------------------------------------------------------------- the reading


def _median(values) -> float:
    return latent_generation._median(values)


def _gap_closed(sample: float, floor: float, reference: float) -> float:
    if any(np.isnan([sample, floor, reference])) or reference == floor:
        return float("nan")
    return float((sample - floor) / (reference - floor))


def _atlas(path: Path) -> dict[str, dict]:
    return {
        t["id"]: {"scTM": t.get("scTM"), "plddt": t.get("plddt")}
        for t in json.loads(Path(path).read_text())["per_target"]
    }


def margin_over_shuffles(
    table: list[dict],
    atlas: dict[str, dict],
    key: str = "scTM",
    *,
    non_degenerate: bool = True,
) -> dict:
    """The margin of a set over its own shuffles.

    The median of AtlasFold's ``key`` (``scTM`` or ``plddt``) of the sequences minus the median of their shuffles', and the one-sided rank ``p`` that the sequences are above. With
    ``non_degenerate`` the sequences that are degenerate are left out with their shuffles: a low-complexity sequence folds confidently and a shuffle of it does not, which moves
    a margin without the model having learnt anything. ``table`` is a summary's rows (``id``, ``degenerate``); ``atlas`` maps ``samples`` and ``floor`` to per-target tables by
    id (``_atlas``); the shuffle of sample ``sNNN`` is floor ``sNNN``.
    """
    ids = [
        r["id"]
        for r in table
        if not (non_degenerate and r["degenerate"])
        and r["id"] in atlas["samples"]
        and r["id"] in atlas["floor"]
    ]
    pairs = [(atlas["samples"][i].get(key), atlas["floor"][i].get(key)) for i in ids]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    x, y = [a for a, _ in pairs], [b for _, b in pairs]
    sample, floor = (_median(x), _median(y)) if pairs else (float("nan"),) * 2
    return {
        "key": key,
        "non_degenerate": non_degenerate,
        "sequences": len(x),
        "samples": sample,
        "floor": floor,
        "margin": sample - floor,
        "p": mannwhitney_greater(x, y),
    }


def run_margin(args) -> dict:
    fold_dir = Path(args.fold_dir)
    table = json.loads((fold_dir / "summary.json").read_text())["table"]
    atlas = {k: _atlas(fold_dir / f"atlas_{k}.json") for k in ("samples", "floor")}
    out = {
        f"{key}{'' if nd else '_all'}": margin_over_shuffles(
            table, atlas, key, non_degenerate=nd
        )
        for key in ("scTM", "plddt")
        for nd in (True, False)
    }
    if args.out:
        latent_generation._write(Path(args.out), out)
    for name, m in out.items():
        print(
            f"  {fold_dir.name} {name:10s} {m['sequences']:3d} sequences: {m['samples']:.3f} against {m['floor']:.3f}, "
            f"margin {m['margin']:+.3f}, rank p {m['p']:.4f}"
        )
    return out


def paired_deltas(
    table: list[dict],
    atlas: dict[str, dict],
    key: str = "scTM",
    *,
    non_degenerate: bool = True,
) -> np.ndarray:
    """Each sequence's AtlasFold ``key`` minus its own shuffle's: the paired differences.

    The shuffle of sample ``sNNN`` is floor ``sNNN`` and has its length and composition; a sequence and its shuffle are one pair, and a pair with a value missing on either side is left
    out, as is a degenerate sequence with its shuffle when ``non_degenerate``.
    """
    pairs = [
        (atlas["samples"][r["id"]].get(key), atlas["floor"][r["id"]].get(key))
        for r in table
        if not (non_degenerate and r["degenerate"])
        and r["id"] in atlas["samples"]
        and r["id"] in atlas["floor"]
    ]
    return np.array(
        [a - b for a, b in pairs if a is not None and b is not None], dtype=np.float64
    )


def paired_summary(
    delta,
    *,
    draws: int = PAIRED_DRAWS,
    resamples: int = PAIRED_RESAMPLES,
    seed: int = PAIRED_SEED,
) -> dict:
    """The size of a set's paired margin and what it is worth: the median and the mean of the differences, the share above zero, a bootstrap 95 % interval of each
    (``resamples`` resamples of the pairs), and the one-sided ``p`` of a sign-flip permutation of the mean (``draws`` flips, the observed arrangement counted): under the null
    the difference of a pair is as likely to be one sign as the other."""
    delta = np.asarray(delta, dtype=np.float64)
    n = len(delta)
    if n == 0:
        nan = float("nan")
        return {
            "pairs": 0,
            "median": nan,
            "mean": nan,
            "share_above_zero": nan,
            "ci_median": [nan, nan],
            "ci_mean": [nan, nan],
            "p": nan,
        }
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, n, size=(resamples, n))]
    flips = rng.choice(np.array([-1.0, 1.0]), size=(draws, n))
    at_least = int((((flips * delta).mean(axis=1)) >= delta.mean() - 1e-12).sum())
    return {
        "pairs": n,
        "median": float(np.median(delta)),
        "mean": float(delta.mean()),
        "share_above_zero": float((delta > 0).mean()),
        "ci_median": [
            float(v) for v in np.percentile(np.median(boot, axis=1), [2.5, 97.5])
        ],
        "ci_mean": [float(v) for v in np.percentile(boot.mean(axis=1), [2.5, 97.5])],
        "p": (at_least + 1) / (draws + 1),
    }


def paired_comparison(
    a,
    b,
    *,
    resamples: int = PAIRED_RESAMPLES,
    seed: int = PAIRED_SEED,
) -> dict:
    """One group of paired differences against another (two seeds, a flow and a sequence-only model): the difference of their medians with a bootstrap 95 % interval (each group
    resampled on its own) and the one-sided rank ``p`` that ``a`` is above ``b``."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        nan = float("nan")
        return {"difference_of_medians": nan, "ci": [nan, nan], "p": nan}
    rng = np.random.default_rng(seed)
    ma = np.median(a[rng.integers(0, len(a), size=(resamples, len(a)))], axis=1)
    mb = np.median(b[rng.integers(0, len(b), size=(resamples, len(b)))], axis=1)
    return {
        "difference_of_medians": float(np.median(a) - np.median(b)),
        "ci": [float(v) for v in np.percentile(ma - mb, [2.5, 97.5])],
        "p": mannwhitney_greater(a, b),
    }


def atlas_repeat(first: dict, second: dict, key: str = "scTM") -> dict:
    """How far AtlasFold's ``key`` of the same sequences moves between two runs of it (two seeds), sequence by sequence: the mean and the median of the absolute
    difference, the standard deviation of the difference and the correlation of the two runs. ``first`` and ``second`` map ids to per-target values (``_atlas``)."""
    ids = [
        i
        for i in first
        if i in second
        and first[i].get(key) is not None
        and second[i].get(key) is not None
    ]
    a = np.array([first[i][key] for i in ids], dtype=np.float64)
    b = np.array([second[i][key] for i in ids], dtype=np.float64)
    d = b - a
    nan = float("nan")
    return {
        "key": key,
        "sequences": len(ids),
        "mean_abs_difference": float(np.abs(d).mean()) if len(d) else nan,
        "median_abs_difference": float(np.median(np.abs(d))) if len(d) else nan,
        "sd_of_difference": float(d.std(ddof=1)) if len(d) > 1 else nan,
        "pearson_r": float(np.corrcoef(a, b)[0, 1])
        if len(d) > 2 and a.std() > 0 and b.std() > 0
        else nan,
    }


def run_repeat(args) -> dict:
    """``repeat``: a set's AtlasFold evaluations against a second run of them (``--repeat-dir``, made with another seed on the same records): how far the scTM of its sequences, its shuffles and
    its reference chains moves, and what the paired margin is in each run."""
    first_dir, second_dir = Path(args.fold_dir), Path(args.repeat_dir)
    table = json.loads((first_dir / "summary.json").read_text())["table"]
    out: dict = {"agreement": {}, "margin": {}}
    atlas = {"first": {}, "second": {}}
    for label in ("samples", "floor", "reference"):
        a = _atlas(first_dir / f"atlas_{label}.json")
        b = _atlas(second_dir / f"atlas_{label}.json")
        out["agreement"][label] = {
            key: atlas_repeat(a, b, key) for key in ("scTM", "plddt")
        }
        if label != "reference":
            atlas["first"][label], atlas["second"][label] = a, b
    for run in ("first", "second"):
        out["margin"][run] = paired_summary(paired_deltas(table, atlas[run], "scTM"))
    if args.out:
        latent_generation._write(Path(args.out), out)
    for label, per_key in out["agreement"].items():
        for key, m in per_key.items():
            print(
                f"  {label:9s} {key:6s} {m['sequences']:3d} sequences: mean |difference| {m['mean_abs_difference']:.4f}, "
                f"sd of the difference {m['sd_of_difference']:.4f}, r {m['pearson_r']:.4f}"
            )
    for run, m in out["margin"].items():
        print(
            f"  paired margin, {run} run: median {m['median']:+.3f} ({m['ci_median'][0]:+.3f} to {m['ci_median'][1]:+.3f}), "
            f"mean {m['mean']:+.3f}, sign-flip p {m['p']:.4f}"
        )
    return out


def _set_label(directory: Path, taken) -> str:
    """The name of a set in a record: its directory's name, or its path when another set already has that name (two seeds of one arm are both ``z``)."""
    for label in (directory.name, str(directory)):
        if label not in taken:
            return label
    return f"{directory}#{len(taken)}"


def run_paired(args) -> dict:
    """``paired``: each directory of ``--fold-dir`` read as a set (``summary.json``, ``atlas_samples.json``, ``atlas_floor.json``); with two or more, the first against the rest pooled."""
    out: dict = {"sets": {}}
    groups = []
    for directory in args.fold_dir:
        directory = Path(directory)
        table = json.loads((directory / "summary.json").read_text())["table"]
        atlas = {k: _atlas(directory / f"atlas_{k}.json") for k in ("samples", "floor")}
        delta = paired_deltas(table, atlas, "scTM")
        groups.append(delta)
        out["sets"][_set_label(directory, out["sets"])] = {
            "scTM": paired_summary(delta),
            "scTM_all": paired_summary(
                paired_deltas(table, atlas, "scTM", non_degenerate=False)
            ),
            "plddt": paired_summary(paired_deltas(table, atlas, "plddt")),
        }
    if len(groups) > 1:
        out["first_against_the_rest"] = paired_comparison(
            groups[0], np.concatenate(groups[1:])
        )
    if args.out:
        latent_generation._write(Path(args.out), out)
    for name, s in out["sets"].items():
        for kind, m in s.items():
            print(
                f"  {name:14s} {kind:9s} {m['pairs']:3d} pairs: median {m['median']:+.3f} "
                f"({m['ci_median'][0]:+.3f} to {m['ci_median'][1]:+.3f}), mean {m['mean']:+.3f} "
                f"({m['ci_mean'][0]:+.3f} to {m['ci_mean'][1]:+.3f}), above zero {m['share_above_zero']:.2f}, "
                f"sign-flip p {m['p']:.4f}"
            )
    if "first_against_the_rest" in out:
        c = out["first_against_the_rest"]
        print(
            f"  first against the rest: difference of medians {c['difference_of_medians']:+.3f} "
            f"({c['ci'][0]:+.3f} to {c['ci'][1]:+.3f}), rank p {c['p']:.4f}"
        )
    return out


def _succeeds(chain: dict, table: dict | None, route: str = ROUTE) -> bool:
    got = (table or {}).get(chain["id"], {}).get("scTM")
    plddt = chain["plddt"].get(route)
    return bool(
        plddt is not None
        and plddt >= CONFIDENT
        and got is not None
        and got >= CONSISTENT
    )


def summarize_set(
    fold: dict,
    atlas: dict[str, dict],
    novelty: dict | None = None,
    direct: dict | None = None,
) -> dict:
    """One set read by the ruler: ``fold`` is ``fold.json``, ``atlas`` maps ``samples``, ``floor`` and ``reference`` to AtlasFold's
    per-target tables, ``novelty`` the record of the search of this set against the training chains (``hits`` by sample id), ``direct`` the pilot's own
    ``{"fold": fold.json, "atlas": per-target table}`` for the sample ids, when there is one."""
    by_kind: dict[str, list[dict]] = {"sample": [], "floor": [], "reference": []}
    for chain in fold["chains"]:
        by_kind[chain["kind"]].append(chain)
    samples, floors, refs = by_kind["sample"], by_kind["floor"], by_kind["reference"]
    tables = {k: atlas.get(k) for k in ("samples", "floor", "reference")}
    hits = (novelty or {}).get("hits", {})

    def share(chains, table):
        return (
            float(np.mean([_succeeds(c, table) for c in chains]))
            if chains and table is not None
            else float("nan")
        )

    rows = []
    for c in samples:
        a = (tables["samples"] or {}).get(c["id"], {})
        hit = hits.get(c["id"])
        row = {
            "id": c["id"],
            "length": c["length"],
            "plddt": c["plddt"].get(ROUTE),
            "atlas_plddt": a.get("plddt"),
            "sctm": a.get("scTM"),
            "success": _succeeds(c, tables["samples"]),
            "degenerate": degenerate(c["sequence"]),
            "entropy": composition_entropy(c["sequence"]),
            "distinct_3mers": distinct_share(c["sequence"]),
            "copy": None if hit is None else bool(hit["copy"]),
            "best_identity": None if hit is None else hit["identity"],
            "nll": c.get("nll"),
        }
        row["credited"] = bool(
            row["success"] and not row["degenerate"] and row["copy"] is not True
        )
        rows.append(row)
    n = len(samples)
    successes = int(sum(r["success"] for r in rows))
    credited = int(sum(r["credited"] for r in rows))
    ref_share, floor_share = (
        share(refs, tables["reference"]),
        share(floors, tables["floor"]),
    )
    known = not (np.isnan(ref_share) or np.isnan(floor_share)) and n > 0
    novelty_known = novelty is not None

    def rule(value: float):
        return (
            bool(
                value >= SUPPORTED_SHARE * ref_share
                and value >= floor_share + OVER_FLOOR
            )
            if known
            else None
        )

    def kind_median(chains, table, key):
        return _median((table or {}).get(c["id"], {}).get(key) for c in chains)

    def continuous(key_for_kind):
        sample, floor, ref = (
            key_for_kind(samples, "samples"),
            key_for_kind(floors, "floor"),
            key_for_kind(refs, "reference"),
        )
        return {
            "samples": sample,
            "floor": floor,
            "reference": ref,
            "gap_closed": _gap_closed(sample, floor, ref),
        }

    out = {
        "route": ROUTE,
        "samples": n,
        "floors": len(floors),
        "references": len(refs),
        "success": {"count": successes, "share": successes / n if n else float("nan")},
        "credited": {
            "count": credited,
            "share": credited / n if n else float("nan"),
            "novelty_known": novelty_known,
        },
        "reference_success_share": ref_share,
        "floor_success_share": floor_share,
        "verdict": {
            "success_share": successes / n if n else float("nan"),
            "credited_share": credited / n if n else float("nan"),
            "reference_success_share": ref_share,
            "floor_success_share": floor_share,
            "supported_success": rule(successes / n) if n else None,
            "supported_credited": rule(credited / n) if (n and novelty_known) else None,
        },
        "success_interval": wilson(successes, n),
        "continuous": {
            "atlas_plddt": continuous(
                lambda chains, label: kind_median(chains, tables[label], "plddt")
            ),
            "sctm": continuous(
                lambda chains, label: kind_median(chains, tables[label], "scTM")
            ),
            "model_plddt": continuous(
                lambda chains, label: _median(c["plddt"].get(ROUTE) for c in chains)
            ),
            "esmc_nll": continuous(
                lambda chains, label: _median(c.get("nll") for c in chains)
            ),
        },
        "share_confident": {
            "samples": float(
                np.mean(
                    [r["plddt"] is not None and r["plddt"] >= CONFIDENT for r in rows]
                )
            )
            if rows
            else float("nan"),
            "floor": float(np.mean([c["plddt"][ROUTE] >= CONFIDENT for c in floors]))
            if floors
            else float("nan"),
            "reference": float(np.mean([c["plddt"][ROUTE] >= CONFIDENT for c in refs]))
            if refs
            else float("nan"),
        },
        "sequence": {
            "degenerate_share": float(np.mean([r["degenerate"] for r in rows]))
            if rows
            else float("nan"),
            "degenerate_share_floor": float(
                np.mean([degenerate(c["sequence"]) for c in floors])
            )
            if floors
            else float("nan"),
            "degenerate_share_reference": float(
                np.mean([degenerate(c["sequence"]) for c in refs])
            )
            if refs
            else float("nan"),
            "copy_share": float(
                np.mean([r["copy"] for r in rows if r["copy"] is not None])
            )
            if novelty_known and rows
            else None,
            "identity": (novelty or {}).get("identity"),
            "clusters_30": (novelty or {}).get("clusters_30"),
            "clusters_30_over_n": ((novelty or {}).get("clusters_30") / n)
            if novelty_known and n
            else None,
        },
        "table": rows,
    }
    consistent = [c["consistency"] for c in samples if "consistency" in c]
    if consistent:
        out["consistency"] = {
            "paired": _consistency_medians(consistent),
            "unpaired": _consistency_medians(
                [
                    c["consistency_unpaired"]
                    for c in samples
                    if "consistency_unpaired" in c
                ]
            ),
            "round_trip": _consistency_medians(
                [c["consistency"] for c in refs if "consistency" in c]
            ),
            "pair_bias": {
                "paired_median": _median(
                    c.get("pair_bias_relative_l2") for c in samples
                ),
                "round_trip_median": _median(
                    c.get("pair_bias_relative_l2") for c in refs
                ),
            },
        }
    if direct is not None:
        out["direct_against_reencoded"] = _two_by_two(rows, direct)
    return out


def _consistency_medians(records: list[dict]) -> dict:
    keys = ("relative_l2", "cosine", "anoise_level")
    return {key: _median(r.get(key) for r in records) for key in keys} | {
        "n": len(records)
    }


def _two_by_two(rows: list[dict], direct: dict) -> dict:
    """The table of a sample's direct success (the pilot's own reading) against its success on the re-encoded route."""
    by_id = {c["id"]: c for c in direct["fold"]["chains"] if c.get("kind") == "sample"}
    table = {
        "direct_and_reencoded": 0,
        "direct_only": 0,
        "reencoded_only": 0,
        "neither": 0,
    }
    for r in rows:
        d = by_id.get(r["id"])
        got = direct["atlas"].get(r["id"], {}).get("scTM")
        d_success = bool(
            d is not None
            and d["plddt"].get("direct") is not None
            and d["plddt"]["direct"] >= CONFIDENT
            and got is not None
            and got >= CONSISTENT
        )
        key = {
            (True, True): "direct_and_reencoded",
            (True, False): "direct_only",
            (False, True): "reencoded_only",
            (False, False): "neither",
        }[(d_success, bool(r["success"]))]
        table[key] += 1
    return table


# --------------------------------------------------------------- the classes


def classify(summary: dict, pilot: dict, *, control_supported: bool) -> dict:
    """The class of one set: *supported*, *above the pilot* or *at the pilot's level*.

    ``pilot`` has ``successes`` and ``n`` (the pilot's pooled re-encoded successes) and ``atlas_plddt`` (its sequences' AtlasFold
    pLDDT). ``control_supported``: whether the ESMC reference passes the absolute rule; if it does not, *supported* is withdrawn.
    """
    v = summary["verdict"]
    n = summary["samples"]
    credited = summary["credited"]["count"]
    rule = v.get("supported_credited")
    share_p = fisher_greater(credited, n, pilot["successes"], pilot["n"])
    gain = credited / n - pilot["successes"] / pilot["n"]
    plddt = [r["atlas_plddt"] for r in summary["table"] if r["atlas_plddt"] is not None]
    plddt_gain = _median(plddt) - _median(pilot["atlas_plddt"])
    plddt_p = mannwhitney_greater(plddt, pilot["atlas_plddt"])
    above_share = bool(gain >= ABOVE_SHARE and share_p < ABOVE_SHARE_P)
    above_plddt = bool(plddt_gain >= ABOVE_PLDDT and plddt_p < ABOVE_PLDDT_P)
    if rule and control_supported:
        label = "supported"
    elif above_share or above_plddt:
        label = "above the pilot"
    else:
        label = "at the pilot's level"
    return {
        "class": label,
        "supported_rule_met": rule,
        "supported_withdrawn": bool(rule and not control_supported),
        "credited_share": credited / n,
        "gain_over_pilot": gain,
        "fisher_p": share_p,
        "above_by_share": above_share,
        "atlas_plddt_gain": plddt_gain,
        "mannwhitney_p": plddt_p,
        "above_by_plddt": above_plddt,
    }


# ---------------------------------------------------------------- fold (GPU)


def _read_table_rows(path: Path) -> list[dict]:
    return latent_generation.read_samples(path)


def run_fold(args) -> dict:
    """Fold a set of sequences on the re-encoded route, with their floors and the reference chains: one file each, resumable."""
    import torch

    from foldspace.esmfold2 import validate_interface
    from foldspace.esmfold2 import decoder_structure_check
    from foldspace.esmfold2 import interface as iface
    from foldspace.esmfold2 import perturb as P
    from foldspace.esmfold2 import substrate as sub

    settings = validate_interface.configure(True)
    full = sub.open_substrate(args.checkpoint)
    dropout = validate_interface.lm_dropout_in_force(full)
    print(f"device {full.device}  {settings}  {dropout}", flush=True)
    if dropout["lm_dropout"] != 0.0:
        raise SystemExit(f"lm_dropout in force is {dropout['lm_dropout']}, not 0")

    head, head_note = None, "not asked for"
    if not args.no_plausibility:
        try:
            from foldspace.esmfold2.lm_head import load_lm_head

            head, head_note = load_lm_head(full), "loaded"
        except Exception as exc:  # noqa: BLE001 -- a diagnostic: its absence is recorded and nothing else stops
            head_note = f"unavailable: {type(exc).__name__}: {exc}"
    print(f"ESMC masked-LM head: {head_note}", flush=True)

    norm = np.load(args.normalisation, allow_pickle=False)
    mean, std = norm["mean"].astype(np.float64), norm["std"].astype(np.float64)
    spectrum = P.spectrum_from_covariance(
        np.load(args.covariance, allow_pickle=False)["covariance"]
    )

    samples = _read_table_rows(args.samples)
    singles = None
    if args.singles:
        held = np.load(args.singles, allow_pickle=False)
        singles = {k: held[k] for k in held.files}
    pilot_floors = {}
    if args.floors:
        for c in json.loads(Path(args.floors).read_text())["chains"]:
            if c.get("floor_sequence"):
                pilot_floors[c["id"]] = c["floor_sequence"]
    floors = {
        s["id"]: pilot_floors.get(s["id"])
        or floor_sequence(s["sequence"], args.seed, s["id"])
        for s in samples
    }
    panel = decoder_structure_check.read_panel(args.panel) if args.panel else []
    if args.limit:  # a smoke of the whole path on the first few of each, not a reading
        samples, panel = samples[: args.limit], panel[: args.limit]

    jobs = [("sample", s["id"], s["sequence"], None) for s in samples]
    jobs += [("floor", s["id"], floors[s["id"]], None) for s in samples]
    jobs += [("reference", r["id"], r["sequence"], r["decoded"]) for r in panel]

    out = Path(args.out)
    chain_dir = out / "chains"
    chain_dir.mkdir(parents=True, exist_ok=True)
    device = full.device

    def tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None].to(
            device
        )

    def derive(sequence):
        """The single the model derives from a sequence: ``[L, 256]`` float32, bfloat16 values (``interface.model_single``)."""
        prepared = sub.prepare_sequences(full, sequence)
        with torch.no_grad():
            stack = sub.lm_hidden_states(full, prepared)
            single = iface.model_single(full, stack)
        return prepared, single[0].float().cpu().numpy()

    def pair_bias(a, b) -> float:
        with torch.no_grad():
            pa = iface.pair_prior(full, single=tensor(a)).float()
            pb = iface.pair_prior(full, single=tensor(b)).float()
            return float(
                torch.linalg.norm(pa - pb) / torch.linalg.norm(pb).clamp(min=1e-12)
            )

    began = time.perf_counter()
    for kind, i, sequence, decoded in jobs:
        npz, meta = chain_dir / f"{kind}-{i}.npz", chain_dir / f"{kind}-{i}.json"
        if npz.exists() and meta.exists():
            print(f"  {kind} {i} already folded, not repeated", flush=True)
            continue
        prepared = sub.prepare_sequences(full, sequence)
        folded = iface.fold_native(full, prepared, seed=args.seed)
        record = {
            "id": i,
            "kind": kind,
            "length": len(sequence),
            "sequence": sequence,
            "decoded": decoded,
            "plddt": {ROUTE: folded.metrics.get(validate_interface.PLDDT)},
            "ptm": {ROUTE: folded.metrics.get(validate_interface.PTM)},
            "card": torch.cuda.get_device_name(0),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        }
        arrays = {"ca": folded.ca}
        if head is not None:
            record["nll"] = masked_nll(
                full,
                head,
                prepared,
                np.random.default_rng(args.seed + latent_generation.zlib_crc(f"{kind}:{i}")),
            )
        if kind == "sample" and singles is not None:
            _, derived = derive(sequence)
            arrays["derived"] = derived.astype(np.float32)
            record["consistency"] = consistency(
                singles[i], derived, mean=mean, std=std, spectrum=spectrum
            )
            record["pair_bias_relative_l2"] = pair_bias(singles[i], derived)
        if kind == "reference" and decoded is not None:
            _, derived = derive(sequence)
            _, round_trip = derive(decoded)
            record["consistency"] = consistency(
                derived, round_trip, mean=mean, std=std, spectrum=spectrum
            )
            record["pair_bias_relative_l2"] = pair_bias(derived, round_trip)
        tmp = chain_dir / f"{kind}-{i}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, npz)
        meta.write_text(json.dumps(record, default=float))
        nll_text = f"  nll {record['nll']:.3f}" if "nll" in record else ""
        print(
            f"  {kind} {i} L={len(sequence)}  pLDDT {record['plddt'][ROUTE]:.3f}"
            f"{nll_text}  {time.perf_counter() - began:7.1f}s",
            flush=True,
        )
        del folded, prepared
        torch.cuda.empty_cache()

    chains, coords, derived_all = [], {}, {}
    for kind, i, _, _ in jobs:
        chains.append(json.loads((chain_dir / f"{kind}-{i}.json").read_text()))
        arrays = np.load(chain_dir / f"{kind}-{i}.npz", allow_pickle=False)
        for key in arrays.files:  # each array once
            if key == "ca":
                coords[f"{kind}-{i}"] = arrays[key]
            elif key == "derived" and kind == "sample":
                derived_all[i] = arrays[key]
    if singles is not None and derived_all:
        unpaired = unpaired_consistency(
            {i: singles[i] for i in derived_all},
            derived_all,
            mean=mean,
            std=std,
            spectrum=spectrum,
        )
        for c in chains:
            if c["kind"] == "sample" and c["id"] in unpaired:
                c["consistency_unpaired"] = unpaired[c["id"]]
    tmp = out / "fold_coords.tmp.npz"
    np.savez(tmp, **coords)
    os.replace(tmp, out / "fold_coords.npz")
    if derived_all:
        np.savez(out / "derived_singles.npz", **derived_all)
    for label, kind in (
        ("samples", "sample"),
        ("floor", "floor"),
        ("reference", "reference"),
    ):
        (out / f"records_{label}.json").write_text(
            json.dumps(
                [
                    {
                        "id": c["id"],
                        "sequence": c["sequence"],
                        "coordinates": coords[f"{kind}-{c['id']}"].tolist(),
                    }
                    for c in chains
                    if c["kind"] == kind
                ]
            )
        )
    result = {
        "experiment": "foldspace.esmfold2.seqeval.fold",
        "route": ROUTE,
        "provenance": validate_interface._provenance(args, full, {"mode": "seqeval_fold"}),
        "settings": settings,
        "lm_dropout": dropout,
        "plausibility": head_note,
        "masks": {"count": MASKS, "rate": MASK_RATE},
        "seed": args.seed,
        "chains": chains,
    }
    latent_generation._write(out / "fold.json", result)
    print(f"wrote {out / 'fold.json'}", flush=True)
    return result


def masked_nll(full, head, prepared, rng: np.random.Generator) -> float:
    """ESMC's cross-entropy (nats) of the residues it is made to predict under ``MASKS`` random masks of ``MASK_RATE`` of the chain."""
    import torch

    from foldspace.esmfold2.lm_head import masked_logits
    from foldspace.esmfold2.sequence_directions import chain_slice

    tokens = chain_slice(prepared, 0).to(prepared.features["input_ids"].device)
    truth = prepared.features["input_ids"].reshape(-1)[tokens]
    losses = []
    for chosen in mask_positions(len(tokens), rng):
        positions = tokens[torch.as_tensor(chosen, device=tokens.device)]
        logits = masked_logits(full, prepared, head, positions).float()
        logp = torch.log_softmax(logits, dim=-1)
        target = truth[torch.as_tensor(chosen, device=tokens.device)]
        losses.append(-logp[torch.arange(len(chosen), device=logp.device), target])
    return float(torch.cat(losses).mean())


# -------------------------------------------------------------- summarize


def run_summarize(args) -> dict:
    fold_dir = Path(args.fold_dir)
    fold = json.loads((fold_dir / "fold.json").read_text())
    atlas = {}
    for label in ("samples", "floor", "reference"):
        path = fold_dir / f"atlas_{label}.json"
        if path.exists():
            atlas[label] = _atlas(path)
    novelty = None
    if args.novelty:
        record = json.loads(Path(args.novelty).read_text())
        novelty = record["sets"][args.novelty_set]
    direct = None
    if args.direct:
        direct_dir = Path(args.direct)
        direct = {
            "fold": json.loads((direct_dir / "fold.json").read_text()),
            "atlas": _atlas(direct_dir / "atlas_samples.json"),
        }
    summary = summarize_set(fold, atlas, novelty, direct)
    summary["provenance"] = fold.get("provenance")
    latent_generation._write(Path(args.out), summary)
    v = summary["verdict"]
    print(
        f"{fold_dir.name}: {summary['samples']} sequences, success {summary['success']['count']}"
        f" (share {v['success_share']:.3f}), credited {summary['credited']['count']};"
        f" floor {v['floor_success_share']:.3f}, reference {v['reference_success_share']:.3f};"
        f" supported (success) {v['supported_success']}"
    )
    return summary


def _pilot(paths) -> dict:
    successes, n, plddt = 0, 0, []
    for p in paths:
        summary = json.loads(Path(p).read_text())
        successes += int(summary["success"]["count"])
        n += int(summary["samples"])
        plddt += [
            r["atlas_plddt"] for r in summary["table"] if r["atlas_plddt"] is not None
        ]
    return {"successes": successes, "n": n, "atlas_plddt": plddt}


def run_classes(args) -> dict:
    pilot = _pilot(args.pilot)
    control = json.loads(Path(args.control).read_text())
    control_supported = bool(control["verdict"].get("supported_credited"))
    if control["verdict"].get("supported_credited") is None:
        control_supported = bool(control["verdict"].get("supported_success"))
    per_set: dict[tuple[int, str], dict] = {}
    for spec in args.set:
        head, _, path = spec.partition("=")
        seed, _, temperature = head.partition(":")
        summary = json.loads(Path(path).read_text())
        per_set[(int(seed), temperature)] = {
            "summary": summary,
            "reading": classify(summary, pilot, control_supported=control_supported),
        }
    chosen = {}
    for seed in sorted({s for s, _ in per_set}):
        options = [(t, v) for (s, t), v in per_set.items() if s == seed]
        best = max(
            options,
            key=lambda tv: (
                tv[1]["reading"]["credited_share"],
                _median(r["atlas_plddt"] for r in tv[1]["summary"]["table"]),
            ),
        )
        chosen[seed] = {"temperature": best[0], **best[1]["reading"]}
    classes = [c["class"] for c in chosen.values()]
    order = ["at the pilot's level", "above the pilot", "supported"]
    if len(set(classes)) == 1:
        overall = classes[0]
    elif len(classes) >= 3:
        overall = sorted(classes, key=order.index)[len(classes) // 2]
    else:
        overall = "undecided: the seeds disagree; train seed 2"
    result = {
        "pilot": {k: v for k, v in pilot.items() if k != "atlas_plddt"}
        | {"atlas_plddt_median": _median(pilot["atlas_plddt"])},
        "control_supported": control_supported,
        "per_set": {
            f"seed{s}_temperature{t}": v["reading"] for (s, t), v in sorted(per_set.items())
        },
        "chosen": {f"seed{s}": c for s, c in chosen.items()},
        "overall": overall,
    }
    latent_generation._write(Path(args.out), result)
    for name, c in result["chosen"].items():
        print(f"  {name} (temperature {c['temperature']}): {c['class']}")
    print(f"overall: {overall}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub_ = parser.add_subparsers(dest="command", required=True)
    f = sub_.add_parser("fold")
    f.add_argument(
        "--samples", type=Path, required=True, help="a directory with samples.tsv"
    )
    f.add_argument("--out", type=Path, required=True)
    f.add_argument(
        "--singles",
        type=Path,
        default=None,
        help="samples.npz: the generated singles, by sample id",
    )
    f.add_argument(
        "--floors",
        type=Path,
        default=None,
        help="a fold.json of the pilot, for the shuffles it made",
    )
    f.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="panel_decoded.tsv: the reference chains",
    )
    f.add_argument("--checkpoint", default="ESMFold2")
    f.add_argument(
        "--normalisation",
        type=Path,
        default=Path("cache/esmfold2/model_single_train_normalisation.npz"),
    )
    f.add_argument(
        "--covariance",
        type=Path,
        default=Path("cache/esmfold2/model_single_train_covariance.npz"),
    )
    f.add_argument("--no-plausibility", action="store_true")
    f.add_argument(
        "--limit",
        type=int,
        default=0,
        help="fold only the first N samples and N reference chains (a smoke, not a reading)",
    )
    f.add_argument("--seed", type=int, default=20261122)
    f.add_argument("--allow-dirty", action="store_true")
    s = sub_.add_parser("summarize")
    s.add_argument("--fold-dir", type=Path, required=True)
    s.add_argument("--novelty", type=Path, default=None)
    s.add_argument("--novelty-set", default=None)
    s.add_argument(
        "--direct",
        type=Path,
        default=None,
        help="the pilot's fold directory of the same samples",
    )
    s.add_argument("--out", type=Path, required=True)
    m = sub_.add_parser("margin")
    m.add_argument(
        "--fold-dir",
        type=Path,
        required=True,
        help="a set's directory, with summary.json and AtlasFold's atlas_samples.json and atlas_floor.json",
    )
    m.add_argument("--out", type=Path, default=None)
    pr = sub_.add_parser("paired")
    pr.add_argument(
        "--fold-dir",
        type=Path,
        nargs="+",
        required=True,
        help="one or more sets' directories (each with summary.json, atlas_samples.json and atlas_floor.json); the first is read against the rest pooled",
    )
    pr.add_argument("--out", type=Path, default=None)
    rp = sub_.add_parser("repeat")
    rp.add_argument("--fold-dir", type=Path, required=True)
    rp.add_argument(
        "--repeat-dir",
        type=Path,
        required=True,
        help="the same records evaluated by AtlasFold at another seed (atlas_samples.json, atlas_floor.json, atlas_reference.json)",
    )
    rp.add_argument("--out", type=Path, default=None)
    c = sub_.add_parser("classes")
    c.add_argument(
        "--pilot",
        type=Path,
        action="append",
        required=True,
        help="summary.json of each pilot arm on the re-encoded route",
    )
    c.add_argument(
        "--control", type=Path, required=True, help="summary.json of the ESMC reference"
    )
    c.add_argument(
        "--set", action="append", required=True, help="SEED:TEMPERATURE=summary.json"
    )
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fold":
        run_fold(args)
    elif args.command == "summarize":
        run_summarize(args)
    elif args.command == "margin":
        run_margin(args)
    elif args.command == "paired":
        run_paired(args)
    elif args.command == "repeat":
        run_repeat(args)
    else:
        run_classes(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
