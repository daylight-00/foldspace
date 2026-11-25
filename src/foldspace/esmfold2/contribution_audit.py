"""What the contribution analysis counted, how it counted it, and how far the count moves.

The contribution analysis folds a sequence with the single of another protein and asks where the fold goes. Its
readings rest on four choices that a reader has to be
able to see, and on a classification that was added after the fact. This reads the stored
folds again, on a CPU, and fixes each of them:

``f``
    ``(1 - TM(swap, ref)) / (1 - TM(no prior, ref))``, one value per swap, defined only when
    removing the prior moves the fold by at least 0.02 in ``1 - TM``. It is a ratio of two
    quantities that are both close to 1 whenever both conditions destroy the fold, so it
    separates "the swap recovers the fold" (near 0) from "it does not" (near 1) and says little
    between. The difference ``TM(swap, ref) - TM(no prior, ref)`` is reported beside it.
the TM-score
    ``foldspace.evaluation.geometry.tm_score``: one global Kabsch superposition under the identity
    correspondence, ``d0`` from the target's length, normalised by the target's length. A lower
    bound on what a TM-score search finds, so a threshold on it is a strict one. Three others are
    read beside it: the evaluator's TM-score search (``refold.tm_score``),
    US-align's under the identity correspondence (``-TMscore 1``, the evaluator's scTM) and US-align's
    sequence-independent alignment (``-TMscore 0``), which does not assume that residue ``i`` of one
    fold is residue ``i`` of the other.
the classification
    A swap *follows the single* when its fold reaches TM >= 0.5 against the single's protein (its
    reference fold or its deposited structure, whichever is higher), *keeps the sequence's fold* when it
    does against the sequence's own, and is *neither* otherwise. Post hoc: the analysis was specified on ``f``, not
    this. The audit reports the counts at 0.3 to 0.7 and under each TM.
the unit
    A pair is the unit of independence: its two swaps share a pair of singles and a pair of lengths,
    so confidence intervals resample pairs.

It also asks whether the two proteins of a pair have folds alike enough that "follows the single" and
"keeps the sequence's own" could not be told apart, and how far apart the two singles of a swap are.

numpy only (US-align is optional, and the audit says so when it is missing).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = [
    "bootstrap_pairs",
    "classify",
    "f_values",
    "load_run",
    "score_rows",
    "swap_rows",
    "tm_backends",
]

SAME_FOLD = 0.5
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MIN_PRIOR_EFFECT = 0.02
CONDITIONS = ("AA", "BB", "AB", "BA", "Am", "Bm", "A0", "B0", "AA_seed")


def load_run(directory: Path) -> list[dict]:
    """The pairs of one contribution run: its first eight and its repeat, with their Cα."""
    directory = Path(directory)
    pairs: list[dict] = []
    for stem in ("contribution_det", "contribution_det_replicate"):
        path = directory / f"{stem}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        held = np.load(directory / f"{stem}_coords.npz", allow_pickle=False)
        arrays = {key: held[key] for key in held.files}  # each array read once
        for pair in record["pairs"]:
            k = pair["pair"]
            cas = {
                name: np.asarray(arrays[f"pair{k}_{name}"], dtype=np.float64)
                for name in (*CONDITIONS, "native_A", "native_B")
            }
            pairs.append({**pair, "run": stem, "cas": cas})
    if not pairs:
        raise FileNotFoundError(f"no contribution readings under {directory}")
    return pairs


def swap_rows(pairs: list[dict]) -> list[dict]:
    """One row per swap: the folds it is read against, and its confidence."""
    rows = []
    for index, p in enumerate(pairs):
        c = p["cas"]
        for swap, own, donor, none in (("AB", "A", "B", "A0"), ("BA", "B", "A", "B0")):
            rows.append(
                {
                    "pair": index,
                    "run": p["run"],
                    "swap": swap,
                    "length": p["length"],
                    "own_id": p[own],
                    "donor_id": p[donor],
                    "swap_ca": c[swap],
                    "none_ca": c[none],
                    "own_ref": c[own * 2],
                    "donor_ref": c[donor * 2],
                    "own_native": c[f"native_{own}"],
                    "donor_native": c[f"native_{donor}"],
                    "ptm": p["ptm"][swap],
                    "plddt": p["plddt"][swap],
                    "sequence_identity": p["sequence_identity"],
                }
            )
    return rows


# --------------------------------------------------------------------- TM


def _load_refold():
    here = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "refold", here / "src" / "foldspace" / "evaluation" / "refold.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _usalign(refold, binary: str, mobile, target, mode: int) -> float:
    """US-align's TM-score normalised by the target, in the mode asked for.

    ``-TMscore 1`` superposes under the identity correspondence (the evaluator's scTM);
    ``-TMscore 0`` aligns without assuming any correspondence.
    """
    with tempfile.TemporaryDirectory(prefix="contribution-usalign-") as work:
        work = Path(work)
        refold.write_ca_pdb(work / "mobile.pdb", mobile)
        refold.write_ca_pdb(work / "target.pdb", target)
        out = subprocess.run(
            [
                binary,
                str(work / "mobile.pdb"),
                str(work / "target.pdb"),
                "-TMscore",
                str(mode),
                "-outfmt",
                "2",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    row = None
    for line in out.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        row = line.split("\t")
    if row is None or len(row) < 11:
        raise RuntimeError(f"could not parse US-align output:\n{out.stdout}")
    return float(row[3])  # TM2: normalised by the second structure, the target


def tm_backends(binary: str | None = "USalign") -> dict:
    """Name -> ``f(mobile, target) -> TM normalised by the target's length``."""
    from foldspace.evaluation import geometry

    refold = _load_refold()
    backends = {
        "global_kabsch": lambda m, t: geometry.tm_score(m, t),
        "tmscore_search": lambda m, t: refold.tm_score(t, m)[0],
    }
    found = shutil.which(binary) if binary else None
    if found:
        backends["usalign_identity"] = lambda m, t: _usalign(refold, found, m, t, 1)
        backends["usalign_free"] = lambda m, t: _usalign(refold, found, m, t, 0)
    return backends


def score_rows(rows: list[dict], backends: dict) -> dict:
    """For each backend, the TM of every swap against each of the folds it is read against."""
    out = {}
    for name, tm in backends.items():
        out[name] = {
            "donor_ref": np.array([tm(r["swap_ca"], r["donor_ref"]) for r in rows]),
            "donor_native": np.array(
                [tm(r["swap_ca"], r["donor_native"]) for r in rows]
            ),
            "own_ref": np.array([tm(r["swap_ca"], r["own_ref"]) for r in rows]),
            "own_native": np.array([tm(r["swap_ca"], r["own_native"]) for r in rows]),
            "none_ref": np.array([tm(r["none_ca"], r["own_ref"]) for r in rows]),
        }
    return out


# -------------------------------------------------------------- readings


def classify(scores: dict, threshold: float = SAME_FOLD) -> dict:
    """Counts of swaps that follow the single, keep the sequence's fold, neither or both."""
    donor = np.maximum(scores["donor_ref"], scores["donor_native"]) >= threshold
    own = np.maximum(scores["own_ref"], scores["own_native"]) >= threshold
    return {
        "swaps": len(donor),
        "follow_the_single": int((donor & ~own).sum()),
        "keep_the_sequences_fold": int((own & ~donor).sum()),
        "both": int((donor & own).sum()),
        "neither": int((~donor & ~own).sum()),
    }


def f_values(scores: dict) -> dict:
    """``f`` per swap (NaN where undefined) and the difference of TMs."""
    swap = scores["own_ref"]
    none = scores["none_ref"]
    lost = 1.0 - none
    moved = 1.0 - swap
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(lost >= MIN_PRIOR_EFFECT, moved / lost, np.nan)
    return {"f": f, "delta_tm": swap - none, "lost": lost}


def _spread(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return {
        "n": len(x),
        "q25": float(q[0]),
        "median": float(q[1]),
        "q75": float(q[2]),
    }


def bootstrap_pairs(
    pair_of_row: np.ndarray, statistic, n: int = 10000, seed: int = 0
) -> dict:
    """A 95 % interval for ``statistic(row_indices)``, resampling pairs and not swaps.

    The two swaps of a pair share its two singles and its length, so the pair is the unit.
    """
    rng = np.random.default_rng(seed)
    pairs = np.unique(pair_of_row)
    members = {p: np.flatnonzero(pair_of_row == p) for p in pairs}
    draws = []
    for _ in range(n):
        chosen = rng.choice(pairs, size=len(pairs), replace=True)
        draws.append(statistic(np.concatenate([members[p] for p in chosen])))
    draws = np.array([d for d in draws if np.isfinite(d)])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"low": float(lo), "high": float(hi), "resamples": len(draws)}


def single_distances(rows: list[dict], targets: Path, stats: Path) -> np.ndarray:
    """Mean per-row distance between the two singles of a swap, in standardised units.

    Two independent rows of unit variance in 256 channels are about sqrt(2 * 256) apart, so the
    distance is divided by sqrt(256): identical singles give 0 and unrelated ones about 1.4.
    """
    from foldspace.esmfold2 import validate_interface

    norm = np.load(stats, allow_pickle=False)
    std = norm["std"]
    ids = sorted({r["own_id"] for r in rows} | {r["donor_id"] for r in rows})
    cached = validate_interface._cached_rows(Path(targets), ids)
    out = []
    for r in rows:
        a, b = cached[r["own_id"]], cached[r["donor_id"]]
        out.append(
            float(np.linalg.norm((a - b) / std, axis=1).mean() / np.sqrt(a.shape[1]))
        )
    return np.array(out)


def audit(
    runs: list[Path], *, binary: str | None, targets: Path | None, stats: Path | None
) -> dict:
    pairs = [p for run in runs for p in load_run(run)]
    rows = swap_rows(pairs)
    backends = tm_backends(binary)
    scores = score_rows(rows, backends)
    pair_of_row = np.array([r["pair"] for r in rows])
    n_pairs = len(pairs)

    g = scores["global_kabsch"]
    fv = f_values(g)
    undefined = [
        {
            "pair": int(rows[i]["pair"]),
            "swap": rows[i]["swap"],
            "length": rows[i]["length"],
            "sequence": rows[i]["own_id"],
            "tm_no_prior_to_ref": float(g["none_ref"][i]),
        }
        for i in np.flatnonzero(~np.isfinite(fv["f"]))
    ]

    def median_f(idx):
        return (
            float(np.nanmedian(fv["f"][idx]))
            if np.isfinite(fv["f"][idx]).any()
            else float("nan")
        )

    def follow_share(idx):
        c = classify({k: v[idx] for k, v in g.items()})
        return c["follow_the_single"] / c["swaps"]

    ambiguity = np.array(
        [backends["global_kabsch"](r["own_ref"], r["donor_ref"]) for r in rows]
    )
    ambiguity_native = np.array(
        [backends["global_kabsch"](r["own_native"], r["donor_native"]) for r in rows]
    )
    classification = {
        name: {f"{t:g}": classify(s, t) for t in THRESHOLDS}
        for name, s in scores.items()
    }
    # the pairs' two swaps, side by side
    kinds = np.where(
        np.maximum(g["donor_ref"], g["donor_native"]) >= SAME_FOLD,
        "follow",
        np.where(
            np.maximum(g["own_ref"], g["own_native"]) >= SAME_FOLD, "keep", "neither"
        ),
    )
    pair_level: dict[str, int] = {}
    for p in range(n_pairs):
        key = "+".join(sorted(kinds[pair_of_row == p]))
        pair_level[key] = pair_level.get(key, 0) + 1

    report = {
        "runs": [str(r) for r in runs],
        "pairs": n_pairs,
        "swaps": len(rows),
        "backends": list(backends),
        "usalign_available": "usalign_identity" in backends,
        "tm_definition": "TM of the first fold against the second, normalised by the second's length; "
        "identity correspondence for global_kabsch, tmscore_search and usalign_identity",
        "f": {
            "definition": "(1 - TM(swap, ref)) / (1 - TM(no prior, ref)); undefined when 1 - TM(no prior, ref) < 0.02",
            "defined": int(np.isfinite(fv["f"]).sum()),
            "undefined": undefined,
            "spread": _spread(fv["f"]),
            "median_ci95_over_pairs": bootstrap_pairs(pair_of_row, median_f),
            "by_backend": {
                name: _spread(f_values(s)["f"]) for name, s in scores.items()
            },
        },
        "delta_tm": {
            "definition": "TM(swap, ref) - TM(no prior, ref)",
            "spread": _spread(fv["delta_tm"]),
        },
        "tm_swap_to_own_ref": {
            name: _spread(s["own_ref"]) for name, s in scores.items()
        },
        "classification_by_threshold": classification,
        "follow_share_ci95_over_pairs": bootstrap_pairs(
            pair_of_row, follow_share, seed=1
        ),
        "pair_level_outcomes": pair_level,
        "ambiguity": {
            "tm_between_the_two_reference_folds": _spread(ambiguity),
            "tm_between_the_two_deposited_structures": _spread(ambiguity_native),
            "swaps_with_reference_folds_alike_at_0.3": int((ambiguity >= 0.3).sum()),
            "swaps_with_reference_folds_alike_at_0.4": int((ambiguity >= 0.4).sum()),
            "swaps_with_reference_folds_alike_at_0.5": int((ambiguity >= 0.5).sum()),
            "max_sequence_identity_of_a_pair": float(
                max(r["sequence_identity"] for r in rows)
            ),
        },
        "confidence_by_outcome": {
            kind: {
                "n": int((kinds == kind).sum()),
                "ptm": _spread(
                    [r["ptm"] for r, k in zip(rows, kinds, strict=True) if k == kind]
                ),
                "plddt": _spread(
                    [r["plddt"] for r, k in zip(rows, kinds, strict=True) if k == kind]
                ),
            }
            for kind in ("follow", "keep", "neither")
        },
    }
    if targets is not None and stats is not None and Path(targets).exists():
        d = single_distances(rows, targets, stats)
        report["single_distance"] = {
            "unit": "mean row distance between the swap's two singles / sqrt(256) in standardised units",
            "by_outcome": {
                kind: _spread(d[kinds == kind])
                for kind in ("follow", "keep", "neither")
            },
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--usalign", default="USalign")
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()
    report = audit(
        args.runs, binary=args.usalign, targets=args.targets, stats=args.stats
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
