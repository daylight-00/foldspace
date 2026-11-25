"""The dev panel the structure check is made on, and the controls it is read against: numpy only.

The question is whether a sequence read back from a single still folds. That is asked on a fixed
small panel of dev chains rather than on the whole split: a fold costs about 20 seconds. The panel is chosen from the
dev metadata alone (length and cluster), before any decoder is read, so that it cannot be picked to flatter one.

The **control**: a decoder's errors are not random errors. A conservative substitution costs a fold less than a
random one, and the question that matters is whether the decoder's are cheaper than chance would make them. So each
decoded chain has a twin with the *same number* of substitutions at random positions, to residues drawn from the
dev composition (never the true one); both are folded the same way.

Selection is by hashing the ids with a seed, not by anything that depends on a model.
"""

from __future__ import annotations

import numpy as np

from foldspace.esmfold2.decode_eval import ALPHABET

__all__ = ["random_substitutions", "select_panel", "substitutions"]


def select_panel(
    ids: list[str],
    lengths: np.ndarray,
    clusters: np.ndarray,
    *,
    n: int = 24,
    lo: int = 70,
    hi: int = 220,
    seed: int = 20261120,
) -> list[int]:
    """Indices of ``n`` chains spread over ``[lo, hi]`` residues, at most one from any cluster."""
    lengths = np.asarray(lengths)
    clusters = np.asarray(clusters)
    rng = np.random.default_rng(seed)
    best: dict[int, int] = {}
    for i in rng.permutation(len(ids)):
        if lo <= lengths[i] <= hi and int(clusters[i]) not in best:
            best[int(clusters[i])] = int(i)
    candidates = sorted(best.values(), key=lambda i: (int(lengths[i]), ids[i]))
    if len(candidates) < n:
        raise ValueError(
            f"{len(candidates)} clusters have a chain of {lo} to {hi} residues, {n} asked for"
        )
    positions = np.linspace(0, len(candidates) - 1, n).round().astype(int)
    return [candidates[p] for p in positions]


def substitutions(true: str, decoded: str) -> list[int]:
    """Positions at which two sequences of one length differ."""
    if len(true) != len(decoded):
        raise ValueError(f"{len(true)} and {len(decoded)} residues")
    return [i for i, (a, b) in enumerate(zip(true, decoded, strict=True)) if a != b]


def random_substitutions(
    true: str, k: int, rng: np.random.Generator, composition: np.ndarray | None = None
) -> str:
    """``true`` with exactly ``k`` residues replaced, at random positions, by random other types.

    ``composition`` is the weight of each of the twenty types (uniform when None); the true residue is
    excluded at each position.
    """
    if not 0 <= k <= len(true):
        raise ValueError(f"{k} substitutions in a chain of {len(true)}")
    weights = (
        np.ones(len(ALPHABET))
        if composition is None
        else np.asarray(composition, dtype=np.float64)
    )
    out = list(true)
    for position in rng.choice(len(true), size=k, replace=False):
        w = weights.copy()
        w[ALPHABET.index(true[position])] = 0.0
        out[position] = ALPHABET[int(rng.choice(len(ALPHABET), p=w / w.sum()))]
    return "".join(out)
