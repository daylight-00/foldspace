"""Which chains a step sees, and how they are packed: numpy only.

The sequence head reads one residue at a time out of whole-chain features, so a batch is a set of
whole chains padded to the longest, and the cost of a step is the padded tokens, not the chains.
Chains are drawn by **a cluster
uniformly, then a chain inside it uniformly**, because the train split is 3,373 30 % clusters of
which one holds 981 chains, and chain-uniform draws hand the loss to a few families. The other
policy (chain-uniform) is kept as the one ablation.

Packing sorts the drawn chains by length with a little noise and fills batches up to a padded-token
budget, then shuffles the batches: the lengths in a batch are alike, so little is padding, and the
order of batches is not the order of lengths.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cluster_probabilities", "draw", "pack"]


def cluster_probabilities(clusters: np.ndarray) -> np.ndarray:
    """Per-chain draw probability for 'a cluster uniformly, then a chain of it uniformly'."""
    clusters = np.asarray(clusters)
    _, inverse, counts = np.unique(clusters, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse]
    return weight / weight.sum()


def draw(
    n_chains: int,
    n_draws: int,
    rng: np.random.Generator,
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    """``n_draws`` chain indices, with replacement; uniform when no probabilities are given."""
    if probabilities is not None and len(probabilities) != n_chains:
        raise ValueError(f"{len(probabilities)} probabilities for {n_chains} chains")
    return rng.choice(n_chains, size=n_draws, replace=True, p=probabilities)


def pack(
    indices: np.ndarray,
    lengths: np.ndarray,
    max_tokens: int,
    rng: np.random.Generator | None = None,
    *,
    noise: float = 4.0,
) -> list[np.ndarray]:
    """Batches of whole chains, each at most ``max_tokens`` padded tokens.

    With a generator the chains are sorted by length plus noise and the batches shuffled; without
    one the order is exactly by length (what evaluation wants: deterministic). A chain longer than
    the budget forms a batch alone rather than being dropped.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return []
    key = lengths[indices].astype(np.float64)
    if rng is not None:
        key = key + rng.uniform(-noise, noise, size=len(indices))
    ordered = indices[np.argsort(key, kind="stable")]

    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for i in ordered:
        n = int(lengths[i])
        if current and max(longest, n) * (len(current) + 1) > max_tokens:
            batches.append(current)
            current, longest = [], 0
        current.append(int(i))
        longest = max(longest, n)
    batches.append(current)

    out = [np.array(b, dtype=np.int64) for b in batches]
    if rng is not None:
        out = [out[j] for j in rng.permutation(len(out))]
    return out
