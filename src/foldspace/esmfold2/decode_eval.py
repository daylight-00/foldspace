"""How well a decoder reads residue types back, on chains whose clusters it never saw: numpy only.

The input is what a decoder emits for a set of whole chains: log-probabilities of the 20 residue
types per residue, the true types, and where each chain starts. Everything the decoder is judged by
is computed here, in one place, from those arrays.

**Cluster-weighted is the primary reading.** The dev split is 400 clusters, and one of them holds
1,070 of its 3,509 chains (30.5 %); a micro-averaged accuracy is therefore largely that one family's.
So each cluster counts once (its residues pooled inside it), and the micro figures are reported beside,
with and without the largest cluster. Checkpoints are chosen by the cluster-weighted cross-entropy.

Beside accuracy: chain-exact recovery (a chain with every residue right), the substitutions a chain
gets, accuracy by length, by distance from either end, per residue type with the confusions that
cost most, and calibration (a decoder that is sure when wrong is worse than one that is not).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

ALPHABET = "ARNDCQEGHILKMFPSTWYV"
LENGTH_EDGES = (60, 100, 140, 180, 220, 260, 10_000)
TERMINAL_OFFSETS = 8

__all__ = ["ALPHABET", "selection_stats", "summarize"]


def _row_values(logp: np.ndarray, labels: np.ndarray):
    rows = np.arange(len(labels))
    nll = -logp[rows, labels].astype(np.float64)
    predicted = logp.argmax(axis=1)
    return nll, predicted


def _per_chain(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, starts)


def _per_cluster(
    values: np.ndarray, inverse: np.ndarray, n_clusters: int
) -> np.ndarray:
    return np.bincount(inverse, weights=values, minlength=n_clusters)


def selection_stats(
    logp: np.ndarray,
    labels: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    clusters: np.ndarray,
) -> dict:
    """The few numbers a checkpoint is chosen by: cluster-weighted cross-entropy and accuracy."""
    nll, predicted = _row_values(logp, labels)
    correct = (predicted == labels).astype(np.float64)
    _, inverse = np.unique(clusters, return_inverse=True)
    k = int(inverse.max()) + 1
    n = _per_cluster(lengths.astype(np.float64), inverse, k)
    return {
        "cw_ce": float((_per_cluster(_per_chain(nll, starts), inverse, k) / n).mean()),
        "cw_acc": float(
            (_per_cluster(_per_chain(correct, starts), inverse, k) / n).mean()
        ),
        "micro_ce": float(nll.mean()),
        "micro_acc": float(correct.mean()),
    }


def summarize(
    logp: np.ndarray,
    labels: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    clusters: np.ndarray,
    *,
    top_confusions: int = 10,
    calibration_bins: int = 15,
) -> dict:
    """Every reading of a decoder on one split."""
    logp = np.asarray(logp)
    labels = np.asarray(labels, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    clusters = np.asarray(clusters)
    if logp.shape != (len(labels), len(ALPHABET)):
        raise ValueError(f"logp {logp.shape} for {len(labels)} residues")
    if int(lengths.sum()) != len(labels):
        raise ValueError("lengths do not cover the residues")

    nll, predicted = _row_values(logp, labels)
    correct = predicted == labels
    top2 = (np.argsort(-logp, axis=1)[:, :2] == labels[:, None]).any(axis=1)
    n_chains = len(lengths)
    unique, inverse, cluster_sizes = np.unique(
        clusters, return_inverse=True, return_counts=True
    )
    k = len(unique)

    chain_correct = _per_chain(correct.astype(np.float64), starts)
    chain_nll = _per_chain(nll, starts)
    chain_acc = chain_correct / lengths
    exact = chain_correct == lengths
    substitutions = lengths - chain_correct

    residues_per_cluster = _per_cluster(lengths.astype(np.float64), inverse, k)
    cluster_acc = _per_cluster(chain_correct, inverse, k) / residues_per_cluster
    cluster_ce = _per_cluster(chain_nll, inverse, k) / residues_per_cluster
    cluster_exact = _per_cluster(exact.astype(np.float64), inverse, k) / cluster_sizes

    largest = int(np.argmax(cluster_sizes))
    in_largest = inverse == largest
    rest = ~in_largest
    rest_rows = np.repeat(rest, lengths)
    largest_rows = np.repeat(in_largest, lengths)

    # by length (per residue, over the chain it belongs to)
    row_chain_length = np.repeat(lengths, lengths)
    by_length = {}
    for lo, hi in pairwise(LENGTH_EDGES):
        take = (row_chain_length >= lo) & (row_chain_length < hi)
        if take.any():
            label = f"{lo}-{hi - 1}" if hi < 10_000 else f"{lo}+"
            by_length[label] = {
                "residues": int(take.sum()),
                "acc": float(correct[take].mean()),
            }

    # by distance from either end
    position = np.arange(len(labels)) - np.repeat(starts, lengths)
    from_c = np.repeat(lengths, lengths) - 1 - position
    terminal = {
        "N": [float(correct[position == j].mean()) for j in range(TERMINAL_OFFSETS)],
        "C": [float(correct[from_c == j].mean()) for j in range(TERMINAL_OFFSETS)],
    }
    interior = (position >= TERMINAL_OFFSETS) & (from_c >= TERMINAL_OFFSETS)
    terminal["interior"] = (
        float(correct[interior].mean()) if interior.any() else float("nan")
    )

    # per residue type, and the confusions that cost most
    k_types = len(ALPHABET)
    confusion = np.zeros((k_types, k_types), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    per_residue = {}
    for i, letter in enumerate(ALPHABET):
        support = int(confusion[i].sum())
        called = int(confusion[:, i].sum())
        per_residue[letter] = {
            "n": support,
            "recall": float(confusion[i, i] / support) if support else float("nan"),
            "precision": float(confusion[i, i] / called) if called else float("nan"),
        }
    off = confusion.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:top_confusions]
    confusions = []
    for f in flat:
        t, p = divmod(int(f), k_types)
        if off[t, p] == 0:
            break
        confusions.append(
            {
                "true": ALPHABET[t],
                "predicted": ALPHABET[p],
                "count": int(off[t, p]),
                "share_of_true": float(off[t, p] / confusion[t].sum()),
            }
        )

    # calibration: confidence is the largest probability
    confidence = np.exp(logp.max(axis=1))
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    bin_of = np.clip(np.digitize(confidence, edges[1:-1]), 0, calibration_bins - 1)
    ece = 0.0
    bins = []
    for b in range(calibration_bins):
        take = bin_of == b
        if not take.any():
            continue
        gap = abs(float(correct[take].mean()) - float(confidence[take].mean()))
        ece += take.mean() * gap
        bins.append(
            {
                "low": float(edges[b]),
                "n": int(take.sum()),
                "confidence": float(confidence[take].mean()),
                "accuracy": float(correct[take].mean()),
            }
        )
    sure = confidence >= 0.9

    return {
        "residues": len(labels),
        "chains": int(n_chains),
        "clusters": int(k),
        "micro": {
            "ce": float(nll.mean()),
            "acc": float(correct.mean()),
            "top2_acc": float(top2.mean()),
        },
        "cluster_weighted": {
            "ce": float(cluster_ce.mean()),
            "acc": float(cluster_acc.mean()),
            "chain_exact_rate": float(cluster_exact.mean()),
        },
        "chain_weighted": {
            "acc": float(chain_acc.mean()),
            "chain_exact_rate": float(exact.mean()),
        },
        "without_largest_cluster": {
            "chains": int(rest.sum()),
            "acc": float(correct[rest_rows].mean()) if rest.any() else float("nan"),
            "ce": float(nll[rest_rows].mean()) if rest.any() else float("nan"),
        },
        "largest_cluster": {
            "chains": int(in_largest.sum()),
            "share_of_chains": float(in_largest.mean()),
            "acc": float(correct[largest_rows].mean()),
        },
        "substitutions_per_chain": {
            "mean": float(substitutions.mean()),
            "median": float(np.median(substitutions)),
            "p90": float(np.quantile(substitutions, 0.9)),
            "max": int(substitutions.max()),
        },
        "by_length": by_length,
        "by_terminal_offset": terminal,
        "per_residue": per_residue,
        "top_confusions": confusions,
        "calibration": {
            "ece": float(ece),
            "mean_confidence": float(confidence.mean()),
            "share_at_least_0.9": float(sure.mean()),
            "accuracy_when_at_least_0.9": float(correct[sure].mean())
            if sure.any()
            else float("nan"),
            "bins": bins,
        },
    }
