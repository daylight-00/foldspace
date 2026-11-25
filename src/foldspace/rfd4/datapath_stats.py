"""The arithmetic of the data-path check: which chains are read, how a reconstruction is scored, what the
native latent looks like. NumPy only, so that every number the check reports is a function tested without the model.

`src/foldspace/rfd4/ae_datapath.py` runs RFD4-Proteina's own pipeline and autoencoder over the chains chosen here and calls these functions on
what comes out. Nothing here knows a model, a file format or a path.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

#: the length strata of the corpus (60 to 300 residues): each holds about a quarter of the chains
LENGTH_EDGES = (60, 110, 176, 245, 301)
#: what a chain can be that a plain random draw rarely shows, named by the column of the corpus's split table that finds it
STRESS = {
    "nonstandard": lambda row: int(row["nonstandard_in_sequence"]) > 0,
    "breaks": lambda row: int(row["frame_breaks"]) > 0,
    "copies": lambda row: int(row["polymer_chains"]) > 1,
    "unobserved": lambda row: float(row["coverage"]) < 0.95,
}


def choose_chains(
    rows: list[dict],
    lengths: dict[str, int],
    *,
    per_bin: int = 32,
    stress: int = 16,
    seed: int = 20261111,
    edges: tuple[int, ...] = LENGTH_EDGES,
) -> dict:
    """`per_bin` chains from each length stratum and `stress` from each pool of unusual chains, the pools drawn after the strata and
    from what they left, so that no chain is chosen twice.

    `rows` are the corpus's split table (`data/corpus/split.tsv.gz`), `lengths` maps `<entry>_<entity>` to the
    length of that chain in the corpus array. Returns `{"ids": [...], "why": {id: stratum or pool}}`, the ids in corpus order of
    `rows`, so that the choice is a function of the table and the seed and of nothing else.
    """
    rng = np.random.default_rng(seed)
    by_id = {f"{row['entry']}_{row['entity']}": row for row in rows}
    names = [name for name in by_id if name in lengths]
    chosen: dict[str, str] = {}
    for low, high in pairwise(edges):
        pool = [n for n in names if low <= lengths[n] < high and n not in chosen]
        pick = rng.choice(len(pool), size=min(per_bin, len(pool)), replace=False)
        for i in sorted(int(i) for i in pick):
            chosen[pool[i]] = f"length {low}-{high - 1}"
    for pool_name, test in STRESS.items():
        pool = [n for n in names if n not in chosen and test(by_id[n])]
        pick = rng.choice(len(pool), size=min(stress, len(pool)), replace=False)
        for i in sorted(int(i) for i in pick):
            chosen[pool[i]] = pool_name
    order = {name: i for i, name in enumerate(by_id)}
    ids = sorted(chosen, key=order.__getitem__)
    return {"ids": ids, "why": {name: chosen[name] for name in ids}}


def kabsch_rmsd(moving: np.ndarray, fixed: np.ndarray) -> float:
    """RMSD of `moving` onto `fixed` ([n, 3] each) after the best rotation and translation."""
    p = np.asarray(moving, dtype=np.float64)
    q = np.asarray(fixed, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"two [n, 3] arrays of one shape, not {p.shape} and {q.shape}")
    p = p - p.mean(0)
    q = q - q.mean(0)
    u, s, vt = np.linalg.svd(p.T @ q)
    sign = np.sign(np.linalg.det(u @ vt))
    s[-1] *= sign
    return float(
        np.sqrt(max((p**2).sum() + (q**2).sum() - 2.0 * s.sum(), 0.0) / len(p))
    )


def residue_scores(
    target: np.ndarray,
    predicted: np.ndarray,
    target_mask: np.ndarray,
    residues: np.ndarray,
) -> dict:
    """The distance of predicted atoms from the target's, in the units of the arrays (angstroms), over the atoms the target has.

    `target`, `predicted` are `[n, 37, 3]` in one frame (the decoder is given the Cα, so there is nothing to superpose),
    `target_mask` `[n, 37]` marks the atoms the structure has and `residues` `[n]` is the set of residues scored. Per residue:
    the root mean square over its atoms; reported as the mean and the median over residues, as the root mean square over all atoms
    (the larger, dominated by the long side chains) and for the backbone slots N, Cα, C, O and the side chain apart.
    """
    t = np.asarray(target, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    m = np.asarray(target_mask, dtype=bool) & np.asarray(residues, dtype=bool)[:, None]
    if not m.any():
        return {"residues": 0}
    # the target of an atom the structure lacks may be NaN: it is dropped by `where`, not multiplied by zero
    sq = np.where(m, ((t - p) ** 2).sum(-1), 0.0)
    count = m.sum(-1)
    scored = count > 0
    per_residue = np.sqrt(sq.sum(-1)[scored] / count[scored])
    backbone = np.zeros(37, dtype=bool)
    backbone[[0, 1, 2, 4]] = True
    side = m & ~backbone
    bb = m & backbone
    return {
        "residues": int(scored.sum()),
        "atoms": int(m.sum()),
        "per_residue_mean": float(per_residue.mean()),
        "per_residue_median": float(np.median(per_residue)),
        "per_residue_p95": float(np.quantile(per_residue, 0.95)),
        "all_atom_rms": float(np.sqrt(sq.sum() / m.sum())),
        "backbone_rms": float(np.sqrt(np.where(bb, sq, 0.0).sum() / max(bb.sum(), 1))),
        "side_chain_rms": float(
            np.sqrt(np.where(side, sq, 0.0).sum() / max(side.sum(), 1))
        ),
        "side_chain_atoms": int(side.sum()),
    }


def presence_scores(
    target_mask: np.ndarray,
    predicted_mask: np.ndarray,
    legal: np.ndarray,
    residues: np.ndarray,
) -> dict:
    """Precision, recall and F1 of the atoms the decoder says are present, over the slots a residue may have (`legal`, `[n, 37]`)."""
    keep = np.asarray(legal, dtype=bool) & np.asarray(residues, dtype=bool)[:, None]
    t = np.asarray(target_mask, dtype=bool) & keep
    p = np.asarray(predicted_mask, dtype=bool) & keep
    tp = float((t & p).sum())
    precision = tp / max(float(p.sum()), 1.0)
    recall = tp / max(float(t.sum()), 1.0)
    f1 = 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_atoms": int(t.sum()),
    }


def sequence_scores(
    true: np.ndarray, predicted: np.ndarray, residues: np.ndarray, n_types: int
) -> dict:
    """Accuracy of the predicted residue types over the residues scored, and the recall of each type that occurs."""
    keep = np.asarray(residues, dtype=bool)
    t = np.asarray(true)[keep]
    p = np.asarray(predicted)[keep]
    if t.size == 0:
        return {"residues": 0}
    recall = {
        int(k): float((p[t == k] == k).mean()) for k in range(n_types) if (t == k).any()
    }
    return {
        "residues": int(t.size),
        "accuracy": float((t == p).mean()),
        "recall_by_type": recall,
    }


def latent_summary(
    mean: np.ndarray,
    log_scale: np.ndarray,
    residue_type: np.ndarray,
    sample: np.ndarray | None = None,
    active_nats: float = 0.1,
) -> dict:
    """What the encoder's posterior looks like over a set of residues: `mean`, `log_scale` `[n, d]`, `residue_type` `[n]`.

    * per channel: the mean and spread of the posterior mean, the typical posterior scale `exp(log_scale)`, and the ratio of the
      two (how far apart residues are next to how far one residue's own posterior reaches);
    * the KL to N(0, I) per channel, `0.5 (m^2 + s^2 - 1 - 2 log s)`, averaged over residues, and the number of channels above
      `active_nats` (the autoencoder's own measure of a used channel);
    * the covariance of the posterior mean: its eigenvalues, the participation ratio `(sum l)^2 / sum l^2` (how many directions the
      mean really spans) and the largest correlation between two channels;
    * how much of the posterior mean the residue type alone fixes: the share of each channel's variance between the types.
    """
    m = np.asarray(mean, dtype=np.float64)
    ls = np.asarray(log_scale, dtype=np.float64)
    rt = np.asarray(residue_type)
    n, d = m.shape
    sigma = np.exp(ls)
    kl = 0.5 * (m**2 + sigma**2 - 1.0 - 2.0 * ls)
    cov = np.cov(m, rowvar=False)
    eig = np.linalg.eigvalsh(cov)[::-1]
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 0.0)
    total = ((m - m.mean(0)) ** 2).sum(0)
    within = np.zeros(d)
    for k in np.unique(rt):
        group = m[rt == k]
        within += ((group - group.mean(0)) ** 2).sum(0)
    between = 1.0 - within / np.maximum(total, 1e-300)
    out = {
        "residues": int(n),
        "channels": int(d),
        "mean_of_mean": m.mean(0).tolist(),
        "std_of_mean": m.std(0).tolist(),
        "scale_mean": sigma.mean(0).tolist(),
        "scale_median": np.median(sigma, axis=0).tolist(),
        "separation": (m.std(0) / sigma.mean(0)).tolist(),
        "kl_per_channel": kl.mean(0).tolist(),
        "kl_total": float(kl.sum(-1).mean()),
        "active_channels": int((kl.mean(0) > active_nats).sum()),
        "covariance_eigenvalues": eig.tolist(),
        "participation_ratio": float(eig.sum() ** 2 / (eig**2).sum()),
        "max_abs_correlation": float(np.abs(corr).max()),
        "variance_between_residue_types": between.tolist(),
        "variance_between_residue_types_overall": float(
            1.0 - within.sum() / max(total.sum(), 1e-300)
        ),
        "overall_std_of_mean": float(m.std()),
        "overall_scale_mean": float(sigma.mean()),
    }
    if sample is not None:
        z = np.asarray(sample, dtype=np.float64)
        out["std_of_sample"] = z.std(0).tolist()
        out["overall_std_of_sample"] = float(z.std())
    return out
