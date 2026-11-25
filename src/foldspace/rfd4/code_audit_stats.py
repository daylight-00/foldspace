"""The statistics of the representation audit (numpy only; `src/foldspace/rfd4/code_audit.py` makes the arrays and calls them).

The audit asks what the learned code of an ESM-derived latent holds, against the release encoder's own identity channels, without training anything:

* **Alignment.** The per-type centroids of a code's identity channels against the release encoder's: the R² of the best similarity transform (rotation or reflection, one scale, a translation), with the
  permutation null of the type labels and the ceiling that two halves of the same chains give (`similarity_r2`, `permutation_null`, `split_half_ceiling`).
* **Context in the residual.** What is left of a channel once the residue's type is removed, regressed on the neighbours' types and on the Cα geometry with the folds drawn by chain
  (`context_features`, `grouped_ridge_r2`), and a shuffled-feature null.
* **The information dimension.** The share of a channel's variance within residue types and the participation ratio of the within-type residual covariance (`within_type_share`,
  `participation_ratio`).
* **The regularizers.** The exact Gaussian KL of a posterior `N(sqrt(1 - sigma^2) mu, sigma^2)` to `N(0, 1)`, the rate term `-ln sigma` and the term between them (`kl_exact`, `kl_rate`, `kl_omitted`).

Residue types are integers `0 .. n_types - 1`; every function takes the residues of all chains in one array and the chain of each residue where it matters.
"""

from __future__ import annotations

import numpy as np

N_TYPES = 20


def type_centroids(
    x: np.ndarray, t: np.ndarray, n_types: int = N_TYPES
) -> tuple[np.ndarray, np.ndarray]:
    """`[n_types, d]` mean of `x` over the residues of each type and the number of residues of each type (a type with none has a row of NaN)."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t)
    counts = np.bincount(t, minlength=n_types)[:n_types].astype(np.float64)
    sums = np.zeros((n_types, x.shape[1]))
    np.add.at(sums, t, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts[:, None], counts


def within_type_residual(
    x: np.ndarray, t: np.ndarray, n_types: int = N_TYPES
) -> np.ndarray:
    """`x` minus the centroid of the residue's own type."""
    centroids, _ = type_centroids(x, t, n_types)
    return np.asarray(x, dtype=np.float64) - centroids[t]


def within_type_share(
    x: np.ndarray, t: np.ndarray, n_types: int = N_TYPES
) -> np.ndarray:
    """Per channel, the share of its variance that lies within residue types (one minus eta squared): what the type does not fix."""
    x = np.asarray(x, dtype=np.float64)
    resid = within_type_residual(x, t, n_types)
    total = ((x - x.mean(0)) ** 2).sum(0)
    return (resid**2).sum(0) / total


def participation_ratio(cov: np.ndarray) -> float:
    """`(sum of the eigenvalues)^2 / sum of their squares` of a covariance: the number of directions it spreads over (the dimension if isotropic, one if it is a line)."""
    values = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))
    values = np.clip(values, 0.0, None)
    return float(values.sum() ** 2 / (values**2).sum())


def similarity_r2(target: np.ndarray, source: np.ndarray) -> float:
    """The R² of the best similarity transform of `source` onto `target` (both `[n, d]`, rows paired): a rotation or reflection, one uniform scale and a translation, `1 - SSE / SST` with `SST` the target's
    sum of squares about its centroid. Closed form (orthogonal Procrustes with a scale): `SSE = |a|^2 - (sum of the singular values of b^T a)^2 / |b|^2` for the centred `a` and `b`."""
    a = np.asarray(target, dtype=np.float64)
    b = np.asarray(source, dtype=np.float64)
    a = a - a.mean(0)
    b = b - b.mean(0)
    sst = float((a**2).sum())
    norm_b = float((b**2).sum())
    if sst == 0.0:
        raise ValueError("the target has no spread")
    if norm_b == 0.0:
        return 0.0
    singular = np.linalg.svd(b.T @ a, compute_uv=False)
    return float(singular.sum() ** 2 / (norm_b * sst))


def affine_r2(target: np.ndarray, source: np.ndarray) -> float:
    """The R² of the best affine map of `source` onto `target` (least squares, an intercept): the optimistic comparison, with `d * (d + 1)` parameters against the `n * d` numbers it fits."""
    a = np.asarray(target, dtype=np.float64)
    b = np.asarray(source, dtype=np.float64)
    design = np.hstack([b, np.ones((len(b), 1))])
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    sst = float(((a - a.mean(0)) ** 2).sum())
    return float(1.0 - ((a - design @ coef) ** 2).sum() / sst)


def permutation_null(
    target: np.ndarray,
    source: np.ndarray,
    n: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """`similarity_r2` of `target` against `source` with the rows of `source` permuted `n` times: what an alignment of unrelated codes of these sizes reaches."""
    rng = np.random.default_rng(seed)
    return np.array(
        [similarity_r2(target, source[rng.permutation(len(source))]) for _ in range(n)]
    )


def split_half_ceiling(
    x: np.ndarray,
    t: np.ndarray,
    chain: np.ndarray,
    n_types: int = N_TYPES,
    seed: int = 0,
    repeats: int = 20,
) -> np.ndarray:
    """`similarity_r2` of the centroids from one random half of the chains against the other half's, `repeats` times: the most that two codes of the same kind can agree, given the noise of centroids that
    are means of a few thousand residues."""
    rng = np.random.default_rng(seed)
    chains = np.unique(chain)
    out = []
    for _ in range(repeats):
        half = set(rng.permutation(chains)[: len(chains) // 2].tolist())
        left = np.array([c in half for c in chain.tolist()])
        a, _ = type_centroids(x[left], t[left], n_types)
        b, _ = type_centroids(x[~left], t[~left], n_types)
        keep = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
        out.append(similarity_r2(a[keep], b[keep]))
    return np.array(out)


def one_hot(types: np.ndarray, n_types: int = N_TYPES) -> np.ndarray:
    """`[n, n_types]` rows of zeros and a one; a negative type (outside the chain, or not a standard residue) is a row of zeros."""
    types = np.asarray(types)
    out = np.zeros((len(types), n_types))
    inside = (types >= 0) & (types < n_types)
    out[np.flatnonzero(inside), types[inside]] = 1.0
    return out


def virtual_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The angle at `b` of the points `a, b, c` (radians)."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The dihedral of the points `a, b, c, d` (radians)."""
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    x = (v * w).sum(-1)
    y = (np.cross(b1, v) * w).sum(-1)
    return np.arctan2(y, x)


def chain_features(
    types: np.ndarray, ca: np.ndarray, valid: np.ndarray, window: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The context features of the residues of one chain: `(sequence, structure, usable)`.

    `types` are the residues' types (negative where not a standard residue), `ca` their Cα `[L, 3]` and `valid` whether a residue has one. A residue is *usable* when it and the `window` residues on each side
    have a Cα (the chain's ends are left out). **sequence**: the one-hot types of the neighbours at -window .. -1 and +1 .. +window (`2 * window * 20`). **structure**: the Cα(i)-Cα(i-2), Cα(i)-Cα(i+2) and
    Cα(i)-Cα(i+3) distances, the virtual bond angle at i, the sine and cosine of the dihedral over i-1 .. i+2, a helix and a strand indicator (the i, i+3 distance within 4.8 and 5.8 Å; the i, i+2 distance above
    6.0 Å), the number of Cα within 10 Å (the neighbours within two residues excluded), the relative position and the two chain-end flags (12 numbers).
    """
    n = len(types)
    index = np.arange(n)
    usable = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        usable[i] = bool(valid[i - window : i + window + 1].all())
    seq = np.zeros((n, 2 * window * N_TYPES))
    offsets = [k for k in range(-window, window + 1) if k != 0]
    for slot, k in enumerate(offsets):
        at = index + k
        inside = (at >= 0) & (at < n)
        t = np.where(inside, types[np.clip(at, 0, n - 1)], -1)
        seq[:, slot * N_TYPES : (slot + 1) * N_TYPES] = one_hot(t)
    structure = np.zeros((n, 12))
    rows = np.flatnonzero(usable)
    if len(rows):
        c = ca[rows]
        m1, p1, m2, p2, p3 = (ca[rows + k] for k in (-1, 1, -2, 2, 3))
        d2m = np.linalg.norm(c - m2, axis=-1)
        d2p = np.linalg.norm(p2 - c, axis=-1)
        d3p = np.linalg.norm(p3 - c, axis=-1)
        theta = virtual_angle(m1, c, p1)
        phi = dihedral(m1, c, p1, p2)
        dist = np.linalg.norm(ca[valid][None, :, :] - c[:, None, :], axis=-1)
        position_of_valid = index[valid]
        far = np.abs(position_of_valid[None, :] - rows[:, None]) > 2
        burial = ((dist < 10.0) & far).sum(1)
        structure[rows, 0] = d2m
        structure[rows, 1] = d2p
        structure[rows, 2] = d3p
        structure[rows, 3] = theta
        structure[rows, 4] = np.sin(phi)
        structure[rows, 5] = np.cos(phi)
        structure[rows, 6] = ((d3p > 4.8) & (d3p < 5.8)).astype(float)
        structure[rows, 7] = (d2p > 6.0).astype(float)
        structure[rows, 8] = burial
        structure[rows, 9] = rows / max(n - 1, 1)
        structure[rows, 10] = (rows < 2 * window).astype(float)
        structure[rows, 11] = (rows >= n - 2 * window).astype(float)
    return seq, structure, usable


def grouped_ridge_r2(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    folds: int = 5,
    alpha: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """The held-out R² of a ridge regression of each column of `targets` on `features`, the `folds` drawn by `groups` (a chain's residues are all in one fold, so nothing that is particular to a chain is
    learned from it and read on it). The features are standardised by the training folds, the pooled `1 - SSE / SST` is taken against the training folds' mean."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    fold_of = dict(
        zip(
            unique.tolist(),
            (rng.permutation(len(unique)) % folds).tolist(),
            strict=True,
        )
    )
    fold = np.array([fold_of[g] for g in groups.tolist()])
    sse = np.zeros(y.shape[1])
    sst = np.zeros(y.shape[1])
    for k in range(folds):
        test = fold == k
        train = ~test
        if not test.any() or not train.any():
            continue
        mu, sd = x[train].mean(0), x[train].std(0)
        sd[sd == 0] = 1.0
        a = (x[train] - mu) / sd
        b = (x[test] - mu) / sd
        y_mean = y[train].mean(0)
        coef = np.linalg.solve(
            a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y[train] - y_mean)
        )
        pred = b @ coef + y_mean
        sse += ((y[test] - pred) ** 2).sum(0)
        sst += ((y[test] - y_mean) ** 2).sum(0)
    return 1.0 - sse / sst


def kl_rate(sigma: np.ndarray) -> float:
    """The rate term the first learned-posterior autoencoders were trained on: the sum over channels of `-ln sigma`, averaged over residues (`sigma` is `[residues, channels]`)."""
    return float((-np.log(np.asarray(sigma, dtype=np.float64))).sum(-1).mean())


def kl_exact(mu: np.ndarray, sigma: np.ndarray) -> float:
    """The exact Gaussian KL of `N(sqrt(1 - sigma^2) mu, sigma^2)` to `N(0, 1)`: the sum over channels of `1/2 [(1 - sigma^2) mu^2 + sigma^2 - 1 - ln sigma^2]`, averaged over residues."""
    mu = np.asarray(mu, dtype=np.float64)
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    return float((0.5 * ((1.0 - s2) * mu**2 + s2 - 1.0 - np.log(s2))).sum(-1).mean())


def kl_omitted(mu: np.ndarray, sigma: np.ndarray) -> float:
    """What the rate term leaves out of the exact KL: the sum over channels of `1/2 (1 - sigma^2)(mu^2 - 1)`, averaged over residues; `kl_exact - kl_rate`."""
    mu = np.asarray(mu, dtype=np.float64)
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    return float((0.5 * (1.0 - s2) * (mu**2 - 1.0)).sum(-1).mean())
