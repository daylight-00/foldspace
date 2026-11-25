"""How far the structure moved.

A comparison is always between two predictions of the **same** sequence, produced by
the same model from a perturbed and an unperturbed state. That makes the residue correspondence the identity -- no alignment, no
alignment-dependent score -- which is worth stating, because it changes what the
usual numbers mean here.

Three quantities, deliberately not one:

- **RMSD** is superposition-dependent and dominated by whichever part moved
  most. A hinge between two rigid domains gives a large RMSD from a change that
  is locally nothing.
- **TM-score** is length-normalised and saturates: it answers "same fold?", and
  at the small displacements the gate is about it will sit near 1 and say little.
  Computed here under a single global Kabsch superposition, **not** maximised
  over superpositions the way TM-align does -- so it is a lower bound, and is
  comparable across this corpus but not against published TM-align values.
- **lDDT** is superposition-free and local. At small σ it is the one that moves,
  which is why the gate reads primarily off it.

Reporting only one of the three is how a hinge gets mistaken for a collapse, or
a local unravelling for stability.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kabsch_rmsd", "lddt_ca", "superimpose", "tm_score"]


def _as_coords(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(
            f"expected two matching (L, 3) arrays, got {x.shape} {y.shape}"
        )
    if len(x) == 0:
        raise ValueError("no coordinates")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(
            "non-finite coordinates reached a geometry function; drop them "
            "where they enter, so the loss shows up in a coverage number "
            "rather than here where it cannot"
        )
    return x, y


def _kabsch_rotation(xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Optimal rotation from centred ``xc`` onto centred ``yc``.

    ``np.linalg.svd`` can fail to converge on a near-degenerate covariance -- an
    almost collinear chain, or two structures sharing almost no spread. A tiny
    isotropic ridge makes it converge without moving a well-conditioned answer;
    if that fails too the caller is told, rather than one entry killing a run.
    """
    covariance = xc.T @ yc
    try:
        u, _, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        scale = float(np.abs(covariance).max()) or 1.0
        try:
            u, _, vt = np.linalg.svd(covariance + np.eye(3) * scale * 1e-9)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Kabsch superposition did not converge; the structures share "
                "too little spread to align"
            ) from exc
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def superimpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``mobile`` rotated and translated onto ``target`` (Kabsch)."""
    x, y = _as_coords(mobile, target)
    xc, yc = x - x.mean(0), y - y.mean(0)
    return (_kabsch_rotation(xc, yc) @ xc.T).T + y.mean(0)


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD after optimal superposition, in Å."""
    fitted = superimpose(mobile, target)
    _, y = _as_coords(fitted, target)
    return float(np.sqrt(((fitted - y) ** 2).sum(axis=1).mean()))


def tm_score(mobile: np.ndarray, target: np.ndarray) -> float:
    """TM-score under one global superposition (see the module docstring).

    ``d0 = 1.24 * (L - 15)**(1/3) - 1.8``, floored at 0.5 as TM-score does for
    short chains, where the formula goes negative.
    """
    fitted = superimpose(mobile, target)
    _, y = _as_coords(fitted, target)
    length = len(y)
    d0 = max(1.24 * (length - 15) ** (1 / 3) - 1.8, 0.5) if length > 15 else 0.5
    distances = np.linalg.norm(fitted - y, axis=1)
    return float((1.0 / (1.0 + (distances / d0) ** 2)).mean())


def lddt_ca(
    mobile: np.ndarray,
    target: np.ndarray,
    *,
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    exclude_neighbours: int = 0,
) -> float:
    """Cα lDDT of ``mobile`` against ``target``, superposition-free.

    The fraction of target Cα–Cα distances under ``inclusion_radius`` that are
    preserved to within each threshold, averaged over thresholds. 1.0 means
    every local distance survived.

    ``exclude_neighbours`` drops pairs closer than that many residues in
    sequence; 0 keeps them, which is what makes a local unravelling visible.
    """
    x, y = _as_coords(mobile, target)
    dx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    dy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)

    mask = (dy < inclusion_radius) & ~np.eye(len(y), dtype=bool)
    if exclude_neighbours:
        index = np.arange(len(y))
        mask &= np.abs(index[:, None] - index[None, :]) > exclude_neighbours
    if not mask.any():
        raise ValueError("no pairs within the inclusion radius")

    deviation = np.abs(dx[mask] - dy[mask])
    return float(np.mean([(deviation < t).mean() for t in thresholds]))
