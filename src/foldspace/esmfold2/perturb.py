"""Perturbations of a single, for the robustness experiment: errors a generator's output could carry, each with a level and a measured size.

All of it works in the coordinates the decoders read, ``z = (H - mean) / std`` with the training split's channel
statistics, where a channel has unit variance over the corpus. A level is therefore a fraction of the corpus's own
spread and does not depend on the units of the single. Numpy only.

Five families, each one a way a trained generator is wrong that the others are not:

``noise``
    ``z + level * eps``: independent Gaussian error of the same size in every channel. The level is the error's standard
    deviation in channel standard deviations. The single is not spread evenly over its directions (a few carry most of the
    variance), so this error is small against the leading ones and large against the others.
``anoise``
    the same error shaped like the data: Gaussian with the covariance of the training singles times ``level ** 2``, so
    that in every direction the error is ``level`` times that direction's own standard deviation. It is the error of a
    generator that is as accurate, relatively, in the small directions as in the large ones.
``shrink``
    ``(1 - level) * z``: a fraction of the deviation from the mean single removed. A generator trained to minimise a
    squared error drifts to the mean when it is unsure, and the mean single is the one a position-aware average would give.
``rank``
    ``z`` projected on the ``level`` leading directions of the training covariance: what a truncated representation loses.
    It is deterministic, and is a measurement of a projection and not a proposal to compress.
``dropout``
    a share ``level`` of the positions of a chain, chosen at random and always the same number, set to the mean single: a
    generator that is right in most places and has no idea in a few.

A perturbed single is a different single of the same shape; whether the folding model still folds it, and whether the
sequence is still readable from it, are what the robustness experiment measures.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    "COARSE_LEVELS",
    "FAMILIES",
    "LEVELS",
    "Spectrum",
    "basis_from_covariance",
    "conditions",
    "levels_of",
    "name_of",
    "perturb",
    "relative_size",
    "rng_for",
    "severity_order",
    "spectrum_from_covariance",
]

#: ``anoise`` is last: it was added after the first four had been read by the decoders, and a family's generator is keyed by
#: its position here, so the realisations of the first four are the ones that were read
FAMILIES = ("noise", "shrink", "rank", "dropout", "anoise")

#: fixed in advance
LEVELS: dict[str, tuple[float, ...]] = {
    "noise": (0.1, 0.2, 0.4, 0.8),
    "shrink": (0.1, 0.25, 0.5),
    "rank": (128, 64, 32, 16),
    "dropout": (0.05, 0.15, 0.30),
    "anoise": (0.05, 0.1, 0.2, 0.4),
}


#: levels beyond the standard ones, extending them to where the fold breaks. ``shrink`` 1.0 is the mean single at every position.
COARSE_LEVELS: dict[str, tuple[float, ...]] = {
    "noise": (1.6, 3.2),
    "shrink": (0.75, 0.9, 1.0),
    "rank": (8, 4, 2, 1),
    "dropout": (0.6, 0.9),
    "anoise": (0.8, 1.6),
}


@dataclass(frozen=True)
class Spectrum:
    """The eigen-decomposition of the training covariance of the standardised single."""

    vectors: np.ndarray
    """``[D, D]``, one direction per column, by descending eigenvalue."""
    values: np.ndarray
    """``[D]``, the variance along each direction, descending and never negative."""


def levels_of(family: str, which: str = "standard") -> tuple[float, ...]:
    """The levels of a family: the standard ones, or the coarser ones."""
    if which == "standard":
        return LEVELS[family]
    if which == "coarse":
        return COARSE_LEVELS[family]
    raise ValueError(f"levels are standard or coarse, not {which!r}")


def conditions(which: str = "standard") -> list[tuple[str, float]]:
    """Every (family, level), in the order they are folded and reported."""
    return [(f, level) for f in FAMILIES for level in levels_of(f, which)]


def name_of(family: str, level: float) -> str:
    """A key for a condition, stable across runs: ``noise_0.4``, ``rank_64``."""
    return f"{family}_{int(level) if family == 'rank' else level:g}"


def severity_order(family: str, which: str = "standard") -> list[float]:
    """The levels of a family from the mildest to the most severe: a smaller rank is a more severe one."""
    levels = sorted(levels_of(family, which))
    return levels[::-1] if family == "rank" else levels


def rng_for(
    seed: int, chain: int | str, family: str, level: float
) -> np.random.Generator:
    """The generator of one chain under one condition: the same in every run, independent between conditions.

    A chain is keyed by its id, so that the singles perturbed for folding and for reading the sequence are the same
    perturbation of the same chain, whichever order the chains are held in.
    """
    key = zlib.crc32(chain.encode()) if isinstance(chain, str) else int(chain)
    # the coarse levels follow the standard ones in the index, so the realisations of the standard ones are the same with or without them
    place = (LEVELS[family] + COARSE_LEVELS[family]).index(level)
    return np.random.default_rng(
        np.random.SeedSequence([seed, key, FAMILIES.index(family), place])
    )


def spectrum_from_covariance(covariance: np.ndarray) -> Spectrum:
    """Eigenvectors (as columns) and eigenvalues of a symmetric matrix, by descending eigenvalue."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"not a square matrix: {covariance.shape}")
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    order = np.argsort(values)[::-1]
    return Spectrum(vectors[:, order], np.clip(values[order], 0.0, None))


def basis_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """The eigenvectors of ``spectrum_from_covariance`` alone: ``[D, D]``."""
    return spectrum_from_covariance(covariance).vectors


def perturb(
    z: np.ndarray,
    family: str,
    level: float,
    rng: np.random.Generator,
    *,
    spectrum: Spectrum | None = None,
) -> np.ndarray:
    """``z`` (``[L, D]``, standardised) under one condition, as float32.

    ``anoise`` and ``rank`` need the ``spectrum`` of the training covariance (``spectrum_from_covariance``).
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"a chain's single is [L, D], not {z.shape}")
    if family == "noise":
        out = z + level * rng.standard_normal(z.shape)
    elif family == "anoise":
        if spectrum is None:
            raise ValueError("anoise needs the spectrum of the training covariance")
        eps = rng.standard_normal(z.shape) * np.sqrt(spectrum.values)
        out = z + level * (eps @ spectrum.vectors.T)
    elif family == "shrink":
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"shrink is a fraction, not {level}")
        out = (1.0 - level) * z
    elif family == "rank":
        if spectrum is None:
            raise ValueError("rank needs the spectrum of the training covariance")
        k = int(level)
        if not 1 <= k <= spectrum.vectors.shape[1]:
            raise ValueError(f"rank {k} of {spectrum.vectors.shape[1]} directions")
        leading = spectrum.vectors[:, :k]
        out = (z @ leading) @ leading.T
    elif family == "dropout":
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"dropout is a share, not {level}")
        n = z.shape[0]
        k = round(level * n)
        out = z.copy()
        if k:
            out[rng.choice(n, size=k, replace=False)] = 0.0
    else:
        raise ValueError(f"unknown family {family!r}; one of {FAMILIES}")
    return out.astype(np.float32)


def relative_size(z: np.ndarray, perturbed: np.ndarray) -> float:
    """``|perturbed - z| / |z|`` over the chain: how large the change is against the single's own deviation from the mean."""
    z = np.asarray(z, dtype=np.float64)
    denominator = float(np.linalg.norm(z))
    if denominator == 0.0:
        raise ValueError("a single of zeros has no scale")
    return float(
        np.linalg.norm(np.asarray(perturbed, dtype=np.float64) - z) / denominator
    )
