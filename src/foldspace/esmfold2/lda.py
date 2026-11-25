"""Linear discriminant analysis of the residue type from a row of the single: the earlier probe, made again.

The first reading of how much of the sequence a mixed single holds was a linear discriminant analysis
(97.9 % of residues) fitted on three train shards and scored on
the fourth. A shard is not a cluster: 88 to 89 % of the chains of each share a 30 % cluster with the other
three, so that number is not a held-out one. This fits the same discriminant
two ways on the model's own singles -- on the shards that were not held out and scored on the held-out
shard (the old protocol), and on the whole train split and scored on the dev and test splits (clusters it
never saw) -- so that the difference is a measurement and not an argument.

The fit is closed-form: class means, the pooled within-class covariance with a small ridge, equal-covariance
Gaussian classes, priors from the counts. It is exact and has no seed. Accumulation is in float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from foldspace.esmfold2 import batching
from foldspace.esmfold2.decoder_data import IGNORE, Normaliser, Split, gather

__all__ = ["LDA", "Projected", "Statistics", "fit_lda", "solve", "statistics"]

N_CLASSES = 20


class LDA(nn.Module):
    """The fitted discriminant, as a decoder: logits are the discriminant scores."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        return x @ self.weight.T + self.bias


@dataclass
class Statistics:
    """What a discriminant is made of: class counts, class means and the pooled within-class covariance (no ridge)."""

    counts: torch.Tensor
    """``[20]``, float64."""
    means: torch.Tensor
    """``[20, d]``."""
    covariance: torch.Tensor
    """``[d, d]``."""


class Projected(nn.Module):
    """A decoder that reads the single through a fixed projection: ``x @ basis`` and then ``inner``."""

    def __init__(self, inner: nn.Module, basis: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("basis", basis)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.inner(x @ self.basis, lengths)


@torch.no_grad()
def statistics(
    split: Split,
    norm: Normaliser,
    chain_indices: np.ndarray,
    *,
    max_tokens: int = 65536,
) -> Statistics:
    """The class counts, means and pooled within-class covariance of the given chains, accumulated in float64."""
    device = split.z.device
    d = split.z.shape[1]
    counts = torch.zeros(N_CLASSES, dtype=torch.float64, device=device)
    sums = torch.zeros(N_CLASSES, d, dtype=torch.float64, device=device)
    scatter = torch.zeros(d, d, dtype=torch.float64, device=device)
    for batch in batching.pack(np.asarray(chain_indices), split.lengths, max_tokens):
        x, y, _ = gather(split, batch, norm)
        valid = y != IGNORE
        rows = x[valid].double()
        onehot = torch.nn.functional.one_hot(y[valid], N_CLASSES).double()
        counts += onehot.sum(dim=0)
        sums += onehot.T @ rows
        scatter += rows.T @ rows
    if (counts == 0).any():
        raise ValueError("a residue type has no example in the chains given")
    means = sums / counts[:, None]
    within = scatter - (means.T * counts) @ means  # sum_c n_c mu_c mu_c^T removed
    return Statistics(counts, means, within / (counts.sum() - N_CLASSES))


@torch.no_grad()
def solve(
    stats: Statistics, basis: torch.Tensor | None = None, *, ridge: float = 1e-3
) -> nn.Module:
    """The discriminant of ``stats``; with ``basis`` (``[d, k]``) the one that reads only the ``k`` directions given.

    ``ridge`` is a fraction of the mean variance (of the directions read). The classes' priors are the counts'.
    """
    means, covariance = stats.means, stats.covariance
    if basis is not None:
        basis = basis.to(means.dtype)
        means = means @ basis
        covariance = basis.T @ covariance @ basis
    k = means.shape[1]
    covariance = covariance + (
        ridge
        * covariance.diagonal().mean()
        * torch.eye(k, dtype=torch.float64, device=means.device)
    )
    weight = torch.linalg.solve(
        covariance, means.T
    ).T  # (Sigma^-1 mu_c)^T, one row per class
    bias = -0.5 * (weight * means).sum(dim=1) + torch.log(
        stats.counts / stats.counts.sum()
    )
    model = LDA(weight.float(), bias.float())
    return model if basis is None else Projected(model, basis.float())


@torch.no_grad()
def fit_lda(
    split: Split,
    norm: Normaliser,
    chain_indices: np.ndarray,
    *,
    ridge: float = 1e-3,
    max_tokens: int = 65536,
) -> LDA:
    """Fit on the given chains of ``split``. ``ridge`` is a fraction of the mean variance."""
    stats = statistics(split, norm, chain_indices, max_tokens=max_tokens)
    return solve(stats, ridge=ridge)  # type: ignore[return-value]
