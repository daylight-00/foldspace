"""The adapted autoencoder: a latent anchored on the sequence and a decoder fine-tuned to read it.

The release latent is the encoding of all atoms. Here the latent of a residue is a function of its sequence alone: `z = M(W LN(h))`,
`h` the residue's ESMC state (arm `esmc`) or a learned embedding of its type at the same width (arm `onehot`, the control: the same
parameters downstream), `LN` a LayerNorm, `W` a linear map 2560 -> 32 and `M` a per-channel moment matching to zero mean and unit variance
over the residues of the training set (BatchNorm without affine parameters, whose running statistics are what a later job applies: the
channels the pretrained flow was trained on have a standard deviation of 0.99). `z` is **deterministic**: nothing in it
is a variance and no KL acts on it. The decoder is a copy of the release decoder, all of it trained, reading `z` and the Cα
coordinates it is given; during training `z` carries Gaussian noise (`--noise`) so that the decoder tolerates what a flow generates, and
every score is at zero noise. The objective is the autoencoder's coordinate, residue and atom-presence terms, with the sequence
condition masked on every residue (the decoder reads it as an input). Reconstruction is scored on the dev chains as for the reading adapters: residues with
every atom and truncated residues apart, per-residue and side-chain RMSD, residue types recovered, atom-presence F1.

With `--residual R` the last R of the 32 channels are not sequence-derived: `r = M(W_r mu)`, a map of the release encoder's posterior
mean on the same chain (all atoms; frozen, a training-only geometry encoder, absent at generation, where the flow would generate it),
so that the width the decoder reads stays 32 and the pretrained flow's tensors need no splice.

With `--inplace` (in-place layout) the 14 informative channels of the release latent stay and the code takes the other 18; with `--source fixed --inplace` the code is a fixed Gaussian vector of 18 channels
for each residue type (the identity control in the in-place layout, `separated_code`: the best of 64 seeded draws by the smallest distance between two standard residues). The in-place latents also have a
**posterior**: the release flow is trained to generate a posterior *sample*, half of whose channels are noise, and the first in-place layout arm gave its flow means and a noise-free code. With
`--posterior-native` the channels kept are, in training, the release encoder's samples (its own scales, resampled at every step) and not its means; with `--code-sigma S` the code channels carry Gaussian
noise of scale S; the two replace `--noise` (independent noise on every channel), which is refused beside them. With `--stochastic-code --kl β` (the ESMC code only) the width is learned
instead: `z = sqrt(1 - σ²) μ + σ ε` with `σ = 0.99 sigmoid(head(LN(h)))` per residue and channel (unit variance whatever σ is), and the loss adds β times the mean over residues of the sum over channels of
`-ln σ`, the width part of the Gaussian KL to N(0, 1) (equal to it where μ² = 1; `--kl-exact` puts the exact KL, `½ Σ[(1 - σ²) μ² + σ² - 1 - ln σ²]`, in its place, and both are logged). The checkpoint records the posterior
(`posterior`, and the head `sigma_head` among the latent's tensors), a flow's target is made the same way (`src/foldspace/rfd4/flow_model.py`), and every dev reading is made at the posterior's mean (`sqrt(1 - σ²) μ` for a
learned width; `read_dev`, which a saved checkpoint can be read with) and again sampled, the learned posterior's unshrunk μ being the reading `end_code`.

What this prototype asks is whether a decoder can rebuild a structure from the Cα trace and a latent that holds the sequence and nothing
else. The side chain's conformation is not in such a latent, so the side-chain error says how much of it the sequence and the Cα
trace imply and how much a geometry residual `r` would have to carry.

    .venv-rfd4/bin/python -P -B src/foldspace/rfd4/train_autoencoder.py --shards cache/rfd4esmc/shards --ae <ckpt> --source esmc --out runs/rfd4esmc/autoencoder/esmc
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foldspace.rfd4.datapath_stats as ds
import foldspace.rfd4.flow_data as fd
from foldspace.rfd4.train_reading_adapter import KEYS, Shards, collate

SOURCES = ("esmc", "onehot", "codebook", "none", "fixed")
#: in-place layout (`--inplace`): the informative channels of the release latent that stay where the pretrained flow expects them, the release
#: latent's identity channels (0 and 10) and the 16 channels the release decoder does not read (`src/foldspace/rfd4/channel_audit.py`) carry the code
INPLACE_KEEP = (1, 4, 5, 8, 13, 14, 15, 16, 17, 22, 25, 27, 28, 31)
#: the widest a learned posterior may be: at 1 the sample is pure noise and the gradient of `sqrt(1 - sigma^2)` is infinite (runs went to NaN where a width saturated)
SIGMA_MAX = 0.99


def code_separation(code: np.ndarray, rows: tuple[int, ...] = ()) -> float:
    """The smallest distance between the codes of two residue types (`rows` of the table; every row when none are given)."""
    c = np.asarray(code, dtype=np.float64)
    if rows:
        c = c[list(rows)]
    distance = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    return float(distance[np.triu_indices(len(c), k=1)].min())


def separated_code(
    n_rows: int,
    width: int,
    rows: tuple[int, ...] = (),
    seed: int = 20261112,
    tries: int = 64,
) -> np.ndarray:
    """A fixed Gaussian code of `width` channels for each of `n_rows` residue types: the best of `tries` seeded draws by the smallest distance between two of `rows` (the standard residues).

    It is a code that nothing learns and whose types lie far enough apart for a decoder to tell them (a code whose types are 0.04 to 0.7 apart is not read);
    the draw is a function of the seed, so the same code is made again."""
    best, best_distance = None, -1.0
    for k in range(tries):
        code = (
            np.random.default_rng(seed + k)
            .standard_normal((n_rows, width))
            .astype(np.float32)
        )
        distance = code_separation(code, rows)
        if distance > best_distance:
            best, best_distance = code, distance
    assert best is not None
    return best


def standardize_table(table: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """A code table moment-matched to zero mean and unit variance in every channel over a residue distribution: `weight[row]` is how much of the residues have that row's type.

    It is what the ESMC latent's BatchNorm does to its own channels over the training residues, so that a fixed code of the same width has the same marginals as the latent it controls (the type-only
    control of the full-width latent). Rows that no residue has (weight 0) are moved with the rest and count for nothing; a channel without variance is an error and not a division."""
    table = np.asarray(table, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if weight.shape != (table.shape[0],) or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError(
            "the weights are one non-negative number for each row, not all zero"
        )
    weight = weight / weight.sum()
    mean = (weight[:, None] * table).sum(0)
    std = np.sqrt((weight[:, None] * (table - mean) ** 2).sum(0))
    if (std <= 1e-12).any():
        raise ValueError(
            "a channel of the code has no variance over the residues: nothing to match"
        )
    return ((table - mean) / std).astype(np.float32)


def file_sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_native_code(latent, path: str) -> None:
    """Fill a native-calibrated latent's tables from the file `src/foldspace/rfd4/native_identity_stats.py` writes: `mean [rows, channels]`, `chol [rows, channels, channels]` and `channels`, the release's channels that
    the code takes (which are the latent's own)."""
    import torch

    stats = np.load(path)
    mean, chol = stats["mean"], stats["chol"]
    if [int(c) for c in stats["channels"]] != list(latent.code_idx):
        raise ValueError(
            f"{path} is the statistics of channels {stats['channels'].tolist()}, not of this latent's code channels {latent.code_idx}"
        )
    if mean.shape != tuple(latent.code.shape) or chol.shape != tuple(
        latent.code_chol.shape
    ):
        raise ValueError(
            f"{path} has tables of {mean.shape} and {chol.shape}, the latent's are {tuple(latent.code.shape)} and {tuple(latent.code_chol.shape)}"
        )
    with torch.no_grad():
        latent.code.copy_(torch.from_numpy(mean).to(latent.code.dtype))
        latent.code_chol.copy_(torch.from_numpy(chol).to(latent.code_chol.dtype))


def load_code_bank(latent, path: str) -> None:
    """Fill a banked latent's tables from the file `src/foldspace/rfd4/code_bank.py` writes: `code_bank [N, channels]` sorted by type, `bank_start` and `bank_count` `[rows]` (where each type's donors are), `mean [rows, channels]` (the
    type centroids, the code's mean reading) and `channels`, the release's channels that the code takes (which are the latent's own)."""
    import torch

    bank = np.load(path)
    if [int(c) for c in bank["channels"]] != list(latent.code_idx):
        raise ValueError(
            f"{path} is the bank of channels {bank['channels'].tolist()}, not of this latent's code channels {latent.code_idx}"
        )
    shapes = {
        "code_bank": tuple(latent.code_bank.shape),
        "bank_start": tuple(latent.bank_start.shape),
        "bank_count": tuple(latent.bank_count.shape),
        "mean": tuple(latent.code.shape),
    }
    for key, shape in shapes.items():
        if bank[key].shape != shape:
            raise ValueError(
                f"{path}: {key} has shape {bank[key].shape}, the latent's is {shape}"
            )
    start, count = bank["bank_start"], bank["bank_count"]
    if (
        (count < 1).any()
        or (start < 0).any()
        or (start + count > len(bank["code_bank"])).any()
    ):
        raise ValueError(f"{path}: a type's donors are not rows of the bank")
    with torch.no_grad():
        latent.code.copy_(torch.from_numpy(bank["mean"]).to(latent.code.dtype))
        latent.code_bank.copy_(
            torch.from_numpy(bank["code_bank"]).to(latent.code_bank.dtype)
        )
        latent.bank_start.copy_(torch.from_numpy(start).to(latent.bank_start.dtype))
        latent.bank_count.copy_(torch.from_numpy(count).to(latent.bank_count.dtype))


def anchored_class():
    import torch
    from rfproteina.model_masks import compute_ae_masks
    from torch import nn

    class AnchoredLatent(nn.Module):
        """`z = M(W LN(h))` for the residues that carry a latent; zero elsewhere."""

        def __init__(
            self,
            source: str,
            n_types: int,
            width: int = 32,
            residual: int = 0,
            window: int = 0,
            freeze_embed: bool = False,
            inplace: bool = False,
            in_dim: int = 2560,
            standard_rows: tuple[int, ...] = (),
            stochastic: bool = False,
            calibrated: bool = False,
            banked: int = 0,
        ):
            super().__init__()
            self.source = source
            self.in_dim = in_dim
            self.residual = residual
            self.window = window
            self.n_types = n_types
            self.inplace = inplace
            # a fixed in-place code whose table is the release encoder's own per-type statistics: `code` is the mean of each type's code,
            # `code_chol` the Cholesky factor of its covariance, and a sample is `code + code_chol @ eps`; the checkpoint carries `code_chol`, which is how a loader knows
            self.calibrated = bool(calibrated)
            if calibrated and not (source == "fixed" and inplace):
                raise ValueError("a native-calibrated code is a fixed code in place")
            # a fixed in-place code whose draws are the codes of other residues of the same type, from a bank: `code` is each type's centroid (the mean reading), `code_bank` the donors
            # sorted by type and `bank_start` and `bank_count` say where a type's are; a sample is a donor of the residue's type, drawn at random, with the additive noise of `code_sigma`. The number of donors
            # is the size of the checkpoint's `code_bank`, which is how a loader knows
            self.banked = int(banked)
            if self.banked and not (source == "fixed" and inplace):
                raise ValueError("a banked code is a fixed code in place")
            if self.banked and calibrated:
                raise ValueError(
                    "a code has one table or the other: a bank or a statistics file"
                )
            # the scale of the Gaussian noise a sampled code carries (a posterior of fixed width; zero is the deterministic code): not a weight, so it is
            # set by whoever builds the module and recorded beside the checkpoint's tensors
            self.code_sigma = 0.0
            if stochastic and not (inplace and source == "esmc"):
                raise ValueError("a learned posterior is for the in-place ESMC code")
            if inplace:
                if residual:
                    raise ValueError("an in-place latent has no residual channels")
                self.keep = list(INPLACE_KEEP)
                self.code_idx = [c for c in range(width) if c not in INPLACE_KEEP]
                self.register_buffer(
                    "inplace_flag", torch.ones(1)
                )  # marks a checkpoint's latent as in place
            n_code = len(self.code_idx) if inplace else width - residual
            if source == "fixed":
                # a code that nothing learns: a fixed random vector of the latent's width for each residue type. Twenty points of a
                # Gaussian in 32 dimensions lie 5 to 8 apart (ESMC's types, 3.4 at the closest), so a decoder that does not read it
                # is not excused by a code that is too close to tell apart
                if inplace:
                    # the in-place layout: the code takes the channels that ESMC's code takes there (the release latent's identity channels and the 16 it does not
                    # read), the 14 others come from the release encoder; the types lie as far apart as a draw of a few dozen seeds can make them
                    if calibrated:
                        # the tables are filled from the release encoder's statistics by whoever builds the module (`native_identity_stats.py`)
                        self.register_buffer("code", torch.zeros(n_types + 1, n_code))
                        self.register_buffer(
                            "code_chol", torch.zeros(n_types + 1, n_code, n_code)
                        )
                        return
                    if self.banked:
                        # the tables are filled from the bank file by whoever builds the module (`load_code_bank`)
                        self.register_buffer("code", torch.zeros(n_types + 1, n_code))
                        self.register_buffer(
                            "code_bank", torch.zeros(self.banked, n_code)
                        )
                        self.register_buffer(
                            "bank_start", torch.zeros(n_types + 1, dtype=torch.long)
                        )
                        self.register_buffer(
                            "bank_count", torch.ones(n_types + 1, dtype=torch.long)
                        )
                        return
                    self.register_buffer(
                        "code",
                        torch.from_numpy(
                            separated_code(n_types + 1, n_code, standard_rows)
                        ),
                    )
                    return
                # with `residual` channels of the release encoder's posterior mean (as `esmc` has them) the code takes the rest
                self.register_buffer(
                    "code",
                    torch.randn(
                        n_types + 1,
                        width - residual,
                        generator=torch.Generator().manual_seed(20261112),
                    ),
                )
                if residual:
                    self.res_proj = nn.Linear(width, residual)
                    self.res_moment = nn.BatchNorm1d(
                        residual, affine=False, momentum=0.01
                    )
                return
            if source == "onehot":
                # one table for each offset of a residue's neighbours in the chain (the extra row is "outside the chain");
                # the control without ESMC that still sees sequence context when `window` is not zero
                self.embed = nn.ModuleList(
                    nn.Embedding(n_types + 1, 2560) for _ in range(2 * window + 1)
                )
                # the residue's own table at the scale of the plain one-hot arm, its neighbours' at a fifth of it, so that the
                # control starts as that arm with context added and the identity is not diluted by it
                for k, table in zip(
                    range(-window, window + 1), self.embed, strict=True
                ):
                    nn.init.normal_(table.weight, std=1.0 if k == 0 else 0.2)
                    # a fixed random code of each type, as ESMC's state is a fixed input: the control no longer depends on
                    # how well a learned table is conditioned
                    table.weight.requires_grad_(not freeze_embed)
            if source == "codebook":
                # identity (and, with a window, the neighbours' identities) mapped straight to the 32 channels: a codebook that
                # nothing but gradient descent on 32 numbers a type has to find, the best a code of sequence identity can be
                self.proj = nn.Linear((2 * window + 1) * (n_types + 1), n_code)
            else:
                # `in_dim` is the width of the rows the source is read from: ESMC's last state (2,560, the default), or a narrower one
                # (ESMFold2's mixed single, 256) put in the shards' `esmc` key
                self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)
                self.proj = nn.Linear(in_dim, n_code)
            self.moment = nn.BatchNorm1d(n_code, affine=False, momentum=0.01)
            # the learned posterior of the in-place ESM code: a width `sigma = 0.99 sigmoid(head(LN(h)))` per residue and channel, starting at 0.2 with
            # zero weights, so that the code is `sqrt(1 - sigma^2) mu + sigma eps` and keeps a unit variance whatever the width. The posterior is q = N(sqrt(1 - sigma^2) mu, sigma^2):
            # its mean is the code scaled down and a sample adds the noise. What the code gives up is kept two ways at every sampled step: `last_rate`, the width part of the
            # Gaussian KL to N(0, 1), sum over channels of -ln sigma (the term the first autoencoders were trained on, equal to the KL where mu^2 = 1), and `last_kl_exact`, the
            # exact KL, 1/2 sum[(1 - sigma^2) mu^2 + sigma^2 - 1 - ln sigma^2]; `last_kl` is the one in the loss, the rate term unless `kl_exact_in_loss`
            self.sigma_head = None
            self.last_kl = None
            self.last_rate = None
            self.last_kl_exact = None
            self.last_sigma = None
            self.kl_exact_in_loss = False
            if stochastic:
                self.sigma_head = nn.Linear(in_dim, n_code)
                nn.init.zeros_(self.sigma_head.weight)
                start = 0.2 / SIGMA_MAX
                nn.init.constant_(self.sigma_head.bias, math.log(start / (1.0 - start)))
            if residual:
                self.res_proj = nn.Linear(width, residual)
                self.res_moment = nn.BatchNorm1d(residual, affine=False, momentum=0.01)

        def forward(
            self,
            batch: dict,
            noise: float = 0.0,
            native=None,
            sample: bool = False,
            unshrunk: bool = False,
        ) -> torch.Tensor:
            """The latent of the batch. `noise` is independent Gaussian noise on every channel (the first prototypes' way of making a decoder tolerate what a flow generates);
            `sample` gives the code channels of an in-place latent a draw of its posterior (the width `code_sigma`, or the learned one), and `native` is whatever the caller keeps in the other
            channels (the release encoder's posterior mean or one of its samples). Without `sample` the code is the posterior's **mean**: `mu` for a fixed or a deterministic code, and
            `sqrt(1 - sigma^2) mu` for a learned posterior; `unshrunk` gives a learned posterior's `mu` itself (what this method returned for it before the shrinkage)."""
            _, emit = compute_ae_masks(batch)
            if self.source == "fixed" and self.inplace:
                z = torch.zeros(*batch["residue_type"].shape, 32, device=emit.device)
                if emit.any():
                    rows = torch.zeros(int(emit.sum()), 32, device=emit.device)
                    types = batch["residue_type"].clamp(min=0)[emit]
                    code = self.code[types]
                    if sample and self.banked:
                        # the code of another residue of this type, drawn at random from the bank, and the noise: what this residue's type says and nothing of its own context
                        start, count = self.bank_start[types], self.bank_count[types]
                        pick = torch.minimum(
                            (
                                torch.rand(types.shape, device=code.device) * count
                            ).long(),
                            count - 1,
                        )
                        code = self.code_bank[start + pick]
                        if self.code_sigma:
                            code = code + self.code_sigma * torch.randn_like(code)
                    elif sample and self.calibrated:
                        # a draw of the release encoder's own distribution of this type's code: its mean and its covariance
                        code = code + torch.einsum(
                            "nij,nj->ni", self.code_chol[types], torch.randn_like(code)
                        )
                    elif sample and self.code_sigma:
                        code = code + self.code_sigma * torch.randn_like(code)
                    rows[:, self.code_idx] = code
                    rows[:, self.keep] = native.float()[emit][:, self.keep]
                    z[emit] = rows
                if noise:
                    z = z + noise * torch.randn_like(z) * emit[..., None]
                return z
            if self.source == "fixed":
                n_code = self.code.shape[1]
                z = torch.zeros(
                    *batch["residue_type"].shape,
                    n_code + self.residual,
                    device=emit.device,
                )
                z[..., :n_code] = (
                    self.code[batch["residue_type"].clamp(min=0)] * emit[..., None]
                )
                if self.residual and emit.any():
                    r = self.res_proj(native.float()[emit]).float()
                    z[emit, n_code:] = self.res_moment(r).float()
                if noise:
                    z = z + noise * torch.randn_like(z) * emit[..., None]
                return z
            if self.source == "onehot":
                types = batch["residue_type"].clamp(min=0)
                length = batch["padding_mask"].sum(1, keepdim=True)
                index = torch.arange(types.shape[1], device=types.device)[None, :]
                x = 0.0
                for k, table in zip(
                    range(-self.window, self.window + 1), self.embed, strict=True
                ):
                    at = index + k
                    inside = (at >= 0) & (at < length)
                    shifted = torch.gather(types, 1, at.clamp(0, types.shape[1] - 1))
                    outside = torch.full_like(shifted, self.n_types)
                    x = x + table(torch.where(inside, shifted, outside))
            elif self.source == "none":
                # no information about the sequence at all: what the decoder gives from the Cα trace alone
                x = torch.zeros(*batch["residue_type"].shape, 2560, device=emit.device)
            elif self.source == "codebook":
                types = batch["residue_type"].clamp(min=0)
                length = batch["padding_mask"].sum(1, keepdim=True)
                index = torch.arange(types.shape[1], device=types.device)[None, :]
                parts = []
                for k in range(-self.window, self.window + 1):
                    at = index + k
                    inside = (at >= 0) & (at < length)
                    shifted = torch.gather(types, 1, at.clamp(0, types.shape[1] - 1))
                    outside = torch.full_like(shifted, self.n_types)
                    parts.append(
                        torch.nn.functional.one_hot(
                            torch.where(inside, shifted, outside), self.n_types + 1
                        ).float()
                    )
                x = torch.cat(parts, dim=-1)
            else:
                x = batch["esmc"].float()
            seen = x if self.source == "codebook" else self.norm(x)
            raw = self.proj(seen).float()
            if self.inplace:
                # the native channels that stay in place come from the release encoder's posterior mean, the rest from the code
                z = torch.zeros(*raw.shape[:-1], 32, device=raw.device)
                if emit.any():
                    rows = torch.zeros(int(emit.sum()), 32, device=raw.device)
                    code = self.moment(raw[emit])
                    if self.sigma_head is not None:
                        sigma = SIGMA_MAX * torch.sigmoid(
                            self.sigma_head(seen[emit]).float()
                        )
                        mu = code
                        shrink = torch.sqrt(1.0 - sigma**2)
                        code = mu if unshrunk else shrink * mu
                        if sample:
                            code = shrink * mu + sigma * torch.randn_like(mu)
                            self.last_rate = (
                                (-0.5 * torch.log(sigma**2 + 1e-12)).sum(-1).mean()
                            )
                            self.last_kl_exact = (
                                (
                                    0.5
                                    * (
                                        (1.0 - sigma**2) * mu**2
                                        + sigma**2
                                        - 1.0
                                        - torch.log(sigma**2 + 1e-12)
                                    )
                                )
                                .sum(-1)
                                .mean()
                            )
                            self.last_kl = (
                                self.last_kl_exact
                                if self.kl_exact_in_loss
                                else self.last_rate
                            )
                            self.last_sigma = sigma.detach()
                    elif sample and self.code_sigma:
                        code = code + self.code_sigma * torch.randn_like(code)
                    rows[:, self.code_idx] = code
                    rows[:, self.keep] = native.float()[emit][:, self.keep]
                    z[emit] = rows
                if noise:
                    z = z + noise * torch.randn_like(z) * emit[..., None]
                return z
            z = torch.zeros(
                *raw.shape[:-1], raw.shape[-1] + self.residual, device=raw.device
            )
            if emit.any():
                z[emit, : raw.shape[-1]] = self.moment(raw[emit])
                if self.residual:
                    r = self.res_proj(native.float()[emit]).float()
                    z[emit, raw.shape[-1] :] = self.res_moment(r).float()
            if noise:
                z = z + noise * torch.randn_like(z) * emit[..., None]
            return z

        @torch.no_grad()
        def posterior_scale(self, batch: dict) -> torch.Tensor:
            """The learned widths `sigma` `[residues that carry a latent, channels]` of a latent that has a learned posterior."""
            if self.sigma_head is None:
                raise ValueError("this latent has no learned posterior")
            _, emit = compute_ae_masks(batch)
            seen = self.norm(batch["esmc"].float())
            return SIGMA_MAX * torch.sigmoid(self.sigma_head(seen[emit]).float())

    return AnchoredLatent


def rotated_items(items: list[dict], rng: np.random.Generator) -> list[dict]:
    """The items with their coordinates rotated uniformly at random, one rotation each (the frame the flow's batches are in)."""
    import torch

    out_items = []
    for it in items:
        r = torch.from_numpy(fd.random_rotation(rng))
        out_items.append({**it, "coords": it["coords"] @ r.T})
    return out_items


READINGS = ("mean", "code", "sampled")


def read_dev(
    *,
    latent,
    decoder,
    dev,
    native_fn,
    defaults: dict,
    mask_token: int,
    legal,
    device,
    batch_size: int,
    seed: int,
    mode: str = "mean",
    rotate: bool = False,
    posterior_native: bool = False,
    label: str = "",
) -> dict:
    """A reading of an autoencoder on the dev chains, by the trainer's own metrics (the reading-adapter ones: residues with every atom and truncated residues apart, per-residue and side-chain RMSD, residue types
    recovered, atom-presence F1).

    `mode` says what the decoder is given. **mean**: the channels kept are the release encoder's posterior means and the code is the posterior's mean (`sample=False`: `mu`, or `sqrt(1 - sigma^2) mu`
    for a learned posterior). **code**: the same with a learned posterior's unshrunk `mu`, the reading that the first autoencoders of this layout were given as "at the means" (for any other latent it is the mean reading).
    **sampled**: the posterior the run trains under (the release encoder's samples in the channels kept when `posterior_native`, and the code with its noise).
    """
    import torch
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    if mode not in READINGS:
        raise ValueError(f"mode is one of {READINGS}, not {mode}")
    was = (latent.training, decoder.training)
    latent.eval()
    decoder.eval()
    rows = {"all": [], "complete": [], "truncated": []}
    seq = {"all": [], "complete": [], "truncated": []}
    presence = []
    with torch.no_grad():
        for start in range(0, len(dev), batch_size):
            items = dev.items[start : start + batch_size]
            if (
                rotate
            ):  # the same rotations at every reading: the batch's start seeds them
                items = rotated_items(items, np.random.default_rng([seed, 4242, start]))
            batch = collate(items, defaults, mask_token, device)
            sampled = mode == "sampled"
            native = native_fn(batch, sample=sampled and posterior_native)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                z = latent(
                    batch, native=native, sample=sampled, unshrunk=mode == "code"
                )
                feed = dict(batch)
                feed["z_latent"] = z
                dec = decoder(feed)
            emit = (dec["emit_mask"] & batch["is_resolved_mask"].bool()).cpu().numpy()
            complete = batch["complete"].bool().cpu().numpy()
            coords = batch["coords"].float().cpu().numpy()
            tmask = batch["atom37_mask"].bool().cpu().numpy()
            pred = dec["coors_nm"].float().cpu().numpy() * 10.0
            ptype = dec["aatype_max"].cpu().numpy()
            pres = (dec["atom_presence_logits"].float() > 0).cpu().numpy()
            true = batch["residue_type"].cpu().numpy()
            legal_np = legal[batch["residue_type"]].cpu().numpy()
            for i in range(len(items)):
                parts = {
                    "all": emit[i],
                    "complete": emit[i] & complete[i],
                    "truncated": emit[i] & ~complete[i],
                }
                for part, scored in parts.items():
                    s = ds.residue_scores(coords[i], pred[i], tmask[i], scored)
                    if s.get("residues"):
                        rows[part].append(s)
                    q = ds.sequence_scores(true[i], ptype[i], scored, ENC.n_tokens)
                    if q.get("residues"):
                        seq[part].append(q)
                presence.append(
                    ds.presence_scores(tmask[i], pres[i], legal_np[i], emit[i])["f1"]
                )

    def pooled(group: list[dict], key: str) -> float:
        w = np.array([g["residues"] for g in group], dtype=float)
        return float(np.average([g[key] for g in group], weights=w))

    result = {"label": label, "sampled": mode == "sampled", "mode": mode}
    for part in ("all", "complete", "truncated"):
        if rows[part]:
            result[part] = {
                "residues": int(sum(g["residues"] for g in rows[part])),
                "per_residue_rmsd": pooled(rows[part], "per_residue_mean"),
                "backbone_rms": pooled(rows[part], "backbone_rms"),
                "side_chain_rms": pooled(rows[part], "side_chain_rms"),
                "sequence_accuracy": float(
                    sum(g["accuracy"] * g["residues"] for g in seq[part])
                    / sum(g["residues"] for g in seq[part])
                ),
            }
    result["presence_f1"] = float(np.mean(presence))
    latent.train(was[0])
    decoder.train(was[1])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="the decoder's")
    parser.add_argument("--lr-new", type=float, default=1e-3, help="the latent's")
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="independent Gaussian noise on every channel in training (default 0.2; none with --posterior-native or --code-sigma, which are the two noises that replace it)",
    )
    parser.add_argument(
        "--posterior-native",
        action="store_true",
        help="in-place latents: in training the channels kept from the release latent are the release encoder's posterior *samples* (its own scales), as the release flow's target is, and not its means",
    )
    parser.add_argument(
        "--code-sigma",
        type=float,
        default=0.0,
        help="in-place latents: Gaussian noise of this scale on the code channels in training (a posterior of fixed width; every reading is made at the means and again sampled)",
    )
    parser.add_argument(
        "--stochastic-code",
        action="store_true",
        help="the in-place ESMC code gets a learned posterior: z = sqrt(1 - sigma^2) mu + sigma eps, sigma = 0.99 sigmoid(head(LN(h))) per residue and channel; it needs --posterior-native and --kl",
    )
    parser.add_argument(
        "--kl",
        type=float,
        default=0.0,
        help="with --stochastic-code: the weight of the information the code gives up, the mean over residues of the sum over channels of -ln(sigma) (the width part of the Gaussian KL; it is the KL where the code's variance is one and independent of the width)",
    )
    parser.add_argument(
        "--kl-exact",
        action="store_true",
        help="with --stochastic-code: the term is the exact Gaussian KL of the posterior to N(0, 1), 1/2 sum[(1 - sigma^2) mu^2 + sigma^2 - 1 - ln sigma^2], in place of -ln(sigma); both are logged either way",
    )
    parser.add_argument(
        "--native-code",
        default=None,
        help="--source fixed --inplace --posterior-native: the code's table is the release encoder's own statistics (`src/foldspace/rfd4/native_identity_stats.py`), for each residue type the mean and the covariance of the 18 code channels in the encoder's samples, so that a code is a draw of what the release's latent holds in those channels for a residue of that type and nothing about its context",
    )
    parser.add_argument(
        "--code-bank",
        default=None,
        help="--source fixed --inplace --posterior-native: the code of a residue is the code of another residue of its type drawn at random from this bank (`src/foldspace/rfd4/code_bank.py`), re-drawn at every presentation, with the "
        "additive noise of --code-sigma: the type geometry and the within-type distribution of the bank's code and nothing of a residue's own context",
    )
    parser.add_argument(
        "--standardize-code",
        action="store_true",
        help="--source fixed, not in place, no residual: the code table is moment-matched to zero mean and unit variance in every channel over the training residues (weighted by how often each type occurs), as the "
        "ESMC latent's BatchNorm does for its own: the type-only control of the full-width latent has the marginals of the latent it controls",
    )
    parser.add_argument(
        "--residual",
        type=int,
        default=0,
        help="channels of the 32 that come from the all-atom encoder",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="in-place layout: the 14 informative channels of the release latent other than its identity channels stay in place (the release encoder's "
        "posterior mean) and the code, 18 channels, takes the channels the release decoder does not read and the identity channels",
    )
    parser.add_argument(
        "--freeze-embed",
        action="store_true",
        help="one-hot control only: the tables are fixed random codes, not learned",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="one-hot control only: the residue types of this many neighbours on each side enter too (sequence context without ESMC)",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--train-chains", type=int, default=0)
    parser.add_argument("--dev-chains", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="every training chain is rotated at random in every update (the flow's batches are), and the dev chains are read rotated as well as unrotated",
    )
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    posterior = bool(
        args.posterior_native or args.code_sigma > 0 or args.stochastic_code
    )
    if posterior and not args.inplace:
        parser.error(
            "--posterior-native, --code-sigma and --stochastic-code are for the in-place latents"
        )
    if args.stochastic_code and not (
        args.source == "esmc" and args.posterior_native and args.kl > 0
    ):
        parser.error(
            "--stochastic-code is for --source esmc with --posterior-native and a --kl above zero"
        )
    if args.stochastic_code and args.code_sigma:
        parser.error(
            "--stochastic-code learns the width that --code-sigma fixes: give one"
        )
    if args.kl and not args.stochastic_code:
        parser.error(
            "--kl is the weight of the learned posterior: it needs --stochastic-code"
        )
    if args.kl_exact and not args.stochastic_code:
        parser.error(
            "--kl-exact is for the learned posterior: it needs --stochastic-code"
        )
    if args.native_code and not (
        args.source == "fixed" and args.inplace and args.posterior_native
    ):
        parser.error(
            "--native-code is for --source fixed --inplace with --posterior-native"
        )
    if args.native_code and args.code_sigma:
        parser.error(
            "--native-code carries the covariances of its own: it replaces --code-sigma"
        )
    if args.code_bank and not (
        args.source == "fixed" and args.inplace and args.posterior_native
    ):
        parser.error(
            "--code-bank is for --source fixed --inplace with --posterior-native"
        )
    if args.code_bank and args.native_code:
        parser.error(
            "--code-bank and --native-code are two tables for the same code: give one"
        )
    if args.standardize_code and not (
        args.source == "fixed" and not args.inplace and not args.residual
    ):
        parser.error(
            "--standardize-code is for --source fixed without --inplace and without --residual (the type-only code of the full-width latent)"
        )
    if args.noise is None:
        args.noise = 0.0 if posterior else 0.2
    if posterior and args.noise:
        parser.error(
            "--noise and the posterior (--posterior-native, --code-sigma, --stochastic-code) are two noises on the same channels: give one"
        )

    import torch
    from rfproteina.datapipes.schema.proteina import SPECS
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.model_masks import compute_ae_supervision_mask
    from rfproteina.partial_autoencoder.autoencoder import (
        AutoEncoder,
        blended_token_weights,
    )

    began = time.time()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.shards, args.ae = (str(Path(p).resolve()) for p in (args.shards, args.ae))
    if args.native_code:
        args.native_code = str(Path(args.native_code).resolve())
    if args.code_bank:
        args.code_bank = str(Path(args.code_bank).resolve())
    os.chdir(out)  # the autoencoder makes ./tmp_ae/val_stuff in the working directory
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    index = json.loads((Path(args.shards) / "index.json").read_text())
    train_ids = [r["id"] for r in index["train"]][: args.train_chains or None]
    dev_ids = [r["id"] for r in index["dev"]][: args.dev_chains or None]
    train = Shards(Path(args.shards), train_ids)
    dev = Shards(Path(args.shards), dev_ids)
    print(
        f"{len(train_ids)} train, {len(dev_ids)} dev chains read in {time.time() - began:.0f} s",
        flush=True,
    )

    ae = AutoEncoder.load_from_checkpoint(
        args.ae, strict=True, weights_only=False, map_location="cpu"
    ).to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    cfg = ae.cfg_ae.loss
    mask_token = ENC.token_to_idx["<M>"]
    defaults = {k: SPECS[k][1] for k in KEYS if k in SPECS}
    legal = ae.decoder.residue_atom_mask

    # the width of the rows the source reads: the shards' `esmc` key holds ESMC's last state (2,560) unless the shards were made with
    # another source's rows
    in_dim = int(train.items[0]["esmc"].shape[-1]) if args.source == "esmc" else 2560
    # the rows of the standard residues, among which a fixed code's types are kept apart
    import foldspace.rfd4.ae_datapath as dp

    standard_rows = tuple(
        i for i in range(ENC.n_tokens) if ENC.idx_to_token[i] in dp.THREE_TO_ONE
    )
    latent = anchored_class()(
        args.source,
        ENC.n_tokens,
        residual=args.residual,
        window=args.window,
        freeze_embed=args.freeze_embed,
        inplace=args.inplace,
        in_dim=in_dim,
        standard_rows=standard_rows,
        stochastic=args.stochastic_code,
        calibrated=bool(args.native_code),
        banked=int(np.load(args.code_bank)["code_bank"].shape[0])
        if args.code_bank
        else 0,
    ).to(device)
    latent.code_sigma = float(args.code_sigma)
    latent.kl_exact_in_loss = bool(args.kl_exact)
    if args.native_code:
        load_native_code(latent, args.native_code)
        print(
            f"native-calibrated code of {latent.code.shape[1]} channels from {args.native_code}",
            flush=True,
        )
    elif args.code_bank:
        load_code_bank(latent, args.code_bank)
        print(
            f"banked code of {latent.code.shape[1]} channels, {latent.banked} donors, from {args.code_bank}; noise {latent.code_sigma}",
            flush=True,
        )
    elif args.source == "fixed" and args.inplace:
        print(
            f"fixed code of {latent.code.shape[1]} channels; {len(standard_rows)} standard residues, the smallest distance between two of them "
            f"{code_separation(latent.code.cpu().numpy(), standard_rows):.2f}",
            flush=True,
        )
    code_moments = None
    if args.standardize_code:
        # the type-only code of the full-width latent: the same marginals as the ESMC latent's moment-matched channels, over the resolved training residues
        rows = int(latent.code.shape[0])
        counts = np.zeros(rows)
        for item in train.items:
            types = item["residue_type"].numpy()[item["is_resolved_mask"].numpy() > 0]
            counts += np.bincount(types[types >= 0], minlength=rows)[:rows]
        before = latent.code.detach().cpu().numpy().astype(np.float64)
        weight = counts / counts.sum()
        mean = (weight[:, None] * before).sum(0)
        std = np.sqrt((weight[:, None] * (before - mean) ** 2).sum(0))
        code_moments = {
            "mean_abs_max_before": float(np.abs(mean).max()),
            "std_range_before": [float(std.min()), float(std.max())],
            "residues": int(counts.sum()),
        }
        with torch.no_grad():
            latent.code.copy_(
                torch.from_numpy(standardize_table(before, counts)).to(latent.code)
            )
        print(
            f"fixed code of {latent.code.shape[1]} channels moment-matched over {int(counts.sum())} resolved training residues "
            f"(channel means up to {code_moments['mean_abs_max_before']:.3f} and spreads {std.min():.3f} to {std.max():.3f} before); "
            f"the smallest distance between two standard residues {code_separation(latent.code.cpu().numpy(), standard_rows):.2f}",
            flush=True,
        )

    def native_channels(batch: dict, sample: bool = False):
        """The release encoder's posterior mean on the all-atom batch (the geometry residual's input, and the channels kept in place), one of its samples when `sample`, or nothing."""
        if not (args.residual or args.inplace):
            return None
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            out = ae.encoder(batch)
            return out["z_latent"] if sample else out["mean"]

    decoder = copy.deepcopy(ae.decoder).to(device)
    for p in decoder.parameters():
        p.requires_grad_(True)
    n_new = sum(p.numel() for p in latent.parameters())
    print(
        f"decoder {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f} M parameters, latent {n_new / 1e6:.2f} M",
        flush=True,
    )

    rotated = rotated_items

    def evaluate(
        label: str,
        rotate: bool = False,
        sampled: bool = False,
        mode: str | None = None,
    ) -> dict:
        """A reading on the dev chains (`read_dev`): at the posterior's mean, or, with `sampled`, with the posterior the run trains under (the release encoder's samples in the channels kept and the
        code with its noise); `mode="code"` is a learned posterior's unshrunk code."""
        return read_dev(
            latent=latent,
            decoder=decoder,
            dev=dev,
            native_fn=native_channels,
            defaults=defaults,
            mask_token=mask_token,
            legal=legal,
            device=device,
            batch_size=args.batch,
            seed=args.seed,
            mode=mode or ("sampled" if sampled else "mean"),
            rotate=rotate,
            posterior_native=args.posterior_native,
            label=label,
        )

    @torch.no_grad()
    def latent_report() -> dict:
        """The anchored latent over the dev residues: per-channel spread, the directions it spans, how much of it the residue type fixes."""
        latent.eval()
        zs, types = [], []
        for start in range(0, len(dev), args.batch):
            batch = collate(
                dev.items[start : start + args.batch], defaults, mask_token, device
            )
            native = native_channels(batch)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                z = latent(batch, native=native)
            emit = (dec_emit(batch)) & batch["is_resolved_mask"].bool()
            zs.append(z.float()[emit].cpu().numpy().astype(np.float64))
            types.append(batch["residue_type"][emit].cpu().numpy())
        z, t = np.concatenate(zs), np.concatenate(types)
        latent.train()
        return ds.latent_summary(z, np.zeros_like(z), t)

    @torch.no_grad()
    def sigma_report() -> dict | None:
        """The learned widths over the dev residues: their mean and median, the share above 0.8 (a channel at that width has become the noise that the release flow is pretrained to generate) and the mean of each channel."""
        if not args.stochastic_code:
            return None
        latent.eval()
        scales = []
        for start in range(0, len(dev), args.batch):
            batch = collate(
                dev.items[start : start + args.batch], defaults, mask_token, device
            )
            scales.append(latent.posterior_scale(batch).float().cpu().numpy())
        latent.train()
        s = np.concatenate(scales)
        by_channel = s.mean(0)
        return {
            "mean": float(s.mean()),
            "median": float(np.median(s)),
            "share_above_0.8": float((s > 0.8).mean()),
            "channels_with_mean_above_0.8": int((by_channel > 0.8).sum()),
            "by_channel": [round(float(x), 4) for x in by_channel],
        }

    def dec_emit(batch: dict):
        from rfproteina.model_masks import compute_ae_masks

        return compute_ae_masks(batch)[1]

    log = (out / "log.jsonl").open("a")
    results = []

    def record(entry: dict) -> None:
        entry["seconds"] = round(time.time() - began, 1)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry), flush=True)

    sampled_results = []
    latent.train()
    decoder.train()
    first = evaluate("start")
    record(first)
    results.append(first)

    groups = [
        {"params": list(decoder.parameters()), "lr": args.lr},
        {"params": list(latent.parameters()), "lr": args.lr_new},
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.0)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = args.epochs * steps_per_epoch
    if args.max_steps:
        total = min(total, args.max_steps)
    rng = np.random.default_rng(args.seed)
    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        running: dict[str, list] = {}
        for s in range(steps_per_epoch):
            if step >= total:
                break
            items = [
                train.items[int(i)]
                for i in order[s * args.batch : (s + 1) * args.batch]
            ]
            if args.rotate:
                items = rotated(items, rng)
            batch = collate(items, defaults, mask_token, device)
            scale = min(1.0, (step + 1) / args.warmup)
            for g, base in zip(optimizer.param_groups, base_lrs, strict=True):
                g["lr"] = base * scale
            native = native_channels(batch, sample=args.posterior_native)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                z = latent(batch, noise=args.noise, native=native, sample=posterior)
                feed = dict(batch)
                feed["z_latent"] = z
                dec = decoder(feed)
            dec = {
                k: (v.float() if torch.is_floating_point(v) else v)
                for k, v in dec.items()
            }
            loss_mask = compute_ae_supervision_mask(batch)
            weights = blended_token_weights(
                batch["residue_type"], loss_mask, cfg.balanced_fraction
            )
            losses = {}
            losses.update(
                ae.compute_struct_rec_loss(
                    dec,
                    batch,
                    loss_mask,
                    reduce_mode="sum",
                    loss_ty=cfg.struct.type,
                    weight=cfg.struct.weight,
                    token_weights=weights,
                )
            )
            losses.update(
                ae.compute_seq_rec_loss(
                    dec, batch, loss_mask, weight=cfg.seq.weight, token_weights=weights
                )
            )
            losses.update(
                ae.compute_atom_presence_loss(
                    dec,
                    batch,
                    loss_mask,
                    weight=cfg.atom_presence.weight,
                    token_weights=weights,
                )
            )
            loss = sum(
                losses[k].mean() for k in ("struct_l2", "ce_seq", "atom_presence_bce")
            )
            if args.stochastic_code:
                # the information the code gives up, on the residues of this batch (the width the reconstruction pushes down and this pushes up)
                loss = loss + args.kl * latent.last_kl
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"the loss is {float(loss)} at step {step + 1}: a run that has diverged is stopped, not carried to its end"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) + list(latent.parameters()), 1.0
            )
            optimizer.step()
            step += 1
            for k, v in (
                ("loss", loss),
                ("ce_seq", losses["ce_seq"].mean()),
                ("struct_l2", losses["struct_l2"].mean()),
                ("presence", losses["atom_presence_bce"].mean()),
            ):
                running.setdefault(k, []).append(float(v.detach()))
            if args.stochastic_code:
                running.setdefault("kl", []).append(float(latent.last_kl.detach()))
                running.setdefault("rate", []).append(float(latent.last_rate.detach()))
                running.setdefault("kl_exact", []).append(
                    float(latent.last_kl_exact.detach())
                )
                running.setdefault("sigma", []).append(float(latent.last_sigma.mean()))
            if step % 25 == 0:
                record(
                    {
                        "step": step,
                        "epoch": epoch,
                        **{k: round(float(np.mean(v)), 5) for k, v in running.items()},
                    }
                )
                running = {}
        result = evaluate(f"epoch {epoch + 1}")
        record(result)
        results.append(result)
        if posterior:
            sampled_result = evaluate(f"epoch {epoch + 1} sampled", sampled=True)
            record(sampled_result)
            sampled_results.append(sampled_result)
        if args.rotate:
            result = evaluate(f"epoch {epoch + 1} rotated", rotate=True)
            record(result)
            results.append(result)
        if step >= total:
            break
    torch.save(
        {
            "latent": {k: v.detach().cpu() for k, v in latent.state_dict().items()},
            "decoder": {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
            # what a flow's target is made of when this checkpoint's encoder runs: absent (deterministic) in the first prototypes
            "posterior": {
                "native_sample": bool(args.posterior_native),
                "code_sigma": float(args.code_sigma),
                "learned_sigma": bool(args.stochastic_code),
            },
        },
        out / "autoencoder.pt",
    )
    summary = {
        "source": args.source,
        "in_dim": in_dim,
        "steps": step,
        "epochs": epoch + 1,
        "train_chains": len(train_ids),
        "dev_chains": len(dev_ids),
        "batch": args.batch,
        "lr": args.lr,
        "lr_new": args.lr_new,
        "noise": args.noise,
        "posterior_native": bool(args.posterior_native),
        "code_sigma": float(args.code_sigma),
        "stochastic_code": bool(args.stochastic_code),
        "kl": float(args.kl),
        "kl_term": "exact" if args.kl_exact else "rate",
        "sigma": sigma_report(),
        "residual": args.residual,
        "rotate": args.rotate,
        "inplace": args.inplace,
        "window": args.window,
        "freeze_embed": args.freeze_embed,
        "ae": args.ae,
        "start": results[0],
        "end": results[-1],
        "end_sampled": sampled_results[-1] if sampled_results else None,
        # a learned posterior's `end` is read at its mean (`sqrt(1 - sigma^2) mu`); the unshrunk code, which the first autoencoders of this layout were read at, is a reading of its own
        "end_code": evaluate("end code", mode="code") if args.stochastic_code else None,
        "code_separation": (
            code_separation(latent.code.cpu().numpy(), standard_rows)
            if args.source == "fixed"
            and (args.inplace or args.standardize_code)
            and not args.native_code
            and not args.code_bank
            else None
        ),
        # the type-only code of the full-width latent: moment-matched over the training residues, and what its table was before
        "standardize_code": code_moments,
        "native_code": (
            {"path": args.native_code, "sha256": file_sha256(args.native_code)}
            if args.native_code
            else None
        ),
        "code_bank": (
            {
                "path": args.code_bank,
                "sha256": file_sha256(args.code_bank),
                "donors": latent.banked,
            }
            if args.code_bank
            else None
        ),
        "latent": latent_report(),
        "seconds": round(time.time() - began, 1),
        "stack_commit": os.environ.get("FS_STACK_COMMIT"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
