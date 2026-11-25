"""A generator of the single, from scratch: a Transformer denoiser trained by flow matching.

The coordinate is the 256-wide mixed single of the folding model, standardised channel by channel with the training
split's statistics (``z``), or that single rotated onto the directions of the training covariance and divided by
each direction's standard deviation (``y``, whitened, with a floor so that the smallest directions are not blown up). Both
are invertible linear maps of the same single, so they are the same data in two coordinates; what a squared error and a
Gaussian noise mean differs between them, which is what latent generation compares.

``x_1`` is a chain's single in the coordinate in use, ``x_0`` Gaussian noise, ``x_t = (1 - t) x_0 + t x_1``. The network
predicts ``x_1`` from ``(x_t, t)``, and the velocity it implies is ``(x_1_hat - x_t) / (1 - t)``. Sampling integrates that
velocity from ``t = 0`` to ``t = 1`` with Heun steps and ends on the network's own prediction of ``x_1``.

Nothing here knows the sequence: the generator makes a single and the sequence is read off it afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "EMA",
    "Denoiser",
    "Whitening",
    "flow_inputs",
    "flow_loss",
    "sample",
]

MAX_LENGTH = 512
TERMINAL = 32  # distance from either end saturates here, as in the decoder


# --------------------------------------------------------------- coordinate


@dataclass
class Whitening:
    """An invertible linear map between the standardised single ``z`` and the coordinate the generator works in.

    ``Whitening.identity`` leaves ``z`` as it is. Otherwise ``y = (z @ V) / sqrt(values + floor)`` with ``V`` the
    eigenvectors of the training covariance, so that in every direction the data has a variance close to one (below it,
    only for the directions whose variance is no larger than ``floor``).
    """

    vectors: torch.Tensor | None = None
    scale: torch.Tensor | None = None

    @classmethod
    def identity(cls) -> Whitening:
        return cls(None, None)

    @classmethod
    def from_spectrum(cls, vectors, values, *, floor: float, device) -> Whitening:
        vectors = torch.as_tensor(vectors, dtype=torch.float32, device=device)
        values = torch.as_tensor(values, dtype=torch.float32, device=device)
        return cls(vectors, (values + floor).sqrt())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.vectors is None:
            return z
        return (z @ self.vectors) / self.scale

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.vectors is None:
            return y
        return (y * self.scale) @ self.vectors.T


# -------------------------------------------------------------------- model


def _sinusoid(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    rate = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    out = torch.zeros(length, dim)
    out[:, 0::2] = torch.sin(position * rate)
    out[:, 1::2] = torch.cos(position * rate)
    return out


def _time_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """``t`` in [0, 1], one per chain: sinusoids of ``1000 t``."""
    half = dim // 2
    freq = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device)
        * (-math.log(10000.0) / half)
    )
    angle = (1000.0 * t.float())[:, None] * freq[None]
    return torch.cat([angle.sin(), angle.cos()], dim=-1)


class Block(nn.Module):
    """Pre-norm attention and feed-forward, each modulated by the time (adaptive layer norm, zero-initialised)."""

    def __init__(self, d: int, heads: int, d_ff: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, valid):
        b, n, d = x.shape
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c)).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1[:, None]) + b1[:, None]
        q, k, v = (
            self.qkv(h)
            .view(b, n, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        x = x + g1[:, None] * self.proj(a.transpose(1, 2).reshape(b, n, d))
        h = self.norm2(x) * (1 + s2[:, None]) + b2[:, None]
        return x + g2[:, None] * self.mlp(h)


class Denoiser(nn.Module):
    """``(x_t, t, lengths) -> x_1_hat``, ``x_t`` being ``[B, L, d_data]`` with zero padding."""

    def __init__(
        self,
        d_data: int = 256,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 2048,
    ):
        super().__init__()
        self.inp = nn.Linear(d_data, d_model)
        self.n_term = nn.Embedding(TERMINAL, d_model)
        self.c_term = nn.Embedding(TERMINAL, d_model)
        self.register_buffer("pe", _sinusoid(MAX_LENGTH, d_model), persistent=False)
        self.time = nn.Sequential(
            nn.Linear(256, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_ada = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_data)
        nn.init.zeros_(self.out_ada.weight)
        nn.init.zeros_(self.out_ada.bias)

    def forward(self, x_t, t, lengths):
        b, n, _ = x_t.shape
        position = torch.arange(n, device=x_t.device)[None]
        valid = position < lengths[:, None]
        from_end = (lengths[:, None] - 1 - position).clamp(min=0)
        h = (
            self.inp(x_t)
            + self.pe[:n][None]
            + self.n_term(position.clamp(max=TERMINAL - 1).expand(b, -1))
            + self.c_term(from_end.clamp(max=TERMINAL - 1))
        )
        c = self.time(_time_embedding(t))
        for block in self.blocks:
            h = block(h, c, valid)
        scale, shift = self.out_ada(F.silu(c)).chunk(2, dim=-1)
        h = self.out_norm(h) * (1 + scale[:, None]) + shift[:, None]
        return self.out(h) * valid[..., None]


class EMA:
    """An exponential moving average of a model's parameters, kept apart and swapped in to evaluate.

    Given the number of updates made so far, the decay is held to ``(1 + n) / (10 + n)`` while that is under ``decay``, so that
    the average of the first few hundred steps is not mostly the initial random weights (at 0.999 it would be 67 % of them
    after 400 steps).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int | None = None) -> None:
        decay = (
            self.decay if step is None else min(self.decay, (1 + step) / (10 + step))
        )
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


# --------------------------------------------------------------------- flow


def flow_inputs(x1, lengths, generator=None, t=None):
    """``(x_t, t, noise)`` for clean ``x_1`` ``[B, L, D]``: one ``t`` per chain, noise zero on the padding."""
    b, n, _ = x1.shape
    valid = (torch.arange(n, device=x1.device)[None] < lengths[:, None])[..., None]
    noise = (
        torch.randn(x1.shape, device=x1.device, generator=generator, dtype=x1.dtype)
        * valid
    )
    if t is None:
        t = torch.rand(b, device=x1.device, generator=generator)
    tt = t[:, None, None].to(x1.dtype)
    return (1 - tt) * noise + tt * x1, t, noise


def flow_loss(x1_hat, x1, t, lengths, *, t_floor: float = 0.1):
    """Mean squared velocity error per element, over the valid positions, weighted ``1 / max(1 - t, t_floor)^2``.

    For a network that predicts ``x_1`` the velocity error is the prediction's error over ``1 - t``; the weight is that
    factor with the noise-free end held to ``t_floor`` so that it does not dominate. Returns the scalar loss and the
    unweighted mean squared error of ``x_1_hat``.
    """
    _, n, d = x1.shape
    valid = (torch.arange(n, device=x1.device)[None] < lengths[:, None]).float()
    err = ((x1_hat.float() - x1.float()) ** 2).sum(-1)  # [B, L]
    per_chain = (err * valid).sum(1) / (valid.sum(1) * d)  # [B]
    weight = 1.0 / (1.0 - t).clamp(min=t_floor) ** 2
    return (weight * per_chain).mean(), per_chain.mean().detach()


@torch.no_grad()
def sample(
    model,
    lengths,
    d_data: int = 256,
    steps: int = 50,
    generator=None,
    *,
    start=None,
    t_start: float = 0.0,
    shift: float = 1.0,
):
    """Heun steps of the velocity field from noise to a single in the generator's coordinate: ``[B, L, D]``.

    The last step lands on the network's prediction of ``x_1``, so that the division by ``1 - t`` is never made at ``t = 1``.
    With ``start`` (a partly noised single ``x_t`` at ``t_start``, ``[B, L, D]`` with zero padding) the steps run from
    ``t_start`` and not from pure noise: a single that is completed and not made from nothing.

    ``shift`` bends the time grid: ``steps`` equal intervals ``u`` become ``t = u / (u + shift (1 - u))``, so that a
    ``shift`` above 1 spends more of the steps near ``t = 0``, where the noise is large and the global form is chosen.
    """
    device = lengths.device
    b, n = len(lengths), int(lengths.max())
    valid = (torch.arange(n, device=device)[None] < lengths[:, None])[..., None]
    if start is None:
        x = torch.randn(b, n, d_data, device=device, generator=generator) * valid
    else:
        if not 0.0 <= t_start < 1.0:
            raise ValueError(f"t_start is in [0, 1), not {t_start}")
        x = start.to(device).float() * valid
    u = torch.linspace(0.0, 1.0, steps + 1, device=device)
    ts = u / (
        u + shift * (1.0 - u)
    )  # 0 at the start of the grid and 1 at its end, whatever the shift
    ts = float(t_start) + (1.0 - float(t_start)) * ts

    def predict(x, t):
        return model(x, t.expand(b), lengths).float()

    for k in range(steps):
        t, t_next = ts[k], ts[k + 1]
        x1_hat = predict(x, t)
        if k == steps - 1:
            x = x1_hat * valid
            break
        v = (x1_hat - x) / (1 - t)
        euler = x + (t_next - t) * v
        v2 = (predict(euler, t_next) - euler) / (1 - t_next)
        x = (x + (t_next - t) * 0.5 * (v + v2)) * valid
    return x
