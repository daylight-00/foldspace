"""The sequence heads: what reads residue types off a standardised mixed single.

``LinearDecoder``
    One linear map, residue by residue: the baseline that says how much of the sequence is in a row of
    the single and nothing else.
``ContextualDecoder``
    A small pre-norm Transformer encoder over the whole chain, from scratch. Each row of the single
    already carries the context ESMC gave it, so the question is what attention over neighbouring rows
    adds. Besides the projected row it is given the distance to each end (saturating at 31) -- the audit
    found a position-fixed component on the first and last rows -- and a sinusoidal position, so that a
    neighbour at offset k is distinguishable from one at k+1.

Both take ``x`` ``[B, L, 256]`` (standardised, zero on padding) and ``lengths`` ``[B]`` and return
logits ``[B, L, 20]``; the output at a row does not depend on how much padding the batch has.
"""

from __future__ import annotations

import math

import torch
from torch import nn

N_CLASSES = 20
MAX_LENGTH = 512

__all__ = ["ContextualDecoder", "LinearDecoder", "build_decoder", "parameter_count"]


class LinearDecoder(nn.Module):
    def __init__(self, d_in: int = 256, n_classes: int = N_CLASSES, **_ignored):
        super().__init__()
        self.out = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        return self.out(x)


def _sinusoid(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    scale = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(position * scale)
    pe[:, 1::2] = torch.cos(position * scale)
    return pe


class ContextualDecoder(nn.Module):
    def __init__(
        self,
        d_in: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_classes: int = N_CLASSES,
        terminal: int = 32,
        **_ignored,
    ):
        super().__init__()
        self.terminal = terminal
        self.inp = nn.Linear(d_in, d_model)
        self.n_embedding = nn.Embedding(terminal, d_model)
        self.c_embedding = nn.Embedding(terminal, d_model)
        self.register_buffer("pe", _sinusoid(MAX_LENGTH, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_ff,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        if length > MAX_LENGTH:
            raise ValueError(
                f"a chain of {length} residues is beyond the {MAX_LENGTH} positions"
            )
        position = torch.arange(length, device=x.device)
        from_n = position.clamp(max=self.terminal - 1)[None, :].expand(b, length)
        from_c = (lengths[:, None] - 1 - position[None, :]).clamp(0, self.terminal - 1)
        h = (
            self.inp(x)
            + self.n_embedding(from_n)
            + self.c_embedding(from_c)
            + self.pe[:length]
        )
        padding = position[None, :] >= lengths[:, None]
        h = self.encoder(h, src_key_padding_mask=padding)
        return self.out(self.norm(h))


def build_decoder(config: dict) -> nn.Module:
    arch = config["arch"]
    if arch == "linear":
        return LinearDecoder(**config)
    if arch == "contextual":
        return ContextualDecoder(**config)
    raise ValueError(f"unknown decoder {arch!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
