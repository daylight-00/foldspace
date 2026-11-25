"""A learned token-only baseline for the flow's readings: the best predictor of a token's clean latent that sees that
token's own noisy latent and the time and nothing of the chain's context.

`flow_eval.py` reads a flow against two baselines of the token's own noisy latent, and both are weak where it matters. The
per-channel marginals ignore that the channels depend on one another (the 24 ESMC channels are a map of a state that is near one of
twenty kinds), and the empirical joint of 65,536 training tokens is a kernel estimate in 32 dimensions, 16 of which are noise at the
native latent's prior, so that at a high `t` it snaps to one training token and is far worse than the marginals (at t = 0.9 about 48
times, for the pretraining checkpoint on its own latent): a skill over it at a high `t` says that the baseline is poor, not that the flow
learned. What the reading needs is a predictor that is as good as the token's own information allows, so that what is left below it is
the context. This fits one: a small network on the token's whole noisy latent, `x_t = t x_1 + (1 - t) x_0` with `x_0` standard normal,
trained on the clean latents of training tokens to minimise the squared error of `E[x_1 | x_t, t]`, with the posterior mean of a
unit-variance Gaussian prior as its starting point (the last layer is zero, so that it can only improve on it), and checked on held-out
training tokens against the per-channel marginals' posterior mean, which a network of this kind can represent and must not be worse than.

A baseline that is too weak makes a flow look better than it is, and one that is undertrained is too weak; the held-out comparison is the
guard, and a time at which the network is not at least as good as the marginals (within 1 %) is not read with it (`validation` says
which).
"""

from __future__ import annotations

import itertools
import math

import torch


def gain(t: torch.Tensor) -> torch.Tensor:
    """The coefficient of the posterior mean of a unit-variance Gaussian prior: `E[x_1 | x_t] = t / (t^2 + (1 - t)^2) x_t`."""
    return t / (t * t + (1.0 - t) ** 2)


class TokenDenoiser(torch.nn.Module):
    """`x_t, t -> E[x_1 | x_t, t]` for one token: the Gaussian shrinkage plus a learned correction of the same shape as the latent."""

    def __init__(self, dim: int, hidden: int = 512, layers: int = 4) -> None:
        super().__init__()
        self.dim = dim
        sizes = [dim + 2] + [hidden] * layers
        blocks: list[torch.nn.Module] = []
        for a, b in itertools.pairwise(sizes):
            blocks += [torch.nn.Linear(a, b), torch.nn.SiLU()]
        self.body = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Linear(hidden, dim)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, xt: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=xt.dtype, device=xt.device)
        if t.dim() == 0:
            t = t.expand(xt.shape[0])
        tt = t[:, None]
        # the noisy latent seen in the clean latent's units, the noise-to-signal ratio on a log scale, and the time
        scaled = xt / tt
        feats = torch.cat([scaled, torch.log((1.0 - tt) / tt).clamp(-6, 6), tt], dim=-1)
        return gain(tt) * xt + self.head(self.body(feats))


def fit_token_denoiser(
    pool: torch.Tensor,
    steps: int = 6000,
    batch: int = 8192,
    hidden: int = 512,
    layers: int = 4,
    lr: float = 2e-3,
    holdout: float = 0.1,
    seed: int = 0,
    t_range: tuple[float, float] = (0.03, 0.97),
) -> tuple[TokenDenoiser, torch.Tensor, dict]:
    """Fit on the clean latents `pool` `[N, d]` of training tokens (the last `holdout` share is kept out, and returned for the check).
    The noise and the times are drawn afresh at every step from a generator of its own, so that nothing here moves the caller's random
    streams. Returns the network in `eval()`, the held-out tokens and a summary."""
    device = pool.device
    gen = torch.Generator(device=device).manual_seed(seed)
    n = len(pool)
    cut = n - max(1, int(n * holdout))
    train, held = pool[:cut], pool[cut:]
    with torch.random.fork_rng(devices=[]):
        # the initial weights come from the seed and leave the caller's CPU stream where it was (the layers are made on the CPU)
        torch.manual_seed(seed)
        net = TokenDenoiser(pool.shape[1], hidden, layers)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    lo, hi = t_range
    losses = []
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * step / steps))
        idx = torch.randint(0, cut, (batch,), generator=gen, device=device)
        x1 = train[idx]
        t = lo + (hi - lo) * torch.rand(batch, generator=gen, device=device)
        x0 = torch.randn(x1.shape, generator=gen, device=device)
        xt = t[:, None] * x1 + (1.0 - t[:, None]) * x0
        loss = ((net(xt, t) - x1) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            losses.append((step, float(loss.detach())))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    info = {
        "train_tokens": int(cut),
        "heldout_tokens": int(n - cut),
        "steps": steps,
        "batch": batch,
        "hidden": hidden,
        "layers": layers,
        "lr": lr,
        "seed": seed,
        "parameters": int(sum(p.numel() for p in net.parameters())),
        "training_loss_trace": losses,
    }
    return net, held, info


@torch.no_grad()
def validate(
    net: TokenDenoiser,
    held: torch.Tensor,
    times: list[float],
    reference,
    repeats: int = 4,
    seed: int = 1,
    tolerance: float = 0.01,
) -> dict:
    """Held-out error per token (summed over the channels) of the network and of `reference(x_t, t)` (the marginals' posterior mean)
    at each time, on noise that neither saw in training. `ok` is whether the network is at least as good as the reference up to
    `tolerance` (where the marginals are all but exact, at a high `t` for channels that do not depend on one another, a network that
    reproduces them cannot be expected to be a hair better)."""
    gen = torch.Generator(device=held.device).manual_seed(seed)
    out = {}
    for t in times:
        e_net = e_ref = 0.0
        for _ in range(repeats):
            x0 = torch.randn(held.shape, generator=gen, device=held.device)
            xt = t * held + (1.0 - t) * x0
            e_net += float(((net(xt, t) - held) ** 2).sum(-1).mean()) / repeats
            e_ref += float(((reference(xt, t) - held) ** 2).sum(-1).mean()) / repeats
        out[f"{t:g}"] = {
            "network": e_net,
            "reference": e_ref,
            "ok": e_net <= e_ref * (1.0 + tolerance),
        }
    return out
