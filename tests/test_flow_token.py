"""The learned token-only baseline of `src/foldspace/rfd4/flow_token.py`: it starts at the Gaussian shrinkage, it comes
close to the exact posterior mean where that is known, it beats what the per-channel marginals can do when the channels depend on one
another, and its held-out check says so. Needs torch, which the CI environment may not have."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import foldspace.rfd4.flow_token as ft

SIGMA = 0.4


def mixture(n: int, d: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two spherical clusters at +-mu in `d` dimensions, every channel at the same sign: the channels are dependent."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.full((d,), 2.0)
    sign = (torch.rand(n, generator=g) < 0.5).float() * 2 - 1
    return sign[:, None] * mu[None] + SIGMA * torch.randn(n, d, generator=g), mu


def exact_posterior_mean(xt: torch.Tensor, t: float, mu: torch.Tensor) -> torch.Tensor:
    """`E[z | x_t]` for the two-cluster mixture, in closed form."""
    v = t * t * SIGMA**2 + (1 - t) ** 2
    means = torch.stack([mu, -mu])  # [2, d]
    logw = -((xt[:, None, :] - t * means[None]) ** 2).sum(-1) / (2 * v)  # [N, 2]
    w = torch.softmax(logw, dim=1)
    cond = means[None] + (t * SIGMA**2 / v) * (xt[:, None, :] - t * means[None])
    return (w[..., None] * cond).sum(1)


def marginal_posterior_mean(
    xt: torch.Tensor, t: float, mu: torch.Tensor
) -> torch.Tensor:
    """What the per-channel marginals give: each channel's own two-cluster posterior mean, the channels taken as independent."""
    out = torch.empty_like(xt)
    for c in range(xt.shape[1]):
        out[:, c] = exact_posterior_mean(xt[:, c : c + 1], t, mu[c : c + 1])[:, 0]
    return out


def test_it_starts_at_the_gaussian_shrinkage() -> None:
    net = ft.TokenDenoiser(6, hidden=16, layers=2)
    xt = torch.randn(50, 6)
    for t in (0.1, 0.5, 0.9):
        tt = torch.tensor(t)
        assert torch.allclose(net(xt, t), ft.gain(tt) * xt, atol=1e-6)


def test_the_gain_is_the_posterior_mean_of_a_unit_gaussian() -> None:
    # for x_1 ~ N(0, 1) and x_t = t x_1 + (1 - t) x_0: E[x_1 | x_t] = t / (t^2 + (1 - t)^2) x_t
    g = torch.Generator().manual_seed(0)
    x1, x0 = torch.randn(200_000, generator=g), torch.randn(200_000, generator=g)
    t = 0.3
    xt = t * x1 + (1 - t) * x0
    slope = float((xt * x1).mean() / (xt * xt).mean())
    assert slope == pytest.approx(float(ft.gain(torch.tensor(t))), abs=0.01)


def test_it_approaches_the_exact_posterior_mean_and_beats_the_marginals() -> None:
    d = 6
    pool, mu = mixture(12_000, d, seed=0)
    net, held, info = ft.fit_token_denoiser(
        pool, steps=2500, batch=4096, hidden=128, layers=3, seed=0
    )
    assert info["train_tokens"] + info["heldout_tokens"] == len(pool)
    assert not any(p.requires_grad for p in net.parameters())
    gen = torch.Generator().manual_seed(5)
    for t in (0.3, 0.5, 0.7):
        x0 = torch.randn(held.shape, generator=gen)
        xt = t * held + (1 - t) * x0
        e_net = float(((net(xt, t) - held) ** 2).sum(-1).mean())
        e_bayes = float(((exact_posterior_mean(xt, t, mu) - held) ** 2).sum(-1).mean())
        e_marg = float(
            ((marginal_posterior_mean(xt, t, mu) - held) ** 2).sum(-1).mean()
        )
        assert e_net <= 1.15 * e_bayes, (t, e_net, e_bayes)
        # where the marginals are all but exact (0.7) a network can only reproduce them
        assert e_net <= 1.01 * e_marg, (t, e_net, e_marg)
    # where the channels' dependence is what tells the cluster, the marginals lose clearly
    t = 0.3
    x0 = torch.randn(held.shape, generator=gen)
    xt = t * held + (1 - t) * x0
    e_net = float(((net(xt, t) - held) ** 2).sum(-1).mean())
    e_marg = float(((marginal_posterior_mean(xt, t, mu) - held) ** 2).sum(-1).mean())
    assert e_net < 0.8 * e_marg


def test_the_held_out_check_flags_a_baseline_that_is_worse_than_its_reference() -> None:
    pool, mu = mixture(4_000, 4, seed=1)
    net, held, _ = ft.fit_token_denoiser(
        pool, steps=300, batch=1024, hidden=32, layers=2, seed=0
    )
    good = ft.validate(net, held, [0.3, 0.6], lambda x, t: ft.gain(torch.tensor(t)) * x)
    assert set(good) == {"0.3", "0.6"} and all(v["ok"] for v in good.values())
    # a reference that is the exact posterior mean cannot be beaten by a network trained on 300 steps
    best = ft.validate(
        net, held, [0.3], lambda x, t: exact_posterior_mean(x, t, mu), repeats=2
    )
    assert best["0.3"]["ok"] is False


def test_training_does_not_move_the_callers_random_stream() -> None:
    pool, _ = mixture(1_000, 3, seed=2)
    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    ft.fit_token_denoiser(pool, steps=20, batch=256, hidden=16, layers=2, seed=9)
    after = torch.rand(3)
    assert torch.equal(before, after)
