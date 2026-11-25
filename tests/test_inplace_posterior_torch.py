"""The in-place latents' posterior at torch level (`src/foldspace/rfd4/train_autoencoder.py`, `src/foldspace/rfd4/flow_model.py`): a fixed code in place, a code that carries Gaussian noise of a fixed scale or a learned
width when it is sampled, and an encoder that gives a flow the release encoder's *samples* in the channels kept in place. They need torch and the release's model-mask module: they are skipped where the stack is not installed
and run in the stack's environment. The arithmetic that needs numpy only, and the argument combinations the trainer refuses, are in `test_inplace_posterior.py`."""

from __future__ import annotations

import math

import pytest
import foldspace.rfd4.train_autoencoder as autoencoder

pytestmark = pytest.mark.offline

STANDARD = tuple(range(1, 21))

torch = pytest.importorskip("torch")
pytest.importorskip("rfproteina.model_masks")
import foldspace.rfd4.flow_model as fm

BATCH, LENGTH = 4, 400


def batch() -> dict:
    return {
        "padding_mask": torch.ones(BATCH, LENGTH, dtype=torch.bool),
        "is_in_context_conditioning": torch.zeros(BATCH, LENGTH, dtype=torch.bool),
        "supports_local_latent": torch.ones(BATCH, LENGTH, dtype=torch.bool),
        "residue_type": torch.randint(1, 21, (BATCH, LENGTH)),
        "esmc": torch.randn(BATCH, LENGTH, 64),
    }


def fixed_latent():
    return autoencoder.anchored_class()("fixed", 30, inplace=True, standard_rows=STANDARD)


def test_a_fixed_code_in_place_has_its_code_in_the_18_channels_and_the_release_in_the_14() -> (
    None
):
    latent = fixed_latent()
    b = batch()
    native = torch.randn(BATCH, LENGTH, 32)
    z = latent(b, native=native)
    keep, code = latent.keep, latent.code_idx
    assert z.shape == (BATCH, LENGTH, 32)
    assert len(code) == 18 and sorted(keep + code) == list(range(32))
    assert torch.equal(z[..., keep], native[..., keep])
    assert torch.equal(z[..., code], latent.code[b["residue_type"]])
    # the table is a buffer of the checkpoint and its layout says so: the encoder is rebuilt from the state dict
    assert "code" in latent.state_dict() and "inplace_flag" in latent.state_dict()


@pytest.mark.parametrize("source", ["fixed", "esmc"])
def test_a_sampled_code_carries_noise_of_its_scale_and_nothing_else_changes(
    source: str,
) -> None:
    if source == "fixed":
        latent = fixed_latent()
    else:
        latent = autoencoder.anchored_class()("esmc", 30, inplace=True, in_dim=64)
    latent.eval()
    latent.code_sigma = 0.3
    b = batch()
    native = torch.randn(BATCH, LENGTH, 32)
    mean = latent(b, native=native)
    torch.manual_seed(0)
    sampled = latent(b, native=native, sample=True)
    delta = sampled - mean
    keep, code = latent.keep, latent.code_idx
    assert torch.equal(delta[..., keep], torch.zeros_like(delta[..., keep]))
    assert delta[..., code].std().item() == pytest.approx(0.3, rel=0.05)
    assert float(delta[..., code].detach().mean().abs()) < 0.01
    # without `sample` the code is what it was, and a latent with no posterior ignores the request
    assert torch.equal(latent(b, native=native), mean)
    latent.code_sigma = 0.0
    assert torch.equal(latent(b, native=native, sample=True), mean)


class StubNative(torch.nn.Module):
    """A release encoder whose sample differs from its mean by exactly one in every channel."""

    def forward(self, b: dict) -> dict:
        mean = torch.zeros(BATCH, LENGTH, 32)
        return {"mean": mean, "z_latent": mean + 1.0}


def test_the_encoder_gives_the_flow_the_release_encoders_samples_when_the_checkpoint_has_a_posterior() -> (
    None
):
    latent = fixed_latent()
    b = batch()
    keep, code = latent.keep, latent.code_idx
    first_prototype = fm.esm_encoder_class()(StubNative(), latent, 0)
    out = first_prototype(b)
    # no posterior: the release's means in the channels kept and a deterministic code, as the first in-place layout arm's flow was given
    assert torch.equal(out["z_latent"], out["mean"])
    assert torch.all(out["z_latent"][..., keep] == 0.0)
    posterior = fm.esm_encoder_class()(StubNative(), latent, 0, native_sample=True)
    out = posterior(b)
    assert torch.all(out["z_latent"][..., keep] == 1.0)
    assert torch.all(out["mean"][..., keep] == 0.0)
    # the code is the same in both while its scale is zero, and the encoder's output is read by the release's flow as `z_latent`
    assert torch.equal(out["z_latent"][..., code], out["mean"][..., code])
    latent.code_sigma = 0.5
    noisy = fm.esm_encoder_class()(StubNative(), latent, 0, native_sample=True)(b)
    assert not torch.equal(noisy["z_latent"][..., code], noisy["mean"][..., code])
    assert torch.equal(noisy["mean"][..., code], latent.code[b["residue_type"]])


def esmc_latent(stochastic: bool = True):
    latent = autoencoder.anchored_class()(
        "esmc", 30, inplace=True, in_dim=64, stochastic=stochastic
    )
    latent.eval()
    return latent


def test_a_learned_posterior_starts_at_a_width_of_a_fifth_and_only_a_latent_that_has_one_reports_it() -> (
    None
):
    latent = esmc_latent()
    sigma = latent.posterior_scale(batch())
    assert sigma.shape == (BATCH * LENGTH, 18)
    assert torch.allclose(sigma, torch.full_like(sigma, 0.2), atol=1e-6)
    assert "sigma_head.weight" in latent.state_dict()
    with pytest.raises(ValueError, match="no learned posterior"):
        esmc_latent(stochastic=False).posterior_scale(batch())
    with pytest.raises(ValueError, match="in-place ESMC code"):
        autoencoder.anchored_class()("fixed", 30, inplace=True, stochastic=True)


def test_the_learned_posterior_keeps_a_unit_variance_and_the_channels_kept_and_reports_what_it_gives_up() -> (
    None
):
    latent = esmc_latent()
    b = batch()
    native = torch.randn(BATCH, LENGTH, 32)
    # the moment matching takes the statistics of one batch (as its training does), so that the means have zero mean and unit variance
    latent.moment.momentum = 1.0
    latent.train()
    latent(b, native=native)
    latent.eval()
    with torch.no_grad():
        latent.sigma_head.bias.fill_(0.0)  # half of the widest width, everywhere
    mean = latent(
        b, native=native, unshrunk=True
    )  # the code `mu`, whose second moment is one
    torch.manual_seed(1)
    z = latent(b, native=native, sample=True)
    keep, code = latent.keep, latent.code_idx
    assert torch.equal(z[..., keep], native[..., keep])
    mu = mean[..., code].reshape(-1, 18)
    drawn = z[..., code].reshape(-1, 18)
    sigma = latent.posterior_scale(b)
    width = (
        0.5 * autoencoder.SIGMA_MAX
    )  # a head at zero is a sigmoid of one half, of the widest width allowed
    assert torch.allclose(sigma, torch.full_like(sigma, width), atol=1e-6)
    # the sample is sqrt(1 - sigma^2) mu + sigma eps with eps standard normal, and keeps the unit variance of the means
    eps = (drawn - torch.sqrt(1 - sigma**2) * mu) / sigma
    eps, drawn = eps.detach(), drawn.detach()
    assert float(eps.std()) == pytest.approx(1.0, abs=0.02)
    assert abs(float(eps.mean())) < 0.02
    assert float(drawn.var(0).mean()) == pytest.approx(1.0, abs=0.05)
    # what the code gives up: the sum over channels of -ln sigma, averaged over residues
    assert float(latent.last_kl.detach()) == pytest.approx(
        -18 * math.log(width), rel=1e-4
    )
    assert latent.last_sigma.shape == sigma.shape


def test_the_encoder_gives_the_flow_a_sample_of_a_learned_posterior_and_its_mean_beside_it() -> (
    None
):
    latent = esmc_latent()
    b = batch()
    keep, code = latent.keep, latent.code_idx
    out = fm.esm_encoder_class()(StubNative(), latent, 0)(b)
    # the release encoder's means in the channels kept (no sampled native channels asked for), a sampled code in `z_latent` and the code's means in `mean`
    assert torch.all(out["z_latent"][..., keep] == 0.0)
    assert not torch.equal(out["z_latent"][..., code], out["mean"][..., code])
    assert torch.equal(
        out["mean"][..., code],
        latent(b, native=torch.zeros(BATCH, LENGTH, 32))[..., code],
    )
    # and that mean is the posterior's mean, `sqrt(1 - sigma^2) mu`, not the code itself
    mu = latent(b, native=torch.zeros(BATCH, LENGTH, 32), unshrunk=True)[..., code]
    sigma = latent.posterior_scale(b).reshape(BATCH, LENGTH, 18)
    assert torch.allclose(
        out["mean"][..., code], torch.sqrt(1 - sigma**2) * mu, atol=1e-6
    )


def widened(latent, bias: float = 0.0, momentum_batch: dict | None = None):
    """A learned posterior of one width everywhere (the head's bias with zero weights), its moment matching taken from `momentum_batch` so that the code has zero mean and unit variance there."""
    if momentum_batch is not None:
        latent.moment.momentum = 1.0
        latent.train()
        latent(momentum_batch, native=torch.randn(BATCH, LENGTH, 32))
        latent.eval()
    with torch.no_grad():
        latent.sigma_head.bias.fill_(bias)
    return latent


def test_without_sample_a_learned_posterior_is_its_mean_and_unshrunk_gives_the_code() -> (
    None
):
    latent = widened(esmc_latent(), 1.0)
    b = batch()
    native = torch.randn(BATCH, LENGTH, 32)
    code = latent.code_idx
    sigma = latent.posterior_scale(b).reshape(BATCH, LENGTH, 18)
    mu = latent(b, native=native, unshrunk=True)[..., code]
    mean = latent(b, native=native)[..., code]
    # the posterior is N(sqrt(1 - sigma^2) mu, sigma^2): its mean is the code scaled down, by 0.14 where the width is at its widest
    assert torch.allclose(mean, torch.sqrt(1 - sigma**2) * mu, atol=1e-6)
    sigma, mean, mu = sigma.detach(), mean.detach(), mu.detach()
    assert float(sigma.mean()) > 0.6 and float((mean.abs() / mu.abs()).mean()) < 0.8
    # the channels kept are the release encoder's, whatever is asked of the code
    keep = latent.keep
    assert torch.equal(latent(b, native=native)[..., keep], native[..., keep])
    # a fixed or a deterministic code has no shrinking: its mean is the code
    fixed = fixed_latent()
    assert torch.equal(fixed(b, native=native), fixed(b, native=native, unshrunk=True))


def test_the_learned_posterior_has_the_mean_and_the_variance_it_says() -> None:
    latent = widened(esmc_latent(), 0.0)
    b = batch()
    native = torch.randn(BATCH, LENGTH, 32)
    code = latent.code_idx
    mean = latent(b, native=native)[..., code].detach()
    sigma = latent.posterior_scale(b).reshape(BATCH, LENGTH, 18).detach()
    torch.manual_seed(2)
    draws = torch.stack(
        [latent(b, native=native, sample=True)[..., code] for _ in range(300)]
    ).detach()
    # the Monte Carlo mean of the samples is the posterior mean and their variance is sigma^2 (here the same everywhere)
    assert float((draws.mean(0) - mean).abs().mean()) < 0.04
    assert float(draws.var(0).mean()) == pytest.approx(
        float(sigma.pow(2).mean()), rel=0.05
    )


def test_the_rate_term_is_the_kl_where_the_code_has_unit_second_moment_and_the_omitted_term_is_the_rest() -> (
    None
):
    b = batch()
    native = torch.randn(BATCH, LENGTH, 32)
    # a width that does not depend on the residue and a code of unit second moment in every channel: the two regularizers are the same number
    latent = widened(esmc_latent(), 0.5, momentum_batch=b)
    latent(b, native=native, sample=True)
    assert float(latent.last_kl_exact.detach()) == pytest.approx(
        float(latent.last_rate.detach()), rel=2e-3
    )
    assert float(latent.last_kl.detach()) == float(
        latent.last_rate.detach()
    )  # the rate term is the default term
    # where the code's second moment is not one they differ by 1/2 (1 - sigma^2)(mu^2 - 1) summed over the channels, which is what the exact KL adds
    latent = widened(esmc_latent(), 0.5)
    latent(b, native=native, sample=True)
    mu = latent(b, native=native, unshrunk=True)[..., latent.code_idx].reshape(-1, 18)
    sigma = latent.posterior_scale(b)
    omitted = (0.5 * (1 - sigma**2) * (mu**2 - 1)).sum(-1).mean()
    exact = (
        (0.5 * ((1 - sigma**2) * mu**2 + sigma**2 - 1 - torch.log(sigma**2)))
        .sum(-1)
        .mean()
    )
    assert float(latent.last_kl_exact.detach()) == pytest.approx(float(exact), rel=1e-4)
    assert float((latent.last_kl_exact - latent.last_rate).detach()) == pytest.approx(
        float(omitted), rel=1e-3, abs=1e-4
    )
    assert (
        abs(float(omitted)) > 0.1
    )  # a code whose moment is not matched: the two are different numbers here


def test_the_exact_kl_can_be_the_term_in_the_loss() -> None:
    latent = widened(esmc_latent(), 0.5)
    latent.kl_exact_in_loss = True
    latent(batch(), native=torch.randn(BATCH, LENGTH, 32), sample=True)
    assert float(latent.last_kl.detach()) == float(latent.last_kl_exact.detach())
    assert float(latent.last_kl.detach()) != float(latent.last_rate.detach())


def calibrated_latent(seed: int = 0):
    """A fixed in-place code whose table is a Gaussian for each type: the mean of the 18 channels and the Cholesky factor of their covariance."""
    latent = autoencoder.anchored_class()("fixed", 30, inplace=True, calibrated=True)
    rng = torch.Generator().manual_seed(seed)
    mean = torch.randn(31, 18, generator=rng)
    factor = torch.tril(0.3 * torch.randn(31, 18, 18, generator=rng))
    factor = factor + torch.diag_embed(0.5 + torch.rand(31, 18, generator=rng))
    with torch.no_grad():
        latent.code.copy_(mean)
        latent.code_chol.copy_(factor)
    return latent, mean, factor


def test_a_native_calibrated_code_is_a_draw_of_its_types_gaussian_and_its_mean_is_the_table() -> (
    None
):
    latent, mean, factor = calibrated_latent()
    latent.eval()
    b = batch()
    native = torch.randn(BATCH, LENGTH, 32)
    keep, code = latent.keep, latent.code_idx
    # without a sample the code is the table's mean for the residue's type, and the channels kept are the release's
    z = latent(b, native=native)
    assert torch.equal(z[..., code], mean[b["residue_type"]])
    assert torch.equal(z[..., keep], native[..., keep])
    # sampled, the residues of one type are draws of N(mean, L L^T): their mean and covariance are the table's
    torch.manual_seed(3)
    t = 5
    b["residue_type"] = torch.full((BATCH, LENGTH), t)
    drawn = torch.cat(
        [
            latent(b, native=native, sample=True)[..., code].reshape(-1, 18)
            for _ in range(20)
        ]
    ).detach()
    assert float((drawn.mean(0) - mean[t]).abs().max()) < 0.05
    covariance = factor[t] @ factor[t].T
    assert float((torch.cov(drawn.T) - covariance).abs().max()) < 0.12
    assert torch.equal(
        latent(b, native=native, sample=True)[..., keep], native[..., keep]
    )


def test_a_calibrated_latent_says_so_in_its_state_and_loads_only_the_statistics_of_its_own_channels(
    tmp_path,
) -> None:
    import numpy as np

    latent, mean, factor = calibrated_latent()
    plain = fixed_latent()
    assert "code_chol" in latent.state_dict() and "code_chol" not in plain.state_dict()
    with pytest.raises(ValueError, match="a fixed code in place"):
        autoencoder.anchored_class()("esmc", 30, inplace=True, in_dim=64, calibrated=True)
    other = autoencoder.anchored_class()("fixed", 30, inplace=True, calibrated=True)
    path = tmp_path / "stats.npz"
    np.savez(
        path, mean=mean.numpy(), chol=factor.numpy(), channels=np.array(other.code_idx)
    )
    autoencoder.load_native_code(other, str(path))
    assert torch.equal(other.code, mean) and torch.equal(other.code_chol, factor)
    np.savez(path, mean=mean.numpy(), chol=factor.numpy(), channels=np.arange(18))
    with pytest.raises(ValueError, match="not of this latent's code channels"):
        autoencoder.load_native_code(other, str(path))
    np.savez(
        path,
        mean=mean[:10].numpy(),
        chol=factor[:10].numpy(),
        channels=np.array(other.code_idx),
    )
    with pytest.raises(ValueError, match="has tables of"):
        autoencoder.load_native_code(other, str(path))


def test_the_encoder_gives_the_flow_a_draw_of_the_calibrated_code_and_its_mean_beside_it() -> (
    None
):
    latent, mean, _ = calibrated_latent()
    latent.eval()
    b = batch()
    code = latent.code_idx
    out = fm.esm_encoder_class()(StubNative(), latent, 0, native_sample=True)(b)
    assert torch.equal(out["mean"][..., code], mean[b["residue_type"]])
    assert not torch.equal(out["z_latent"][..., code], out["mean"][..., code])
    assert torch.all(
        out["z_latent"][..., latent.keep] == 1.0
    )  # the release encoder's sample in the channels kept
