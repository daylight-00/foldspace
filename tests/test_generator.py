"""The generator of latent generation on arrays whose answer is known, and on the synthetic world: coordinates, the denoiser, the flow, the sampler.

Runs on a CPU with tiny models. What is checked is what a run cannot check about itself: that the whitened coordinate is an
invertible map that does what it says, that the denoiser does not depend on padding and does depend on time once its
modulation is not zero, that the flow is what the docstring says at both ends, that the sampler lands on the target when the
network is an oracle, and that a few steps of training reduce the error on the world's singles.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esmfold2_world import build

from foldspace.esmfold2 import batching, generator
from foldspace.esmfold2.decoder_data import Normaliser, gather, load_split

CPU = torch.device("cpu")
D = 16


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build(tmp_path_factory.mktemp("world"))


def _spectrum(seed=0, floor=1e-3):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    values = 0.7 ** np.arange(D) * 4.0
    return (
        q,
        values,
        generator.Whitening.from_spectrum(q, values, floor=floor, device=CPU),
    )


def _tiny(**kw):
    base = {"d_data": D, "d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64}
    return generator.Denoiser(**{**base, **kw})


# --------------------------------------------------------------- coordinate


def test_the_identity_coordinate_leaves_the_single_alone():
    z = torch.randn(2, 5, D)
    w = generator.Whitening.identity()
    assert torch.equal(w.forward(z), z) and torch.equal(w.inverse(z), z)


def test_the_whitened_coordinate_is_invertible_and_has_unit_variance_in_every_direction():
    q, values, w = _spectrum()
    g = torch.Generator().manual_seed(0)
    z = (
        torch.randn(20000, D, generator=g) * torch.tensor(values).sqrt().float()
    ) @ torch.tensor(q.T).float()
    y = w.forward(z)
    assert torch.allclose(w.inverse(y), z, atol=1e-4)
    variance = y.var(dim=0)
    assert torch.allclose(variance, torch.ones(D), atol=0.1)
    # a direction no larger than the floor is not blown up: its variance is the share the floor leaves
    _, tiny_values, w_tiny = _spectrum(floor=1.0)
    y_tiny = w_tiny.forward(z)
    expected = torch.tensor(tiny_values / (tiny_values + 1.0)).float()
    assert torch.allclose(y_tiny.var(dim=0), expected, atol=0.1)


# -------------------------------------------------------------------- model


def test_the_denoiser_answers_in_the_shape_it_was_given_and_is_zero_on_the_padding():
    model = _tiny()
    x = torch.randn(3, 9, D)
    lengths = torch.tensor([9, 5, 7])
    out = model(x, torch.rand(3), lengths)
    assert out.shape == x.shape
    assert torch.all(out[1, 5:] == 0) and torch.all(out[2, 7:] == 0)


def test_the_denoiser_does_not_depend_on_what_pads_the_batch():
    torch.manual_seed(0)
    model = _tiny().eval()
    for p in (
        model.parameters()
    ):  # a modulation of zero would hide a dependence on time and so on padding too
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(1, 6, D)
    lengths = torch.tensor([6])
    alone = model(x, torch.tensor([0.4]), lengths)
    padded = torch.cat([x, torch.randn(1, 5, D)], dim=1)
    batch = torch.cat([padded, torch.randn(1, 11, D)], dim=0)
    batch[0, 6:] = 0.0
    out = model(batch, torch.tensor([0.4, 0.9]), torch.tensor([6, 11]))
    assert torch.allclose(out[0, :6], alone[0], atol=1e-4)


def test_time_is_ignored_at_initialisation_and_used_once_the_modulation_is_not_zero():
    torch.manual_seed(1)
    model = _tiny().eval()
    x = torch.randn(1, 5, D)
    lengths = torch.tensor([5])
    a, b = (
        model(x, torch.tensor([0.1]), lengths),
        model(x, torch.tensor([0.9]), lengths),
    )
    assert torch.allclose(
        a, b
    )  # zero-initialised modulation: the blocks are the identity
    for block in model.blocks:
        torch.nn.init.normal_(block.ada.weight, std=0.1)
    torch.nn.init.normal_(model.out_ada.weight, std=0.1)
    a, b = (
        model(x, torch.tensor([0.1]), lengths),
        model(x, torch.tensor([0.9]), lengths),
    )
    assert not torch.allclose(a, b, atol=1e-5)


# --------------------------------------------------------------------- flow


def test_the_flow_is_noise_at_zero_and_the_data_at_one_and_zero_on_the_padding():
    x1 = torch.randn(3, 8, D)
    lengths = torch.tensor([8, 5, 6])
    valid = (torch.arange(8)[None] < lengths[:, None])[..., None]
    x1 = x1 * valid
    x_t, t, noise = generator.flow_inputs(x1, lengths, t=torch.zeros(3))
    assert torch.equal(x_t, noise) and torch.all(noise[~valid.expand_as(noise)] == 0)
    x_t, _, _ = generator.flow_inputs(x1, lengths, t=torch.ones(3))
    assert torch.allclose(x_t, x1)
    # one t per chain, in [0, 1)
    _, t, _ = generator.flow_inputs(x1, lengths)
    assert t.shape == (3,) and torch.all((t >= 0) & (t < 1))
    # a generator makes the same draw again
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    a = generator.flow_inputs(x1, lengths, g1)
    b = generator.flow_inputs(x1, lengths, g2)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_the_loss_is_zero_for_a_perfect_prediction_and_weights_the_clean_end_more():
    x1 = torch.randn(2, 6, D)
    lengths = torch.tensor([6, 4])
    loss, mse = generator.flow_loss(x1.clone(), x1, torch.tensor([0.3, 0.6]), lengths)
    assert loss == 0 and mse == 0
    bad = x1 + 1.0
    # the same error costs 1 at t=0, 4 at t=0.5 and is held to 100 from t=0.9 on
    costs = []
    for t in (0.0, 0.5, 0.9, 0.99):
        value, plain = generator.flow_loss(bad, x1, torch.full((2,), t), lengths)
        costs.append(float(value / plain))
    assert costs == pytest.approx([1.0, 4.0, 100.0, 100.0], rel=1e-4)
    # padding does not count: an error only on the padding costs nothing
    padded = x1.clone()
    padded[1, 4:] += 5.0
    assert generator.flow_loss(padded, x1, torch.zeros(2), lengths)[0] == 0


class _Oracle(torch.nn.Module):
    """A network that knows the answer: it returns the single it was told to."""

    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, x_t, t, lengths):
        return self.target[: x_t.shape[0], : x_t.shape[1]].clone()


@pytest.mark.parametrize("steps", [2, 5, 50])
def test_the_sampler_lands_on_the_target_when_the_network_is_an_oracle(steps):
    lengths = torch.tensor([7, 4])
    target = torch.randn(2, 7, D)
    valid = (torch.arange(7)[None] < lengths[:, None])[..., None]
    target = target * valid
    out = generator.sample(_Oracle(target), lengths, d_data=D, steps=steps)
    assert torch.allclose(out, target, atol=1e-4)


def test_the_sampler_is_reproducible_from_its_generator_and_pads_with_zeros():
    model = _tiny().eval()
    lengths = torch.tensor([6, 3])
    a = generator.sample(
        model, lengths, d_data=D, steps=4, generator=torch.Generator().manual_seed(1)
    )
    b = generator.sample(
        model, lengths, d_data=D, steps=4, generator=torch.Generator().manual_seed(1)
    )
    assert torch.equal(a, b) and torch.all(a[1, 3:] == 0)


# ---------------------------------------------------------------------- EMA


def test_the_moving_average_follows_the_parameters_and_comes_back_when_asked():
    model = _tiny()
    ema = generator.EMA(model, decay=0.5)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    name = next(iter(before))
    assert torch.allclose(
        ema.shadow[name], 0.5 * before[name] + 0.5 * model.state_dict()[name]
    )
    ema.copy_to(model)
    assert torch.allclose(model.state_dict()[name], ema.shadow[name])


def test_the_average_is_not_mostly_the_initial_weights_in_the_first_steps():
    model = _tiny()
    ema = generator.EMA(model, decay=0.999)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    name = next(iter(initial))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, step=0)  # the first update: a decay of 0.1, not 0.999
    assert torch.allclose(
        ema.shadow[name], 0.1 * initial[name] + 0.9 * model.state_dict()[name]
    )
    for step in range(1, 400):
        ema.update(model, step=step)
    # after 400 updates the share of the initial weights is what the schedule leaves, far under the 67 % of a fixed 0.999
    share = float(
        (
            (ema.shadow[name] - model.state_dict()[name])
            / (initial[name] - model.state_dict()[name])
        ).mean()
    )
    assert abs(share) < 0.05


# ----------------------------------------------------------------- training


def test_a_few_steps_of_flow_matching_reduce_the_error_on_the_worlds_singles(world):
    torch.manual_seed(0)
    norm = Normaliser.load(
        world["cache"] / "esmfold2" / "model_single_train_normalisation.npz", CPU
    )
    train = load_split(
        "train",
        cache=world["cache"],
        chains=world["chains"],
        train_table=world["train_table"],
        device=CPU,
    )
    d = train.z.shape[1]
    model = generator.Denoiser(d_data=d, d_model=48, n_layers=2, n_heads=2, d_ff=96)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    chains = np.arange(train.n_chains)

    def mse_at(t_value):
        g = torch.Generator().manual_seed(5)
        total = 0.0
        for batch in batching.pack(chains[:8], train.lengths, 4096):
            x1, _, lens = gather(train, batch, norm)
            x_t, t, _ = generator.flow_inputs(
                x1, lens, g, t=torch.full((len(batch),), t_value)
            )
            with torch.no_grad():
                total += float(
                    generator.flow_loss(model.eval()(x_t, t, lens), x1, t, lens)[1]
                )
        return total

    before = mse_at(0.5)
    rng = np.random.default_rng(0)
    for _ in range(60):
        batch = next(
            iter(
                batching.pack(
                    rng.choice(chains, 6, replace=False), train.lengths, 4096, rng
                )
            )
        )
        x1, _, lens = gather(train, batch, norm)
        x_t, t, _ = generator.flow_inputs(x1, lens)
        loss, _ = generator.flow_loss(model.train()(x_t, t, lens), x1, t, lens)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert mse_at(0.5) < 0.8 * before


def test_the_sampler_can_start_from_a_partly_noised_single():
    lengths = torch.tensor([7, 4])
    target = (
        torch.randn(2, 7, D) * (torch.arange(7)[None] < lengths[:, None])[..., None]
    )
    start = 0.5 * torch.randn(2, 7, D) + 0.5 * target  # a single noised to t = 0.5
    out = generator.sample(
        _Oracle(target), lengths, d_data=D, steps=4, start=start, t_start=0.5
    )
    assert torch.allclose(out, target, atol=1e-4)
    # the steps run from the start whatever the network is
    model = _tiny().eval()
    near = generator.sample(model, lengths, d_data=D, steps=3, start=start, t_start=0.9)
    assert near.shape == start.shape
    with pytest.raises(ValueError, match="t_start"):
        generator.sample(
            _Oracle(target), lengths, d_data=D, steps=3, start=start, t_start=1.0
        )


def test_a_shift_spends_more_of_the_steps_where_the_noise_is_and_keeps_both_ends():
    lengths = torch.tensor([6, 4])
    target = (
        torch.randn(2, 6, D) * (torch.arange(6)[None] < lengths[:, None])[..., None]
    )
    seen = []

    class Recorder(torch.nn.Module):
        def forward(self, x_t, t, lengths):
            seen.append(float(t[0]))
            return target[:, : x_t.shape[1]].clone()

    # the first prediction is at the first grid point and the grid is the same length whatever the shift
    for shift in (1.0, 3.0):
        seen.clear()
        out = generator.sample(Recorder(), lengths, d_data=D, steps=20, shift=shift)
        assert torch.allclose(out, target, atol=1e-4)
        times = sorted(set(seen))
        assert times[0] == 0.0 and times[-1] < 1.0
        if shift == 1.0:
            even = times
        else:
            bent = times
    # on the bent grid more of the points lie in the first fifth of the time
    assert sum(t < 0.2 for t in bent) > sum(t < 0.2 for t in even)
    # a shift of 1 is the even grid
    assert even[3] == pytest.approx(3 / 20, abs=1e-6)
