"""ESMC alignment on RFD4's native flow (`src/foldspace/rfd4/flow_align.py`): the term, and the five things that a training run stands on.

The toy module has the release's shape where it matters: a trunk that makes a final token state `s`, a `local_latents_linear` that reads it, a flow matcher with
`compute_loss` and `optimization_terms`, and a config with the Muon exclusion patterns. The real network is exercised by the smoke run, on real batches."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import foldspace.rfd4.flow_align as fa
import foldspace.rfd4.flow_data as fd

pytestmark = pytest.mark.offline

D = 16  # the toy's token width


class ToyNN(torch.nn.Module):
    """A trunk (two linear maps, the second a stand-in for the shared denoiser trunk) and the heads that read its final token state."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(4, D), torch.nn.GELU(), torch.nn.Linear(D, D)
        )
        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(D), torch.nn.Linear(D, 3, bias=False)
        )
        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(D), torch.nn.Linear(D, 3, bias=False)
        )

    def forward(self, batch: dict) -> dict:
        s = self.trunk(batch["x"])
        return {"latents": self.local_latents_linear(s), "ca": self.ca_linear(s)}


class ToyFlow:
    """The two methods of the release's flow matcher that the term touches: a loss dictionary, and the terms that are optimised."""

    data_modes = ("latents", "ca")

    def compute_loss(self, batch, nn_out):
        return {
            m: ((nn_out[m] - batch["target"]) ** 2).mean(dim=(1, 2))
            for m in self.data_modes
        }

    def optimization_terms(self, losses):
        return {m: losses[m] for m in self.data_modes}


class ToyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nn = ToyNN()
        self.fm = ToyFlow()
        self.cfg_exp = type("Cfg", (), {})()
        self.cfg_exp.opt = {
            "muon_exclude_param_patterns": [r"^nn\.local_latents_linear\."]
        }


def batch_of(b: int = 3, n: int = 5, esmc_dim: int = fa.ESMC_DIM) -> dict:
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[0, -2:] = False  # a padded example
    return {
        "x": torch.randn(b, n, 4, generator=g),
        "target": torch.randn(b, n, 3, generator=g),
        "esmc": torch.randn(b, n, esmc_dim, generator=g).half(),
        "padding_mask": mask,
    }


def step(model, batch):
    """One loss computation as the release's training step does it: the network, the loss dictionary, the optimised sum."""
    nn_out = model.nn(batch)
    losses = model.fm.compute_loss(batch=batch, nn_out=nn_out)
    terms = model.fm.optimization_terms(losses)
    return losses, sum(v.mean() for v in terms.values())


def test_the_term_is_one_minus_the_cosine_over_the_residues_that_are_there() -> None:
    esmc = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    # a prediction equal to the normalised target is a loss of 0; the opposite one is 2; padded residues do not count
    target = torch.nn.functional.layer_norm(esmc, (8,))
    assert fa.align_loss(target, esmc, mask).abs().max() < 1e-5
    assert torch.allclose(
        fa.align_loss(-target, esmc, mask), torch.full((2,), 2.0), atol=1e-5
    )
    wrong_only_where_padded = target.clone()
    wrong_only_where_padded[1, 2:] = -target[1, 2:]
    assert fa.align_loss(wrong_only_where_padded, esmc, mask)[1].abs() < 1e-5


def test_esmc_scale_does_not_decide_the_term() -> None:
    # ESMC's last hidden state has a few features of very large magnitude: the target is normalised before the cosine
    esmc = torch.randn(1, 3, 8)
    spiky = esmc.clone()
    spiky[..., 0] *= 100.0
    mask = torch.ones(1, 3, dtype=torch.bool)
    prediction = torch.randn(1, 3, 8)
    plain = fa.align_loss(prediction, esmc, mask)
    scaled = fa.align_loss(prediction, 7.0 * esmc, mask)
    assert torch.allclose(plain, scaled, atol=1e-5)
    assert fa.align_loss(prediction, spiky, mask).item() != pytest.approx(
        plain.item(), abs=1e-3
    )


def test_attach_adds_the_term_to_the_loss_dictionary_and_the_optimised_sum() -> None:
    model = ToyModule()
    batch = batch_of()
    losses, plain = step(model, batch)
    state = fa.attach(model, weight=0.5, hidden=32)
    losses_with, total = step(model, batch)
    assert set(losses_with) == set(losses) | {fa.LOSS_KEY}
    assert losses_with[fa.LOSS_KEY].shape == (3,)
    # the optimised sum is the release's plus the weighted term (the means over the examples)
    expected = plain + 0.5 * losses_with[fa.LOSS_KEY].mean()
    assert total.item() == pytest.approx(expected.item(), rel=1e-5)
    assert state.s is None  # the token state is not kept past the loss


def test_a_weight_of_zero_leaves_the_optimised_sum_as_the_releases() -> None:
    model = ToyModule()
    batch = batch_of()
    _, plain = step(model, batch)
    fa.attach(model, weight=0.0, hidden=32)
    losses, total = step(model, batch)
    assert fa.LOSS_KEY in losses  # still recorded
    assert total.item() == pytest.approx(plain.item(), rel=1e-6)


def test_the_term_reaches_the_shared_trunk_and_the_projector() -> None:
    model = ToyModule()
    state = fa.attach(model, weight=1.0, hidden=32)
    batch = batch_of()
    nn_out = model.nn(batch)
    losses = model.fm.compute_loss(batch=batch, nn_out=nn_out)
    losses[fa.LOSS_KEY].mean().backward()  # the alignment term alone
    trunk_last = model.nn.trunk[2].weight
    assert trunk_last.grad is not None and trunk_last.grad.abs().sum() > 0
    head = getattr(model.nn, fa.HEAD_NAME)
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in head[1:4:2].parameters()
    )
    # the heads that read the same state got nothing from it
    assert model.nn.local_latents_linear[1].weight.grad is None
    assert state.s is None


def test_the_probe_records_the_gradients_of_the_term_and_of_the_flow_loss() -> None:
    model = ToyModule()
    state = fa.attach(
        model, weight=1.0, hidden=32, probe_params={"trunk": model.nn.trunk[2].weight}
    )
    batch = batch_of()
    nn_out = model.nn(batch)
    model.fm.compute_loss(batch=batch, nn_out=nn_out)
    assert state.probe["align"]["trunk"] > 0 and state.probe["flow"]["trunk"] > 0
    # without probe parameters nothing is recorded
    plain = fa.attach(ToyModule(), weight=1.0)
    assert plain.probe == {}


def test_the_projector_is_in_the_modules_parameters_the_optimizer_and_the_checkpoint() -> (
    None
):
    model = ToyModule()
    fa.attach(model, weight=1.0, hidden=32)
    names = [n for n, _ in model.named_parameters()]
    assert any(n.startswith(fa.TRAINING_ONLY_PREFIX) for n in names), names
    # the release's optimizer builder reads `named_parameters` of the module: the projector is kept on Adam by the pattern that `attach` appends
    import re

    patterns = [re.compile(p) for p in model.cfg_exp.opt["muon_exclude_param_patterns"]]
    head_weights = [n for n in names if n.startswith(fa.TRAINING_ONLY_PREFIX)]
    assert head_weights and all(
        any(p.search(n) for p in patterns) for n in head_weights
    )
    # and the checkpoint holds it, and a resume restores it
    state = {k: v.clone() for k, v in model.state_dict().items()}
    assert any(k.startswith(fa.TRAINING_ONLY_PREFIX) for k in state)
    fresh = ToyModule()
    fa.attach(fresh, weight=1.0, hidden=32)
    fresh.load_state_dict(state, strict=True)
    for k, v in model.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k])


def test_an_inference_checkpoint_loads_into_the_release_network_without_the_projector() -> (
    None
):
    model = ToyModule()
    fa.attach(model, weight=1.0, hidden=32)
    weights, dropped = fa.drop_training_only(model.state_dict())
    assert dropped and not any(k.startswith(fa.TRAINING_ONLY_PREFIX) for k in weights)
    release_shaped = ToyModule()  # no projector, as the sampler builds it
    release_shaped.load_state_dict(weights, strict=True)
    # a checkpoint of a model that never had a projector drops nothing
    again, none = fa.drop_training_only(release_shaped.state_dict())
    assert none == [] and set(again) == set(release_shaped.state_dict())


def test_a_second_attach_and_a_batch_without_the_rows_are_refused() -> None:
    model = ToyModule()
    fa.attach(model, weight=1.0)
    with pytest.raises(ValueError, match="already has"):
        fa.attach(model, weight=1.0)
    other = ToyModule()
    fa.attach(other, weight=1.0)
    batch = batch_of()
    del batch["esmc"]
    nn_out = other.nn(batch)
    with pytest.raises(KeyError, match="esmc"):
        other.fm.compute_loss(batch=batch, nn_out=nn_out)


def test_a_forward_without_gradient_leaves_nothing_for_the_loss() -> None:
    model = ToyModule()
    state = fa.attach(model, weight=1.0)
    with torch.no_grad():
        model.nn(batch_of())
    assert state.s is None
    batch = batch_of()
    with torch.no_grad():
        nn_out = model.nn(batch)
    with pytest.raises(RuntimeError, match="token state"):
        model.fm.compute_loss(batch=batch, nn_out=nn_out)


def test_detach_puts_the_release_methods_back() -> None:
    model = ToyModule()
    plain_loss = model.fm.compute_loss
    state = fa.attach(model, weight=1.0)
    assert model.fm.compute_loss != plain_loss
    fa.detach(state, model)
    batch = batch_of()
    losses, _ = step(model, batch)
    assert fa.LOSS_KEY not in losses


def test_the_esmc_rows_of_a_batch_are_cut_with_the_same_window_as_its_coordinates() -> (
    None
):
    n, crop = 40, 16
    # rows that name their own residue: row i is all i
    esmc = np.repeat(np.arange(n, dtype=np.float32)[:, None], 6, axis=1)
    item = {
        "residue_type": np.arange(n),
        "atom37_mask": np.ones((n, 37), dtype=bool),
        "coords": np.zeros((n, 37, 3), dtype=np.float32),
        **{
            k: np.zeros(n)
            for k in fd.TOKEN_KEYS
            if k not in ("residue_type", "atom37_mask")
        },
    }
    for seed in range(5):
        feats = fd.chain_features(
            item, np.random.default_rng(seed), crop=crop, esmc=esmc
        )
        window = feats["residue_type"]  # the residues kept, by their own index
        assert feats["esmc"].shape == (crop, 6)
        assert np.array_equal(
            feats["esmc"][:, 0].astype(int), window
        )  # row j is the row of residue window[j]
    # a chain whose ESMC has another length is refused, not cut
    with pytest.raises(ValueError, match="rows for a chain"):
        fd.chain_features(item, np.random.default_rng(0), crop=crop, esmc=esmc[:-1])


def test_the_projector_reads_a_bf16_state_after_the_autocast_context_has_closed() -> (
    None
):
    """A token state out of a forward under bf16 autocast, given to the float32 projector outside it, is an error for a `LayerNorm`; a reading goes through `predict`."""
    head = fa.projector(token_dim=16, hidden=8, esmc_dim=6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        state = torch.nn.Linear(4, 16)(torch.randn(2, 5, 4))
    assert state.dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="BFloat16|Float"):
        head(state)
    out = fa.predict(head, state)
    assert out.dtype == torch.float32 and out.shape == (2, 5, 6)
