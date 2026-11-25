"""The ESMC context adapter (`src/foldspace/rfd4/flow_adapt.py`) and the sequence conditioning of the flow stream (`src/foldspace/rfd4/flow_data.py`): what the adapter adds and to what, and that an
example given no sequence stays the release's own."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import foldspace.rfd4.flow_adapt as fad
import foldspace.rfd4.flow_data as fd

pytestmark = pytest.mark.offline

D = 8  # the toy's token width
E = 6  # the toy's ESMC width


class ToyFactory(torch.nn.Module):
    """`init_repr_factory` of the release: reads the batch and returns the initial token representation."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, D, bias=False)

    def forward(self, batch: dict):
        return self.linear(batch["x"])


class ToyNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_dim = D
        self.init_repr_factory = ToyFactory()
        self.out = torch.nn.Linear(D, 2, bias=False)

    def forward(self, batch: dict):
        s_init = self.init_repr_factory(batch) * batch["padding_mask"][..., None]
        return self.out(s_init)


class ToyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nn = ToyNN()
        self.cfg_exp = type("Cfg", (), {})()
        self.cfg_exp.opt = {"muon_exclude_param_patterns": [r"^nn\.out\."]}


def batch(rows: bool = True, zero_second: bool = False) -> dict:
    g = torch.Generator().manual_seed(0)
    b = {
        "x": torch.randn(2, 5, 4, generator=g),
        "padding_mask": torch.ones(2, 5, dtype=torch.bool),
    }
    if rows:
        context = torch.randn(2, 5, E, generator=g)
        if zero_second:
            context[1] = 0.0  # the second example is given no sequence
        b[fad.CONTEXT_KEY] = context
    return b


def attach(monkeypatch=None):
    module = ToyModule()
    # the toy's ESMC width is not 2560
    original = fad.adapter
    fad.adapter = lambda esmc_dim=E, token_dim=D: original(
        esmc_dim=E, token_dim=token_dim
    )
    try:
        state = fad.attach(module)
    finally:
        fad.adapter = original
    return module, state


def test_the_adapter_changes_nothing_until_its_weights_move() -> None:
    module, _ = attach()
    plain = ToyModule()
    plain.load_state_dict(
        {k: v for k, v in module.state_dict().items() if not k.startswith(fad.PREFIX)}
    )
    with torch.no_grad():
        assert torch.equal(module.nn(batch()), plain.nn(batch()))


def test_an_example_without_rows_is_the_releases_whatever_the_weights_are() -> None:
    module, _ = attach()
    plain = ToyModule()
    plain.load_state_dict(
        {k: v for k, v in module.state_dict().items() if not k.startswith(fad.PREFIX)}
    )
    with torch.no_grad():
        getattr(module.nn, fad.HEAD_NAME)[1].weight.normal_()
        got, native = module.nn(batch(zero_second=True)), plain.nn(batch())
    # the example that has rows moved; the one that has none is the release's to the bit
    assert not torch.equal(got[0], native[0])
    assert torch.equal(got[1], native[1])


def test_no_key_is_the_native_forward() -> None:
    module, state = attach()
    with torch.no_grad():
        getattr(module.nn, fad.HEAD_NAME)[1].weight.normal_()
        out = module.nn(batch(rows=False))
        plain = ToyModule()
        plain.load_state_dict(
            {
                k: v
                for k, v in module.state_dict().items()
                if not k.startswith(fad.PREFIX)
            }
        )
        assert torch.equal(out, plain.nn(batch(rows=False)))
    assert state.calls == 1 and state.with_rows == 0


def test_the_gradient_reaches_the_adapter_and_the_state_dict_carries_it() -> None:
    module, _ = attach()
    module.nn(batch()).pow(2).sum().backward()
    assert getattr(module.nn, fad.HEAD_NAME)[1].weight.grad.abs().sum() > 0
    assert any(k.startswith(fad.PREFIX) for k in module.state_dict())


def test_the_adapter_is_on_adam_once_and_a_second_attach_is_refused() -> None:
    module, _ = attach()
    patterns = module.cfg_exp.opt["muon_exclude_param_patterns"]
    assert patterns.count(rf"^nn\.{fad.HEAD_NAME}\.") == 1
    with pytest.raises(ValueError, match="already has"):
        fad.attach(module)


def test_detach_gives_back_the_releases_model() -> None:
    module, state = attach()
    fad.detach(state, module)
    assert not hasattr(module.nn, fad.HEAD_NAME) and state.handle is None
    assert module.nn(batch()).shape == (2, 5, 2)


def test_adapter_only_optimizer_preserves_the_native_forward_after_training() -> None:
    pytest.importorskip("rfproteina")
    module, _ = attach()
    plain = ToyModule()
    plain.load_state_dict(
        {k: v for k, v in module.state_dict().items() if not k.startswith(fad.PREFIX)}
    )
    selected = fad.freeze_generator(module)
    assert selected["names"]
    optimizer = torch.optim.Adam(
        [p for p in module.parameters() if p.requires_grad], lr=0.01
    )
    for _ in range(3):
        optimizer.zero_grad()
        module.nn(batch()).square().sum().backward()
        optimizer.step()
    assert torch.equal(module.nn(batch(rows=False)), plain.nn(batch(rows=False)))
    assert not torch.equal(module.nn(batch()), plain.nn(batch()))


# the sequence conditioning of the stream ----------------------------------------------------------------------------------------------------------------------------------


def item(n: int = 12) -> dict:
    rng = np.random.default_rng(1)
    return {
        "coords": rng.normal(size=(n, 37, 3)).astype(np.float32),
        "atom37_mask": np.ones((n, 37), dtype=bool),
        "residue_type": rng.integers(0, 20, size=n).astype(np.int64),
        "token_type": np.zeros(n, dtype=np.int32),
        "chain_iid": np.zeros(n, dtype=np.int16),
        "charge": np.zeros((n, 37), dtype=np.int8),
        "ground_truth_element": np.zeros((n, 37), dtype=np.int8),
        "incident_bond_type_counts": np.zeros((n, 37, 4), dtype=np.int8),
        "supports_local_latent": np.ones(n, dtype=bool),
        "is_in_context_conditioning": np.zeros(n, dtype=bool),
        "is_resolved_mask": np.ones(n, dtype=bool),
    }


def test_the_sequence_is_given_as_the_releases_exact_conditioning() -> None:
    pytest.importorskip("rfproteina")
    from atomworks.ml.conditions import C_SEQ
    from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS

    it = item()
    given = fd.sequence_conditions(it["residue_type"])
    assert np.array_equal(given[C_SEQ.full_name], it["residue_type"])
    assert given[C_SEQ.full_name].dtype == np.int64
    probabilities = np.stack(
        [given[c.full_name] for c in SEQUENCE_PROBABILITY_CONDITIONS], axis=-1
    )
    assert probabilities.shape == (12, len(SEQUENCE_PROBABILITY_CONDITIONS)) == (12, 28)
    # a canonical residue is a one-hot over the standard polymer tokens, and no value is NaN (every residue is given)
    assert np.all(probabilities.sum(-1) == 1) and not np.isnan(probabilities).any()


def test_a_chain_given_its_sequence_has_the_same_crop_rotation_and_rows() -> None:
    pytest.importorskip("rfproteina")
    it = item(20)
    rows = np.random.default_rng(2).normal(size=(20, E)).astype(np.float16)
    plain = fd.chain_features(it, np.random.default_rng(5), crop=8, esmc=rows)
    given = fd.chain_features(
        it,
        np.random.default_rng(5),
        crop=8,
        esmc=rows,
        condition_sequence=True,
        context_rows=True,
    )
    # the conditioning draws nothing from the augmentation stream: the window and the rotation are the unconditioned example's
    assert np.array_equal(plain["coords"], given["coords"])
    assert np.array_equal(plain["esmc"], given["esmc"])
    assert np.array_equal(given[fad.CONTEXT_KEY], given["esmc"])
    none = fd.chain_features(
        it, np.random.default_rng(5), crop=8, esmc=rows, context_rows=True
    )
    assert not none[fad.CONTEXT_KEY].any() and none[fad.CONTEXT_KEY].shape == (8, E)
    with pytest.raises(ValueError, match="need the ESMC rows"):
        fd.chain_features(it, np.random.default_rng(5), context_rows=True)


def test_the_coin_that_gives_a_chain_its_sequence_is_its_own_stream() -> None:
    def stream(fraction: float):
        return fd.FlowStream(
            "shards",
            ["a", "b"],
            batch=2,
            seed=7,
            condition_fraction=fraction,
        )

    assert not any(stream(0.0).conditioned(s, j) for s in range(50) for j in range(2))
    assert all(stream(1.0).conditioned(s, j) for s in range(50) for j in range(2))
    half = [stream(0.5).conditioned(s, j) for s in range(400) for j in range(2)]
    assert 0.4 < np.mean(half) < 0.6
    # a function of (seed, step, slot) and of nothing else
    assert half == [stream(0.5).conditioned(s, j) for s in range(400) for j in range(2)]
