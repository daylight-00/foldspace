"""The two injection levels agree with the module they bypass.

`inject.mixed_single` reimplements the first half of `LanguageModelShim.forward`
so that a probe can perturb the state the trunk is conditioned on. Sharing the
module's weights is not enough -- the order of the LayerNorm, the projection and
the softmax mixture has to match too, and a drift there would move every
mixed-single result without failing anything else.

So it is checked against the real module, with random weights and no
checkpoint: torch only, no GPU, no network.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("esm", reason="needs the esm source tree")

from esm.models.esmfold2.layers import LanguageModelShim

from foldspace.esmfold2 import inject

D_Z, D_MODEL, N_LAYERS, B, L = 8, 16, 4, 1, 5


@pytest.fixture
def shim():
    torch.manual_seed(0)
    module = LanguageModelShim(d_z=D_Z, d_model=D_MODEL, num_layers=N_LAYERS).eval()
    # A uniform mixture would hide an ordering mistake; make the layers differ.
    with torch.no_grad():
        module.base_z_combine.copy_(torch.tensor([-2.0, 0.5, 1.5, -1.0, 0.0]))
    return module


@pytest.fixture
def stack():
    torch.manual_seed(1)
    return torch.randn(B, L, N_LAYERS + 1, D_MODEL)


def test_mixed_single_reproduces_the_shims_own_mixture(shim, stack):
    with torch.no_grad():
        single = inject.mixed_single(shim, stack)
        expected_pair = shim(stack)
        pair_from_single = shim.base_z_mlp(single)

    assert single.shape == (B, L, D_Z)
    assert torch.allclose(pair_from_single, expected_pair, atol=1e-6)


def test_layer_weights_is_the_live_softmax(shim):
    weights = inject.layer_weights(shim)
    assert weights.shape == (N_LAYERS + 1,)
    assert float(weights.sum().detach()) == pytest.approx(1.0, abs=1e-6)
    assert int(weights.argmax()) == 2  # the largest logit above


def test_project_stack_keeps_the_layer_axis(shim, stack):
    with torch.no_grad():
        projected = inject.project_stack(shim, stack)
    assert projected.shape == (B, L, N_LAYERS + 1, D_Z)


def test_substituted_single_replaces_the_output_and_is_removed(shim, stack):
    torch.manual_seed(2)
    substitute = torch.randn(B, L, D_Z)

    with torch.no_grad():
        baseline = shim(stack)
        with inject.substituted_single(shim, substitute):
            patched = shim(stack)
            assert torch.allclose(patched, shim.base_z_mlp(substitute), atol=1e-6)
        restored = shim(stack)

    assert not torch.allclose(patched, baseline)
    assert torch.allclose(restored, baseline, atol=1e-6)
    assert shim._forward_hooks == {}, "the hook outlived its context"


def test_the_hook_is_removed_even_when_the_body_raises(shim, stack):
    with (
        pytest.raises(RuntimeError),
        inject.substituted_single(shim, torch.zeros(B, L, D_Z)),
    ):
        raise RuntimeError("boom")
    assert shim._forward_hooks == {}


def test_a_model_without_the_shim_is_refused():
    class Bare:
        pass

    with pytest.raises(AttributeError):
        inject.layer_weights(Bare())
