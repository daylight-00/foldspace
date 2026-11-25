"""The differentiable route into the mixed single, on a shim with random weights.

The design of a generator in the mixed single rests on a graph — a supplied `[L, 256]` state reaching a structural
loss through the shim's own `base_z_mlp`, downstream of the detach that keeps
ESMC frozen. The GPU experiment measures it on the real model; this pins the
graph itself, with no weights, no GPU and no network, so a change that severs it
fails in seconds rather than in a job.

Both failures of the retracted first version are asserted here directly: that
the tracked tensor is a **leaf** (a non-leaf reports `grad=None` whether or not
it is in the graph), and that the route under test actually **changes the
output** (a route that is computed but never injected measures nothing).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("esm", reason="needs the esm source tree")

from esm.models.esmfold2.layers import LanguageModelShim

from foldspace.esmfold2 import inject

D_Z, D_MODEL, N_LAYERS, B, L = 8, 16, 4, 1, 6


@pytest.fixture
def shim():
    torch.manual_seed(0)
    return LanguageModelShim(d_z=D_Z, d_model=D_MODEL, num_layers=N_LAYERS).eval()


@pytest.fixture
def stack():
    torch.manual_seed(1)
    return torch.randn(B, L, N_LAYERS + 1, D_MODEL)


def test_a_supplied_single_carries_gradient_to_a_downstream_loss(shim, stack):
    """The claim, as a graph property."""
    single = inject.mixed_single(shim, stack).detach().clone().requires_grad_(True)
    assert single.is_leaf, "a non-leaf would report grad=None regardless"

    with inject.substituted_single(shim, single):
        pair = shim(stack.detach())
    pair.sum().backward()

    assert single.grad is not None
    assert float(single.grad.norm()) > 0.0


def test_a_detached_stack_carries_nothing(shim, stack):
    """The other half: upstream detaches the stack, so ESMC stays frozen."""
    tracked = stack.detach().clone().requires_grad_(True)
    pair = shim(tracked.detach())
    pair.sum().backward()
    assert tracked.grad is None


def test_an_uninjected_single_is_not_in_the_graph(shim, stack):
    """The retracted experiment's exact mistake, pinned.

    Computing the single and then running the forward on the stack leaves the
    single outside the differentiated graph. Without this assertion the failure
    is invisible: `grad is None` reads identically to "the path is blocked".
    """
    single = inject.mixed_single(shim, stack).detach().clone().requires_grad_(True)

    pair = shim(stack.detach())  # the single is never injected
    pair.sum().backward()

    assert single.grad is None, (
        "if this ever passes with a gradient, the test itself is wrong"
    )


def test_the_route_under_test_changes_the_output(shim, stack):
    """A route that does not change the output cannot be what was measured."""
    torch.manual_seed(2)
    substitute = torch.randn(B, L, D_Z)

    with torch.no_grad():
        baseline = shim(stack)
        with inject.substituted_single(shim, substitute):
            patched = shim(stack)

    assert not torch.allclose(patched, baseline), (
        "the hook did not take effect; any gradient verdict through it is void"
    )


def test_gradient_flows_only_while_the_hook_is_installed(shim, stack):
    single = inject.mixed_single(shim, stack).detach().clone().requires_grad_(True)

    with inject.substituted_single(shim, single):
        shim(stack.detach()).sum().backward()
    first = float(single.grad.norm())

    single.grad = None
    shim(stack.detach()).sum().backward()  # hook removed

    assert first > 0.0
    assert single.grad is None, "the hook outlived its context"
