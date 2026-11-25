"""Two levels at which a state can be put into the folding model.

ESMC does not reach the trunk as a state. The stack
``[B, L, 81, 2560]`` is projected and collapsed by a learned softmax into
``[B, L, 256]``, then expanded to a pair bias. That gives two distinct places to
perturb, and they answer different questions.

**The stack** — ``forward(lm_hidden_states=h')``. Upstream's own argument, so
this is exactly what the model would have computed, with nothing bypassed. Use
it to ask *what ESMC's output space looks like*. Costs ~124 MB per 300-residue
structure in bf16 to cache.

**The mixed single** — ``[B, L, 256]``, the state the folding trunk is actually
conditioned on. Reached by a forward hook on ``net.language_model`` that returns
the shim's own pair expansion of a substituted single. Use it to ask *whether
the coordinate the model uses is generative*. 800× smaller to store, and it is
the candidate a generator would have to emit.

Perturbing the stack and perturbing the mixed single are **not** the same
experiment: the projection is a LayerNorm followed by a rank-256 linear map, so
most stack directions are in its null space and never reach geometry. Results
from the two levels are reported separately and never pooled.

Neither level patches anything. The stack is upstream's argument; the hook is
removed on exit and the module is left as it was.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

__all__ = [
    "layer_weights",
    "mixed_single",
    "project_stack",
    "substituted_single",
]


def _shim(target: Any) -> Any:
    """``LanguageModelShim``, from a Substrate, a model, or the shim itself.

    Identified by ``base_z_combine`` rather than by type, so a probe keeps
    working against a subclass or a distributed replacement of the shim.
    """
    if hasattr(target, "base_z_combine"):
        return target
    net = getattr(target, "net", target)
    if hasattr(net, "base_z_combine"):
        return net
    shim = getattr(net, "language_model", None)
    if shim is None:
        raise AttributeError(
            "no `language_model` shim here: pass a Substrate, an EsmFold2Model, "
            "or the shim itself"
        )
    return shim


def layer_weights(substrate_or_net: Any):
    """The live softmax over ESMC entries, ``[n_layers + 1]``.

    The weights of the mixture, taken from the loaded module -- so a probe run records what it
    actually used rather than what a file on disk says.
    """
    return _shim(substrate_or_net).base_z_combine.softmax(0)


def project_stack(substrate_or_net: Any, stack):
    """``base_z_linear`` applied to a stack: ``[B, L, 81, 2560] -> [B, L, 81, 256]``."""
    shim = _shim(substrate_or_net)
    return shim.base_z_linear(stack.to(shim.base_z_combine.dtype))


def mixed_single(substrate_or_net: Any, stack):
    """The state the trunk is conditioned on: ``[B, L, 81, 2560] -> [B, L, 256]``.

    Reproduces ``LanguageModelShim.forward`` up to, but not including, the pair
    expansion -- the two share the module's own weights, so this cannot drift
    from what the model does.
    """
    shim = _shim(substrate_or_net)
    projected = project_stack(shim, stack)
    weights = shim.base_z_combine.softmax(0)
    return (weights @ projected).squeeze(-2)


@contextmanager
def substituted_single(substrate_or_net: Any, single):
    """Fold as if the layer mixture had produced ``single``.

    ``single`` is ``[B, L, 256]``. Inside the block, ``net.language_model``
    returns ``base_z_mlp(single)`` -- the shim's own expansion -- whatever stack
    it is handed. The hook is removed on exit.

    The stack still has to be computed or supplied, because the model decides
    whether to call ESMC by whether ``lm_hidden_states`` is None
    (``model.py:1164``); a substrate opened with ``load_esmc=False`` therefore
    still needs some stack passed in. Any correctly shaped tensor will do -- it
    is discarded -- but passing the real one keeps the run honest if the hook is
    ever removed by mistake.
    """
    shim = _shim(substrate_or_net)

    def _hook(module, args, kwargs, output):
        return module.base_z_mlp(single.to(output.dtype))

    handle = shim.register_forward_hook(_hook, with_kwargs=True)
    try:
        yield shim
    finally:
        handle.remove()
