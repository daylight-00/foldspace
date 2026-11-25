"""Folding from a mixed single, with and without the backbone that made it.

The generator's output is a ``[L, 256]`` mixed single; the folding model wants an
ESMC stack. ``foldspace.esmfold2.inject.substituted_single`` bridges the
two with a forward hook on the layer-mixing shim, which has two consequences this
module makes explicit instead of leaving to the caller:

1. The shim still runs on whatever stack it is handed before the hook replaces its
   output, and the model only reads a pair prior when ``lm_hidden_states`` is not
   None. A deployed generator has no ESMC backbone, so it needs a stack of the
   right shape that carries no information: :func:`placeholder_stack`.
2. The same model with ``lm_hidden_states=None`` and no backbone resident is the
   upstream's own *no language-model prior* fold. It is the lower bound against
   which a prior's effect is read, so it is exposed as a path of its own
   (:func:`fold_without_prior`) rather than approximated by zeroing a tensor.

Nothing is patched: each path is the substrate's ``fold`` under a context that
the substrate already provides.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from foldspace.esmfold2 import inject
from foldspace.esmfold2 import substrate as sub

__all__ = [
    "Folded",
    "coord_diff",
    "drop_backbone",
    "fold_native",
    "fold_single",
    "fold_stack",
    "fold_without_prior",
    "model_context",
    "model_single",
    "observe_single",
    "pair_prior",
    "placeholder_stack",
    "tensor_diff",
]


@contextmanager
def model_context():
    """The numerical context the folding model runs its forward in.

    The model runs inside a CUDA autocast region in bfloat16, so the single it
    conditions on is *computed there*, in bfloat16, and is not the float32 one that
    ``inject.mixed_single`` makes outside it: they differ by about 0.25 % in relative L2,
    which is the rounding of bf16, and a fold from the float32 one is 0.08 to 0.35 Å from
    native. Anything that is to equal what the model uses is computed inside this context.
    Without a CUDA device there is no region, and this is a no-op.
    """
    import torch

    if torch.cuda.is_available():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def model_single(substrate, stack):
    """The mixed single as the model computes it: ``[1, L, 256]``, bfloat16.

    This is the definition of ``H(S)``. It equals, to every bit, the
    tensor entering ``base_z_mlp`` during the model's own forward
    (``observe_single``), and a fold from it equals native to every bit
    (deterministic mode). A bfloat16 value is exactly representable in float32, so it
    can be stored and loaded as float32 without loss.
    """
    import torch

    with torch.no_grad(), model_context():
        return inject.mixed_single(substrate.net, stack)


def drop_backbone(substrate):
    """The substrate with its ESMC backbone removed, and what removing it freed.

    ``open_substrate(load_esmc=False)`` is not enough for the release checkpoint:
    it bundles ESMC, and a bundled backbone comes in with the trunk whatever
    ``load_esmc`` says (``esmfold2_atomworks.model.esmfold2``, the constructor). A
    model that has to prove it needs no backbone has to have it taken out.

    Returns ``(substrate, info)``: a shallow copy sharing the model, with
    ``has_esmc`` False, and ``info`` with the backbone's presence before and after and
    the CUDA memory it held. Raises if the backbone is still reachable afterwards.
    """
    import gc

    import torch

    net = substrate.net
    had = substrate.esmc is not None
    cuda = torch.cuda.is_available()
    before = torch.cuda.memory_allocated() if cuda else 0
    net.esmc = None
    if hasattr(net, "_esmc"):
        net._esmc = None
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated() if cuda else 0
    if substrate.esmc is not None:
        raise RuntimeError("the ESMC backbone is still reachable after it was dropped")
    info = {
        "present_before": bool(had),
        "present_after": False,
        "freed_gib": (before - after) / 2**30,
    }
    return dataclasses.replace(substrate, has_esmc=False, esmc_source="none"), info


@contextmanager
def observe_single(target):
    """Record what the layer-mixing shim sees and makes during a forward.

    Yields a dict that a forward fills in: ``stack_dtype`` (what enters the shim),
    ``autocast`` (whether a CUDA autocast region is active where it runs),
    ``single`` (the mixed single entering ``base_z_mlp``, in the dtype the model
    computed it in) and ``output_dtype``. The hooks only read: nothing is returned
    from them, so the forward is unchanged.

    The model runs its forward inside an autocast region, so the single it conditions
    on is not necessarily the one ``inject.mixed_single`` makes outside it; this is how
    to read the one it used.
    """
    import torch

    shim = inject._shim(target)
    seen: dict = {}

    def before_shim(module, args):
        seen["stack_dtype"] = str(args[0].dtype)
        seen["autocast"] = bool(torch.is_autocast_enabled())

    def before_mlp(module, args):
        seen["single"] = args[0].detach().clone()

    def after_shim(module, args, output):
        seen["output_dtype"] = str(output.dtype)

    handles = [
        shim.register_forward_pre_hook(before_shim),
        shim.base_z_mlp.register_forward_pre_hook(before_mlp),
        shim.register_forward_hook(after_shim),
    ]
    try:
        yield seen
    finally:
        for handle in handles:
            handle.remove()


@dataclass
class Folded:
    """One fold reduced to what a comparison needs, so the raw output can go."""

    coord: np.ndarray
    """All atoms, ``[n_atoms, 3]``, in the order the substrate emits them."""
    ca: np.ndarray
    """Cα, ``[L, 3]``."""
    metrics: dict


def _reduce(out: sub.FoldOutput) -> Folded:
    atoms = out.atoms()
    coord = np.asarray(atoms.coord, dtype=np.float64)
    ca = coord[np.asarray(atoms.atom_name) == "CA"]
    return Folded(coord=coord, ca=ca, metrics=dict(out.metrics()))


def placeholder_stack(substrate: sub.Substrate, prepared: sub.Prepared):
    """A zero ESMC stack of the shape and dtype the shim expects.

    Its values are discarded by the hook, and zeros keep the shim's LayerNorm
    finite. It exists because the model decides whether to read a pair prior by
    whether a stack was passed, not by whether the backbone is loaded.
    """
    import torch

    shim = inject._shim(substrate.net)
    d_model = shim.base_z_linear[0].normalized_shape[0]
    n_layers = shim.base_z_combine.shape[0]
    reference = shim.base_z_linear[1].weight
    return torch.zeros(
        1,
        prepared.n_tokens,
        n_layers,
        d_model,
        dtype=reference.dtype,
        device=reference.device,
    )


def fold_native(substrate, prepared, *, seed: int, **kwargs) -> Folded:
    """The model computing its own ESMC stack: the reference path."""
    if not substrate.has_esmc:
        raise RuntimeError("the native path needs the ESMC backbone resident")
    return _reduce(
        sub.fold(substrate, prepared, lm_hidden_states=None, seed=seed, **kwargs)
    )


def fold_stack(substrate, prepared, stack, *, seed: int, **kwargs) -> Folded:
    """A stack computed outside the model and handed to it."""
    return _reduce(
        sub.fold(substrate, prepared, lm_hidden_states=stack, seed=seed, **kwargs)
    )


def fold_single(
    substrate, prepared, single, *, seed: int, stack: Any = None, **kwargs
) -> Folded:
    """Fold as if the layer mixture had produced ``single`` (``[1, L, 256]``).

    With no ``stack`` a placeholder is used, which is the path a generator takes:
    no backbone, no stack, only the feature it produced.
    """
    if stack is None:
        stack = placeholder_stack(substrate, prepared)
    with inject.substituted_single(substrate.net, single):
        return _reduce(
            sub.fold(substrate, prepared, lm_hidden_states=stack, seed=seed, **kwargs)
        )


def fold_without_prior(substrate, prepared, *, seed: int, **kwargs) -> Folded:
    """The upstream's own fold with no language-model pair prior at all.

    Needs a substrate opened with ``load_esmc=False``. ``sub.fold`` refuses
    ``lm_hidden_states=None`` there because, for a probe, it would be a mistake;
    here it is the point, so the guard is lifted on a shallow copy that shares the
    model.
    """
    if substrate.has_esmc:
        raise RuntimeError(
            "with the backbone resident, lm_hidden_states=None means 'compute it'; "
            "open the substrate with load_esmc=False for a fold without a prior"
        )
    lifted = dataclasses.replace(substrate, has_esmc=True)
    return _reduce(
        sub.fold(lifted, prepared, lm_hidden_states=None, seed=seed, **kwargs)
    )


def pair_prior(substrate, *, stack=None, single=None):
    """The pair bias the model is conditioned on, ``[1, L, L, 256]``.

    From a stack it is the shim's own forward, from a single it is the
    expansion the hook substitutes; comparing the two is the tensor-level half of
    the injection parity. Both are computed in :func:`model_context`, the numerical
    region the model computes them in; outside it they differ at bf16 rounding.
    """
    import torch

    shim = inject._shim(substrate.net)
    if (stack is None) == (single is None):
        raise ValueError("give exactly one of stack and single")
    with torch.no_grad(), model_context():
        if stack is not None:
            return shim(stack)
        reference = shim.base_z_linear[1].weight
        return shim.base_z_mlp(single.to(reference.dtype))


# --------------------------------------------------------------- comparison


def tensor_diff(a, b) -> dict:
    """How far two tensors are apart, in the units that decide whether it matters."""
    x = np.asarray(
        a.detach().float().cpu() if hasattr(a, "detach") else a, dtype=np.float64
    )
    y = np.asarray(
        b.detach().float().cpu() if hasattr(b, "detach") else b, dtype=np.float64
    )
    if x.shape != y.shape:
        raise ValueError(f"shapes {x.shape} and {y.shape} differ")
    delta = np.abs(x - y)
    scale = float(np.abs(y).max()) or 1.0
    norm = float(np.linalg.norm(y)) or 1.0
    return {
        "max_abs": float(delta.max()),
        "max_abs_over_max": float(delta.max() / scale),
        "rel_l2": float(np.linalg.norm(x - y) / norm),
        "exact": bool(delta.max() == 0.0),
    }


def coord_diff(a: Folded, b: Folded) -> dict:
    """Two folds of the same sequence, compared atom for atom and by Cα geometry.

    The correspondence is the identity (same sequence, same atom order), so the
    atom-wise numbers need no alignment; the Cα ones use the repository's own
    metrics, which is the one definition of TM-score here (a global-superposition
    lower bound, not TM-align) and of lDDT.
    """
    from foldspace.evaluation import geometry

    if a.coord.shape != b.coord.shape:
        raise ValueError(
            f"atom counts {a.coord.shape[0]} and {b.coord.shape[0]} differ"
        )
    d = np.linalg.norm(a.coord - b.coord, axis=1)
    return {
        "max_atom_shift": float(d.max()),
        "rms_atom_shift": float(np.sqrt((d**2).mean())),
        "ca_rmsd": geometry.kabsch_rmsd(a.ca, b.ca),
        "ca_tm": geometry.tm_score(a.ca, b.ca),
        "ca_lddt": geometry.lddt_ca(a.ca, b.ca),
        "identical": bool(d.max() == 0.0),
    }
