"""The masked-LM head ESMFold2 throws away, put back.

ESMFold2 loads its language model as ``EsmcModel`` (``model.py:690``) — the
backbone only. That class declares
``_keys_to_ignore_on_load_unexpected = [..., r"^lm_head\\."]``, so the six
``lm_head.*`` tensors in the published ``ESMC-6B`` checkpoint are read and
discarded. The consequence for a probe is quiet and total: ``net.esmc`` has no
``lm_head`` attribute, so code that reaches for one finds ``None`` **always**,
not "for this checkpoint". A variant generator guarded by
``if head is None: return []`` therefore generates nothing, on every checkpoint,
forever, while reporting success.

The weights are present. This module loads those six tensors into the same
``nn.Sequential`` upstream would have built and applies it to the backbone that
is already resident. Nothing is duplicated: a second ``EsmcForMaskedLM`` would
mean a second copy of a 6-billion-parameter backbone to reach a head of about
seven million.

## Why the last slice of the stack is the right input

``EsmcForMaskedLM.forward`` computes ``lm_head(encoder_outputs.last_hidden_state)``
(``model.py:643``), and ``last_hidden_state`` is the **post-final-norm** state.
The stack this repository already computes for every entry ends on the same
tensor — ``esm/models/esmc/layers.py:742``::

    norm_x = self.norm(x)
    if len(self.blocks) in layers_to_collect:
        collected.append(norm_x)
    return norm_x, x, tuple(collected), attentions

so ``stack[..., -1, :]`` *is* ``last_hidden_state``, not an approximation of it.
Logits from this module are the logits ``EsmcForMaskedLM`` would produce, and
they cost no extra backbone pass beyond the one the mask requires.
"""

from __future__ import annotations

from typing import Any

__all__ = ["load_lm_head", "masked_logits", "sequence_logits"]


def _esmc_source(substrate: Any) -> str:
    """Where the ESMC weights live, by the substrate's own resolution."""
    from esmfold2_atomworks import paths

    source = getattr(substrate, "esmc_source", None)
    if source and source not in {"bundled", "none"}:
        return str(source)
    esmc_id = getattr(substrate.net.config, "esmc_id", None)
    if esmc_id is None:
        raise RuntimeError(
            "this checkpoint bundles its backbone and names no esmc_id, so the "
            "masked-LM head cannot be located; pass the checkpoint directory "
            "explicitly"
        )
    return str(paths.resolve_esmc(esmc_id))


def load_lm_head(substrate: Any, *, source: str | None = None) -> Any:
    """Build ESMC's masked-LM head and load the published weights into it.

    Placed on the substrate's device in the backbone's dtype, in ``eval`` mode.
    Raises if the tensors are absent rather than returning an untrained head —
    a randomly initialised projection would produce plausible-looking logits
    and silently define every variant.
    """
    import torch
    from esm.models.esmc.checkpoint_layout import published_to_native
    from esm.models.esmc.model import _esmc_lm_head
    from safetensors.torch import load_file

    directory = source or _esmc_source(substrate)
    config = substrate.esmc.config
    head = _esmc_lm_head(config.hidden_size, config.vocab_size)

    from pathlib import Path

    wanted: dict[str, torch.Tensor] = {}
    for shard in sorted(Path(directory).glob("*.safetensors")):
        for key, value in load_file(str(shard)).items():
            if key.startswith("lm_head."):
                wanted[key] = value
    if not wanted:
        raise FileNotFoundError(
            f"no lm_head.* tensors under {directory}; this checkpoint ships no "
            "masked-LM head, so the esmc variant family is unavailable rather "
            "than empty"
        )

    # `published_to_native` names the tensors as they sit inside
    # `EsmcForMaskedLM` -- `lm_head.0.weight`. What is being loaded here is that
    # `nn.Sequential` on its own, whose own keys have no such prefix.
    native = {
        key.removeprefix("lm_head."): value
        for key, value in published_to_native(wanted).items()
    }
    missing, unexpected = head.load_state_dict(native, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"masked-LM head did not load cleanly from {directory}: missing "
            f"{sorted(missing)}, unexpected {sorted(unexpected)}. Either is a "
            "layout drift, and a partly loaded head is an untrained projection "
            "that would still produce plausible-looking logits"
        )

    parameter = next(substrate.esmc.parameters())
    return head.to(device=parameter.device, dtype=parameter.dtype).eval()


def sequence_logits(substrate: Any, prepared: Any, head: Any) -> Any:
    """Per-token vocabulary logits for the sequence as featurized, ``[B, L, V]``.

    The whole ESMC stack is computed by the substrate's own path, so the
    BOS/EOS wrapping, the modified-residue collapse and the scatter back to
    per-token layout are upstream's, not a copy of them.
    """
    import torch

    from foldspace.esmfold2.substrate import lm_hidden_states

    with torch.no_grad():
        stack = lm_hidden_states(substrate, prepared)
        return head(stack[..., -1, :].to(next(head.parameters()).dtype))


def masked_logits(substrate: Any, prepared: Any, head: Any, positions: Any) -> Any:
    """Logits at ``positions`` with those positions masked, ``[n, V]``.

    All of ``positions`` are masked **together**, in one pass. That is the
    quantity a mutation load asks for -- what fits at these sites given the
    rest of the chain -- and it costs one backbone forward rather than one per
    site. The substitutions it proposes are conditioned on the unmasked
    remainder, not on each other, which is the standard reading of a
    multi-mask and is why they are proposals rather than a joint sample.
    """
    from dataclasses import replace

    import torch

    from foldspace.esmfold2.vocabulary import mask_token_id

    features = dict(prepared.features)
    ids = features["input_ids"].clone()
    flat = ids.reshape(-1)
    index = torch.as_tensor(positions, device=ids.device, dtype=torch.long)
    flat[index] = mask_token_id()
    features["input_ids"] = ids

    logits = sequence_logits(substrate, replace(prepared, features=features), head)
    return logits.reshape(-1, logits.shape[-1])[index]
