"""ESMC as a conditioning input of RFD4-Proteina's denoiser: a small adapter that adds a reading of the given sequence to the denoiser's initial token representation
.

The denoiser's input representation `s_init` is the projection of the features of a residue (`init_repr_factory`: time-independent, one bias-free linear map of the
concatenated features, masked); a residue's identity is already one of them when the sequence is given (the release's `folding` task: `C_SEQ` exact, the 28
sequence-probability conditions). The adapter adds to it

    s_init  +=  A(h)        A = LayerNorm(2560) -> Linear(2560, 768, no bias), zero at the start, gated by whether a residue has a row at all

where `h` is ESMC-6B's last hidden state of the sequence the denoiser is given, carried by the batch as `esmc_context` (`src/foldspace/rfd4/flow_data.py`: the crop's rows when the
sequence is given, zeros when it is not). Without rows the hook leaves the underlying denoiser unchanged. This is the release model only if its weights have remained frozen;
jointly fine-tuned checkpoints can drift even with rows removed. ESMC is *used at inference* here, unlike the alignment term of `flow_align.py`, which only teaches.

It is attached as a forward hook on `nn.init_repr_factory`, which sees the batch it reads (so the rows come from the batch and no state is passed around), with the adapter
inside `nn` so that the release's optimizer builder, EMA callback and checkpoints carry it (on Adam, as the alignment projector is). Nothing of the release is edited. An
adapter checkpoint is **not** one the release's own generator can load (`strict=True` refuses the extra module): the fold script builds the model, attaches the adapter
and loads it.

Where the rows come from at inference is an input of that script (a sequence's rows, computed by ESMC through `esmfold2-atomworks`' substrate), not of the hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: the module whose output is the initial token representation (`LocalLatentsTransformer._forward_impl`: `s_init = init_repr_factory(input) * mask`)
CONSUMER = "init_repr_factory"
#: where the adapter lives, so that its parameters are `nn.esmc_adapter.*` for the optimizer's patterns, the EMA and the checkpoint
HEAD_NAME = "esmc_adapter"
PREFIX = f"nn.{HEAD_NAME}."
#: the batch's key for the rows the adapter reads
CONTEXT_KEY = "esmc_context"
ESMC_DIM = 2560


def adapter(esmc_dim: int = ESMC_DIM, token_dim: int = 768):
    """`LayerNorm -> Linear` with the linear map at zero: the model it is added to is unchanged until the first step."""
    import torch

    linear = torch.nn.Linear(esmc_dim, token_dim, bias=False)
    torch.nn.init.zeros_(linear.weight)
    return torch.nn.Sequential(torch.nn.LayerNorm(esmc_dim), linear)


@dataclass
class AdaptState:
    handle: object = None
    calls: int = 0
    with_rows: int = 0
    originals: dict = field(default_factory=dict)


@dataclass
class AddContext:
    """Callable hooks deepcopy with their head, as required by upstream EMA."""

    head: object
    state: AdaptState

    def __call__(self, _module, args, output):
        batch = args[0]
        rows = batch.get(CONTEXT_KEY) if hasattr(batch, "get") else None
        self.state.calls += 1
        if rows is None:
            return output
        self.state.with_rows += 1
        rows = rows.to(device=output.device)
        present = (rows != 0).any(dim=-1, keepdim=True)
        delta = self.head(rows.float()) * present
        return output + delta.to(output.dtype)


def attach(model) -> AdaptState:
    """Add the adapter to a `Proteina` (anything with `model.nn.init_repr_factory`, `model.nn.token_dim` and `model.cfg_exp.opt`). Call it before the trainer builds its optimizers
    and before a checkpoint that carries the adapter is loaded."""

    if hasattr(model.nn, HEAD_NAME):
        raise ValueError(f"the module already has nn.{HEAD_NAME}")
    consumer = getattr(model.nn, CONSUMER)
    head = adapter(token_dim=int(model.nn.token_dim))
    device = next(model.nn.parameters()).device
    setattr(model.nn, HEAD_NAME, head.to(device))
    state = AdaptState()

    state.handle = consumer.register_forward_hook(AddContext(head, state))
    # the adapter's weights stay on Adam: a new head is not what Muon is for
    opt = model.cfg_exp.opt
    patterns = list(opt.get("muon_exclude_param_patterns") or [])
    pattern = rf"^nn\.{HEAD_NAME}\."
    if pattern not in patterns:
        opt["muon_exclude_param_patterns"] = [*patterns, pattern]
    return state


def detach(state: AdaptState, model) -> None:
    """Remove the hook and the adapter (the model is the release's again)."""
    if state.handle is not None:
        state.handle.remove()
        state.handle = None
    if hasattr(model.nn, HEAD_NAME):
        delattr(model.nn, HEAD_NAME)


def freeze_generator(model) -> dict:
    """Freeze the release generator/AE, retaining gradients through the trunk to the adapter."""
    from rfproteina.utils.weights import (
        ParameterFreezingConfig,
        freeze_parameters_with_config,
    )

    if not hasattr(model.nn, HEAD_NAME):
        raise ValueError("attach the ESMC adapter before selecting trainable weights")
    policy = ParameterFreezingConfig(
        freeze_by_default=True, param_policies={f"{PREFIX}*": False}
    )
    freeze_parameters_with_config(model, policy, verbose=False)
    trainable = {
        name: p.numel() for name, p in model.named_parameters() if p.requires_grad
    }
    if not trainable or any(not name.startswith(PREFIX) for name in trainable):
        raise RuntimeError(f"unexpected trainable parameters: {list(trainable)}")
    return {"names": list(trainable), "parameters": sum(trainable.values())}
