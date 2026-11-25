"""ESMC as a training signal on RFD4-Proteina's native flow: an alignment of the denoiser's final token state with ESMC's reading of the same residues
.

The flow stays what the release trained: its autoencoder, its latent, its denoiser weights (the pretraining checkpoint's, as in the native continuation A) and
its sampler. One term is added to the release's own training step:

    loss = flow loss (as released) + weight * mean over residues of (1 - cos(P(s), LayerNorm(h_ESMC)))

where `s` is the denoiser's final token state, the input of its output heads (`nn.local_latents_linear`, `nn.ca_linear`), `P` is a small projector (the
only new weights, `nn.esmc_align`) and `h_ESMC` is the frozen ESMC-6B reading of the chain's clean sequence, whose rows the batches already carry
(`src/foldspace/rfd4/flow_data.py`: `esmc`, cut to the same crop window as the coordinates). ESMC is a training-time teacher: nothing here calls it at generation, and
the projector is dropped from a checkpoint that is exported for inference (`drop_training_only`).

What is *not* built: a trainer, a loss framework, or an edit of the release's code. `attach` puts a forward pre-hook on `nn.local_latents_linear` (which
sees `s` and nothing else), puts the projector inside `nn` (so that the release's own optimizer builder, EMA callback and checkpoint carry it like any other
weight, `build_optimizers(self.named_parameters(), ...)`), and wraps two methods of the flow matcher: `compute_loss` adds the per-example term `esmc_align` beside
the release's, and `optimization_terms` weights it into the sum that `Proteina.training_step` already backpropagates, steps and averages. The release's
`exploration_k` keeps the best of its noise draws for every loss at once, and the term is kept or dropped with them.

The projector is kept on Adam (`muon_exclude_param_patterns`): a new randomly initialised head is not what Muon is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: the module whose input is the final token state: `LocalLatentsTransformer._forward_impl` runs `s = trunk(...)` and then `local_latents_linear(s)`
S_CONSUMER = "local_latents_linear"
#: where the projector lives, so that its parameters are `nn.esmc_align.*` for the optimizer's patterns, the EMA and the checkpoint
HEAD_NAME = "esmc_align"
#: the prefix of the weights that exist for training only
TRAINING_ONLY_PREFIX = f"nn.{HEAD_NAME}."
#: the loss term's name in the flow matcher's loss dictionary
LOSS_KEY = "esmc_align"
#: ESMC-6B's hidden width
ESMC_DIM = 2560
TOKEN_DIM = 768


def projector(token_dim: int = TOKEN_DIM, hidden: int = 1024, esmc_dim: int = ESMC_DIM):
    """The projector from the denoiser's token state to ESMC's width: a normalisation, two linear maps and a GELU between them (about 3.4 M weights)."""
    import torch

    return torch.nn.Sequential(
        torch.nn.LayerNorm(token_dim),
        torch.nn.Linear(token_dim, hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, esmc_dim),
    )


def align_loss(predicted, esmc, mask):
    """The per-example alignment loss: one minus the cosine between the projected token state and ESMC's row of the same residue, averaged over the
    residues of the example that are there (`mask`, the batch's padding mask). `esmc` is normalised across its features first (no learned scale): ESMC's
    last hidden state has a few features of very large magnitude, which a cosine would otherwise be about."""
    import torch.nn.functional as F

    target = F.layer_norm(esmc.float(), (esmc.shape[-1],))
    cos = F.cosine_similarity(predicted.float(), target, dim=-1)  # [b, n]
    valid = mask.to(cos.dtype)
    return ((1.0 - cos) * valid).sum(1) / valid.sum(1).clamp(min=1.0)


@dataclass
class AlignState:
    """What a hooked forward leaves for the loss: the token state of the last pass, and the diagnostics of the term."""

    weight: float
    #: keep the token state of a forward without gradient too (read by the mechanism analysis; a training run leaves it off)
    keep_always: bool = False
    s: object = None
    handle: object = None
    originals: dict = field(default_factory=dict)
    probe_params: dict = field(default_factory=dict)
    probe: dict = field(default_factory=dict)


def attach(
    model,
    weight: float,
    hidden: int = 1024,
    probe_params: dict | None = None,
    keep_always: bool = False,
) -> AlignState:
    """Add the alignment term to a `Proteina` (or anything shaped like one: `model.nn` with `local_latents_linear`, `model.fm` with `compute_loss` and
    `optimization_terms`). Call it before the trainer builds its optimizers and before a checkpoint is loaded into the module.

    `probe_params` (`{name: parameter}`) makes every loss computation also record, in `state.probe`, the gradient norm of the alignment term and of the
    release's own loss with respect to each of them: the check that the term reaches the shared trunk and the ratio from which a weight is chosen. It costs
    two extra backward passes of a micro-batch and is for the smoke run."""
    import torch

    if hasattr(model.nn, HEAD_NAME):
        raise ValueError(f"the module already has nn.{HEAD_NAME}")
    consumer = getattr(model.nn, S_CONSUMER)
    state = AlignState(
        weight=float(weight),
        probe_params=dict(probe_params or {}),
        keep_always=keep_always,
    )
    # the token width is read from the module that consumes the state (its linear map's input), not assumed
    token_dim = next(
        m for m in consumer.modules() if isinstance(m, torch.nn.Linear)
    ).in_features
    head = projector(token_dim=token_dim, hidden=hidden)
    device = next(model.nn.parameters()).device
    setattr(model.nn, HEAD_NAME, head.to(device))

    def keep_s(_module, args):
        # a forward under no_grad (a recycle pass without gradient, a sampler) leaves nothing that a loss could use, unless a reading asked to keep it
        state.s = args[0] if (state.keep_always or torch.is_grad_enabled()) else None

    state.handle = consumer.register_forward_pre_hook(keep_s)

    fm = model.fm
    state.originals = {
        "compute_loss": fm.compute_loss,
        "optimization_terms": fm.optimization_terms,
    }

    def compute_loss(batch, nn_out):
        losses = state.originals["compute_loss"](batch, nn_out)
        if state.s is None:
            raise RuntimeError(
                "the alignment term needs the token state of a pass with gradient, and the hook saw none"
            )
        if "esmc" not in batch:
            raise KeyError(
                "the batch carries no `esmc` rows: the stream was built without the ESMC array"
            )
        predicted = getattr(model.nn, HEAD_NAME)(state.s)
        per_example = align_loss(predicted, batch["esmc"], batch["padding_mask"])
        if state.probe_params:
            state.probe = gradient_norms(
                per_example,
                losses,
                state.originals["optimization_terms"],
                state.probe_params,
            )
        state.s = None
        return {**losses, LOSS_KEY: per_example}

    def optimization_terms(losses):
        terms = state.originals["optimization_terms"](losses)
        if LOSS_KEY in losses:
            terms = {**terms, LOSS_KEY: state.weight * losses[LOSS_KEY]}
        return terms

    fm.compute_loss = compute_loss
    fm.optimization_terms = optimization_terms
    # the projector's weights stay on Adam
    opt = model.cfg_exp.opt
    patterns = list(opt.get("muon_exclude_param_patterns") or [])
    pattern = rf"^nn\.{HEAD_NAME}\."
    if pattern not in patterns:
        opt["muon_exclude_param_patterns"] = [*patterns, pattern]
    return state


def gradient_norms(per_example, losses, release_terms, params: dict) -> dict:
    """The gradient norm of the alignment term (mean over the examples) and of the release's own optimised loss (`release_terms` is the flow matcher's own
    `optimization_terms`, before the alignment term is weighted in), with respect to each of `params`."""
    import torch

    names = list(params)
    tensors = [params[n] for n in names]
    flow = sum(v.mean() for v in release_terms(losses).values())
    out = {}
    for label, loss in (("align", per_example.mean()), ("flow", flow)):
        grads = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
        out[label] = {
            n: (float(g.norm()) if g is not None else 0.0)
            for n, g in zip(names, grads, strict=True)
        }
    return out


def detach(state: AlignState, model) -> None:
    """Remove the hook and put back the flow matcher's own methods (the projector is left where it is)."""
    if state.handle is not None:
        state.handle.remove()
    for name, original in state.originals.items():
        setattr(model.fm, name, original)


def predict(head, token_state):
    """The projector's reading of a token state, in float32 whatever the state's dtype.

    The state comes out of a forward under bf16 autocast and a reading calls the projector after that context has closed, where a float32 `LayerNorm` given a bf16 input is an error
    (a CPU run, which has no autocast, does not show it)."""
    return head(token_state.float()).float()


def drop_training_only(state_dict: dict) -> tuple[dict, list[str]]:
    """The weights of a checkpoint without the projector (`nn.esmc_align.*`), and the names that were dropped: what an inference checkpoint holds."""
    kept = {
        k: v for k, v in state_dict.items() if not k.startswith(TRAINING_ONLY_PREFIX)
    }
    return kept, [k for k in state_dict if k.startswith(TRAINING_ONLY_PREFIX)]


def load_head(checkpoint, state: str = "ema", expect_step: int | None = None) -> dict:
    """The projector's weights of a training checkpoint (the EMA's or the raw ones), without the `nn.esmc_align.` prefix, for `nn.esmc_align.load_state_dict`.

    A checkpoint without a projector, or at another update than the one asked for, is refused: a reading of the wrong weights is a result that looks like a finding."""
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if expect_step is not None and int(ck["global_step"]) != int(expect_step):
        raise SystemExit(
            f"{checkpoint} is at update {int(ck['global_step'])}, and update {int(expect_step)} was asked for"
        )
    weights = (
        ck["callbacks"]["SimpleEMA"]["ema_state_dict"]
        if state == "ema"
        else ck["state_dict"]
    )
    head = {
        k[len(TRAINING_ONLY_PREFIX) :]: v
        for k, v in weights.items()
        if k.startswith(TRAINING_ONLY_PREFIX)
    }
    if not head:
        raise SystemExit(f"{checkpoint} carries no alignment projector ({state})")
    return head
