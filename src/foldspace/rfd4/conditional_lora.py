"""Small context-gated PEFT adaptation of the existing RFD4 denoiser.

RFD4 owns LoRA injection, optimizer, EMA and checkpointing. Foldspace only chooses
four projections in the last two attention blocks and gates the PEFT B outputs
per example. A target row enables adaptation for the entire complex, including
binder tokens; examples without any target rows retain the frozen release.

The gate is set at the eager root forward. Recycle activation checkpointing is
disabled (pair checkpointing stays enabled), so backward never rereads mutable
gate state. This prototype supports eager, single-device sequential execution.
Adapters must remain unmerged: merging would destroy rows-off equality.
"""

from __future__ import annotations

from dataclasses import dataclass

import foldspace.rfd4.flow_adapt as fad

CONFIG_KEY = "foldspace_conditional_lora"


def configuration(nn, rank: int = 8) -> dict:
    if rank < 1 or len(nn.transformer_layers) < 2:
        raise ValueError("positive rank and at least two transformer blocks required")
    return {
        "version": 1,
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "train_bias": "none",
        "backend": "peft",
        "target_modules": [
            f"transformer_layers.{i}.mhba.mha.{projection}"
            for i in range(len(nn.transformer_layers) - 2, len(nn.transformer_layers))
            for projection in ("to_qkv", "to_out_node")
        ],
        "gate": "any_nonzero_esmc_row_per_example",
        "recycle_checkpoint": False,
    }


@dataclass
class Gate:
    present: object = None

    def __call__(self, module, args, kwargs):
        if getattr(module, "recycle_checkpoint", False):
            raise RuntimeError("conditional LoRA requires recycle checkpointing off")
        batch = args[0] if args else kwargs["input"]
        rows = batch.get(fad.CONTEXT_KEY)
        mask = batch["padding_mask"]
        if rows is None:
            self.present = mask.new_zeros((mask.shape[0], 1, 1))
        else:
            if rows.shape[:2] != mask.shape or rows.ndim != 3:
                raise ValueError("ESMC rows do not match the denoiser batch")
            supplied = rows.ne(0).any(-1).to(device=mask.device)
            self.present = (supplied & mask.bool()).any(-1)[:, None, None]


@dataclass
class GateOutput:
    gate: Gate

    def __call__(self, module, args, output):
        if self.gate.present is None or output.shape[0] != self.gate.present.shape[0]:
            raise RuntimeError("conditional LoRA gate has no matching root batch")
        return output * self.gate.present.to(device=output.device, dtype=output.dtype)


def is_trainable(name: str) -> bool:
    return name.startswith(fad.PREFIX) or (
        name.startswith("nn.transformer_layers.")
        and any(part in name for part in (".lora_A.default.", ".lora_B.default."))
    )


def canonical_base_name(name: str) -> str:
    return name.replace(".base_layer.", ".")


def attach(model, config: dict | None = None) -> dict:
    from omegaconf import OmegaConf, open_dict
    from rfproteina.utils.lora import add_lora_adapter
    from torch import nn

    if getattr(model.nn, "use_torch_compile", False):
        raise ValueError("conditional LoRA requires eager execution")
    config = configuration(model.nn) if config is None else dict(config)
    expected = configuration(model.nn, int(config["r"]))
    if config != expected:
        raise ValueError("unsupported conditional LoRA configuration")
    # Exact names are also an upstream contract: suffix matching must not expand
    # this deliberately small adaptation to other projections.
    for name in config["target_modules"]:
        layer = model.nn.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{name} is not the expected plain Linear")
    add_lora_adapter(model, OmegaConf.create(config))
    gate = Gate()
    model.nn.recycle_checkpoint = False
    model.nn.register_forward_pre_hook(gate, with_kwargs=True)
    for name in config["target_modules"]:
        layer = model.nn.get_submodule(name)
        if list(layer.lora_B) != ["default"] or layer.merged:
            raise RuntimeError("expected one unmerged PEFT adapter")
        layer.lora_B["default"].register_forward_hook(GateOutput(gate))
    with open_dict(model.cfg_exp):
        model.cfg_exp[CONFIG_KEY] = config
    patterns = list(model.cfg_exp.opt.get("muon_exclude_param_patterns") or [])
    model.cfg_exp.opt["muon_exclude_param_patterns"] = [*patterns, r"\.lora_[AB]\."]
    # Upstream injection freezes all weights, including the existing ESMC head.
    return freeze(model)


def freeze(model) -> dict:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_trainable(name))
    selected = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if not any(".lora_B." in n for n in selected) or not any(
        n.startswith(fad.PREFIX) for n in selected
    ):
        raise RuntimeError("both ESMC adapter and LoRA parameters are required")
    return {
        "names": list(selected),
        "parameters": sum(selected.values()),
        "lora_parameters": sum(
            v for n, v in selected.items() if not n.startswith(fad.PREFIX)
        ),
    }


def checkpoint_config(checkpoint: dict) -> dict | None:
    cfg = checkpoint.get("hyper_parameters", {}).get("cfg_exp", {})
    value = cfg.get(CONFIG_KEY)
    from omegaconf import OmegaConf

    return None if value is None else OmegaConf.to_container(value, resolve=True)
