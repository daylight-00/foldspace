"""RFD4-Proteina's flow with the autoencoder it was trained against, or with an ESM-derived latent in the autoencoder's place.

The release flow is trained on pairs `(Cα trace, z)` where `z = autoencoder.encode(batch)["z_latent"]` (`Proteina._get_clean_sample`),
and a sample is turned into atoms by `autoencoder.decode(...)`. Those are the two places the autoencoder enters, and both are methods
of one object, so an ESM-derived latent replaces it by an autoencoder whose encoder and decoder are the adapted autoencoder ones (`src/foldspace/rfd4/train_autoencoder.py`):

* **encoder**: `z = M(W LN(h_ESMC))` for the sequence channels, and for the residual channels a map of the release encoder's posterior
  mean on the same atoms (a training-only target: the flow generates those channels, and nothing reads them from a structure at
  sampling). It is deterministic and runs under `no_grad` in `eval()` for as long as the model trains: the moment matching is a
  BatchNorm whose running statistics are what defines the target, and a batch's own statistics would make a chain's target depend on the
  chains it is batched with;
* **decoder**: the adapted decoder (the release decoder fine-tuned to read that latent). The flow's own sequence readout is a separate
  output; `condition_decoder_on_sequence_prediction` feeds it to the decoder as the sequence condition, which the adapted decoder was
  trained without, so a sampler for this latent leaves it off and reads the sequence from the decoder (`src/foldspace/rfd4/sample_flow.py`).

`load_flow` builds the module from the release flow checkpoint's own configuration, loads its denoiser weights only (a fresh optimizer,
as `pretrain_ckpt_path` does) and, given an adapter, swaps the autoencoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def esm_encoder_class():
    """The encoder of an ESM-derived autoencoder: the adapted autoencoder latent over the batch's ESMC rows, frozen, in `eval()` for good."""
    import torch
    from torch import nn

    class EsmLatentEncoder(nn.Module):
        def __init__(
            self,
            native: nn.Module,
            latent: nn.Module,
            residual: int,
            native_sample: bool = False,
        ) -> None:
            super().__init__()
            self.native = native  # the release encoder; its posterior mean (or sample) feeds the residual channels or the channels kept in place
            self.latent = latent
            self.residual = residual
            # a posterior in this sense: the channels kept in place are the release encoder's own *samples*, and the code carries the latent's `code_sigma`
            self.native_sample = native_sample
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

        def train(self, mode: bool = True) -> EsmLatentEncoder:
            # `Module.train()` of the Lightning module reaches this one: the running statistics of the moment matching are the
            # definition of the target and are never updated here
            return super().train(False)

        @torch.no_grad()
        def forward(self, batch: dict) -> dict[str, torch.Tensor]:
            # the release encoder's posterior mean feeds the residual channels, or the channels kept in place (in-place layout)
            needs_native = bool(self.residual) or bool(
                getattr(self.latent, "inplace", False)
            )
            posterior = (
                self.native_sample
                or bool(getattr(self.latent, "code_sigma", 0.0))
                or getattr(self.latent, "sigma_head", None) is not None
            )
            out = self.native(batch) if needs_native else None
            mean = out["mean"] if out is not None else None
            kept = out["z_latent"] if (out is not None and self.native_sample) else mean
            # the release's encoder returns a sample as `z_latent`, which is what the flow is trained on; so does this one when it has a posterior
            z = self.latent(batch, native=kept, sample=posterior)
            z_mean = self.latent(batch, native=mean) if posterior else z
            return {"z_latent": z, "mean": z_mean, "log_scale": torch.zeros_like(z)}

    return EsmLatentEncoder


def adapted_autoencoder(ae, adapter_path: str | Path):
    """`ae` with the adapted encoder and decoder of `adapter_path` in place of its own, in place; returns it, frozen."""
    import torch
    from foldspace.rfd4.train_autoencoder import anchored_class
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    state = torch.load(adapter_path, map_location="cpu", weights_only=False)
    residual = (
        int(state["latent"]["res_proj.weight"].shape[0])
        if "res_proj.weight" in state["latent"]
        else 0
    )
    # the layout is read from the checkpoint: a fixed identity code carries its table (`code`), an in-place latent (in-place layout) its flag, and
    # the ESMC latent neither
    source = "fixed" if "code" in state["latent"] else "esmc"
    latent = anchored_class()(
        source,
        ENC.n_tokens,
        residual=residual,
        inplace="inplace_flag" in state["latent"],
        # a learned posterior carries its width head
        stochastic="sigma_head.weight" in state["latent"],
        # a native-calibrated code carries the Cholesky factors of its types' covariances
        calibrated="code_chol" in state["latent"],
        # a banked code carries its donors, and their number is the bank's size
        banked=int(state["latent"]["code_bank"].shape[0])
        if "code_bank" in state["latent"]
        else 0,
    )
    latent.load_state_dict(state["latent"])
    # the posterior a checkpoint was trained under (absent in the first prototypes: deterministic, the release encoder's means in the channels kept)
    posterior = state.get("posterior") or {}
    latent.code_sigma = float(posterior.get("code_sigma", 0.0))
    ae.decoder.load_state_dict(state["decoder"])
    ae.encoder = esm_encoder_class()(
        ae.encoder,
        latent,
        residual,
        native_sample=bool(posterior.get("native_sample", False)),
    )
    for p in ae.parameters():
        p.requires_grad_(False)
    ae.eval()
    return ae


def flow_config(
    base_ckpt: str | Path, nn_overrides=None, opt_overrides=None, ae_ckpt=None
):
    """The configuration a flow checkpoint carries (`hyper_parameters.cfg_exp`), with overrides of its `nn` and `opt` entries. The release's own flow
    checkpoint names its autoencoder through an environment variable of the release's cluster (`${oc.env:DATA_PATH}/...`), which does not resolve here and
    which the model's constructor replaces by the path it is given in any case: `ae_ckpt` is put in its place before the configuration is resolved."""
    import torch
    from omegaconf import OmegaConf

    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["hyper_parameters"]["cfg_exp"]
    cfg = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    if ae_ckpt is not None:
        cfg.autoencoder_ckpt_path = str(ae_ckpt)
    # the release's dataset and loader specifications name paths through environment variables of its own cluster (`${oc.env:...}`) and
    # resolvers of its launcher; a logger resolves the whole configuration, and nothing here reads them
    for split in (
        "train",
        "val",
    ):  # the other entries of `datasets` are referred to from elsewhere in the configuration
        cfg.datasets[split] = {}
    cfg.dataloaders = {}
    cfg.metrics = None
    cfg.callbacks = {}
    for key, value in (nn_overrides or {}).items():
        cfg.nn[key] = value
    for key, value in (opt_overrides or {}).items():
        cfg.opt[key] = value
    OmegaConf.to_container(
        cfg, resolve=True
    )  # fails here, and not in a logger, on an interpolation that does not resolve
    return cfg, ck


def nn_state_from_bundle(path: str | Path) -> dict:
    """The denoiser's tensors of a release bundle (`model.safetensors`: the flow under `nn.` and the autoencoder under `autoencoder.`), as NumPy arrays
    keyed as the denoiser's own state dict is. The release's flow checkpoint (`flow-ema.ckpt`) is the LoRA training format, with each adapted
    layer split into `base_layer` and `lora_A`/`lora_B`, and the denoiser built here has plain layers; the bundle is the same release with the
    adapter merged into the base, which is what the release's own generation reads."""
    from safetensors.numpy import load_file

    return {
        k[len("nn.") :]: v
        for k, v in load_file(str(path)).items()
        if k.startswith("nn.")
    }


def load_flow(
    base_ckpt: str | Path,
    ae_ckpt: str | Path,
    store_dir: str | Path,
    adapter: str | Path | None = None,
    nn_overrides=None,
    opt_overrides=None,
    init: str = "pretrained",
    bundle: str | Path | None = None,
):
    """The flow of `base_ckpt` (denoiser weights only) over the release autoencoder, or over its ESM-derived replacement.

    `init="scratch"` keeps the checkpoint's configuration and leaves the denoiser at its own initialisation: the reference against
    which what pretraining carries over to a latent is measured. With `bundle` (a release `model.safetensors`) the denoiser's weights are the
    bundle's and `base_ckpt` gives only the configuration: the way to load the release's own flow, whose checkpoint holds its adapter unmerged.
    """
    import torch
    from rfproteina.proteina import Proteina

    if init not in ("pretrained", "scratch"):
        raise ValueError(f"init is 'pretrained' or 'scratch', not {init!r}")
    cfg, ck = flow_config(base_ckpt, nn_overrides, opt_overrides, ae_ckpt=ae_ckpt)
    model = Proteina(
        cfg_exp=cfg,
        store_dir=str(store_dir),
        autoencoder_ckpt_path=str(ae_ckpt),
        load_training_metrics=False,
    )
    if bundle is not None:
        state = {
            k: torch.from_numpy(v) for k, v in nn_state_from_bundle(bundle).items()
        }
    else:
        state = {
            k[len("nn.") :]: v
            for k, v in ck["state_dict"].items()
            if k.startswith("nn.")
        }
    if init == "pretrained":
        model.nn.load_state_dict(state, strict=True)
    if adapter is not None:
        adapted_autoencoder(model.autoencoder, adapter)
    return model, ck
