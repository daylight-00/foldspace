"""The reading adapter: an encoder that sees a backbone, and in two arms the sequence, trained into the frozen release decoder
.

The release encoder reads all atoms and takes the identity of a residue from which atom slots are occupied; the decoder reads the latent
and rebuilds the residue's type, atoms and coordinates. Here the encoder is a copy of the release encoder whose input is cut to what a
design has before it has a side chain: N, Cα, C and O (the bond counts and the element and charge of every other slot are zeroed, since
each of them tells the residue's type). Three arms differ in one thing and are otherwise the same network and budget:

* `backbone`: nothing else; what the geometry of the backbone alone gives.
* `onehot`: the residue type through a learned embedding of the width of ESMC's state (2560), then LayerNorm, then the same
  projection (2560 -> 768, zero at the start) that
* `esmc`: applies to ESMC's last hidden state of the sequence (the corpus array, `src/foldspace/rfd4/esmc_corpus_features.py`).

So `onehot` and `esmc` differ in what enters and not in the parameters downstream. The native arm (`native`) is the release encoder on all atoms,
evaluated and not trained: the upper bound and the pipeline's own check. The decoder is the release decoder, frozen, with the sequence
condition `C_SEQ` set to the mask token on every residue, so that nothing but the latent tells it the sequence. The objective is the
autoencoder's own (KL 1e-4, coordinate L2, residue cross-entropy, atom-presence BCE, balanced token weights) plus the KL from the
adapter's posterior to the release encoder's on the same chain (`--match`): a dense signal that moves the adapter toward the latent the
decoder reads. Evaluation on the dev chains is `src/foldspace/rfd4/datapath_stats.py` on the posterior mean decoded, split into residues with
every legal atom and residues without.

    .venv-rfd4/bin/python -P -B src/foldspace/rfd4/train_reading_adapter.py --shards cache/rfd4esmc/shards --ae <ckpt> --arm esmc --out runs/rfd4esmc/reading_adapter/e1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foldspace.rfd4.datapath_stats as ds

BACKBONE_SLOTS = (0, 1, 2, 4)  # N, CA, C, O of the atom37 layout
ARMS = ("native", "backbone", "onehot", "esmc", "sham")
KEYS = (
    "coords",
    "atom37_mask",
    "residue_type",
    "token_type",
    "chain_iid",
    "charge",
    "ground_truth_element",
    "incident_bond_type_counts",
    "supports_local_latent",
    "is_in_context_conditioning",
    "is_resolved_mask",
    "complete",
    "esmc",
)


class Shards:
    """The shards of a set of chains, read into memory once.

    With `sham_seed`, each chain's ESMC state is replaced by the first rows of another chain's that is at least as long (a fixed
    draw): a valid state of another sequence, the pathway without the information.
    """

    def __init__(self, root: Path, ids: list[str], sham_seed: int | None = None):
        import torch

        self.items = [torch.load(root / f"{i}.pt", weights_only=True) for i in ids]
        if sham_seed is not None:
            rng = np.random.default_rng(sham_seed)
            original = [x["esmc"] for x in self.items]
            lengths = np.array([len(e) for e in original])
            for i, item in enumerate(self.items):
                longer = np.flatnonzero(
                    (lengths >= lengths[i]) & (np.arange(len(lengths)) != i)
                )
                j = int(rng.choice(longer)) if len(longer) else int(np.argmax(lengths))
                item["esmc"] = original[j][: lengths[i]].clone()

    def __len__(self) -> int:
        return len(self.items)


def collate(items: list[dict], defaults: dict, mask_token: int, device) -> dict:
    """The items padded to the longest, stacked, with the masks the networks read; the sequence condition is the mask token."""
    import torch

    n = max(int(x["residue_type"].shape[0]) for x in items)
    batch = {}
    for key in KEYS:
        padded = []
        for x in items:
            v = x[key]
            pad = n - v.shape[0]
            fill = defaults.get(key, 0)
            if pad:
                v = torch.cat([v, v.new_full((pad, *v.shape[1:]), fill)])
            padded.append(v)
        batch[key] = torch.stack(padded)
    lengths = torch.tensor([int(x["residue_type"].shape[0]) for x in items])
    batch["padding_mask"] = torch.arange(n)[None, :] < lengths[:, None]
    batch["condition_sequence_1_residue"] = torch.full(
        (len(items), n), mask_token, dtype=torch.int64
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["coords_nm"] = batch["coords"] / 10.0
    batch["ca_coors_nm"] = batch["coords_nm"][..., 1, :]
    batch["coord_mask"] = batch["atom37_mask"]
    return batch


def sanitize(batch: dict) -> dict:
    """The batch as a design has it before a side chain: backbone atoms only, and nothing else that depends on the residue's type."""
    import torch

    keep = torch.zeros(37, dtype=torch.bool, device=batch["coords"].device)
    keep[list(BACKBONE_SLOTS)] = True
    mask = batch["atom37_mask"] & keep
    out = dict(batch)
    out["atom37_mask"] = mask
    out["coord_mask"] = mask
    out["coords"] = batch["coords"] * mask[..., None]
    out["coords_nm"] = out["coords"] / 10.0
    out["ca_coors_nm"] = out["coords_nm"][..., 1, :]
    out["incident_bond_type_counts"] = torch.zeros_like(
        batch["incident_bond_type_counts"]
    )
    out["ground_truth_element"] = batch["ground_truth_element"] * keep.to(
        batch["ground_truth_element"].dtype
    )
    out["charge"] = batch["charge"] * keep.to(batch["charge"].dtype)
    return out


def adapter_class():
    import torch
    from rfproteina.model_masks import compute_ae_masks
    from torch import nn

    class AdapterEncoder(nn.Module):
        """The release encoder's trunk and heads behind an input cut to the backbone, plus, in two arms, the sequence."""

        def __init__(self, release_encoder, arm: str, n_types: int):
            super().__init__()
            self.arm = arm
            self.base = copy.deepcopy(release_encoder)
            if arm == "onehot":
                self.embed = nn.Embedding(n_types, 2560)
                nn.init.normal_(self.embed.weight, std=1.0)
            if arm in ("onehot", "esmc", "sham"):
                self.norm = nn.LayerNorm(2560, elementwise_affine=False)
                self.proj = nn.Linear(2560, self.base.token_dim, bias=False)
                nn.init.zeros_(
                    self.proj.weight
                )  # the arm starts as the backbone-only encoder

        def new_parameters(self) -> list:
            return [p for n, p in self.named_parameters() if not n.startswith("base.")]

        def forward(self, batch: dict) -> dict:
            enc = self.base
            b = sanitize(batch)
            attn_mask, emit_mask = compute_ae_masks(b)
            c = enc.cond_factory(b)
            c = enc.transition_c_2(enc.transition_c_1(c, attn_mask), attn_mask)
            seqs = enc.init_repr_factory(b)
            if self.arm == "onehot":
                # a padded position carries -1, which an embedding refuses; the padding is masked out below
                ids = batch["residue_type"].clamp(min=0)
                seqs = seqs + self.proj(self.norm(self.embed(ids)))
            elif self.arm in ("esmc", "sham"):
                seqs = seqs + self.proj(self.norm(batch["esmc"].float()))
            seqs = seqs * attn_mask[..., None]
            for layer in enc.transformer_layers:
                seqs = layer(seqs, None, c, attn_mask)
            flat = enc.latent_decoder_mean_n_log_scale(seqs) * emit_mask[..., None]
            mean, log_scale = torch.chunk(flat, chunks=2, dim=-1)
            z = (mean + torch.randn_like(log_scale) * torch.exp(log_scale)) * emit_mask[
                ..., None
            ]
            return {
                "mean": mean,
                "log_scale": log_scale,
                "z_latent": z,
                "emit_mask": emit_mask,
            }

    return AdapterEncoder


def posterior_kl(mean, log_scale, mean_ref, log_scale_ref, mask):
    """KL(q || q_ref) of two diagonal Gaussians, summed over channels, averaged over the residues of the mask."""
    var_ratio = torch_exp(2 * (log_scale - log_scale_ref))
    kl = 0.5 * (
        var_ratio
        + (mean - mean_ref) ** 2 * torch_exp(-2 * log_scale_ref)
        - 1.0
        - 2 * (log_scale - log_scale_ref)
    )
    kl = kl.sum(-1) * mask
    return kl.sum() / mask.sum().clamp(min=1)


def torch_exp(x):
    import torch

    return torch.exp(x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-new", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--match",
        type=float,
        default=1.0,
        help="weight of KL(adapter posterior || release posterior)",
    )
    parser.add_argument(
        "--train-chains", type=int, default=0, help="only the first N training chains"
    )
    parser.add_argument(
        "--dev-chains", type=int, default=0, help="only the first N dev chains"
    )
    parser.add_argument(
        "--max-steps", type=int, default=0, help="stop after N steps (a smoke run)"
    )
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--adapter",
        default="",
        help="evaluate this saved adapter (adapter.pt) and train nothing",
    )
    args = parser.parse_args(argv)

    import torch
    from rfproteina.datapipes.schema.proteina import SPECS
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.model_masks import compute_ae_supervision_mask
    from rfproteina.partial_autoencoder.autoencoder import (
        AutoEncoder,
        blended_token_weights,
        optimization_loss,
    )

    began = time.time()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.shards, args.ae = (str(Path(p).resolve()) for p in (args.shards, args.ae))
    os.chdir(out)  # the autoencoder makes ./tmp_ae/val_stuff in the working directory
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    index = json.loads((Path(args.shards) / "index.json").read_text())
    train_ids = [r["id"] for r in index["train"]][: args.train_chains or None]
    dev_ids = [r["id"] for r in index["dev"]][: args.dev_chains or None]
    sham = args.seed if args.arm == "sham" else None
    train = Shards(Path(args.shards), train_ids, sham) if args.arm != "native" else None
    dev = Shards(Path(args.shards), dev_ids, sham)
    print(
        f"{len(train_ids) if train else 0} train, {len(dev_ids)} dev chains read in {time.time() - began:.0f} s",
        flush=True,
    )

    ae = AutoEncoder.load_from_checkpoint(
        args.ae, strict=True, weights_only=False, map_location="cpu"
    ).to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    cfg = ae.cfg_ae.loss
    mask_token = ENC.token_to_idx["<M>"]
    defaults = {k: SPECS[k][1] for k in KEYS if k in SPECS}
    legal = ae.decoder.residue_atom_mask

    if args.arm == "native":
        model = None
    else:
        model = adapter_class()(ae.encoder, args.arm, ENC.n_tokens).to(device)
        for p in model.parameters():
            p.requires_grad_(True)
        n_new = sum(p.numel() for p in model.new_parameters())
        print(
            f"adapter: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, {n_new / 1e6:.2f} M new",
            flush=True,
        )

    def encode(batch: dict) -> dict:
        if model is None:
            with torch.no_grad():
                return ae.encoder(batch)
        return model(batch)

    @torch.no_grad()
    def evaluate(label: str) -> dict:
        if model is not None:
            model.eval()
        rows = {"all": [], "complete": [], "truncated": []}
        seq = {"all": [], "complete": [], "truncated": []}
        presence = []
        match_l2, match_n = 0.0, 0
        for start in range(0, len(dev), args.batch):
            items = dev.items[start : start + args.batch]
            batch = collate(items, defaults, mask_token, device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                enc = encode(batch)
                teacher = ae.encoder(batch)
                feed = dict(batch)
                feed["z_latent"] = enc["mean"]
                dec = ae.decoder(feed)
            emit = (enc["emit_mask"] & batch["is_resolved_mask"].bool()).cpu().numpy()
            complete = batch["complete"].bool().cpu().numpy()
            coords = batch["coords"].float().cpu().numpy()
            tmask = batch["atom37_mask"].bool().cpu().numpy()
            pred = dec["coors_nm"].float().cpu().numpy() * 10.0
            ptype = dec["aatype_max"].cpu().numpy()
            pres = (dec["atom_presence_logits"].float() > 0).cpu().numpy()
            true = batch["residue_type"].cpu().numpy()
            legal_np = legal[batch["residue_type"]].cpu().numpy()
            for i in range(len(items)):
                parts = {
                    "all": emit[i],
                    "complete": emit[i] & complete[i],
                    "truncated": emit[i] & ~complete[i],
                }
                for part, scored in parts.items():
                    s = ds.residue_scores(coords[i], pred[i], tmask[i], scored)
                    if s.get("residues"):
                        rows[part].append(s)
                    q = ds.sequence_scores(true[i], ptype[i], scored, ENC.n_tokens)
                    if q.get("residues"):
                        seq[part].append(q)
                presence.append(
                    ds.presence_scores(tmask[i], pres[i], legal_np[i], emit[i])["f1"]
                )
            m = emit & (batch["supports_local_latent"].cpu().numpy())
            diff = (
                (enc["mean"].float() - teacher["mean"].float())
                .norm(dim=-1)
                .cpu()
                .numpy()
            )
            ref = teacher["mean"].float().norm(dim=-1).cpu().numpy()
            match_l2 += float((diff / np.maximum(ref, 1e-6))[m].sum())
            match_n += int(m.sum())

        def pooled(group: list[dict], key: str) -> float:
            w = np.array([g["residues"] for g in group], dtype=float)
            return float(np.average([g[key] for g in group], weights=w))

        result = {"label": label}
        for part in ("all", "complete", "truncated"):
            if rows[part]:
                result[part] = {
                    "residues": int(sum(g["residues"] for g in rows[part])),
                    "per_residue_rmsd": pooled(rows[part], "per_residue_mean"),
                    "backbone_rms": pooled(rows[part], "backbone_rms"),
                    "side_chain_rms": pooled(rows[part], "side_chain_rms"),
                    "sequence_accuracy": float(
                        sum(g["accuracy"] * g["residues"] for g in seq[part])
                        / sum(g["residues"] for g in seq[part])
                    ),
                }
        result["presence_f1"] = float(np.mean(presence))
        result["mean_relative_l2_to_release_posterior_mean"] = match_l2 / max(
            match_n, 1
        )
        if model is not None:
            model.train()
        return result

    @torch.no_grad()
    def latent_report() -> dict:
        """The adapter's posterior against the release encoder's on the dev chains, in the release latent's own terms: how much of
        each native channel's posterior mean the adapter predicts (R^2), the KL between the two posteriors per channel, the ratio of
        posterior scales, and the summary of each latent (`datapath_stats.latent_summary`)."""
        if model is not None:
            model.eval()
        keep: dict[str, list] = {k: [] for k in ("am", "al", "nm", "nl", "type")}
        for start in range(0, len(dev), args.batch):
            batch = collate(
                dev.items[start : start + args.batch], defaults, mask_token, device
            )
            with torch.autocast(device.type, dtype=torch.bfloat16):
                enc = encode(batch)
                teacher = ae.encoder(batch)
            emit = enc["emit_mask"] & batch["is_resolved_mask"].bool()
            for key, value in (
                ("am", enc["mean"]),
                ("al", enc["log_scale"]),
                ("nm", teacher["mean"]),
                ("nl", teacher["log_scale"]),
            ):
                keep[key].append(value.float()[emit].cpu().numpy().astype(np.float64))
            keep["type"].append(batch["residue_type"][emit].cpu().numpy())
        c = {k: np.concatenate(v) for k, v in keep.items()}
        resid = ((c["am"] - c["nm"]) ** 2).sum(0)
        total = ((c["nm"] - c["nm"].mean(0)) ** 2).sum(0)
        sigma_a, sigma_n = np.exp(c["al"]), np.exp(c["nl"])
        kl = 0.5 * (
            (sigma_a / sigma_n) ** 2
            + (c["am"] - c["nm"]) ** 2 / sigma_n**2
            - 1.0
            - 2.0 * (c["al"] - c["nl"])
        )
        release = ds.latent_summary(c["nm"], c["nl"], c["type"])
        active = np.array(release["kl_per_channel"]) > 0.1
        r2 = 1.0 - resid / np.maximum(total, 1e-12)
        if model is not None:
            model.train()
        return {
            "residues": len(c["am"]),
            "r2_of_posterior_mean_per_channel": r2.tolist(),
            "r2_mean_over_the_release_active_channels": float(r2[active].mean()),
            "kl_adapter_to_release_per_channel": kl.mean(0).tolist(),
            "scale_ratio_per_channel": (sigma_a.mean(0) / sigma_n.mean(0)).tolist(),
            "adapter": ds.latent_summary(c["am"], c["al"], c["type"]),
            "release": release,
        }

    log = (out / "log.jsonl").open("a")
    results = []

    def record(entry: dict) -> None:
        entry["seconds"] = round(time.time() - began, 1)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry), flush=True)

    first = evaluate("start" if model is not None else "native")
    record(first)
    results.append(first)
    if model is None:
        (out / "summary.json").write_text(
            json.dumps({"arm": "native", "dev": first, "ae": args.ae}, indent=1)
        )
        return 0
    if args.adapter:
        model.load_state_dict(
            torch.load(args.adapter, map_location=device, weights_only=True)
        )
        result = evaluate("adapter")
        record(result)
        (out / "eval.json").write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "adapter": args.adapter,
                    "dev": result,
                    "latent": latent_report(),
                    "ae": args.ae,
                    "stack_commit": os.environ.get("FS_STACK_COMMIT"),
                },
                indent=1,
            )
        )
        return 0

    new = set(map(id, model.new_parameters()))
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in new], "lr": args.lr},
        {"params": model.new_parameters(), "lr": args.lr_new},
    ]
    groups = [g for g in groups if g["params"]]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.0)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = args.epochs * steps_per_epoch
    if args.max_steps:
        total = min(total, args.max_steps)
    rng = np.random.default_rng(args.seed)
    step = 0
    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        running = {}
        for s in range(steps_per_epoch):
            if step >= total:
                break
            items = [
                train.items[int(i)]
                for i in order[s * args.batch : (s + 1) * args.batch]
            ]
            batch = collate(items, defaults, mask_token, device)
            scale = min(1.0, (step + 1) / args.warmup)
            for g, base in zip(optimizer.param_groups, base_lrs, strict=True):
                g["lr"] = base * scale
            with torch.autocast(device.type, dtype=torch.bfloat16):
                enc = model(batch)
                feed = dict(batch)
                feed["z_latent"] = enc["z_latent"]
                dec = ae.decoder(feed)
                if args.match:
                    with torch.no_grad():
                        teacher = ae.encoder(batch)
            dec = {
                k: (v.float() if torch.is_floating_point(v) else v)
                for k, v in dec.items()
            }
            mean, log_scale = enc["mean"].float(), enc["log_scale"].float()
            loss_mask = compute_ae_supervision_mask(batch)
            weights = blended_token_weights(
                batch["residue_type"], loss_mask, cfg.balanced_fraction
            )
            losses = {}
            losses.update(
                ae.compute_kl_penalty(mean, log_scale, loss_mask, w=cfg.kl.weight)
            )
            losses.update(
                ae.compute_struct_rec_loss(
                    dec,
                    batch,
                    loss_mask,
                    reduce_mode="sum",
                    loss_ty=cfg.struct.type,
                    weight=cfg.struct.weight,
                    token_weights=weights,
                )
            )
            losses.update(
                ae.compute_seq_rec_loss(
                    dec, batch, loss_mask, weight=cfg.seq.weight, token_weights=weights
                )
            )
            losses.update(
                ae.compute_atom_presence_loss(
                    dec,
                    batch,
                    loss_mask,
                    weight=cfg.atom_presence.weight,
                    token_weights=weights,
                )
            )
            loss = optimization_loss(losses, cfg.struct.type)
            matched = None
            if args.match:
                matched = posterior_kl(
                    mean,
                    log_scale,
                    teacher["mean"].float(),
                    teacher["log_scale"].float(),
                    loss_mask.float(),
                )
                loss = loss + args.match * matched
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            for k, v in (
                ("loss", loss),
                ("ce_seq", losses["ce_seq"].mean()),
                ("struct_l2", losses["struct_l2"].mean()),
                ("presence", losses["atom_presence_bce"].mean()),
                ("match", matched if matched is not None else loss * 0),
            ):
                running.setdefault(k, []).append(float(v))
            if step % 25 == 0:
                record(
                    {
                        "step": step,
                        "epoch": epoch,
                        **{k: round(float(np.mean(v)), 5) for k, v in running.items()},
                    }
                )
                running = {}
        result = evaluate(f"epoch {epoch + 1}")
        record(result)
        results.append(result)
        if step >= total:
            break
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "adapter.pt"
    )
    summary = {
        "arm": args.arm,
        "steps": step,
        "epochs": epoch + 1,
        "train_chains": len(train_ids),
        "dev_chains": len(dev_ids),
        "batch": args.batch,
        "lr": args.lr,
        "lr_new": args.lr_new,
        "match": args.match,
        "ae": args.ae,
        "start": results[0],
        "end": results[-1],
        "latent": latent_report(),
        "seconds": round(time.time() - began, 1),
        "stack_commit": os.environ.get("FS_STACK_COMMIT"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
