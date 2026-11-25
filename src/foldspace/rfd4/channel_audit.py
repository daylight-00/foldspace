"""Which of the release latent's 32 channels does the release decoder read?

Half of the channels sit at the prior.
A variant of the ESM-derived latent that keeps the informative native channels where the pretrained flow expects them and puts the ESM
code in the channels at the prior ("in-place layout") assumes the decoder does not read those. That is a measurement and not a premise: a channel
at the prior in distribution can still carry a small signal the decoder uses, and replacing it with something else would then cost
reconstruction before any ESM code is learned.

The release encoder and decoder are run on the dev chains as the release runs them (the sequence condition masked), and the latent
is changed before the decoder reads it:

* **as encoded** (the reference);
* **one channel at a time**: channel `c` replaced by an independent standard-normal draw (what the flow's prior would put there), for
  each of the 32 channels; the change in per-residue RMSD (all atoms and side chains) and in the residue types recovered is the
  channel's weight for the decoder;
* **groups**: the 16 channels at the prior replaced together by draws and by zeros; the 16 informative channels replaced together
  (the control: it must destroy the reconstruction, or the test does not discriminate).

A channel that costs nothing when replaced is free for a code; one that costs something is not, and the cost is the price of the
code's placing there.

    .venv-rfd4/bin/python -P -B src/foldspace/rfd4/channel_audit.py --shards cache/rfd4esmc/shards --ae <ckpt> --out runs/rfd4esmc/audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foldspace.rfd4.datapath_stats as ds
from foldspace.rfd4.train_reading_adapter import KEYS, Shards, collate

#: the channels with a spread above the prior's, from the 5,600 dev residues
ACTIVE = [0, 1, 4, 5, 8, 10, 13, 14, 15, 16, 17, 22, 25, 27, 28, 31]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chains", type=int, default=96, help="dev chains read")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch
    from rfproteina.datapipes.schema.proteina import SPECS
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.model_masks import compute_ae_masks
    from rfproteina.partial_autoencoder.autoencoder import AutoEncoder

    began = time.time()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.shards, args.ae = (str(Path(p).resolve()) for p in (args.shards, args.ae))
    os.chdir(out)  # the autoencoder makes ./tmp_ae/val_stuff in the working directory
    device = torch.device(args.device)
    index = json.loads((Path(args.shards) / "index.json").read_text())
    dev = Shards(Path(args.shards), [r["id"] for r in index["dev"]][: args.chains])
    ae = AutoEncoder.load_from_checkpoint(
        args.ae, strict=True, weights_only=False, map_location="cpu"
    ).to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    mask_token = ENC.token_to_idx["<M>"]
    defaults = {k: SPECS[k][1] for k in KEYS if k in SPECS}
    batches = []
    for start in range(0, len(dev), args.batch):
        batch = collate(
            dev.items[start : start + args.batch], defaults, mask_token, device
        )
        with (
            torch.no_grad(),
            torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            z = ae.encode(batch)["z_latent"].float()
        batches.append((batch, z))

    @torch.no_grad()
    def reading(change) -> dict:
        """The reconstruction scores over the dev chains when `change(z, emit)` is applied to the latent before the decoder reads it."""
        rows = {"all": [], "side": []}
        seq = []
        for k, (batch, z0) in enumerate(batches):
            _, emit = compute_ae_masks(batch)
            gen = torch.Generator(device=device).manual_seed(args.seed + k)
            z = change(z0.clone(), emit, gen)
            feed = dict(batch)
            feed["z_latent"] = z
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                dec = ae.decoder(feed)
            scored = (
                (
                    dec["emit_mask"]
                    & batch["is_resolved_mask"].bool()
                    & batch["complete"].bool()
                )
                .cpu()
                .numpy()
            )
            coords = batch["coords"].float().cpu().numpy()
            tmask = batch["atom37_mask"].bool().cpu().numpy()
            pred = dec["coors_nm"].float().cpu().numpy() * 10.0
            ptype = dec["aatype_max"].cpu().numpy()
            true = batch["residue_type"].cpu().numpy()
            for i in range(len(coords)):
                s = ds.residue_scores(coords[i], pred[i], tmask[i], scored[i])
                if s.get("residues"):
                    rows["all"].append(s)
                q = ds.sequence_scores(true[i], ptype[i], scored[i], ENC.n_tokens)
                if q.get("residues"):
                    seq.append(q)

        def pooled(group: list[dict], key: str) -> float:
            w = np.array([g["residues"] for g in group], dtype=float)
            return float(np.average([g[key] for g in group], weights=w))

        return {
            "per_residue_rmsd": pooled(rows["all"], "per_residue_mean"),
            "side_chain_rms": pooled(rows["all"], "side_chain_rms"),
            "sequence_accuracy": float(
                sum(g["accuracy"] * g["residues"] for g in seq)
                / sum(g["residues"] for g in seq)
            ),
        }

    def replace(channels: list[int], how: str):
        idx = torch.tensor(channels, device=device)

        def change(z, emit, gen):
            if how == "draw":
                draw = torch.randn(
                    z.shape[:-1] + (len(channels),), device=device, generator=gen
                )
            else:
                draw = torch.zeros(z.shape[:-1] + (len(channels),), device=device)
            z[..., idx] = torch.where(emit[..., None], draw, z[..., idx])
            return z

        return change

    result = {"as_encoded": reading(lambda z, emit, gen: z)}
    base = result["as_encoded"]
    print("as encoded", base, flush=True)
    per_channel = {}
    for c in range(32):
        r = reading(replace([c], "draw"))
        per_channel[c] = {
            **r,
            "d_rmsd": r["per_residue_rmsd"] - base["per_residue_rmsd"],
            "d_side_chain": r["side_chain_rms"] - base["side_chain_rms"],
            "d_sequence_accuracy": r["sequence_accuracy"] - base["sequence_accuracy"],
            "informative": c in ACTIVE,
        }
        print(
            f"channel {c:2d} ({'active' if c in ACTIVE else 'prior '}): d_rmsd {per_channel[c]['d_rmsd']:+.4f}"
            f"  d_side_chain {per_channel[c]['d_side_chain']:+.4f}  d_sequence_accuracy {per_channel[c]['d_sequence_accuracy']:+.4f}",
            flush=True,
        )
    prior = [c for c in range(32) if c not in ACTIVE]
    result["prior_channels_drawn"] = reading(replace(prior, "draw"))
    result["prior_channels_zeroed"] = reading(replace(prior, "zero"))
    result["active_channels_drawn_control"] = reading(replace(ACTIVE, "draw"))
    for name in (
        "prior_channels_drawn",
        "prior_channels_zeroed",
        "active_channels_drawn_control",
    ):
        print(name, result[name], flush=True)
    report = {
        "args": vars(args),
        "chains": len(dev),
        "residues": int(sum(int(b["padding_mask"].sum()) for b, _ in batches)),
        "groups": result,
        "per_channel": per_channel,
        "active_channels": ACTIVE,
        "seconds": round(time.time() - began, 1),
        "stack_commit": os.environ.get("FS_STACK_COMMIT"),
    }
    (out / "audit.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
