"""The audit of a full-width latent: what the 32 channels of the ESMC latent and of its type-only control hold, beside the release encoder's own 32 channels. CPU, nothing trained.

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/esmc_only_audit.py --shards cache/rfd4esmc/shards --ae <release ae> \\
        --adapter esmc_only=<autoencoder.pt> --adapter type_only=<autoencoder.pt> --out audit.json

The release encoder reads the dev chains once (its posterior means) and each adapter's latent is computed from the chains' ESMC rows (or, for a fixed code, from the residue types) at the standard residues that carry
a latent; an adapter must be a full-width latent (no residual channels, not in place). For every latent, all 32 channels together:

* **moments**: the largest channel mean and the range of the channel spreads (a moment-matched latent has 0 and 1), and the mean and the covariance themselves (`mean`, `covariance`);
* **dimension**: the participation ratio of the covariance (the number of directions it spreads over) and the number of eigenvalues that hold 90 % of the variance;
* **type**: the share of the total variance that the residue type fixes, the smallest and the median distance between the centroids of two types, and that smallest distance in units of the spread within types;
* **within types**: the share of each channel's variance that lies within types (range over the channels) and the participation ratio of the within-type residual;
* **context**: the held-out ridge R² of the within-type residual on the neighbours' types, on the Cα geometry and on both (five folds by chain), pooled over the channels weighted by their residual variance, and the same
  with the features' rows shuffled (what the regression reaches by chance). A latent with no variance within types (a fixed code) has none to read.

`src/foldspace/rfd4/code_audit_stats.py` has the arithmetic and `src/foldspace/rfd4/code_audit.py` the dev-chain pass this one repeats for the full width; the recorded reading of each adapter's autoencoder (`summary.json` beside its checkpoint) is copied in.
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
import foldspace.rfd4.code_audit_stats as cs
import foldspace.rfd4.code_audit as ca
from foldspace.rfd4.train_reading_adapter import KEYS, Shards, collate


def latent_summary(z: np.ndarray, t: np.ndarray) -> dict:
    """The moments, dimension and type structure of a latent `z` `[n, d]` whose rows are residues of the types `t` (the context is read by `context_summary`)."""
    z = np.asarray(z, dtype=np.float64)
    covariance = np.cov(z.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    share = np.cumsum(eigenvalues) / eigenvalues.sum()
    centroids, counts = cs.type_centroids(z, t)
    keep = ~np.isnan(centroids).any(1)
    kept = centroids[keep]
    gaps = np.linalg.norm(kept[:, None, :] - kept[None, :, :], axis=-1)[
        np.triu_indices(len(kept), k=1)
    ]
    within = cs.within_type_residual(z, t)
    within_variance = within.var(0)
    total_variance = z.var(0)
    spread = float(np.sqrt(within_variance.sum()))
    channel_share = cs.within_type_share(z, t)
    return {
        "residues": len(z),
        "channels": int(z.shape[1]),
        "types_with_residues": int(keep.sum()),
        "channel_mean_abs_max": float(np.abs(z.mean(0)).max()),
        # the first two moments whole, for a reading of what a flow generates against what the latent is (`read_diagnostics.py`)
        "mean": z.mean(0).tolist(),
        "covariance": covariance.tolist(),
        "channel_std_range": [
            float(np.sqrt(total_variance.min())),
            float(np.sqrt(total_variance.max())),
        ],
        "participation_ratio": float(cs.participation_ratio(covariance)),
        "eigenvalues_for_90_percent": int(np.searchsorted(share, 0.9) + 1),
        "type_share_of_variance": float(
            1.0 - within_variance.sum() / total_variance.sum()
        ),
        "centroid_distance_min": float(gaps.min()),
        "centroid_distance_median": float(np.median(gaps)),
        "within_type_spread": spread,
        "centroid_distance_min_in_spreads": float(gaps.min() / spread)
        if spread > 1e-9
        else None,
        "within_type_share_range": [
            float(channel_share.min()),
            float(channel_share.max()),
        ],
        "within_type_participation_ratio": float(
            cs.participation_ratio(np.cov(within.T))
        )
        if spread > 1e-9
        else None,
        "type_counts": counts.tolist(),
    }


def pooled_r2(r2: np.ndarray, variance: np.ndarray) -> float:
    """The R² of all channels together: each channel's R² weighted by the variance it has to explain."""
    return float((np.asarray(r2) * variance).sum() / variance.sum())


def context_summary(
    z: np.ndarray,
    t: np.ndarray,
    seq: np.ndarray,
    structure: np.ndarray,
    usable: np.ndarray,
    chain: np.ndarray,
    seed: int,
) -> dict | None:
    """The held-out R² of the within-type residual of `z` on the context, pooled over the channels (see the module's header); None where there is no variance within types."""
    residual = cs.within_type_residual(np.asarray(z, dtype=np.float64), t)[usable]
    variance = residual.var(0)
    if variance.sum() <= 1e-9:
        return None
    measured = ca.context_r2(z, t, seq, structure, usable, chain, seed)
    out = {"residues": measured["residues"]}
    for name in ("sequence", "structure", "both"):
        out[name] = pooled_r2(measured[name], variance)
        out[name + "_shuffled"] = pooled_r2(measured[name + "_shuffled"], variance)
    out["per_channel_both"] = [float(x) for x in measured["both"]]
    return out


def build_latent(path: str, in_dim: int, enc, torch):
    """A saved adapted autoencoder checkpoint as its latent in evaluation mode; it must be a full-width latent."""
    import foldspace.rfd4.train_autoencoder as autoencoder

    state = torch.load(path, map_location="cpu", weights_only=False)
    keys = state["latent"]
    if "inplace_flag" in keys or "res_proj.weight" in keys:
        raise ValueError(
            f"{path} is an in-place or a residual latent: this audit is of the full-width latents"
        )
    source = "fixed" if "code" in keys else "esmc"
    latent = autoencoder.anchored_class()(source, enc.n_tokens, in_dim=in_dim)
    latent.load_state_dict(keys)
    latent.eval()
    return latent


def collect(
    ae,
    dev,
    latents: dict,
    defaults: dict,
    mask_token: int,
    device,
    batch_size: int,
    std_of_token: np.ndarray,
    torch,
) -> dict:
    """One pass over the dev chains: the release encoder's posterior means and each latent, at the standard residues that carry a latent, and the chains' Cα and types for the context features."""
    from rfproteina.model_masks import compute_ae_masks

    native, types, chain, position, chain_inputs = [], [], [], [], []
    per_latent = {key: [] for key in latents}
    for start in range(0, len(dev), batch_size):
        items = dev.items[start : start + batch_size]
        batch = collate(items, defaults, mask_token, device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            out = ae.encoder(batch)
        mean = out["mean"].float()
        emit = compute_ae_masks(batch)[1] & batch["is_resolved_mask"].bool()
        std = torch.from_numpy(std_of_token)[batch["residue_type"].cpu()].to(
            emit.device
        )
        select = emit & (std >= 0)
        native.append(mean[select].cpu().numpy())
        types.append(std[select].cpu().numpy())
        for i, item in enumerate(items):
            n = int(item["residue_type"].shape[0])
            chain.append(np.full(int(select[i].sum()), start + i))
            position.append(np.flatnonzero(select[i].cpu().numpy()))
            valid = (emit[i, :n] & batch["atom37_mask"][i, :n, 1].bool()).cpu().numpy()
            t_std = np.where(valid, std[i, :n].cpu().numpy(), -1)
            chain_inputs.append(
                (t_std, batch["coords"][i, :n, 1, :].float().cpu().numpy(), valid)
            )
        for key, latent in latents.items():
            with torch.no_grad():
                z = latent(batch).float()
            per_latent[key].append(z[select].cpu().numpy())
    return {
        "native": np.concatenate(native).astype(np.float64),
        "types": np.concatenate(types),
        "chain": np.concatenate(chain),
        "position": np.concatenate(position),
        "chain_inputs": chain_inputs,
        "latents": {
            key: np.concatenate(parts).astype(np.float64)
            for key, parts in per_latent.items()
        },
    }


def analyse(data: dict, seed: int) -> dict:
    t, chain = data["types"], data["chain"]
    seq, structure, usable = ca.context_matrix(data)
    report: dict = {
        "residues": len(t),
        "chains": len(np.unique(chain)),
        "usable_for_context": int(usable.sum()),
        "latents": {},
    }
    everything = {"native": data["native"], **data["latents"]}
    for key, z in everything.items():
        entry = latent_summary(z, t)
        entry["context"] = context_summary(z, t, seq, structure, usable, chain, seed)
        report["latents"][key] = entry
    return report


def text(report: dict) -> str:
    lines = [
        f"{report['chains']} chains, {report['residues']} standard residues ({report['usable_for_context']} usable for the context features)"
    ]
    for key, e in report["latents"].items():
        c = e["context"]
        context = (
            "no variance within types: no context to read"
            if c is None
            else "context R² within types (held out): sequence {:.3f}, structure {:.3f}, both {:.3f} (shuffled both {:.3f})".format(
                c["sequence"], c["structure"], c["both"], c["both_shuffled"]
            )
        )
        in_spreads = e["centroid_distance_min_in_spreads"]
        lines.append(
            f"{key:7s} 32 channels: mean at most {e['channel_mean_abs_max']:.3f} (abs), spread {e['channel_std_range'][0]:.2f} to {e['channel_std_range'][1]:.2f}; "
            f"participation ratio {e['participation_ratio']:.1f}, {e['eigenvalues_for_90_percent']} eigenvalues hold 90 %; "
            f"the type fixes {e['type_share_of_variance']:.3f} of the variance, centroids {e['centroid_distance_min']:.2f} apart at the closest "
            f"(median {e['centroid_distance_median']:.2f}"
            + (f", {in_spreads:.2f} spreads" if in_spreads is not None else "")
            + f"); within-type share {e['within_type_share_range'][0]:.2f} to {e['within_type_share_range'][1]:.2f}"
            + (
                f", within-type participation ratio {e['within_type_participation_ratio']:.1f}"
                if e["within_type_participation_ratio"] is not None
                else ""
            )
            + f"; {context}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="LABEL=path of an adapted autoencoder checkpoint of a full-width latent (autoencoder.pt; its summary.json beside it is copied into the report)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--dev-chains", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.adapter:
        parser.error("give at least one --adapter LABEL=path")
    labels = [a.split("=", 1)[0] for a in args.adapter]
    if len(set(labels)) != len(labels) or "native" in labels:
        parser.error("two adapters have the same label, or one is called native")

    import torch
    from rfproteina.datapipes.schema.proteina import SPECS
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.partial_autoencoder.autoencoder import AutoEncoder

    began = time.time()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    args.shards, args.ae = (str(Path(p).resolve()) for p in (args.shards, args.ae))
    given = {
        a.split("=", 1)[0]: str(Path(a.split("=", 1)[1]).resolve())
        for a in args.adapter
    }
    os.chdir(
        out.parent
    )  # the autoencoder makes ./tmp_ae/val_stuff in the working directory
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    index = json.loads((Path(args.shards) / "index.json").read_text())
    dev_ids = [r["id"] for r in index["dev"]][: args.dev_chains or None]
    dev = Shards(Path(args.shards), dev_ids)
    ae = AutoEncoder.load_from_checkpoint(
        args.ae, strict=True, weights_only=False, map_location="cpu"
    ).to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    defaults = {k: SPECS[k][1] for k in KEYS if k in SPECS}
    mask_token = ENC.token_to_idx["<M>"]
    std_of_token = ca.standard_map(ENC)
    in_dim = int(dev.items[0]["esmc"].shape[-1])
    latents = {k: build_latent(p, in_dim, ENC, torch) for k, p in given.items()}
    print(
        f"{len(dev_ids)} dev chains and {len(latents)} latents read in {time.time() - began:.0f} s",
        flush=True,
    )
    data = collect(
        ae, dev, latents, defaults, mask_token, device, args.batch, std_of_token, torch
    )
    print(
        f"collected {len(data['types'])} residues in {time.time() - began:.0f} s",
        flush=True,
    )
    report = analyse(data, args.seed)
    for key, path in given.items():
        summary_path = Path(path).with_name("summary.json")
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            end = summary["end"]["complete"]
            report["latents"][key]["recorded_reading"] = {
                "per_residue_rmsd": end["per_residue_rmsd"],
                "backbone_rms": end["backbone_rms"],
                "side_chain_rms": end["side_chain_rms"],
                "sequence_accuracy": end["sequence_accuracy"],
                "noise": summary.get("noise"),
                "epochs": summary.get("epochs"),
                "steps": summary.get("steps"),
                "stack_commit": summary.get("stack_commit"),
            }
    report["args"] = {k: v for k, v in vars(args).items()}
    report["seconds"] = round(time.time() - began, 1)
    out.write_text(json.dumps(report, indent=1))
    out.with_suffix(".txt").write_text(text(report))
    print(text(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
