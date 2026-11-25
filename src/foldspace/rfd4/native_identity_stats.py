"""The release encoder's own statistics of the channels an in-place code takes, for each residue type.

The in-place latents give the code the release latent's two identity channels (0 and 10) and the 16 channels its decoder does not read. The fixed identity control of that round put a random, well-separated code
there (`separated_code`); it generated well and designed nothing, and what it lacked was the release's own semantics in those channels as much as context. This reads, on training chains, the release encoder's posterior
*samples* in those 18 channels at the residues that carry a latent, and keeps for each residue type the mean and the covariance of the 18-vector: a code drawn from that is what the release's latent holds in those channels
for a residue of that type, with nothing of its context (`train_autoencoder.py --source fixed --inplace --posterior-native --native-code`).

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/native_identity_stats.py --shards cache/rfd4esmc/shards --ae <ckpt> --out native_identity_stats.npz

The file holds `mean [rows, 18]`, `chol [rows, 18, 18]` (the Cholesky factor of the covariance, with a small ridge), `cov`, `counts`, `channels` (the release's channel of each of the 18) and `pooled` (the rows that had too few
residues and were given the statistics of all residues). A residue type with fewer than `--min-count` residues is given the pooled statistics.
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

RIDGE = 1e-6
MIN_COUNT = 500


def finalize(
    counts: np.ndarray, sums: np.ndarray, sumsq: np.ndarray, min_count: int = MIN_COUNT
) -> dict:
    """The mean and the covariance of each row from its count, sum and sum of outer products, and the Cholesky factors.

    `counts [rows]`, `sums [rows, d]`, `sumsq [rows, d, d]` (sums of `z z^T`). The covariance is the unbiased one. A row with fewer than `min_count` observations gets the statistics of all the observations pooled
    (the rows' own means are not mixed in: the pooled mean and covariance are those of every residue together).
    """
    counts = np.asarray(counts, dtype=np.float64)
    rows, d = sums.shape
    total = counts.sum()
    if total < 2:
        raise ValueError("fewer than two residues were read")
    pooled_mean = sums.sum(0) / total
    pooled_cov = (sumsq.sum(0) - total * np.outer(pooled_mean, pooled_mean)) / (
        total - 1
    )
    mean = np.zeros((rows, d))
    cov = np.zeros((rows, d, d))
    pooled = counts < max(min_count, 2)
    for r in range(rows):
        if pooled[r]:
            mean[r], cov[r] = pooled_mean, pooled_cov
        else:
            mean[r] = sums[r] / counts[r]
            cov[r] = (sumsq[r] - counts[r] * np.outer(mean[r], mean[r])) / (
                counts[r] - 1
            )
    chol = np.stack([np.linalg.cholesky(c + RIDGE * np.eye(d)) for c in cov])
    return {"mean": mean, "cov": cov, "chol": chol, "pooled": pooled, "counts": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--chains",
        type=int,
        default=0,
        help="train chains read (default all of the shards' train set)",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    import torch
    from foldspace.rfd4.train_reading_adapter import KEYS, collate
    from foldspace.rfd4.train_autoencoder import INPLACE_KEEP
    from rfproteina.datapipes.schema.proteina import SPECS
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.model_masks import compute_ae_masks
    from rfproteina.partial_autoencoder.autoencoder import AutoEncoder

    began = time.time()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    shards, ae_path = (str(Path(p).resolve()) for p in (args.shards, args.ae))
    os.chdir(
        out.parent
    )  # the autoencoder makes ./tmp_ae/val_stuff in the working directory
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    index = json.loads((Path(shards) / "index.json").read_text())
    ids = [r["id"] for r in index["train"]][: args.chains or None]
    ae = AutoEncoder.load_from_checkpoint(
        ae_path, strict=True, weights_only=False, map_location="cpu"
    ).to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    defaults = {k: SPECS[k][1] for k in KEYS if k in SPECS}
    mask_token = ENC.token_to_idx["<M>"]
    channels = [c for c in range(32) if c not in INPLACE_KEEP]
    rows = ENC.n_tokens + 1
    counts = np.zeros(rows)
    sums = np.zeros((rows, len(channels)))
    sumsq = np.zeros((rows, len(channels), len(channels)))
    for start in range(0, len(ids), args.batch):
        items = [
            torch.load(Path(shards) / f"{i}.pt", weights_only=True)
            for i in ids[start : start + args.batch]
        ]
        batch = collate(items, defaults, mask_token, device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            z = ae.encoder(batch)["z_latent"].float()
        emit = compute_ae_masks(batch)[1] & batch["is_resolved_mask"].bool()
        types = batch["residue_type"][emit].cpu().numpy()
        code = z[emit][:, channels].cpu().numpy().astype(np.float64)
        np.add.at(counts, types, 1.0)
        np.add.at(sums, types, code)
        np.add.at(sumsq, types, code[:, :, None] * code[:, None, :])
        if (start // args.batch) % 25 == 0:
            print(
                f"   {start + len(items)} of {len(ids)} chains, {time.time() - began:.0f} s",
                flush=True,
            )
    stats = finalize(counts, sums, sumsq, args.min_count)
    np.savez(
        out,
        mean=stats["mean"].astype(np.float32),
        chol=stats["chol"].astype(np.float32),
        cov=stats["cov"].astype(np.float32),
        counts=stats["counts"],
        pooled=stats["pooled"],
        channels=np.array(channels),
        chains=np.array(len(ids)),
        seed=np.array(args.seed),
    )
    print(
        f"{len(ids)} chains, {int(counts.sum())} residues, {int((~stats['pooled']).sum())} of {rows} rows with their own statistics, in {time.time() - began:.0f} s; written {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
