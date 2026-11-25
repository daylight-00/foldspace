"""The code bank of an ESM-derived autoencoder, and the checks of the control that draws from it.

`contextual_hybrid`'s code of a residue is a function of ESMC's state at that residue. The control keeps the code's type centroids and the whole distribution of its variation within a type, and takes away the correspondence between a code
and its residue's own context: the code of a residue is the code of *another* residue of the same type, drawn at random from a bank, at every presentation (`train_autoencoder.py --source fixed --inplace --posterior-native --code-bank`).
The bank is the posterior mean (no noise) of the autoencoder's code at every standard residue that carries a latent in the first train chains of the P1 pool; a type with fewer than `--min-count` residues draws from all of them.

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/code_bank.py --shards cache/rfd4esmc/shards --adapter <contextual_hybrid autoencoder.pt> --out code_bank.npz
    python -P -B src/foldspace/rfd4/code_bank.py --check audit.json --key type_shuffled_hybrid --reference contextual_hybrid          (the two checks of the control, on the audit of `src/foldspace/rfd4/code_audit.py --bank`)

The file holds `code_bank [N, 18]` (sorted by residue type), `bank_start` and `bank_count` `[rows]` (the donors of a type; a type with too few residues has all `N`), `mean [rows, 18]` (the type centroids, the code's mean
reading), `counts`, `pooled`, `bank_types`, `channels` (the release's channel of each of the 18) and the provenance of the adapter and the chains.

The checks, on the held dev chains: **no_context** the control's code has no context: the held-out context R² of its identity slots is not above `NO_CONTEXT_LIMIT` on either (a held-out R² below zero is the ridge's overfitting and says nothing
of context: a code with none has -0.015 on 9,600 synthetic residues and -0.004 on the audit's 43,931 real ones); **type_structure_kept** it keeps what was to be kept: its type-centroid similarity (alignment) is within `ALIGNMENT_LIMIT` of the reference's and the
within-type share of its identity slots within `TYPE_STRUCTURE_SHARE_LIMIT`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MIN_COUNT = 500
NO_CONTEXT_LIMIT = 0.01
ALIGNMENT_LIMIT = 0.02
TYPE_STRUCTURE_SHARE_LIMIT = 0.01


def bank_layout(
    types: np.ndarray, codes: np.ndarray, rows: int, min_count: int = MIN_COUNT
) -> dict:
    """The bank of `codes` (one row a residue, `types` its residue type, a row index below `rows`) sorted by type, with where each type's donors are and the centroids.

    A type with fewer than `min_count` residues (and a type with none) has all the residues for donors and their pooled mean for its centroid, as the control's statistics did (`native_identity_stats.finalize`)."""
    types = np.asarray(types, dtype=np.int64)
    codes = np.asarray(codes, dtype=np.float64)
    if types.ndim != 1 or codes.ndim != 2 or len(types) != len(codes):
        raise ValueError(
            f"types {types.shape} and codes {codes.shape} are not one row a residue"
        )
    if len(types) < 2:
        raise ValueError("fewer than two residues were read")
    if types.min() < 0 or types.max() >= rows:
        raise ValueError(
            f"residue types run from {types.min()} to {types.max()}, the table has {rows} rows"
        )
    order = np.argsort(types, kind="stable")
    counts = np.bincount(types, minlength=rows).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    pooled = counts < max(min_count, 1)
    sums = np.zeros((rows, codes.shape[1]))
    np.add.at(sums, types, codes)
    pooled_mean = codes.mean(0)
    mean = np.where(
        pooled[:, None], pooled_mean[None, :], sums / np.maximum(counts, 1)[:, None]
    )
    total = len(types)
    return {
        "code_bank": codes[order].astype(np.float32),
        "bank_types": types[order],
        "bank_start": np.where(pooled, 0, starts).astype(np.int64),
        "bank_count": np.where(pooled, total, counts).astype(np.int64),
        "mean": mean.astype(np.float32),
        "counts": counts,
        "pooled": pooled,
    }


def draw_donors(
    layout: dict, token_types: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """A donor's code for each residue type of `token_types`: a row drawn uniformly from that type's donors."""
    t = np.asarray(token_types, dtype=np.int64)
    start, count = layout["bank_start"][t], layout["bank_count"][t]
    pick = np.minimum((rng.random(len(t)) * count).astype(np.int64), count - 1)
    return np.asarray(layout["code_bank"])[start + pick]


def verdict(report: dict, key: str, reference: str) -> dict:
    """no_context and type_structure_kept from an audit report (`src/foldspace/rfd4/code_audit.py`): the control `key` against the adapter `reference`."""
    control, ref = report["adapters"][key], report["adapters"][reference]
    if "context" not in control:
        raise ValueError(
            f"the audit has no context reading for {key}: its identity slots have no variation within types"
        )
    both = control["context"]["both"]
    no_context_holds = all(float(x) <= NO_CONTEXT_LIMIT for x in both)
    alignment_gap = abs(
        float(control["alignment"]["similarity_r2"]) - float(ref["alignment"]["similarity_r2"])
    )
    share_gaps = [
        abs(float(control["within_type_share"][s]) - float(ref["within_type_share"][s]))
        for s in ref["slots"]
    ]
    type_structure_kept_holds = alignment_gap <= ALIGNMENT_LIMIT and all(g <= TYPE_STRUCTURE_SHARE_LIMIT for g in share_gaps)
    return {
        "no_context": {
            "passes": bool(no_context_holds),
            "context_r2_both": [float(x) for x in both],
            "limit": NO_CONTEXT_LIMIT,
            "reference_context_r2_both": [float(x) for x in ref["context"]["both"]],
        },
        "type_structure_kept": {
            "passes": bool(type_structure_kept_holds),
            "alignment_gap": alignment_gap,
            "alignment_limit": ALIGNMENT_LIMIT,
            "within_type_share_gaps": share_gaps,
            "share_limit": TYPE_STRUCTURE_SHARE_LIMIT,
            "control_alignment": float(control["alignment"]["similarity_r2"]),
            "reference_alignment": float(ref["alignment"]["similarity_r2"]),
        },
        "passes": bool(no_context_holds and type_structure_kept_holds),
    }


def build(args: argparse.Namespace) -> int:
    import os

    import torch
    from foldspace.rfd4.train_reading_adapter import KEYS, collate
    from foldspace.rfd4.train_autoencoder import anchored_class, file_sha256
    from rfproteina.datapipes.schema.proteina import SPECS
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.model_masks import compute_ae_masks

    began = time.time()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    shards, adapter = (str(Path(p).resolve()) for p in (args.shards, args.adapter))
    os.chdir(out.parent)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    index = json.loads((Path(shards) / "index.json").read_text())
    ids = [r["id"] for r in index["train"]][: args.chains or None]
    state = torch.load(adapter, map_location="cpu", weights_only=False)
    if (
        "inplace_flag" not in state["latent"]
        or "code" in state["latent"]
        or "sigma_head.weight" in state["latent"]
    ):
        raise SystemExit(
            f"{adapter} is not a deterministic or fixed-width in-place ESMC code: a bank is made of that"
        )
    first = torch.load(Path(shards) / f"{ids[0]}.pt", weights_only=True)
    latent = anchored_class()(
        "esmc",
        ENC.n_tokens,
        residual=0,
        inplace=True,
        in_dim=int(first["esmc"].shape[-1]),
    )
    latent.load_state_dict(state["latent"])
    latent.to(device).eval()
    defaults = {k: SPECS[k][1] for k in KEYS if k in SPECS}
    mask_token = ENC.token_to_idx["<M>"]
    types, codes = [], []
    for start in range(0, len(ids), args.batch):
        items = [
            torch.load(Path(shards) / f"{i}.pt", weights_only=True)
            for i in ids[start : start + args.batch]
        ]
        batch = collate(items, defaults, mask_token, device)
        emit = compute_ae_masks(batch)[1] & batch["is_resolved_mask"].bool()
        with torch.no_grad():
            z = latent(
                batch,
                native=torch.zeros(*batch["residue_type"].shape, 32, device=device),
                sample=False,
            )
        types.append(batch["residue_type"][emit].cpu().numpy())
        codes.append(z[emit][:, latent.code_idx].float().cpu().numpy())
        if (start // args.batch) % 25 == 0:
            print(
                f"   {start + len(items)} of {len(ids)} chains, {time.time() - began:.0f} s",
                flush=True,
            )
    types, codes = np.concatenate(types), np.concatenate(codes)
    layout = bank_layout(types, codes, ENC.n_tokens + 1, args.min_count)
    np.savez(
        out,
        **layout,
        channels=np.array(latent.code_idx),
        chains=np.array(len(ids)),
        adapter_sha256=np.array(file_sha256(adapter)),
        seed=np.array(args.seed),
    )
    print(
        f"{len(ids)} chains, {len(types)} residues, {int((~layout['pooled']).sum())} of {len(layout['pooled'])} rows with donors of their own, "
        f"in {time.time() - began:.0f} s; written {out}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards")
    parser.add_argument("--adapter", help="the adapted autoencoder checkpoint whose code is the bank")
    parser.add_argument("--out")
    parser.add_argument(
        "--chains",
        type=int,
        default=1024,
        help="train chains read (the first of the pool's train set)",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--check",
        help="an audit report (`code_audit.py --bank`): print and return the verdict no_context and type_structure_kept",
    )
    parser.add_argument("--key", default="type_shuffled_hybrid")
    parser.add_argument("--reference", default="contextual_hybrid")
    args = parser.parse_args(argv)
    if args.check:
        result = verdict(
            json.loads(Path(args.check).read_text()), args.key, args.reference
        )
        print(json.dumps(result, indent=1))
        return 0 if result["passes"] else 1
    if not (args.shards and args.adapter and args.out):
        parser.error("--shards, --adapter and --out, or --check")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
