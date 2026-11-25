"""The training shards: corpus chains read once through RFD4's pipeline into small files a training job can read from local disk
.

A shard holds, for one chain, the tensors the release encoder and decoder read (and nothing else: a trace of both on a chain shows 14
keys of the batch, and a batch cut to those gives the same latent and the same coordinates bit for bit), the residue's ESMC state from the
corpus array (`src/foldspace/rfd4/esmc_corpus_features.py`) and which residues have every legal atom. A chain enters only if it maps 1:1 onto the
corpus's residues (`src/foldspace/rfd4/ae_datapath.py`: the same count, the same standard residues, the backbone within a tolerance), the
pipeline gives no residue's identity as a sequence condition (a modified residue in RFD4's vocabulary would), and every token is
resolved: a chain that needs a note is left out and counted, so that the arms read the same standard chains.

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/prepare_shards.py \\
        --chains data/corpus/chains.npz --split-table data/corpus/split.tsv.gz \\
        --esmc cache/rfd4esmc/esmc_last.f16.npy --out cache/rfd4esmc/shards --train 2048 --dev 256
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: the keys of the collated batch the release encoder and decoder read, stored per chain without the batch dimension
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
)

_STATE: dict = {}


def choose(
    splits: np.ndarray,
    lengths: np.ndarray,
    ids: list[str],
    split: str,
    count: int,
    seed: int,
) -> list[str]:
    """`count` chains of one split, in a fixed random order; the caller takes the first that pass."""
    pool = [ids[i] for i in range(len(ids)) if str(splits[i]) == split]
    order = np.random.default_rng([seed, sum(map(ord, split))]).permutation(len(pool))
    return [pool[int(i)] for i in order[:count]]


def _init(
    chains: str,
    split_table: str,
    esmc: str,
    mirror: str,
    reuse: str = "",
    keep_esmc: bool = True,
) -> None:
    import torch

    torch.set_num_threads(1)
    _STATE.update(reuse=Path(reuse) if reuse else None, keep_esmc=keep_esmc)
    import foldspace.rfd4.ae_datapath as dp
    from atomworks.io import parse
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.partial_autoencoder.decoder import get_atom_mask
    from rfproteina.utils.parse_presets import RCSB_WITH_METALS

    z = np.load(chains, allow_pickle=False)
    ids = [str(i) for i in z["ids"]]
    rows = dp.split_rows(Path(split_table))
    _STATE.update(
        dp=dp,
        z={k: z[k] for k in ("xyz", "tokens", "starts", "lengths")},
        index_of={n: i for i, n in enumerate(ids)},
        row_of={f"{r['entry']}_{r['entity']}": r for r in rows},
        mirror=Path(mirror).expanduser(),
        parse=parse,
        preset=RCSB_WITH_METALS,
        pipeline=build_design_validation_pipeline(),
        collate=collate_batch,
        legal=get_atom_mask(),
        esmc=np.load(esmc, mmap_mode="r"),
    )


def _one(args: tuple[str, str, str]) -> dict:
    """Read one chain and write its shard; returns what became of it."""
    import torch

    name, split, out = args
    s = _STATE
    dp = s["dp"]
    record: dict = {"id": name, "split": split}
    t0 = time.time()
    earlier = s["reuse"] / f"{name}.pt" if s.get("reuse") else None
    if earlier is not None and earlier.is_file():
        # a chain an earlier run already read and kept: the same file (a hard link), the same record, no second read
        shard = torch.load(earlier, weights_only=False)
        target = Path(out) / f"{name}.pt"
        target.unlink(missing_ok=True)
        os.link(earlier, target)
        record["tokens"] = int(shard["residue_type"].shape[0])
        record["residues_complete"] = float(shard["complete"].float().mean())
        record["reused"] = True
        record["seconds"] = round(time.time() - t0, 2)
        return record
    try:
        batch, reference, _letters = dp.read_chain(
            name,
            s["z"],
            s["index_of"],
            s["row_of"],
            s["mirror"],
            s["parse"],
            s["preset"],
            s["pipeline"],
            s["collate"],
            "rfproteina.datapipes.schema.proteina.ProteinaDataSample",
            record,
        )
        n = int(batch["residue_type"].shape[1])
        i = s["index_of"][name]
        start, length = int(s["z"]["starts"][i]), int(s["z"]["lengths"][i])
        reasons = []
        if n != length:
            reasons.append("not one to one")
        if record.get("sequence_mismatches_standard", 1) != 0:
            reasons.append("sequence differs from the corpus")
        if record.get("nonstandard_residues", 0) != 0:
            reasons.append("a modified residue")
        if int((batch["condition_sequence_1_residue"][0] != _mask_token()).sum()) != 0:
            reasons.append("the pipeline gives an identity")
        if not bool(batch["is_resolved_mask"][0].all()):
            reasons.append("a token is unresolved")
        if not bool(batch["supports_local_latent"][0].all()):
            reasons.append("a token has no latent")
        if n == length:
            from foldspace.rfd4.datapath_stats import kabsch_rmsd

            rmsd = kabsch_rmsd(
                batch["coords"][0][:, [0, 1, 2]]
                .reshape(-1, 3)
                .numpy()
                .astype(np.float64),
                reference["xyz"].reshape(-1, 3),
            )
            record["backbone_rmsd_to_corpus"] = rmsd
            if rmsd >= dp.BACKBONE_TOLERANCE:
                reasons.append("backbone differs from the corpus")
        if reasons:
            record["left_out"] = reasons
            record["seconds"] = round(time.time() - t0, 2)
            return record
        required = s["legal"][batch["residue_type"][0]].clone()
        required[:, 36] = False
        complete = ~(required & ~batch["atom37_mask"][0].bool()).any(-1)
        shard = {k: batch[k][0].clone() for k in KEYS}
        shard["complete"] = complete
        if s["keep_esmc"]:
            shard["esmc"] = torch.from_numpy(
                np.array(s["esmc"][start : start + length], dtype=np.float16)
            )
        shard["id"] = name
        torch.save(shard, Path(out) / f"{name}.pt")
        record["tokens"] = n
        record["residues_complete"] = float(complete.float().mean())
    except Exception as error:  # noqa: BLE001 -- a chain that fails is a row
        record["left_out"] = [
            f"{type(error).__name__}: {str(error).splitlines()[0][:160]}"
        ]
    record["seconds"] = round(time.time() - t0, 2)
    return record


def _mask_token() -> int:
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    return ENC.token_to_idx["<M>"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chains", required=True)
    parser.add_argument("--split-table", required=True)
    parser.add_argument("--esmc", required=True)
    parser.add_argument(
        "--mirror", default="~/workspaces/databases/alphafold3/mmcif_files"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", type=int, default=2048)
    parser.add_argument("--dev", type=int, default=256)
    parser.add_argument(
        "--surplus",
        type=float,
        default=1.15,
        help="chains read per chain wanted (some are left out)",
    )
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reuse",
        default="",
        help="a directory of shards of an earlier run: a chain with a shard there is linked, not read again",
    )
    parser.add_argument(
        "--no-esmc",
        action="store_true",
        help="leave the ESMC rows out of the shards (a reader takes them from the corpus array, which is their source)",
    )
    args = parser.parse_args(argv)

    began = time.time()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    chains, table, esmc = (
        str(Path(p).resolve()) for p in (args.chains, args.split_table, args.esmc)
    )
    z = np.load(chains, allow_pickle=False)
    ids = [str(i) for i in z["ids"]]
    wanted = {"train": args.train, "dev": args.dev}
    tasks: list[tuple[str, str, str]] = []
    for split, count in wanted.items():
        tasks += [
            (n, split, str(out))
            for n in choose(
                z["splits"],
                z["lengths"],
                ids,
                split,
                int(count * args.surplus) + 8,
                args.seed,
            )
        ]
    print(f"{len(tasks)} chains to read with {args.workers} workers", flush=True)
    records: list[dict] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_init,
        initargs=(
            chains,
            table,
            esmc,
            args.mirror,
            str(Path(args.reuse).resolve()) if args.reuse else "",
            not args.no_esmc,
        ),
    ) as pool:
        for k, record in enumerate(pool.map(_one, tasks, chunksize=4)):
            records.append(record)
            if (k + 1) % 100 == 0:
                print(
                    f"{k + 1}/{len(tasks)} read, {sum('left_out' not in r for r in records)} kept, {time.time() - began:.0f} s",
                    flush=True,
                )

    kept: dict[str, list[dict]] = {"train": [], "dev": []}
    for r in records:
        if "left_out" not in r and len(kept[r["split"]]) < wanted[r["split"]]:
            kept[r["split"]].append(r)
    keep_ids = {r["id"] for rs in kept.values() for r in rs}
    for r in (
        records
    ):  # a shard beyond the count wanted is removed so that the directory is the set
        if "left_out" not in r and r["id"] not in keep_ids:
            (out / f"{r['id']}.pt").unlink(missing_ok=True)
    reasons: dict[str, int] = {}
    for r in records:
        for why in r.get("left_out", []):
            reasons[why] = reasons.get(why, 0) + 1
    index = {
        "label": "shards: chains of the corpus read through RFD4's pipeline",
        "seed": args.seed,
        "keys": list(KEYS) + ["complete", "id"] + ([] if args.no_esmc else ["esmc"]),
        "esmc_in_shards": (
            "none (the corpus array is the source)" if args.no_esmc else "all"
        ),
        "reused": sum(bool(r.get("reused")) for r in records),
        "train": [
            {"id": r["id"], "tokens": r["tokens"], "complete": r["residues_complete"]}
            for r in kept["train"]
        ],
        "dev": [
            {"id": r["id"], "tokens": r["tokens"], "complete": r["residues_complete"]}
            for r in kept["dev"]
        ],
        "read": len(records),
        "left_out": reasons,
        "seconds": round(time.time() - began, 1),
        "stack_commit": os.environ.get("FS_STACK_COMMIT"),
    }
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(
        json.dumps(
            {k: v for k, v in index.items() if k not in ("train", "dev")}, indent=1
        )
    )
    print(
        {s: len(v) for s, v in kept.items()},
        "tokens",
        {s: sum(r["tokens"] for r in v) for s, v in kept.items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
