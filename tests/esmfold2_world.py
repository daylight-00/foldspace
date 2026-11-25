"""A small synthetic world with the layout the sequence decoding reads: shards, sequence tables, clusters and a normalisation.

Each residue type has a random prototype in 256 dimensions; a row of a chain is its residue's prototype plus a
little context (the mean of its two neighbours' prototypes) plus noise, rounded to bfloat16 as the real singles
are. So the residue type can be read from a row (a linear decoder does well) and a little better with the
neighbours (a contextual one can do better still), and every file the loader expects exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foldspace.esmfold2.decode_eval import ALPHABET

D = 256


def _rows(
    sequence: str,
    prototypes: np.ndarray,
    rng: np.random.Generator,
    noise: float,
    context: float,
) -> np.ndarray:
    codes = np.array([ALPHABET.index(c) for c in sequence])
    own = prototypes[codes]
    left = np.vstack([own[:1], own[:-1]])
    right = np.vstack([own[1:], own[-1:]])
    z = own + context * 0.5 * (left + right) + rng.standard_normal(own.shape) * noise
    return z.astype(np.float32)


def _bf16(a: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bfloat16 and back: the real singles hold bfloat16 values."""
    bits = a.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32)


def build(
    root: Path,
    *,
    train_chains: int = 160,
    dev_chains: int = 60,
    test_chains: int = 40,
    noise: float = 0.6,
    context: float = 0.3,
    seed: int = 0,
    lengths: tuple[int, int] = (70, 140),
) -> dict:
    """Write ``root/cache/...``, a chains table and a train sequence table; return the paths."""
    rng = np.random.default_rng(seed)
    prototypes = rng.standard_normal((len(ALPHABET), D)).astype(np.float32) * 2.0
    cache = root / "cache"
    ef = cache / "esmfold2"
    (ef / "sequences").mkdir(parents=True)

    ids_all, clusters_all, splits_all = [], [], []
    tables: dict[str, list[tuple[str, str]]] = {}
    shard_rows: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    plan = {"train": train_chains, "dev": dev_chains, "test": test_chains}
    next_cluster = 0
    for split, n in plan.items():
        tables[split] = []
        shard_rows[split] = []
        for i in range(n):
            length = int(rng.integers(lengths[0], lengths[1] + 1))
            sequence = "".join(rng.choice(list(ALPHABET), size=length))
            chain = f"{split[:2]}{i:04d}_1"
            # one big family in dev so that micro and cluster-weighted accuracy differ
            cluster = 10_000 if (split == "dev" and i < 20) else next_cluster
            if not (split == "dev" and i < 20):
                next_cluster += 1
            ids_all.append(chain)
            clusters_all.append(cluster)
            splits_all.append(split)
            tables[split].append((chain, sequence))
            shard_rows[split].append(
                (
                    chain,
                    sequence,
                    _bf16(_rows(sequence, prototypes, rng, noise, context)),
                )
            )

    shards = {"train": 4, "dev": 1, "test": 1}
    for split, n_shards in shards.items():
        directory = ef / "targets" / split
        directory.mkdir(parents=True)
        rows = shard_rows[split]
        for s in range(n_shards):
            part = rows[s::n_shards]
            lengths_s = np.array([len(seq) for _, seq, _ in part], dtype=np.int64)
            np.savez(
                directory / f"z_shard{s}.npz",
                ids=np.array([c for c, _, _ in part]),
                lengths=lengths_s,
                starts=np.concatenate(([0], np.cumsum(lengths_s)[:-1])).astype(
                    np.int64
                ),
                z=np.concatenate([z for _, _, z in part]),
                failed_ids=np.array([]),
                failed_lengths=np.array([], dtype=np.int64),
                failed_reasons=np.array([]),
                checkpoint="ESMFold2",
                provenance=json.dumps({"checkpoint": {"weights_sha256": "synthetic"}}),
            )

    # sequence tables: train's is the corpus's, in shard-strided order like the real one
    def write_table(path: Path, rows: list[tuple[str, str]]) -> None:
        path.write_text("seq_id\tsequence\n" + "".join(f"{c}\t{s}\n" for c, s in rows))

    train_table = root / "train_sequences.tsv"
    write_table(train_table, tables["train"])
    write_table(ef / "sequences" / "dev.tsv", tables["dev"])
    write_table(ef / "sequences" / "test.tsv", tables["test"])

    chains = root / "chains.npz"
    np.savez(
        chains,
        ids=np.array(ids_all),
        clusters=np.array(clusters_all, dtype=np.int64),
        splits=np.array(splits_all),
    )

    all_train = np.concatenate([z for _, _, z in shard_rows["train"]])
    np.savez(
        ef / "model_single_train_normalisation.npz",
        mean=all_train.mean(axis=0).astype(np.float64),
        std=all_train.std(axis=0).astype(np.float64),
    )
    return {
        "cache": cache,
        "chains": chains,
        "train_table": train_table,
        "tables": tables,
    }
