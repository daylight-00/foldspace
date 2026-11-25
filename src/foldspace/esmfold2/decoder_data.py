"""The data of the sequence head: the singles of a split resident on the device, and the batches cut out of them.

Each array of each shard is read from disk **once**, at load; after that a batch is an index into a
tensor already on the card, so an epoch reads nothing and decompresses nothing. (An ``NpzFile``
re-reads and decodes the whole array on every ``held[key]``; indexing one inside a loop cost half a
terabyte once. Nothing here indexes a file inside a loop.)

The singles are stored bfloat16 on the device: the model computes them in bfloat16 and a float32 copy
holds exactly those values, so the rows are checked to be bfloat16-exact at load and no precision is
lost. A batch is standardised in float32 with the training split's mean and standard deviation
(``cache/esmfold2/model_single_train_normalisation.npz``) at the moment it is cut.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from foldspace.esmfold2 import target_audit as audit, sequences
from foldspace.esmfold2.decode_eval import ALPHABET

IGNORE = -100
D = 256

__all__ = ["IGNORE", "Normaliser", "Split", "gather", "load_split"]


@dataclass
class Normaliser:
    """Channel-wise ``(x - mean) / std`` of the training split, and its inverse."""

    mean: object
    std: object
    digest: str

    @classmethod
    def load(cls, path: Path, device):
        import torch

        held = np.load(path, allow_pickle=False)
        mean, std = held["mean"], held["std"]  # each array read once
        digest = hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()[:16]
        return cls(
            torch.tensor(mean, dtype=torch.float32, device=device),
            torch.tensor(std, dtype=torch.float32, device=device),
            digest,
        )

    def __call__(self, x):
        return (x.float() - self.mean) / self.std

    def inverse(self, y):
        return y * self.std + self.mean


@dataclass
class Split:
    """A split's chains, back to back: features and labels on the device, bookkeeping in numpy."""

    name: str
    ids: list[str]
    lengths: np.ndarray
    starts: np.ndarray
    clusters: np.ndarray
    shard: np.ndarray
    labels_np: np.ndarray
    z: object
    labels: object
    lengths_t: object
    starts_t: object

    @property
    def n_chains(self) -> int:
        return len(self.ids)

    @property
    def n_rows(self) -> int:
        return int(self.lengths.sum())

    def sequence(self, chain: int) -> str:
        s = int(self.starts[chain])
        codes = self.labels_np[s : s + int(self.lengths[chain])]
        return "".join(ALPHABET[int(c)] for c in codes)


def _letter_codes() -> np.ndarray:
    lookup = np.full(256, -1, dtype=np.int16)
    for i, letter in enumerate(ALPHABET):
        lookup[ord(letter)] = i
    return lookup


def load_split(
    name: str,
    *,
    cache: Path,
    chains: Path,
    train_table: Path,
    device,
    limit_chains: int = 0,
) -> Split:
    """One split: ``cache/esmfold2/targets/<name>/z_shard*.npz`` with its sequences and clusters.

    ``limit_chains`` keeps the first chains only, for a smoke on a login node; it still reads whole
    shards. Refuses a chain whose sequence and features disagree in length, a letter outside the
    twenty, and features that are not exactly representable in bfloat16.
    """
    import torch

    cache = Path(cache)
    table = (
        Path(train_table)
        if name == "train"
        else cache / "esmfold2" / "sequences" / f"{name}.tsv"
    )
    sequence_of = dict(sequences.read_table(table))
    held = np.load(chains, allow_pickle=False)
    ids_all, clusters_all = held["ids"], held["clusters"]  # each read once
    cluster_of = dict(zip(ids_all.tolist(), clusters_all.tolist(), strict=True))
    lookup = _letter_codes()

    ids: list[str] = []
    lengths: list[int] = []
    clusters: list[int] = []
    shard_of: list[int] = []
    label_parts: list[np.ndarray] = []
    z_parts = []
    kept = 0
    for s, path in enumerate(audit.shard_paths(cache / "esmfold2" / "targets" / name)):
        if limit_chains and kept >= limit_chains:
            break
        shard = audit.load_shard(path)
        take = (
            len(shard.ids)
            if not limit_chains
            else min(len(shard.ids), limit_chains - kept)
        )
        rows = int(shard.lengths[:take].sum())
        raw = torch.from_numpy(shard.z[:rows]).to(device)
        z = raw.to(torch.bfloat16)
        if not torch.equal(z.float(), raw):
            raise ValueError(
                f"{path.name}: the singles are not exactly representable in bfloat16"
            )
        del raw
        for i in range(take):
            chain = str(shard.ids[i])
            sequence = sequence_of[chain]
            if len(sequence) != int(shard.lengths[i]):
                raise ValueError(
                    f"{chain}: {len(sequence)} residues, {int(shard.lengths[i])} rows"
                )
            codes = lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
            if (codes < 0).any():
                raise ValueError(f"{chain}: a letter outside the twenty residue types")
            ids.append(chain)
            lengths.append(len(sequence))
            clusters.append(int(cluster_of[chain]))
            shard_of.append(s)
            label_parts.append(codes.astype(np.int8))
        z_parts.append(z)
        kept += take
    if not ids:
        raise ValueError(f"no chains under {cache / 'esmfold2' / 'targets' / name}")

    lengths_np = np.array(lengths, dtype=np.int64)
    starts_np = np.concatenate(([0], np.cumsum(lengths_np)[:-1])).astype(np.int64)
    labels_np = np.concatenate(label_parts)
    z_all = z_parts[0] if len(z_parts) == 1 else torch.cat(z_parts)
    return Split(
        name=name,
        ids=ids,
        lengths=lengths_np,
        starts=starts_np,
        clusters=np.array(clusters, dtype=np.int64),
        shard=np.array(shard_of, dtype=np.int64),
        labels_np=labels_np.astype(np.int64),
        z=z_all,
        labels=torch.from_numpy(labels_np.astype(np.int8)).to(device),
        lengths_t=torch.from_numpy(lengths_np).to(device),
        starts_t=torch.from_numpy(starts_np).to(device),
    )


def gather(split: Split, chain_indices: np.ndarray, norm: Normaliser):
    """``(x, y, lengths)`` for whole chains: ``x`` ``[B, Lmax, 256]`` float32 standardised, ``y`` ``[B, Lmax]``
    residue types with ``IGNORE`` on padding, ``lengths`` ``[B]``. Padding rows of ``x`` are zero."""
    import torch

    chain_indices = np.asarray(chain_indices, dtype=np.int64)
    device = split.z.device
    idx = torch.from_numpy(chain_indices).to(device)
    lengths = split.lengths_t[idx]
    starts = split.starts_t[idx]
    longest = int(split.lengths[chain_indices].max())
    position = torch.arange(longest, device=device)
    valid = position[None, :] < lengths[:, None]
    rows = torch.where(
        valid,
        starts[:, None] + position[None, :],
        torch.zeros_like(valid, dtype=torch.int64),
    )
    x = norm(split.z[rows]) * valid[..., None]
    y = torch.where(valid, split.labels[rows].long(), torch.full_like(rows, IGNORE))
    return x, y, lengths
