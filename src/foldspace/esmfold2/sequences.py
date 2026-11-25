"""The sequence tables the single's extraction reads: ``seq_id <TAB> sequence``, one split each.

The corpus has one for its train split (``data/corpus/train_sequences.tsv``,
the table the cached singles were made from). The dev and test splits, which share no 30 %
cluster with train and are the held-out sets of the decoding experiment, had none, because nothing was extracted for
them. This writes one for any split from the corpus table, in the corpus's own order, so a shard
index (``i/n``, strided) names the same chains whoever reads it.

numpy only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foldspace.esmfold2 import validate_interface

__all__ = ["read_table", "write_table"]


def write_table(path: Path, chains: list[dict]) -> int:
    """``seq_id <TAB> sequence`` for each chain; returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    lines = ["seq_id\tsequence"]
    for chain in chains:
        if chain["id"] in seen:
            raise ValueError(f"{chain['id']} appears twice in one split")
        seen.add(chain["id"])
        lines.append(f"{chain['id']}\t{chain['sequence']}")
    path.write_text("\n".join(lines) + "\n")
    return len(chains)


def read_table(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        if line:
            seq_id, sequence = line.split("\t")
            rows.append((seq_id, sequence))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chains", type=Path, required=True, help="chains.npz")
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = write_table(args.out, validate_interface.corpus_chains(args.chains, args.split))
    print(f"{args.out}  {count} {args.split} chains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
