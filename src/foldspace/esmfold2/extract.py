"""Extract the mixed single of a table of sequences, as the folding model computes it.

The model computes the single inside its autocast region, in bfloat16; computed outside that region, in
float32 (``inject.mixed_single``), it differs by about 0.25 % in relative L2. The
definition of ``H(S)`` is the model's own (``interface.model_single``), and this is
the extraction that makes it: one sequence at a time (the single is a function of the whole
sequence through attention), at its full length, never a crop.

A bfloat16 value is exactly representable in float32, so ``z`` is stored as float32 without loss.
The shard layout is the one ``foldspace.esmfold2.target_audit`` reads. Refuses to run without a CUDA device: on a CPU there is no autocast region, and
the result would carry the label of a definition it does not follow.

    python -m foldspace.esmfold2.extract --sequences TABLE --shard i/n --out DIR/z_shard0.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

__all__ = ["write_shard"]


def write_shard(
    path: Path,
    ids: list[str],
    blocks: list[np.ndarray],
    failures: list[tuple[str, int, str]],
    *,
    checkpoint: str,
    provenance: dict,
) -> int:
    """One shard in ``esm_targets.py``'s layout; returns the number of rows."""
    if not blocks:
        raise ValueError("no sequence produced a target")
    if len(ids) != len(blocks):
        raise ValueError(f"{len(ids)} ids for {len(blocks)} blocks")
    lengths = np.array([b.shape[0] for b in blocks], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ids=np.array(ids),
        lengths=lengths,
        starts=np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64),
        z=np.concatenate([np.asarray(b, dtype=np.float32) for b in blocks], axis=0),
        failed_ids=np.array([f[0] for f in failures]),
        failed_lengths=np.array([f[1] for f in failures], dtype=np.int64),
        failed_reasons=np.array([f[2] for f in failures]),
        checkpoint=checkpoint,
        provenance=json.dumps(provenance),
    )
    return int(lengths.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default="ESMFold2")
    parser.add_argument(
        "--shard", default="", help="i/n, strided so shards finish together"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch

    from foldspace.esmfold2 import interface, sequences
    from foldspace.evaluation.provenance import capture, checkpoint_identity, require_clean
    from foldspace.esmfold2 import substrate as sub

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device: the model's single is computed in an autocast region that "
            "exists only on one"
        )
    rows = sequences.read_table(args.sequences)
    if args.shard.strip():
        i, _, n = args.shard.partition("/")
        i, n = int(i), int(n)
        if not 0 <= i < n:
            raise SystemExit(f"shard {args.shard!r}: need 0 <= i < n")
        rows = rows[i::n]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no sequences selected")
    require_clean(allow=args.allow_dirty)

    substrate = sub.open_substrate(args.checkpoint)
    identity = checkpoint_identity(substrate.weights or args.checkpoint)
    print(f"device {substrate.device}  {len(rows)} sequences", flush=True)

    ids: list[str] = []
    blocks: list[np.ndarray] = []
    failures: list[tuple[str, int, str]] = []
    began = time.perf_counter()
    for count, (seq_id, sequence) in enumerate(rows, 1):
        try:
            prepared = sub.prepare_sequences(substrate, sequence)
            with torch.no_grad():
                stack = sub.lm_hidden_states(substrate, prepared)
                z = interface.model_single(substrate, stack)
            latent = z[0].detach().to("cpu", dtype=torch.float32).numpy()
        except Exception as exc:  # noqa: BLE001 -- one bad sequence is not a run
            failures.append((seq_id, len(sequence), type(exc).__name__))
            print(f"  {seq_id}: FAILED {type(exc).__name__}: {exc}"[:160], flush=True)
            continue
        if latent.shape[0] != len(sequence):
            failures.append((seq_id, len(sequence), "TokenCountMismatch"))
            continue
        ids.append(seq_id)
        blocks.append(latent)
        if count % 100 == 0 or count == len(rows):
            rate = count / (time.perf_counter() - began)
            print(
                f"  [{count}/{len(rows)}] {rate:.1f} seq/s  {len(failures)} failed",
                flush=True,
            )

    if failures and args.require_all:
        listing = ", ".join(f"{i}({n} aa, {why})" for i, n, why in failures[:5])
        raise SystemExit(
            f"{len(failures)} of {len(rows)} sequences produced no target: {listing}"
        )
    rows_written = write_shard(
        args.out,
        ids,
        blocks,
        failures,
        checkpoint=args.checkpoint,
        provenance=capture(
            {
                "sequences": str(args.sequences),
                "shard": args.shard or "0/1",
                "checkpoint_name": args.checkpoint,
                "checkpoint": identity,
                "definition": "foldspace.esmfold2.interface.model_single (bfloat16 values, stored float32)",
                "allow_dirty": bool(args.allow_dirty),
            }
        ),
    )
    print(f"\n{args.out}  {len(ids)} targets  {rows_written} residues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
