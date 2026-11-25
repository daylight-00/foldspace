"""The ESMC iterative-unmasking sequences at the pilot's lengths: the positive reference of the sequence-only comparison.

ESMC is pretrained and is matched in neither the data nor the size of the generators it stands beside. What it shows is whether the
ruler (``foldspace.esmfold2.seqeval``) can be passed by a generated set at all: if a pretrained 6 B model's sequences do not pass the absolute
rule, no generator is called supported by it.

The policy and the arithmetic are ``src/foldspace/esmfold2/esmc_generate.py``'s (20 steps, a cosine schedule, a random order of the positions,
a temperature annealed from 1.0, the choice restricted to the twenty residues the evaluator reads); this takes the lengths from the
pilot's samples and writes the set in the format of every other: ``samples.tsv`` with an undefined confidence.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from foldspace.esmfold2 import latent_generation as latent_generation, sequence_only

__all__ = ["main"]

SEED = 20261120
BUILD = Path(__file__).resolve().parents[3] / "src" / "foldspace" / "rfd4"
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--chains", type=Path, default=Path("data/corpus/chains.npz")
    )
    parser.add_argument("--checkpoint", default="ESMFold2")
    parser.add_argument("--n", type=int, default=latent_generation.N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="unmasking steps (build's official policy: 20); more than a chain's length is one position per step",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="the initial temperature"
    )
    parser.add_argument(
        "--no-anneal",
        action="store_true",
        help="hold the temperature instead of annealing it toward zero (the official policy anneals)",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    started = time.monotonic()
    sys.path.insert(0, str(BUILD))
    import foldspace.esmfold2.esmc_generate as eg  # build/'s policy, imported and not copied

    from foldspace.evaluation import streams
    from foldspace.esmfold2.lm_head import load_lm_head
    from foldspace.evaluation.provenance import capture, require_clean
    from foldspace.esmfold2.substrate import open_substrate

    require_clean(allow=args.allow_dirty)
    lengths = latent_generation.sample_lengths(
        sequence_only.dev_lengths(args.chains), args.n, *latent_generation.SAMPLE_LENGTHS
    )
    changes = {
        key: value
        for key, value in (
            ("num_steps", args.steps),
            ("temperature", args.temperature),
            ("annealed", False if args.no_anneal else None),
        )
        if value is not None
    }
    policy = replace(eg.OFFICIAL, **changes)
    parity = eg.parity(sorted({int(x) for x in lengths}), policy)
    if parity["differing_lengths"] or not parity["temperatures_equal"]:
        print(
            f"the arithmetic differs from the official one: {parity}", file=sys.stderr
        )
        return 2
    substrate = open_substrate(args.checkpoint)
    head = load_lm_head(substrate)
    glue = eg.self_check(eg.esmc_logits(substrate, head, len(eg.UBIQUITIN)))
    print(f"parity {parity}  self-check on ubiquitin {glue}", flush=True)
    if not glue["ok"]:
        print(
            "ESMC does not recover masked residues of ubiquitin: nothing is generated",
            file=sys.stderr,
        )
        return 2

    sequences = []
    for index, length in enumerate(lengths):
        rng = np.random.Generator(
            np.random.PCG64(streams.digest("esmc-generate", args.seed, index))
        )
        sequence, _ = eg.decode(
            eg.esmc_logits(substrate, head, int(length)), int(length), policy, rng
        )
        sequences.append(sequence)
        if (index + 1) % 8 == 0:
            print(
                f"  {index + 1}/{len(lengths)}  {time.monotonic() - started:.0f}s",
                flush=True,
            )

    sequence_only.write_samples(
        args.out,
        sequences,
        [float("nan")] * len(sequences),
        {
            "n": len(sequences),
            "lengths": lengths.tolist(),
            "generator": "ESMC iterative unmasking (src/foldspace/esmfold2/esmc_generate.py)",
            "policy": policy.__dict__,
            "official_policy": eg.OFFICIAL.__dict__,
            "restricted_to": "the twenty residues the evaluator reads",
            "seed": args.seed,
            "parity": parity,
            "self_check": glue,
            "provenance": capture({"allow_dirty": bool(args.allow_dirty)}),
            "seconds": time.monotonic() - started,
        },
    )
    print(f"wrote {args.out / 'samples.tsv'}", flush=True)
    if len(latent_generation.read_samples(args.out)) != len(
        sequences
    ):  # the file reads back in the format of the others
        raise SystemExit("the samples table does not read back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
