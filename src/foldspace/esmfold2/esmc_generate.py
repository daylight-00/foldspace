"""ESMC-6B's sequences by iterative unmasking, on the policy ESM3's own sampler documents.

ESMC has no generation of its own: `esm/models/esmc/` returns logits and hidden states, and was trained with 15 % of the
tokens of a sequence masked at each step. The iterative decoding in `esm/utils/generation.py` is ESM3's. Its default is a
`GenerationConfig()`: 20 steps, a cosine schedule that says how many positions are still masked after each step, a random
choice of which masked positions to fix, and a temperature of 1.0 annealed to `(max(1 - step / (steps - 1), 0.001)) ** 2`;
BOS and EOS are never sampled. This applies that policy to ESMC's logits and changes one thing: the choice is restricted to
the twenty residues the evaluator reads. A forward is skipped on a step that fixes nothing (the official loop runs it and
discards it), and the random stream is numpy's, so nothing here is bit-for-bit the official sampler's draws. What is the same
is the arithmetic that decides how many positions are fixed when and at what temperature, and `parity` checks it against the
official functions before anything is loaded.

A result is a statement about this policy applied to a model that was not trained to be sampled from all masks, and it says
nothing of the entropy preset, which is not run.

    python src/foldspace/esmfold2/esmc_generate.py --seed 20261026 --out DIR [--limit N] [--budget-s S]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

#: the twenty residues in the order of `foldspace.esmfold2.vocabulary.STANDARD_AA`, which `input_ids_in_aa_order` follows
STANDARD = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class Policy:
    """`esm.sdk.api.GenerationConfig`'s fields that decide a sequence's decoding."""

    num_steps: int = 20
    schedule: str = "cosine"
    strategy: str = "random"
    temperature: float = 1.0
    annealed: bool = True


OFFICIAL = Policy()


def schedule_value(t: float, name: str) -> np.float32:
    """The official schedules (`esm/utils/noise_schedules.py`), in float32 as the official tensors are."""
    t = np.float32(t)
    if name == "cosine":
        return np.cos(t * np.float32(math.pi) * np.float32(0.5))
    if name == "linear":
        return np.float32(1.0) - t
    raise ValueError(f"schedule {name!r} is not one of cosine, linear")


def masked_after(length: int, step: int, policy: Policy = OFFICIAL) -> int:
    """How many of `length` positions are still masked after step `step` (from 0), as `int(schedule * total + 0.1)`."""
    share = schedule_value((step + 1) / policy.num_steps, policy.schedule)
    return int(np.float32(share) * np.float32(length) + np.float32(0.1))


def steps_for(length: int, policy: Policy = OFFICIAL) -> int:
    """The official sampler runs no more steps than there are positions to fill."""
    return min(policy.num_steps, length)


def effective(length: int, policy: Policy = OFFICIAL) -> Policy:
    """The policy with the steps it will run: no more than there are positions."""
    return replace(policy, num_steps=steps_for(length, policy))


def unmask_counts(length: int, policy: Policy = OFFICIAL) -> list[int]:
    """How many positions each step fixes, from all masked to none."""
    run = effective(length, policy)
    counts, still = [], length
    for step in range(run.num_steps):
        after = masked_after(length, step, run)
        counts.append(still - after)
        still = after
    return counts


def annealed_temperature(step: int, num_steps: int, initial: float) -> float:
    """`esm/utils/generation.py::_get_annealed_temperature`."""
    return max(initial - step / max(1, num_steps - 1), 0.001) ** 2


def decode(
    logits_of, length: int, policy: Policy, rng: np.random.Generator
) -> tuple[str, list[dict]]:
    """A sequence of `length` residues, and what each step did.

    `logits_of(state)` takes the current state, `[length]` of an index into `STANDARD` or -1 for a masked position, and returns
    `[length, 20]` logits for every position; only the masked ones are read."""
    if policy.strategy != "random":
        raise NotImplementedError(
            f"strategy {policy.strategy!r}: only the official default, random, is implemented"
        )
    steps = steps_for(length, policy)
    counts = unmask_counts(length, policy)
    state = np.full(length, -1, dtype=np.int64)
    trace = []
    for step, fixed in enumerate(counts):
        temperature = (
            annealed_temperature(step, steps, policy.temperature)
            if policy.annealed
            else policy.temperature
        )
        trace.append(
            {
                "step": step,
                "fixed": fixed,
                "temperature": temperature,
                "masked_before": int((state < 0).sum()),
            }
        )
        if fixed == 0:
            continue
        logits = np.asarray(logits_of(state), dtype=np.float64)
        masked = np.flatnonzero(state < 0)
        chosen = rng.permutation(masked)[:fixed]
        trace[-1]["positions"] = [int(i) for i in chosen]
        noise = -np.log(
            -np.log(np.clip(rng.random((len(chosen), len(STANDARD))), 1e-12, 1 - 1e-12))
        )
        state[chosen] = np.argmax(logits[chosen] / temperature + noise, axis=1)
    if (state < 0).any():
        raise RuntimeError("positions left masked after the last step")
    return "".join(STANDARD[i] for i in state), trace


def parity(lengths: list[int], policy: Policy = OFFICIAL) -> dict:
    """This file's arithmetic against the official functions, for every length: counts of positions fixed, and temperatures.

    Needs torch and `esm`. The counts are the official formula evaluated on the official schedule registry's functions, and the
    temperatures are `_get_annealed_temperature` itself."""
    import torch
    from esm.utils.generation import _get_annealed_temperature
    from esm.utils.noise_schedules import NOISE_SCHEDULE_REGISTRY

    schedule = NOISE_SCHEDULE_REGISTRY[policy.schedule]
    differing = []
    for length in lengths:
        steps, still, official = steps_for(length, policy), length, []
        for step in range(steps):
            share = schedule(torch.tensor((step + 1) / steps))
            after = int((share * torch.tensor(length) + 0.1).int())
            official.append(still - after)
            still = after
        if official != unmask_counts(length, policy):
            differing.append(length)
    steps = policy.num_steps
    temperatures = all(
        annealed_temperature(s, steps, policy.temperature)
        == _get_annealed_temperature(s, steps, policy.temperature)
        for s in range(steps)
    )
    return {
        "lengths": len(lengths),
        "differing_lengths": differing,
        "temperatures_equal": temperatures,
    }


def esmc_logits(substrate, head, length: int):
    """`logits_of` for ESMC-6B on this substrate: a chain of `length` residues, masked where the state says so."""
    from dataclasses import replace as replace_features

    import torch

    from foldspace.esmfold2 import vocabulary
    from foldspace.esmfold2.lm_head import sequence_logits
    from foldspace.esmfold2.sequence_directions import chain_slice
    from foldspace.esmfold2.substrate import prepare_sequences

    prepared = prepare_sequences(substrate, "A" * length)
    base = prepared.features["input_ids"].clone()
    tokens = chain_slice(prepared, 0).to(base.device)
    ids = torch.as_tensor(
        vocabulary.input_ids_in_aa_order(), dtype=base.dtype, device=base.device
    )
    mask = vocabulary.mask_token_id()

    def logits_of(state):
        filled = torch.as_tensor(state, device=base.device)
        values = torch.where(
            filled >= 0, ids[filled.clamp(min=0)], torch.full_like(filled, mask)
        )
        flat = base.clone().reshape(-1)
        flat[tokens] = values.to(flat.dtype)
        features = dict(prepared.features)
        features["input_ids"] = flat.reshape(base.shape)
        out = sequence_logits(
            substrate, replace_features(prepared, features=features), head
        )
        return out.reshape(-1, out.shape[-1])[tokens][:, ids].float().cpu().numpy()

    return logits_of


#: ubiquitin, whose masked residues ESMC should recover well; a residue-alphabet or mask error recovers almost none
UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
SELF_CHECK_MASKED = 12
SELF_CHECK_MUST_RECOVER = 5


def self_check(logits_of, seed: int = 0) -> dict:
    """Mask 12 positions of ubiquitin and count how many the model's top choice recovers.

    The glue between this file's alphabet and ESMC's (`foldspace.esmfold2.vocabulary`) is where a fold or a sampler fails
    silently, so it is checked on what ESMC is known to do before it is asked to generate."""
    length = len(UBIQUITIN)
    native = np.array([STANDARD.index(a) for a in UBIQUITIN])
    masked = np.sort(
        np.random.default_rng(seed).choice(
            length, size=SELF_CHECK_MASKED, replace=False
        )
    )
    state = native.copy()
    state[masked] = -1
    top = np.asarray(logits_of(state))[masked].argmax(axis=1)
    recovered = int((top == native[masked]).sum())
    return {
        "masked": SELF_CHECK_MASKED,
        "recovered": recovered,
        "ok": recovered >= SELF_CHECK_MUST_RECOVER,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        type=Path,
        default=Path("data/corpus/split.json"),
    )
    parser.add_argument("--checkpoint", default="ESMFold2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--budget-s",
        type=float,
        default=0.0,
        help="stop starting sequences after this many seconds; 0 = none",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    here = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(here / "src"))
    from foldspace.evaluation import streams
    from foldspace.evaluation.provenance import capture, require_clean

    require_clean(allow=args.allow_dirty)
    provenance = capture({"allow_dirty": bool(args.allow_dirty)})
    lengths = list(json.loads(args.lengths.read_text())["length_policy"]["lengths"])
    if args.limit:
        lengths = lengths[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    checked = parity(lengths)
    if checked["differing_lengths"] or not checked["temperatures_equal"]:
        print(
            f"the arithmetic differs from the official one: {checked}", file=sys.stderr
        )
        return 2
    print(f"parity with the official arithmetic: {checked}", flush=True)

    from foldspace.esmfold2.lm_head import load_lm_head
    from foldspace.esmfold2.substrate import open_substrate

    substrate = open_substrate(args.checkpoint)
    head = load_lm_head(substrate)
    opened = time.monotonic() - started
    print(f"substrate open on {substrate.device} after {opened:.0f} s", flush=True)
    checked_glue = self_check(esmc_logits(substrate, head, len(UBIQUITIN)))
    print(f"self-check on ubiquitin: {checked_glue}", flush=True)
    if not checked_glue["ok"]:
        (args.out / "report.json").write_text(
            json.dumps({"self_check": checked_glue}, indent=1) + "\n"
        )
        print(
            "ESMC does not recover masked residues of ubiquitin: the glue is wrong, nothing is generated",
            file=sys.stderr,
        )
        return 2

    records = []
    for index, length in enumerate(lengths):
        if args.budget_s and time.monotonic() - started > args.budget_s:
            print(f"budget reached before sequence {index}", flush=True)
            break
        began = time.monotonic()
        rng = np.random.Generator(
            np.random.PCG64(streams.digest("esmc-generate", args.seed, index))
        )
        sequence, trace = decode(
            esmc_logits(substrate, head, length), length, OFFICIAL, rng
        )
        records.append(
            {
                "id": f"e{args.seed}-{index:03d}-L{length}",
                "sequence": sequence,
                "length": length,
                "seconds": time.monotonic() - began,
                "forwards": sum(1 for t in trace if t["fixed"]),
            }
        )
        (args.out / "sequences.json").write_text(json.dumps(records))
        if (index + 1) % 16 == 0:
            print(
                f"  {index + 1}/{len(lengths)}  {records[-1]['seconds']:.1f} s",
                flush=True,
            )
    report = {
        "policy": OFFICIAL.__dict__,
        "restricted_to": "the twenty residues the evaluator reads",
        "seed": args.seed,
        "sequences": len(records),
        "asked": len(lengths),
        "parity": checked,
        "self_check": checked_glue,
        "seconds": {
            "open": opened,
            "total": time.monotonic() - started,
            "generating": sum(r["seconds"] for r in records),
        },
        "provenance": provenance,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"{len(records)} of {len(lengths)} sequences in {report['seconds']['total']:.0f} s"
    )
    return 0 if len(records) == len(lengths) else 1


if __name__ == "__main__":
    sys.exit(main())
