"""Stochastic streams, one per thing that is drawn.

Every random quantity gets its own seed, derived by a hash from ``(prefix, seed, step, example_id)``, so that **a change in
how many values one draw consumes cannot move another**. ``digest`` is that derivation. ``slot`` and ``plan`` apply it to one
example and one step of MultiFlow's corruption, which draws six things: the task, the structure clock, the sequence clock,
the translation noise, the rotation noise and the sequence mask. The clocks of the two modalities are separate, as in
MultiFlow (``codesign_separate_t``), and the translation and rotation streams are separate too, so that turning batch
optimal transport off cannot repaint every rotation. A stream is keyed by the example and not by its position in the
batch, so the plan does not depend on batch order. The optimal-transport assignment couples the examples of a batch and is
computed once over the whole batch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

_WORD = 8

#: Default prefix of the streams. A label, kept as it is because it enters every seed derived from it.
DOMAIN = "campaign"

#: Every stream, in the order upstream draws them. Named here so a reader can
#: see the whole set at once and a test can assert none was forgotten.
STREAMS = (
    "task",
    "structure_time",
    "sequence_time",
    "translation_noise",
    "rotation_noise",
    "sequence_corruption",
)


def digest(prefix: str, seed: int, step: int, example_id: str = "") -> int:
    """A stream's seed, from a prefix and the coordinates of one draw."""
    key = f"{prefix}|{seed}|{step}|{example_id}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=_WORD).digest(), "big")


@dataclass(frozen=True, slots=True)
class Slot:
    """One example's place in one step, and its six independent streams."""

    example_id: str
    length: int
    start: int
    crop: int
    task_seed: int
    structure_time_seed: int
    sequence_time_seed: int
    translation_noise_seed: int
    rotation_noise_seed: int
    sequence_corruption_seed: int

    @property
    def stop(self) -> int:
        return self.start + self.crop

    def seeds(self) -> dict[str, int]:
        return {name: getattr(self, f"{name}_seed") for name in STREAMS}


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """One step: which examples, cropped where, and the forward's own streams.

    Two model streams and not one, for the reason every other split here
    exists: upstream tosses a Python coin to decide self-conditioning and
    draws transformer dropout from torch, and under one stream a change in how
    much dropout consumes would move the coin.
    """

    step: int
    slots: tuple[Slot, ...]
    self_condition_seed: int
    dropout_seed: int

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(s.example_id for s in self.slots)

    def fingerprint(self) -> str:
        """Everything the plan decided, as one string.

        Two arms that log the same fingerprint at the same step drew the same
        crops from the same examples under the same six streams. It is the
        cheap half of the paired check; the expensive half compares the
        realised tensors.
        """
        parts = [str(self.step), str(self.self_condition_seed), str(self.dropout_seed)]
        for slot in self.slots:
            parts.append(
                "/".join(
                    [
                        slot.example_id,
                        str(slot.length),
                        str(slot.start),
                        str(slot.crop),
                        *(str(v) for v in slot.seeds().values()),
                    ]
                )
            )
        return hashlib.blake2b("|".join(parts).encode(), digest_size=16).hexdigest()


def slot(
    example_id: str,
    length: int,
    *,
    seed: int,
    step: int,
    crop: int,
    domain: str = DOMAIN,
) -> Slot:
    """One slot, every stream under its own prefix.

    Keyed by the example rather than by its position in the batch, so the plan
    is invariant to batch order: someone will eventually sort a batch by length
    to pack it better, and under position keying that repaints every example's
    corruption while the arms still look matched in aggregate.
    """
    span = min(crop, length)
    room = length - span
    start = digest(f"{domain}-crop", seed, step, example_id) % (room + 1) if room else 0
    return Slot(
        example_id=example_id,
        length=length,
        start=start,
        crop=span,
        **{
            f"{name}_seed": digest(f"{domain}-{name}", seed, step, example_id)
            for name in STREAMS
        },
    )


def plan(
    step: int, *, examples, seed: int, crop: int, domain: str = DOMAIN
) -> BatchPlan:
    """The step's plan for `[(example_id, length)]`, in the order given."""
    return BatchPlan(
        step=step,
        slots=tuple(
            slot(name, length, seed=seed, step=step, crop=crop, domain=domain)
            for name, length in examples
        ),
        # Per step rather than per example: both are drawn per forward pass,
        # and the trainer reseeds from them immediately before each arm's
        # forward so the arms share the coin and the dropout alike.
        self_condition_seed=digest(f"{domain}-self-condition", seed, step),
        dropout_seed=digest(f"{domain}-dropout", seed, step),
    )
