"""Donors for the information-free control: another protein's residues under this slot's mask.

A control that keeps a pathway and removes its information reads the feature of a frozen language model from a *different protein's* residues instead of the example's own. What "different" has to mean is fixed by the build's plan and by the corpus, not by the word.

**The plan repeats proteins.** A micro-batch of this trainer is one exact length filled to `min(max_batch, budget // L**2 + 1)` slots by repeating the proteins it was dealt, so at the recipe's defaults a micro-batch holds
3.6 distinct proteins in 19 slots, 13 % of the micro-batches hold one, and 95 % of the slots have another copy of their own protein in the batch. "The feature of another slot of the batch, none keeping its own" is then, for 31 % of the
slots by loss weight (40 % counting slots), the feature of another copy of the slot's own protein: a different mask over the same clean sequence shows the residues this copy has masked, which is the answer and not a control. A control must be keyed by the protein.

**The corpus's clusters do not separate homologues.** The 30 % clusters are a cover, and in this corpus 39,346 of the 695,416 pairs of equal-length chains of different clusters and different sequences share more than 30 % of their residues at the same positions, 31,706 of
them at least 95 % (point mutants and constructs of one protein in two clusters, up to 99.7 %). A donor that is a near copy of its target, read under the target's mask, gives the target's own feature. So a donor must differ in *sequence by a measure*, not only by cluster id.

**The donor of a protein** is a chain of another cluster, of another sequence, of the same exact length and with no more than `max_identity` of its residues equal to the target's at the same positions, the best over shifts of up to `max_shift` (so that a copy with a small insertion is
not missed; unrelated chains of one length agree at about 6 %). One is drawn for each distinct protein of a micro-batch, from `(seed, micro-batch, protein)`, and every copy of the protein has it, so that the copies of one protein remain copies of one protein and the micro-batch
has the same structure of repeats as the arm's. The exact length lets a donor's residue stand at the position it is read for and costs nothing: the language model reads a donor state of the target's own length, so the control's forward is the arm's in shape and in cost.
What the identity measure cannot see, a remote homologue below `max_identity` or one the shifts do not align, stays: the control is information-free at the level of the corpus's clusters and of this measure, and no further.

**The donor's state is the target's mask over the donor's residues:** where the target's corrupted sequence has a residue the donor's residue stands, where it has the mask token the mask token stands. The number and positions of the revealed residues are therefore exactly
the target's, its revealed fraction is the target's clock, nothing about the target's clean sequence enters (the state is a function of the target's *mask* and the donor's tokens, and the mask is drawn independently of the sequence), and the draw uses a stream of its
own, so the arms' plans, corruptions and noise are the same.

A chain with no eligible donor is an error and not a fallback to the nearest length or to another copy of itself (`NoDonor`).

Nothing here knows about torch.
"""

from __future__ import annotations

import numpy as np

from foldspace.evaluation import streams

PREFIX = "ustyle-donor"
#: the largest share of residues a donor may have in common with its target at the same positions
MAX_IDENTITY = 0.20
#: the shifts over which the best agreement is taken
MAX_SHIFT = 8


class NoDonor(ValueError):
    """A chain with no chain of another cluster, another sequence and no more than `max_identity` in common at its exact length."""


def shifted_identity(a, b, max_shift: int = MAX_SHIFT) -> float:
    """The largest share of equal residues of two sequences of one length, over the shifts of one against the other up to `max_shift` (on the overlap)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("same length")
    n = len(a)
    best = 0.0
    for shift in range(-max_shift, max_shift + 1):
        x, y = (
            (a[shift:], b[: n - shift]) if shift >= 0 else (a[: n + shift], b[-shift:])
        )
        if len(x):
            best = max(best, float(np.mean(x == y)))
    return best


class Pool:
    """The corpus's chains indexed by exact length, with the cluster, the sequence class and the tokens of each.

    `tokens[i]` is the residue tokens of chain `i`; `max_identity=None` switches the identity guard off (stand-ins that have no sequences), and nothing in the trainer does.
    """

    def __init__(
        self,
        lengths,
        clusters,
        sequences,
        tokens,
        *,
        max_identity: float | None = MAX_IDENTITY,
        max_shift: int = MAX_SHIFT,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.clusters = np.asarray(clusters, dtype=np.int64)
        self.sequences = np.asarray(sequences, dtype=np.int64)
        if not (
            len(self.lengths)
            == len(self.clusters)
            == len(self.sequences)
            == len(tokens)
        ):
            raise ValueError(
                "lengths, clusters, sequences and tokens must be one per chain"
            )
        if any(len(t) != n for t, n in zip(tokens, self.lengths, strict=True)):
            raise ValueError("a chain's tokens must be as long as its length")
        self.tokens = tokens
        self.max_identity = max_identity
        self.max_shift = int(max_shift)
        order = np.argsort(self.lengths, kind="stable")
        bounds = np.flatnonzero(np.diff(self.lengths[order])) + 1
        groups = np.split(order, bounds)
        self._at = {int(self.lengths[g[0]]): g for g in groups if len(g)}
        self._place = np.empty(len(self.lengths), dtype=np.int64)
        for g in self._at.values():
            self._place[g] = np.arange(len(g))
        self._alike: dict[int, np.ndarray] = {}

    def of_length(self, length: int) -> np.ndarray:
        """The chains of the corpus of one exact length (a design has no target to be unlike: its donor is one of these)."""
        return self._at.get(int(length), np.empty(0, dtype=np.int64))

    def alike(self, length: int) -> np.ndarray:
        """For the chains of one length: `[i, j]` is true where the two agree at more than `max_identity` of their positions, the best over the shifts."""
        if length not in self._alike:
            group = self._at[length]
            rows = np.stack([np.asarray(self.tokens[i]) for i in group])
            best = np.zeros((len(group), len(group)))
            for shift in range(-self.max_shift, self.max_shift + 1):
                x, y = (
                    (rows[:, shift:], rows[:, : length - shift])
                    if shift >= 0
                    else (rows[:, : length + shift], rows[:, -shift:])
                )
                if x.shape[1]:
                    same = (x[:, None, :] == y[None, :, :]).mean(axis=2)
                    best = np.maximum(best, same)
            self._alike[length] = best > self.max_identity
        return self._alike[length]

    def eligible(self, index: int) -> np.ndarray:
        """The chains of the corpus a chain may have as its donor."""
        length = int(self.lengths[index])
        same = self._at[length]
        keep = (self.clusters[same] != self.clusters[index]) & (
            self.sequences[same] != self.sequences[index]
        )
        if self.max_identity is not None:
            keep &= ~self.alike(length)[self._place[index]]
        return same[keep]

    def draw(self, index: int, *, seed: int, step: int) -> int:
        """The donor of a protein at a micro-batch: a function of `(seed, step, index)` and of nothing else."""
        candidates = self.eligible(index)
        if len(candidates) == 0:
            raise NoDonor(
                f"chain {index} (length {int(self.lengths[index])}) has no eligible donor at its exact length"
            )
        rng = np.random.default_rng(streams.digest(PREFIX, seed, step, str(int(index))))
        return int(candidates[rng.integers(len(candidates))])

    def micro_batch(self, indices, *, seed: int, step: int) -> np.ndarray:
        """The donor of every slot of a micro-batch (corpus indices, repeats included); the copies of a protein have one donor."""
        chosen: dict[int, int] = {}
        out = []
        for index in indices:
            index = int(index)
            if index not in chosen:
                chosen[index] = self.draw(index, seed=seed, step=step)
            out.append(chosen[index])
        return np.asarray(out, dtype=np.int64)

    def check(self, indices, donors) -> None:
        """Raise if any donor is not what `eligible` allows, or if two copies of a protein have two donors."""
        indices = np.asarray(indices, dtype=np.int64)
        donors = np.asarray(donors, dtype=np.int64)
        if indices.shape != donors.shape:
            raise ValueError("one donor per slot")
        seen: dict[int, int] = {}
        for index, donor in zip(indices.tolist(), donors.tolist(), strict=True):
            if self.lengths[donor] != self.lengths[index]:
                raise ValueError(f"donor {donor} of {index} is not of its length")
            if self.clusters[donor] == self.clusters[index]:
                raise ValueError(f"donor {donor} of {index} is in its cluster")
            if self.sequences[donor] == self.sequences[index]:
                raise ValueError(f"donor {donor} of {index} has its sequence")
            if (
                self.max_identity is not None
                and shifted_identity(
                    self.tokens[index], self.tokens[donor], self.max_shift
                )
                > self.max_identity
            ):
                raise ValueError(
                    f"donor {donor} of {index} shares too many of its residues"
                )
            if seen.setdefault(index, donor) != donor:
                raise ValueError(f"two copies of {index} have two donors")


def donor_state(target_tokens, donor_tokens, mask_token: int) -> np.ndarray:
    """The donor's state under the target's mask: the donor's residue where the target's sequence has one, the mask token where it has the mask token.

    A function of the target's *mask* and of the donor's tokens: what the target's residues are, where it has them, is not read.
    """
    target_tokens = np.asarray(target_tokens)
    donor_tokens = np.asarray(donor_tokens)
    if target_tokens.shape != donor_tokens.shape:
        raise ValueError("a donor's state is built position by position: same length")
    return np.where(target_tokens == mask_token, mask_token, donor_tokens)


def identity(a, b) -> float:
    """The share of positions at which two sequences of one length have the same residue (no alignment, no gaps, no shifts)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("same length")
    return float(np.mean(a == b))


def sequence_classes(tokens, starts, lengths) -> np.ndarray:
    """An integer for every chain, equal for equal sequences (the residue tokens compared as bytes): `Pool`'s `sequences`."""
    seen: dict[bytes, int] = {}
    out = np.empty(len(lengths), dtype=np.int64)
    for i, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        out[i] = seen.setdefault(
            np.ascontiguousarray(tokens[start : start + length]).tobytes(), len(seen)
        )
    return out
