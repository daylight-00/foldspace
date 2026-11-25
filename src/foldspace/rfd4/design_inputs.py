"""The input contract of Foldspace's sequence design: which residues of a chain are given and which are to be generated (`design_mask`), and what is read of the result.

A design is a chain whose sequence is partly known. The input is a FASTA whose record names the residues to generate in either of two ways, or both:

    >ubiquitin design=28-48,60          the complete sequence, with spans (1-based, inclusive) of residues to generate: they are masked here, before any model reads it
    MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
    >novel
    MKVLA________GQDLIAVSPE            `_` is a residue to generate (ESMC's own mask character); no reference sequence exists

When a record gives the complete sequence, the residues under its spans are kept as a *reference* for what is read afterwards (how many of the generated residues are the native ones)
and nowhere else: `Design.masked` is the only sequence the rest of the path takes, and ESMC and RFD4's conditions are built from it. The contract is that **a residue to generate is
never given**: ESMC reads `[known residues, <mask> at the others]` (not the complete sequence with the masked positions' rows zeroed: every other position's hidden state would
still hold what was under the mask), and RFD4's `C_SEQ` and its 28 sequence-probability conditions name the known residues only (`src/foldspace/rfd4/flow_data.py`).

A record can name a structure of the chain too, `structure=PATH` or `structure=PATH:CHAIN` (a CIF, the path relative to the FASTA): the atoms of the *known* residues are then given to the
denoiser as coordinate conditions (`src/foldspace/rfd4/structure_condition.py`), and the structure at the residues to be generated is kept only to read the result against.

`sample_mask` draws the masks of the fine-tuning stream (one to three spans of a tenth to a half of the chain), and `metrics` / `group_diversity` read a generated sequence against its
design: the known residues kept, the residues generated, how many of them are the reference's, and how different the samples of one input are.

Numpy and the standard library only: the stages that read this run in two environments.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: the character of a residue to generate in a sequence (ESM's own mask character, `esm.utils.constants.MASK_STR_SHORT`)
MASK_CHAR = "_"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_spans(text: str, n: int) -> np.ndarray:
    """`[n]` bool, True where a span names the residue: `28-48,60` is residues 28 to 48 and 60 (1-based, inclusive). An empty text names none."""
    mask = np.zeros(n, dtype=bool)
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        low, dash, high = part.partition("-")
        try:
            first, last = int(low), int(high) if dash else int(low)
        except ValueError:
            raise ValueError(
                f"{part!r} is not a span: write 28-48 or a single residue 60"
            ) from None
        if not 1 <= first <= last <= n:
            raise ValueError(
                f"the span {part} is not inside 1-{n} (residues count from 1, last >= first)"
            )
        mask[first - 1 : last] = True
    return mask


def spans_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """The runs of True of a mask as 0-based half-open `(start, stop)` pairs."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def spans_text(mask: np.ndarray) -> str:
    """`28-48,60`: the mask's runs as `parse_spans` reads them."""
    return ",".join(
        f"{a + 1}-{b}" if b - a > 1 else f"{a + 1}" for a, b in spans_of(mask)
    )


def mask_from_spans(spans: list[tuple[int, int]], n: int) -> np.ndarray:
    """The mask of 0-based half-open spans (what `spans_of` returns)."""
    mask = np.zeros(n, dtype=bool)
    for a, b in spans:
        if not 0 <= a < b <= n:
            raise ValueError(f"the span {(a, b)} is not inside 0-{n}")
        mask[a:b] = True
    return mask


@dataclass(frozen=True)
class Design:
    """One chain of a design: `masked` is its sequence with `_` at every residue to generate, and nothing else of the sequence may be given to a model."""

    name: str
    masked: str
    designed: np.ndarray
    #: the letters under the masked positions when the record gave them (`_` where it did not), for what is read afterwards; no model sees it
    reference: str | None = None
    #: `PATH` or `PATH:CHAIN` of a structure of the complete chain whose known residues' atoms are the structure condition (None: the sequence alone)
    structure: str | None = None

    @property
    def n(self) -> int:
        return len(self.masked)

    @property
    def known(self) -> np.ndarray:
        return ~self.designed

    @property
    def spans(self) -> str:
        return spans_text(self.designed)


def make_design(
    name: str, sequence: str, spans: str = "", structure: str | None = None
) -> Design:
    """The design of one record: `sequence` in the 20 letters and `_`, and `spans` of residues to generate besides the `_`; `structure` names a structure of the chain (the condition)."""
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError(f"{name}: an empty sequence")
    bad = sorted(set(sequence) - set(AMINO_ACIDS) - {MASK_CHAR})
    if bad:
        raise ValueError(
            f"{name}: {bad} are not amino acids of the 20-letter alphabet (write _ for a residue to generate)"
        )
    underscore = np.array([c == MASK_CHAR for c in sequence])
    designed = underscore | parse_spans(spans, len(sequence))
    if not designed.any():
        raise ValueError(
            f"{name}: nothing to generate (give design=SPANS in the header, or _ in the sequence)"
        )
    masked = "".join(
        MASK_CHAR if d else c for c, d in zip(sequence, designed, strict=True)
    )
    reference = None
    under_spans = designed & ~underscore
    if under_spans.any():
        reference = "".join(
            c if u else MASK_CHAR for c, u in zip(sequence, under_spans, strict=True)
        )
    return Design(name, masked, designed, reference, structure or None)


def split_structure(value: str) -> tuple[str, str | None]:
    """`(path, chain)` of a `structure=` value: `PATH` or `PATH:CHAIN` (a chain id is one to four letters or digits after the last colon, with no dot or slash in it)."""
    head, colon, tail = value.rpartition(":")
    if colon and head and tail and len(tail) <= 4 and tail.isalnum():
        return head, tail
    return value, None


def resolve_structure(value: str, base: Path) -> str:
    """The `structure=` value with its path made absolute against `base` (the FASTA's directory) when it is relative; the chain is kept."""
    file, chain = split_structure(value)
    resolved = Path(file)
    if not resolved.is_absolute():
        resolved = base / resolved
    return str(resolved) + (f":{chain}" if chain else "")


def read_design_fasta(path) -> dict[str, Design]:
    """`{id: Design}` of a FASTA: an id is the first word of its header, `design=SPANS` and `structure=PATH[:CHAIN]` other words of it (a relative path is the FASTA's directory's), and an
    id used twice is an error."""
    records: dict[str, Design] = {}
    name, spans, structure, parts = None, "", None, []
    base = Path(path).resolve().parent

    def close() -> None:
        if name is not None:
            records[name] = make_design(name, "".join(parts), spans, structure)

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            close()
            words = line[1:].split()
            if not words:
                raise ValueError("a record without an id")
            name, parts = words[0], []
            if name in records:
                raise ValueError(f"the id {name} is used twice")
            spans, structure = "", None
            for word in words[1:]:
                if word.startswith("design="):
                    spans = word[len("design=") :]
                elif word.startswith("structure="):
                    structure = resolve_structure(word[len("structure=") :], base)
        else:
            if name is None:
                raise ValueError("a sequence before the first header")
            parts.append(line)
    close()
    return records


def tokens_for_esmc(design: Design, letters: str, mask_index: int) -> np.ndarray:
    """The tokens ESMC's glue reads for a design, in the alphabet `letters` (AlphaFold's, `foldspace.evaluation.residues.RESIDUES`) with `mask_index` for `_`: built from
    `design.masked`, so that no residue to generate can reach ESMC."""
    table = {c: i for i, c in enumerate(letters)}
    return np.array(
        [mask_index if c == MASK_CHAR else table[c] for c in design.masked],
        dtype=np.int64,
    )


def sample_mask(
    n: int,
    rng: np.random.Generator,
    *,
    low: float = 0.10,
    high: float = 0.50,
    min_span: int = 5,
    max_spans: int = 3,
) -> np.ndarray:
    """A random design mask of a chain of `n` residues: one to `max_spans` spans of at least `min_span` residues that are not adjacent, `low` to `high` of the chain in all.

    One span is the likeliest (0.6; two 0.3, three 0.1): the design of a loop, a helix or a domain's core is one span far more often than several. A function of `rng` and `n`."""
    if n < 2:
        raise ValueError("a chain of fewer than two residues has nothing to split")
    k = round(rng.uniform(low, high) * n)
    k = max(min(min_span, n - 1), min(k, n - 1))
    spans = int(rng.choice(np.arange(1, max_spans + 1), p=_span_odds(max_spans)))
    spans = max(1, min(spans, k // max(1, min_span), n - k + 1))
    extra = k - spans * min_span
    if extra < 0:  # a short chain: spans shorter than `min_span` rather than none
        lengths = np.full(spans, k // spans)
        lengths[: k - lengths.sum()] += 1
    else:
        cuts = np.sort(rng.integers(0, extra + 1, size=spans - 1))
        lengths = np.diff(np.concatenate([[0], cuts, [extra]])) + min_span
    known = n - k
    inner = spans - 1  # at least one known residue between two spans
    room = known - inner
    cuts = np.sort(rng.integers(0, room + 1, size=spans))
    gaps = np.diff(np.concatenate([[0], cuts, [room]]))
    gaps[1:-1] += 1
    mask = np.zeros(n, dtype=bool)
    position = int(gaps[0])
    for length, gap in zip(lengths, gaps[1:], strict=True):
        mask[position : position + int(length)] = True
        position += int(length) + int(gap)
    return mask


def _span_odds(max_spans: int) -> np.ndarray:
    weights = np.array([0.6, 0.3, 0.1] + [0.0] * max(0, max_spans - 3))[:max_spans]
    return weights / weights.sum()


def metrics(design: Design, generated: str) -> dict:
    """What a generated sequence is, read against its design: the known residues it kept (1.0 by construction, since the decode overwrites a given residue: a check, not a result), the
    residues generated, and, when the design has a reference, the share of them that are the reference's. `designed_max_share` is the largest share of one residue among the generated
    ones (a run of one residue is low complexity)."""
    if len(generated) != design.n:
        raise ValueError(
            f"{design.name}: a generated sequence of {len(generated)} residues for a design of {design.n}"
        )
    known = design.known
    kept = [generated[i] == design.masked[i] for i in np.flatnonzero(known)]
    made = [generated[i] for i in np.flatnonzero(design.designed)]
    out = {
        "designed": len(made),
        "known": int(known.sum()),
        "known_preserved": float(np.mean(kept)) if kept else None,
        "designed_distinct": len(set(made)),
        "designed_max_share": max(Counter(made).values()) / len(made),
        "designed_sequence": "".join(made),
    }
    if design.reference is not None:
        compared = [
            (generated[i], design.reference[i])
            for i in np.flatnonzero(design.designed)
            if design.reference[i] != MASK_CHAR
        ]
        out["designed_recovery"] = float(np.mean([a == b for a, b in compared]))
        out["designed_changed"] = 1.0 - out["designed_recovery"]
    return out


def group_diversity(design: Design, samples: list[str]) -> dict:
    """How different the samples of one input are, over the residues generated: the number of distinct generated spans, the mean pairwise identity of the samples there, and the mean
    per-position entropy (nats) of the generated residues across samples."""
    positions = np.flatnonzero(design.designed)
    spans = ["".join(s[i] for i in positions) for s in samples]
    pairs = [
        float(np.mean([a == b for a, b in zip(x, y, strict=True)]))
        for i, x in enumerate(spans)
        for y in spans[i + 1 :]
    ]
    entropies = []
    for column in zip(*spans, strict=True):
        counts = Counter(column)
        total = sum(counts.values())
        entropies.append(-sum(c / total * math.log(c / total) for c in counts.values()))
    return {
        "samples": len(samples),
        "distinct_designed_sequences": len(set(spans)),
        "mean_pairwise_identity": float(np.mean(pairs)) if pairs else None,
        "mean_position_entropy": float(np.mean(entropies)) if entropies else None,
    }
