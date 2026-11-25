"""Directions the sequence supports, as against directions drawn at random.

A displacement of the mixed single along a direction that a sequence produced, ``Δz = z(S') − z(S)``, differs from one along
a random direction: in 256 dimensions a random direction is almost certainly far from anything the data supports, while a
trained generator moves along directions that it does. A variant sequence ``S'`` of a chain defines such a direction, a
*sequence-derived latent direction under fixed wild-type conditioning*: not a tangent of the data manifold, and not a mutant
trajectory, because a sequence enters ESMFold2 twice, as ``res_type`` on the input path and as an ESMC pair prior, and moving
the single alone moves one of the two routes.

Two families of variants:

``esmc``
    Substitutions that ESMC's own masked-LM head scores highly: the directions the model's prior supports, defined by the
    same model whose geometry is read, which is why the second family exists. The head is not loaded by ESMFold2; see
    :mod:`foldspace.esmfold2.lm_head`.

``mpnn``
    Fixed-backbone redesigns from an inverse-folding model, supplied through :func:`variants_from_designs`.

A variant is never filtered for similarity to the wild type: one that folds to something different is the point of a
displacement, and one that folds to something invalid is not a direction at all. Only validity is filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from foldspace.esmfold2 import vocabulary

__all__ = [
    "SequenceVariant",
    "chain_slice",
    "esmc_variants",
    "mutation_positions",
    "native_sequence",
    "pick_substitute",
    "variants_from_designs",
]

#: The twenty standard residues. Canonical one-letter, the source of truth for
#: both integer alphabets -- see :mod:`foldspace.esmfold2.vocabulary`.
STANDARD = vocabulary.STANDARD_AA


@dataclass(frozen=True)
class SequenceVariant:
    """One variant of one chain, as a sequence, and how it was made.

    ``sequence`` is a canonical one-letter string. It is deliberately not an
    array of either model's token ids: the two alphabets share no index, and
    carrying integers here is what let a substitution chosen in one be written
    into the other.
    """

    chain: str
    sequence: str
    family: str
    n_mutations: int
    load: float
    """Mutations as a fraction of chain length -- the stratum axis."""
    positions: tuple[int, ...] = ()
    """Offsets within the chain that differ from wild type."""

    def __post_init__(self) -> None:
        bad = sorted(set(self.sequence) - set(STANDARD))
        if bad:
            raise ValueError(
                f"variant sequence contains {bad}, which are not standard "
                "residues; a proposal that reached one of these is a bug in "
                "the proposer rather than something to fold"
            )


def mutation_positions(
    length: int, load: float, *, seed: int, avoid: set[int] | None = None
) -> list[int]:
    """Positions to mutate for a given load, drawn without replacement."""
    import random

    count = max(1, round(length * load))
    rng = random.Random(seed)
    choices = [i for i in range(length) if not avoid or i not in avoid]
    if count >= len(choices):
        return sorted(choices)
    return sorted(rng.sample(choices, count))


def chain_slice(prepared: Any, chain_index: int) -> Any:
    """Flat token indices of one chain's protein residues, in order."""
    import torch

    features = prepared.features
    asym = features["asym_id"].reshape(-1)
    order = sorted({int(v) for v in asym.tolist()})
    if chain_index >= len(order):
        raise ValueError(f"chain {chain_index} of {len(order)}")

    keep = asym == order[chain_index]
    mol_type = features.get("mol_type")
    if mol_type is not None:
        keep = keep & (mol_type.reshape(-1) == 0)
    return torch.nonzero(keep, as_tuple=False).reshape(-1)


def native_sequence(prepared: Any, positions: Any) -> str:
    """The wild-type sequence at ``positions``, as canonical one-letter.

    Decoded from **both** alphabets and required to agree. They are independent
    encodings of the same residue (``protein_utils.py:398``), so a disagreement
    means the features were edited through one route only -- the incoherent
    state this module exists to keep out of a fold.
    """
    features = prepared.features
    via_res_type = vocabulary.res_type_to_sequence(
        features["res_type"].reshape(-1)[positions].tolist()
    )
    via_input_ids = vocabulary.input_ids_to_sequence(
        features["input_ids"].reshape(-1)[positions].tolist()
    )
    if via_res_type != via_input_ids:
        raise ValueError(
            "res_type and input_ids decode to different sequences:\n"
            f"  res_type : {via_res_type}\n"
            f"  input_ids: {via_input_ids}\n"
            "the two routes have been set independently, which is exactly the "
            "corruption this check exists to catch"
        )
    if "X" in via_res_type:
        raise ValueError(
            "chain contains non-standard residues, which this probe does not "
            "mutate; a modified residue also spans several structure tokens "
            "per residue, so the chain's token layout would not be one "
            "residue per position either"
        )
    return via_res_type


def esmc_variants(
    substrate: Any,
    prepared: Any,
    chain_index: int,
    head: Any,
    *,
    loads: tuple[float, ...] = (0.02, 0.05, 0.10),
    per_load: int = 1,
    seed: int = 0,
    top_k: int = 3,
) -> list[SequenceVariant]:
    """Variants whose substitutions ESMC's own masked-LM head scores highly.

    The chosen positions are masked **together** and the head is read once, so
    each proposal is conditioned on the unmasked remainder of the chain. Only
    the twenty standard residues compete: the head ranges over the full
    vocabulary, which includes BOS, EOS, pad, mask and non-standard letters,
    and an unrestricted ``topk`` will return one of them.

    ``head`` comes from :func:`foldspace.esmfold2.lm_head.load_lm_head`. It is a
    required argument because ESMFold2 discards the head at load time, so
    reaching for ``substrate.esmc.lm_head`` finds ``None`` on every checkpoint
    -- and a family that quietly produces no variants at all is worse than one
    that fails.
    """
    import torch

    from foldspace.esmfold2.lm_head import masked_logits

    if head is None:
        raise ValueError(
            "no masked-LM head; load one with mlm.load_lm_head(substrate) "
            "rather than passing None, which would silently disable the "
            "esmc variant family"
        )

    tokens = chain_slice(prepared, chain_index)
    wild_type = native_sequence(prepared, tokens)
    length = len(wild_type)
    allowed = torch.tensor(vocabulary.input_ids_in_aa_order(), dtype=torch.long)

    asym = prepared.features["asym_id"].reshape(-1)
    chain = str(int(asym[tokens[0]]))

    variants: list[SequenceVariant] = []
    for load in loads:
        for replicate in range(per_load):
            chosen = mutation_positions(
                length, load, seed=seed + replicate * 97 + int(load * 1000)
            )
            logits = masked_logits(substrate, prepared, head, tokens[chosen])
            letters = list(wild_type)
            gathered = logits[:, allowed.to(logits.device)].float().cpu().numpy()
            for row, offset in zip(gathered, chosen):
                letters[offset] = pick_substitute(
                    row, native=wild_type[offset], top_k=top_k
                )
            sequence = "".join(letters)
            changed = tuple(
                i for i, (a, b) in enumerate(zip(sequence, wild_type)) if a != b
            )
            variants.append(
                SequenceVariant(
                    chain=chain,
                    sequence=sequence,
                    family="esmc",
                    n_mutations=len(changed),
                    load=len(changed) / max(length, 1),
                    positions=changed,
                )
            )
    return variants


def pick_substitute(scores: Any, *, native: str, top_k: int = 3) -> str:
    """The residue to substitute, given scores aligned with ``STANDARD``.

    Restricting to the twenty residues *before* the ranking is the point.
    Ranking the full vocabulary and filtering afterwards lets BOS, EOS, pad,
    mask and the non-standard letters consume the top of the list, and leaves
    nothing for a position whose context strongly predicts a real residue.

    Plain arithmetic over a 20-vector, so it is testable without the stack --
    which is the level `CLAUDE.md` asks probe arithmetic to sit at, and the
    level at which the alphabet confusion this module used to carry would have
    been caught.
    """
    import numpy as np

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.shape[0] != len(STANDARD):
        raise ValueError(
            f"expected one score per standard residue ({len(STANDARD)}), got "
            f"{values.shape[0]}; a score vector over the full vocabulary would "
            "silently rank special tokens as residues"
        )
    values = values.copy()
    values[STANDARD.index(native)] = -np.inf
    k = max(1, min(top_k, len(STANDARD) - 1))
    return STANDARD[int(np.argsort(-values)[:k][0])]


def variants_from_designs(
    chain: str,
    sequences: list[str],
    *,
    wild_type: str | None = None,
    family: str = "mpnn",
) -> list[SequenceVariant]:
    """Wrap sequences produced by an external designer as variants.

    The hook for the independent family: an inverse-folding model redesigns the
    binder backbone and its sequences come in here, so the direction owes
    nothing to ESMC.

    ``wild_type`` makes the mutation load measurable. Without it the load is
    unknown and says so, rather than claiming zero -- but a redesign whose load
    is unknown cannot be placed in a stratum, so pass it where it is available.
    """
    out: list[SequenceVariant] = []
    for sequence in sequences:
        if wild_type is None:
            out.append(
                SequenceVariant(
                    chain=chain,
                    sequence=sequence,
                    family=family,
                    n_mutations=-1,
                    load=float("nan"),
                )
            )
            continue
        if len(sequence) != len(wild_type):
            raise ValueError(
                f"redesign is {len(sequence)} residues against a wild type of "
                f"{len(wild_type)}; an indel is not a substitution direction "
                "and has no position-wise Δz"
            )
        changed = tuple(
            i for i, (a, b) in enumerate(zip(sequence, wild_type)) if a != b
        )
        out.append(
            SequenceVariant(
                chain=chain,
                sequence=sequence,
                family=family,
                n_mutations=len(changed),
                load=len(changed) / max(len(wild_type), 1),
                positions=changed,
            )
        )
    return out
