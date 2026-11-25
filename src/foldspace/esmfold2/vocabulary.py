"""One residue, two integer alphabets, and the conversion between them.

ESMFold2 reads every sequence **twice**, by two routes that do not share an
integer alphabet:

``res_type``
    The input path. Alphabetical by three-letter code -- ``ALA`` 2, ``ARG`` 3,
    ``ASN`` 4 ... ``VAL`` 21, with 22 for unknown and 23+ for nucleotides.

``input_ids``
    The ESMC path. Frequency-ordered, the usual ESM alphabet -- ``L`` 4, ``A``
    5, ``G`` 6 ... ``C`` 23.

**No index agrees between them.** Not one of the twenty. ``A`` is 2 in one and
5 in the other; ESMC's ``C`` is 23, which in ``res_type`` is RNA adenine, and
ESMC's ``W`` is 22, which is "unknown residue". A substitution chosen in one
alphabet and written into a tensor of the other is silently a *different
residue* -- the fold still runs, the numbers still look like numbers, and the
experiment measures a scrambled sequence.

Upstream keeps the two apart by deriving both from the residue rather than from
each other (``esm/models/esmfold2/protein_utils.py:398``)::

    res_type = PROTEIN_RESIDUE_TO_RES_TYPE.get(res_3,  PROTEIN_UNK_RES_TYPE)
    input_id = ESM_PROTEIN_VOCAB.get(letter, ESM_PROTEIN_VOCAB["X"])

This module does the same, and that is the whole discipline it enforces: **the
canonical one-letter sequence is the source of truth**, and each alphabet is
reached from it by its own encoder, never by casting one integer to the other.

Every table here is read from the upstream constants at import. A transcribed
copy would be a second source of truth, which is the defect this module exists
to prevent.
"""

from __future__ import annotations

from functools import lru_cache

__all__ = [
    "STANDARD_AA",
    "aa_of_input_id",
    "aa_of_res_type",
    "input_ids_in_aa_order",
    "input_ids_to_sequence",
    "mask_token_id",
    "res_type_to_sequence",
    "sequence_to_input_ids",
    "sequence_to_res_type",
    "standard_input_ids",
    "standard_res_types",
]

#: The twenty standard residues, in the one-letter alphabet this module treats
#: as canonical. Deliberately *not* an alphabet of either model.
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


@lru_cache(maxsize=1)
def _tables() -> tuple[dict[str, int], dict[str, int]]:
    """``(aa -> res_type, aa -> input_id)`` for the twenty standard residues.

    Both are built from the upstream constants, by the same route upstream
    uses: the three-letter code for ``res_type``, the one-letter for
    ``input_ids``.
    """
    from esm.models.esmfold2.constants import (
        ESM_PROTEIN_VOCAB,
        PROTEIN_1TO3,
        PROTEIN_RESIDUE_TO_RES_TYPE,
    )

    res_type: dict[str, int] = {}
    input_id: dict[str, int] = {}
    for aa in STANDARD_AA:
        three = PROTEIN_1TO3[aa]
        res_type[aa] = PROTEIN_RESIDUE_TO_RES_TYPE[three]
        input_id[aa] = ESM_PROTEIN_VOCAB[aa]
    return res_type, input_id


@lru_cache(maxsize=1)
def _inverse() -> tuple[dict[int, str], dict[int, str]]:
    res_type, input_id = _tables()
    return (
        {v: k for k, v in res_type.items()},
        {v: k for k, v in input_id.items()},
    )


def standard_res_types() -> tuple[int, ...]:
    """The ``res_type`` indices of the twenty standard residues, ascending."""
    return tuple(sorted(_tables()[0].values()))


def standard_input_ids() -> tuple[int, ...]:
    """The ESMC ``input_ids`` of the twenty standard residues, ascending.

    This is the set a masked-LM proposal must be restricted to. The head is
    free over the whole vocabulary, which includes BOS, EOS, pad, mask and the
    non-standard letters; an unrestricted ``topk`` can and does return one of
    them, and none of those is a residue to fold.
    """
    return tuple(sorted(_tables()[1].values()))


def input_ids_in_aa_order() -> tuple[int, ...]:
    """The ESMC ``input_ids`` of the standard residues, in ``STANDARD_AA`` order.

    :func:`standard_input_ids` is ascending by index, which is a different
    order again. A score vector gathered with this one is aligned with
    ``STANDARD_AA``, so the selection rule downstream can be plain arithmetic
    over letters instead of carrying either model's integers around.
    """
    table = _tables()[1]
    return tuple(table[aa] for aa in STANDARD_AA)


def sequence_to_res_type(sequence: str) -> list[int]:
    """Canonical sequence -> ``res_type`` indices."""
    table = _tables()[0]
    try:
        return [table[aa] for aa in sequence]
    except KeyError as exc:  # pragma: no cover - message is the point
        raise ValueError(
            f"{exc.args[0]!r} is not one of the twenty standard residues; "
            "this module refuses to guess an encoding for it"
        ) from exc


def sequence_to_input_ids(sequence: str) -> list[int]:
    """Canonical sequence -> ESMC ``input_ids``."""
    table = _tables()[1]
    try:
        return [table[aa] for aa in sequence]
    except KeyError as exc:  # pragma: no cover - message is the point
        raise ValueError(
            f"{exc.args[0]!r} is not one of the twenty standard residues; "
            "this module refuses to guess an encoding for it"
        ) from exc


def aa_of_res_type(index: int) -> str:
    """``res_type`` index -> one-letter residue, or ``X`` when not standard."""
    return _inverse()[0].get(int(index), "X")


def aa_of_input_id(index: int) -> str:
    """ESMC ``input_ids`` index -> one-letter residue, or ``X``."""
    return _inverse()[1].get(int(index), "X")


def res_type_to_sequence(indices) -> str:
    """``res_type`` indices -> canonical sequence, ``X`` for anything else."""
    return "".join(aa_of_res_type(int(i)) for i in indices)


def input_ids_to_sequence(indices) -> str:
    """ESMC ``input_ids`` -> canonical sequence, ``X`` for anything else."""
    return "".join(aa_of_input_id(int(i)) for i in indices)


def mask_token_id() -> int:
    """ESMC's mask token, read from the upstream signature that applies it.

    ``compute_lm_hidden_states`` takes it as ``mask_token_id: int = 32``. Read
    rather than copied, so that a change upstream moves this too instead of
    leaving a stale 32 behind that masks with a real residue.
    """
    import inspect

    from esm.models.esmfold2.layers import compute_lm_hidden_states

    default = (
        inspect.signature(compute_lm_hidden_states).parameters["mask_token_id"].default
    )
    if not isinstance(default, int):  # pragma: no cover - upstream contract
        raise TypeError(
            "compute_lm_hidden_states no longer defaults mask_token_id to an "
            f"int (got {default!r}); the mask token is unknown rather than 32"
        )
    return int(default)
