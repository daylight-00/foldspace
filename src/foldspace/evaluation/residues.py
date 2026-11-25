"""The residue alphabet of the generator side: AlphaFold's order.

The amino-acid vocabularies of this repository share no index:

======================  ====================  =====================
where                   order                 absorbing state
======================  ====================  =====================
`esmfold2.vocabulary`   ESMFold2's two, which disagree with each other
here                    `ARNDCQEGHILKMFPSTWYV`  20 (`X`)
======================  ====================  =====================

`tests/test_vocabulary.py` exists because the first pair were once confused
and `foldspace.esmfold2.sequence_directions` was wrong until it did. This module is the
third, and the failure mode is identical and silent: a model trained on
permuted tokens converges, samples, and emits sequences that mean something
else. Nothing about the shapes objects.

The order here is AlphaFold's `residue_constants.restypes`, because the
generator's apparatus is MultiFlow-shaped and every sequence crossing that
boundary is indexed by it. `X` at 20 is the masking CTMC's absorbing state,
which is also upstream's `MASK_TOKEN_INDEX`. There is no `PAD`: upstream
carries a residue mask instead, and adding a padding token here would be a
third convention rather than a convenience.
"""

from __future__ import annotations

import numpy as np

#: AlphaFold's ordering, via `openfold.np.residue_constants.restypes`.
RESIDUES = "ARNDCQEGHILKMFPSTWYV"

#: The masking CTMC's absorbing state, and upstream's `MASK_TOKEN_INDEX`.
MASK = len(RESIDUES)

#: Rows an embedding table needs: the twenty plus the absorbing state.
VOCAB = len(RESIDUES) + 1

_INDEX = {letter: i for i, letter in enumerate(RESIDUES)}


def tokens(sequence: str) -> np.ndarray:
    """`[L]` of int64. Anything outside the twenty is a corpus bug, not a token."""
    try:
        return np.array([_INDEX[letter] for letter in sequence], dtype=np.int64)
    except KeyError as err:
        raise ValueError(
            f"{err.args[0]!r} is not one of {RESIDUES}; the corpus admits only "
            "standard residues, so this is a bug upstream of here"
        ) from None


def letters(encoded) -> str:
    """The inverse of `tokens`; the absorbing state prints as `X`."""
    return "".join(
        "X" if int(token) == MASK else RESIDUES[int(token)] for token in encoded
    )
