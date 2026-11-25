"""The two alphabets, and the reason they may never be confused.

ESMFold2 reads a sequence twice, through ``res_type`` and through ``input_ids``. What it did not
record -- and what `foldspace.esmfold2.sequence_directions` got wrong until this file
existed -- is that the two alphabets share **no** index.

The first test below is the one that matters: it fails loudly on any build
where the two orderings coincide by accident, and it documents the size of the
disagreement so that nobody reads "both are amino-acid vocabularies" as "both
are the same vocabulary".
"""

from __future__ import annotations

import pytest

from foldspace.esmfold2 import vocabulary as V

pytestmark = pytest.mark.weights


def test_the_two_alphabets_disagree_on_every_single_residue():
    """The defect this module prevents, stated as a measurement.

    Not "the orderings differ somewhere" -- all twenty differ. An index cast
    from one alphabet to the other is therefore always a different residue,
    never coincidentally the right one, which is why the bug produced numbers
    instead of a crash.
    """
    res_type, input_id = V._tables()
    agreeing = [aa for aa in V.STANDARD_AA if res_type[aa] == input_id[aa]]
    assert agreeing == [], (
        f"{agreeing} now share an index between res_type and input_ids; the "
        "conversion is still required but this test's premise has moved"
    )


def test_two_esmc_residues_land_outside_the_protein_range_entirely():
    """The failure is not even confined to the wrong amino acid.

    ESMC's ``C`` and ``W`` exceed the protein ``res_type`` range, so casting
    them writes "unknown residue" and a *nucleotide* into a protein chain.
    """
    _, input_id = V._tables()
    protein_max = max(V.standard_res_types())
    escaping = {aa for aa in V.STANDARD_AA if input_id[aa] > protein_max}
    assert escaping == {"C", "W"}


def test_round_trip_through_each_alphabet_separately():
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    assert V.res_type_to_sequence(V.sequence_to_res_type(sequence)) == sequence
    assert V.input_ids_to_sequence(V.sequence_to_input_ids(sequence)) == sequence


def test_cross_casting_is_what_corrupts_the_sequence():
    """Encode as ESMC, read as res_type: the corruption, made explicit."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    as_esmc = V.sequence_to_input_ids(sequence)
    misread = V.res_type_to_sequence(as_esmc)
    assert misread != sequence
    # Every position is wrong, and two are not residues at all.
    assert sum(a == b for a, b in zip(misread, sequence)) == 0
    assert misread.count("X") == 2


def test_the_standard_sets_have_twenty_distinct_entries_each():
    assert len(V.standard_res_types()) == 20
    assert len(set(V.standard_res_types())) == 20
    assert len(V.standard_input_ids()) == 20
    assert len(set(V.standard_input_ids())) == 20


def test_a_nonstandard_letter_is_refused_rather_than_guessed():
    """``X`` has an encoding upstream; this module still refuses it.

    A proposal that reached ``X`` is a bug in the proposer, and silently
    folding an unknown residue would hide it.
    """
    for bad in ("X", "B", "Z", "U", "-"):
        with pytest.raises(ValueError, match="standard residues"):
            V.sequence_to_res_type("MK" + bad)
        with pytest.raises(ValueError, match="standard residues"):
            V.sequence_to_input_ids("MK" + bad)


def test_the_mask_token_is_read_from_upstream_not_copied():
    """It is 32 today; the point is that it is *read*."""
    assert V.mask_token_id() == 32
    assert V.mask_token_id() not in V.standard_input_ids()


def test_input_ids_agree_with_the_tokenizer_the_cookbook_uses():
    """Two upstream tables, one alphabet -- confirm they have not diverged.

    ``ESM_PROTEIN_VOCAB`` builds the features; ``EsmcTokenizer`` is what
    ``esm/cookbook/tutorials/binder_design.py`` reads. If these ever disagree,
    a variant encoded through one and folded through the other reintroduces
    exactly the bug this file exists to prevent.
    """
    from esm.models.esmc import EsmcTokenizer

    vocab = EsmcTokenizer().vocab
    _, input_id = V._tables()
    for aa in V.STANDARD_AA:
        assert vocab[aa] == input_id[aa], aa
