"""Sequence-derived directions: the arithmetic, with numpy alone.

What matters here is the naming discipline and the mutation bookkeeping. The
direction itself is only as good as the variants that define it, and calling a
substitution a "data-manifold tangent" would overclaim: a substitution guarantees nothing about the manifold.
"""

from __future__ import annotations

import pytest

from foldspace.esmfold2 import sequence_directions as sequence

pytestmark = pytest.mark.offline


def test_a_load_becomes_a_count_and_never_zero():
    assert len(sequence.mutation_positions(100, 0.05, seed=0)) == 5
    assert len(sequence.mutation_positions(100, 0.10, seed=0)) == 10
    # A load too small to round up to one mutation still mutates one: a
    # direction of length zero is not a direction.
    assert len(sequence.mutation_positions(10, 0.001, seed=0)) == 1


def test_positions_are_distinct_sorted_and_in_range():
    picked = sequence.mutation_positions(50, 0.2, seed=7)
    assert picked == sorted(picked)
    assert len(picked) == len(set(picked))
    assert all(0 <= p < 50 for p in picked)


def test_the_same_seed_gives_the_same_positions():
    a = sequence.mutation_positions(80, 0.1, seed=3)
    b = sequence.mutation_positions(80, 0.1, seed=3)
    c = sequence.mutation_positions(80, 0.1, seed=4)
    assert a == b
    assert a != c


def test_avoided_positions_are_never_chosen():
    avoid = set(range(40))
    picked = sequence.mutation_positions(50, 0.2, seed=1, avoid=avoid)
    assert not (set(picked) & avoid)


def test_a_load_larger_than_the_chain_takes_every_position():
    assert sequence.mutation_positions(5, 2.0, seed=0) == [0, 1, 2, 3, 4]


def test_external_designs_are_labelled_as_a_separate_family():
    """The independent control has to be distinguishable in the results."""
    variants = sequence.variants_from_designs("A", ["MKT", "MKS"])
    assert {v.family for v in variants} == {"mpnn"}
    assert all(v.chain == "A" for v in variants)
    # Mutation load is unknown for a full redesign, and says so rather than
    # claiming zero.
    assert all(v.n_mutations == -1 for v in variants)


# --------------------------------------------------------------- substitution


def _scores(**by_letter):
    """A score vector aligned with STANDARD, default low."""
    values = [0.0] * len(sequence.STANDARD)
    for letter, value in by_letter.items():
        values[sequence.STANDARD.index(letter)] = value
    return values


def test_the_substitution_is_the_best_standard_residue_that_is_not_native():
    picked = sequence.pick_substitute(_scores(W=9.0, K=5.0), native="A")
    assert picked == "W"


def test_the_native_residue_can_never_be_proposed():
    """Otherwise a "variant" would be the wild type and Δz would be zero."""
    picked = sequence.pick_substitute(_scores(A=9.0, K=5.0), native="A")
    assert picked == "K"


def test_a_full_vocabulary_score_vector_is_refused():
    """The defect: ranking over the whole vocabulary ranks special tokens.

    The guard is a length check, because a full-vocabulary vector is the shape
    a caller gets by forgetting to gather the standard residues first.
    """
    with pytest.raises(ValueError, match="one score per standard residue"):
        sequence.pick_substitute([0.0] * 64, native="A")


def test_every_proposal_is_a_standard_residue():
    import random

    rng = random.Random(0)
    for _ in range(200):
        values = [rng.gauss(0, 1) for _ in sequence.STANDARD]
        native = rng.choice(sequence.STANDARD)
        picked = sequence.pick_substitute(values, native=native)
        assert picked in sequence.STANDARD
        assert picked != native


# ------------------------------------------------------------------- variants


def test_a_variant_refuses_a_non_standard_letter():
    with pytest.raises(ValueError, match="not standard residues"):
        sequence.SequenceVariant(
            chain="A", sequence="MKX", family="esmc", n_mutations=1, load=0.3
        )


def test_a_redesign_against_a_wild_type_knows_its_own_load():
    variants = sequence.variants_from_designs("B", ["MKTA", "MKSA"], wild_type="MKTA")
    assert variants[0].n_mutations == 0
    assert variants[0].positions == ()
    assert variants[1].n_mutations == 1
    assert variants[1].positions == (2,)
    assert variants[1].load == pytest.approx(0.25)


def test_an_indel_is_refused_rather_than_aligned():
    """There is no position-wise Δz between sequences of different length."""
    with pytest.raises(ValueError, match="not a substitution direction"):
        sequence.variants_from_designs("B", ["MKT"], wild_type="MKTA")
