"""The input contract of Foldspace's sequence design (`src/foldspace/rfd4/design_inputs.py`): what is masked, that nothing under the mask is given to a model, and how a result is read."""

from __future__ import annotations

import foldspace.rfd4.design_inputs as fdz
import numpy as np
import pytest

pytestmark = pytest.mark.offline

SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def test_spans_are_one_based_and_inclusive() -> None:
    mask = fdz.parse_spans("3-5,9", 10)
    assert np.flatnonzero(mask).tolist() == [2, 3, 4, 8]
    assert fdz.spans_text(mask) == "3-5,9"
    assert fdz.spans_of(mask) == [(2, 5), (8, 9)]
    assert np.array_equal(fdz.mask_from_spans(fdz.spans_of(mask), 10), mask)
    assert not fdz.parse_spans("", 10).any()


@pytest.mark.parametrize("text", ["0-3", "5-4", "8-11", "a", "3-"])
def test_a_span_outside_the_chain_or_unreadable_is_refused(text) -> None:
    with pytest.raises(ValueError):
        fdz.parse_spans(text, 10)


def test_a_record_with_spans_is_masked_and_keeps_its_reference() -> None:
    design = fdz.make_design("u", SEQ, "28-48")
    assert design.n == len(SEQ) and design.spans == "28-48"
    assert design.masked[27:48] == "_" * 21
    assert design.masked[:27] == SEQ[:27] and design.masked[48:] == SEQ[48:]
    assert design.reference is not None
    assert design.reference[27:48] == SEQ[27:48] and set(design.reference[:27]) == {"_"}
    assert design.known.sum() == len(SEQ) - 21


def test_a_sequence_with_underscores_has_no_reference() -> None:
    design = fdz.make_design("n", "MKV___LLAG")
    assert design.designed.tolist() == [c == "_" for c in "MKV___LLAG"]
    assert design.reference is None and design.masked == "MKV___LLAG"
    both = fdz.make_design("b", "MKV___LLAG", "8-9")
    assert both.masked == "MKV___L__G" and both.reference == "_______LA_"


def test_a_record_must_ask_for_something_to_generate_and_use_the_alphabet() -> None:
    with pytest.raises(ValueError, match="nothing to generate"):
        fdz.make_design("x", SEQ)
    with pytest.raises(ValueError, match="alphabet"):
        fdz.make_design("x", "MKVXLA", "2")
    with pytest.raises(ValueError, match="alphabet"):
        fdz.make_design("x", "MKV-LA", "2")


def test_nothing_under_the_mask_reaches_a_model() -> None:
    """The contract: two records that differ only at the residues to generate give a model the same input, whatever the reference was."""
    other = SEQ[:27] + "W" * 21 + SEQ[48:]
    a = fdz.make_design("a", SEQ, "28-48")
    b = fdz.make_design("b", other, "28-48")
    assert a.masked == b.masked
    letters = "ARNDCQEGHILKMFPSTWYV"  # AlphaFold's order; any order does the same
    tokens_a = fdz.tokens_for_esmc(a, letters, mask_index=20)
    tokens_b = fdz.tokens_for_esmc(b, letters, mask_index=20)
    assert np.array_equal(tokens_a, tokens_b)
    assert (tokens_a[27:48] == 20).all() and (tokens_a[:27] != 20).all()
    # the residues of the span are not in the tokens at all
    assert [letters[t] for t in tokens_a[:27]] == list(SEQ[:27])


def test_the_fasta_is_read_by_id_with_its_spans_and_a_repeated_id_is_refused(
    tmp_path,
) -> None:
    path = tmp_path / "d.fasta"
    path.write_text(
        f">u design=28-48,60 free text\n{SEQ[:40]}\n{SEQ[40:]}\n\n>n\nMKV__LLAG\n"
    )
    records = fdz.read_design_fasta(path)
    assert list(records) == ["u", "n"]
    assert records["u"].spans == "28-48,60" and records["u"].n == len(SEQ)
    assert records["n"].spans == "4-5"
    path.write_text(">a design=1\nMKV\n>a design=2\nMKV\n")
    with pytest.raises(ValueError, match="used twice"):
        fdz.read_design_fasta(path)
    path.write_text("MKV\n")
    with pytest.raises(ValueError, match="before the first header"):
        fdz.read_design_fasta(path)


def test_a_random_mask_is_a_function_of_its_stream_and_has_the_shape_of_a_design() -> (
    None
):
    seen_spans = []
    for n in (12, 40, 76, 129, 300):
        for seed in range(60):
            mask = fdz.sample_mask(n, np.random.default_rng([seed, n]))
            again = fdz.sample_mask(n, np.random.default_rng([seed, n]))
            assert np.array_equal(mask, again) and mask.shape == (n,)
            spans = fdz.spans_of(mask)
            seen_spans.append(len(spans))
            assert 1 <= len(spans) <= 3
            share = mask.sum() / n
            assert 0.05 <= share <= 0.55, (n, seed, share)
            assert mask.sum() < n  # a design keeps at least one residue
            assert all(b - a >= min(5, n - 1) or n < 20 for a, b in spans), spans
            # the spans are not adjacent: they are as many runs as the mask says
            assert len(spans) == len(fdz.spans_of(mask))
    assert set(seen_spans) == {1, 2, 3}
    assert seen_spans.count(1) > seen_spans.count(3)


def test_a_chain_of_two_residues_still_has_a_mask() -> None:
    mask = fdz.sample_mask(2, np.random.default_rng(1))
    assert mask.sum() == 1


def test_the_reading_of_a_generated_sequence() -> None:
    design = fdz.make_design("u", SEQ, "28-48")
    kept = fdz.metrics(design, SEQ)
    assert kept["known_preserved"] == 1.0 and kept["designed_recovery"] == 1.0
    assert kept["designed_changed"] == 0.0 and kept["designed"] == 21
    # a design that changes every residue under the mask and one known residue
    generated = list(SEQ)
    for i in range(27, 48):
        generated[i] = "G" if SEQ[i] != "G" else "A"
    generated[0] = "A"
    read = fdz.metrics(design, "".join(generated))
    assert read["designed_changed"] == 1.0
    assert read["known_preserved"] == pytest.approx(1 - 1 / 55)
    assert read["designed_max_share"] >= 0.9 and read["designed_distinct"] <= 2
    with pytest.raises(ValueError, match="generated sequence of"):
        fdz.metrics(design, SEQ[:-1])


def test_a_design_without_a_reference_has_no_recovery() -> None:
    design = fdz.make_design("n", "MKV___LLAG")
    read = fdz.metrics(design, "MKVAAALLAG")
    assert "designed_recovery" not in read and read["known_preserved"] == 1.0


def test_the_diversity_of_the_samples_of_one_input() -> None:
    design = fdz.make_design("n", "MKV___LLAG")
    same = fdz.group_diversity(design, ["MKVAAALLAG", "MKVAAALLAG"])
    assert same["distinct_designed_sequences"] == 1
    assert (
        same["mean_pairwise_identity"] == 1.0 and same["mean_position_entropy"] == 0.0
    )
    apart = fdz.group_diversity(design, ["MKVAAALLAG", "MKVCCCLLAG", "MKVDDDLLAG"])
    assert apart["distinct_designed_sequences"] == 3
    assert apart["mean_pairwise_identity"] == 0.0
    assert apart["mean_position_entropy"] > 1.0
    alone = fdz.group_diversity(design, ["MKVAAALLAG"])
    assert alone["mean_pairwise_identity"] is None


def test_a_structure_value_is_a_path_and_an_optional_chain() -> None:
    assert fdz.split_structure("a.cif") == ("a.cif", None)
    assert fdz.split_structure("a.cif:A") == ("a.cif", "A")
    assert fdz.split_structure("/x/y.cif:AB12") == ("/x/y.cif", "AB12")
    # a colon inside a path is not a chain: what follows it has a slash or a dot, or is too long
    assert fdz.split_structure("dir:name/x.cif") == ("dir:name/x.cif", None)
    assert fdz.split_structure("a:b.cif") == ("a:b.cif", None)
    assert fdz.split_structure("a.cif:LONGCHAIN") == ("a.cif:LONGCHAIN", None)


def test_a_record_can_name_a_structure_and_a_relative_path_is_the_fastas(
    tmp_path,
) -> None:
    path = tmp_path / "d.fasta"
    path.write_text(
        f">u design=28-48 structure=models/1ubq.cif:A\n{SEQ}\n>v structure=/abs/x.cif design=3\nMKVLAAG\n>w design=2\nMKV\n"
    )
    records = fdz.read_design_fasta(path)
    assert records["u"].structure == f"{tmp_path.resolve()}/models/1ubq.cif:A"
    assert records["v"].structure == "/abs/x.cif" and records["v"].spans == "3"
    assert records["w"].structure is None
    assert fdz.make_design("x", "MKV___LLAG").structure is None
