"""The pretrained generator's set: how a generation becomes a sequence, which are kept, how the pilot's lengths are met, and the generation loop on a tiny random model."""

from __future__ import annotations

import pytest

from foldspace.esmfold2 import pretrained as pt

pytestmark = pytest.mark.offline


def test_a_generation_is_a_sequence_once_the_end_tokens_and_line_breaks_are_gone():
    text = "<|endoftext|>\nMKVLAAGIVG\nLLTAQWERTY\nAC<|endoftext|>"
    assert pt.clean(text) == "MKVLAAGIVGLLTAQWERTYAC"
    assert pt.clean("<|endoftext|>") == ""
    assert pt.clean("MK VL \n AA") == "MKVLAA"


def test_a_sequence_is_kept_if_it_is_standard_and_of_the_lengths_the_ruler_reads():
    assert pt.keep("A" * 70) and pt.keep("A" * 220)
    assert not pt.keep("A" * 69) and not pt.keep("A" * 221)
    assert not pt.keep("A" * 99 + "X")  # a residue outside the twenty
    assert not pt.keep("")
    assert not pt.keep("a" * 100)  # lower case is not the alphabet the ruler reads


def test_lengths_are_met_by_the_nearest_unused_sequence_within_the_tolerance():
    lengths = [100, 130, 131, 200, 129]
    targets = [130, 130, 129, 300]
    got = pt.match_lengths(lengths, targets)
    assert (
        got == [1, 2, 4, None]
    )  # 129 is served first and takes 129; the two 130s take 130 and 131; 300 has none within 15 %
    used = [g for g in got if g is not None]
    assert len(used) == len(set(used))  # every sequence at most once
    # a candidate outside the tolerance is never taken, however few there are
    assert pt.match_lengths([100], [130]) == [None]  # 30 away from 130, tolerance 19.5
    assert pt.match_lengths([112], [130]) == [0]  # 18 away
    assert pt.match_lengths([], [130, 140]) == [None, None]


def test_the_targets_are_served_shortest_first_so_that_a_scarce_length_is_not_taken_by_a_longer_one():
    # one sequence of 140 could serve a target of 150 or of 130; the shorter target is served first and takes it
    assert pt.match_lengths([140], [150, 130]) == [None, 0]
    # ties in distance go to the earlier sequence
    assert pt.match_lengths([126, 134], [130]) == [0]


def test_the_generation_loop_returns_token_ids_that_start_from_the_prompt_and_are_set_by_the_seed():
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=40, n_positions=32, n_embd=32, n_layer=1, n_head=2)
    )
    settings = {
        "top_k": 10,
        "repetition_penalty": 1.2,
        "temperature": 1.0,
        "max_length": 12,
    }
    a = pt.generate_ids(model, n=7, seed=3, batch=3, device="cpu", settings=settings)
    b = pt.generate_ids(model, n=7, seed=3, batch=3, device="cpu", settings=settings)
    c = pt.generate_ids(model, n=7, seed=4, batch=3, device="cpu", settings=settings)
    assert len(a) == 7 and all(row[0] == 0 and len(row) <= 12 for row in a)
    assert a == b and a != c
