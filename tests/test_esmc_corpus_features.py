"""The corpus's ESMC feature array (`src/foldspace/rfd4/esmc_corpus_features.py`): the plan, the padding, the write into the array, the
contract a record of progress is continued under, the per-vector check, with a stand-in for the model that gives each residue a
state computed from its own token."""

from __future__ import annotations

import json

import foldspace.rfd4.esmc_corpus_features as ecf
import numpy as np
import pytest

pytestmark = pytest.mark.offline

DIM = ecf.DIM
SETTINGS = ecf.settings_of(max_tokens=12, max_chains=2)


def corpus(lengths):
    lengths = np.asarray(lengths, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 20, size=int(lengths.sum()), dtype=np.int8)
    return tokens, starts, lengths


def stand_in(tok, mask):
    """`[B, W, DIM]`: the state of a residue is a function of its token and of nothing else; padding is zero."""
    table = np.random.default_rng(1).standard_normal((21, DIM)).astype(np.float32)
    return table[tok] * mask[..., None]


def expected(tokens):
    """What the stand-in's states are in half precision, rounded the way `to_half` rounds them (through single precision)."""
    table = (
        np.random.default_rng(1)
        .standard_normal((21, DIM))
        .astype(np.float32)
        .astype(np.float16)
    )
    return table[tokens.astype(int)]


def rows(tmp_path, total):
    return ecf.RowFile.create(tmp_path / "a.npy", total)


def contract_of(tokens, lengths, plan, settings=SETTINGS):
    return ecf.make_contract(plan, ecf.digest_of(tokens, lengths), settings)


def finished(tmp_path, lengths, *, max_tokens=14, max_chains=2):
    """A corpus extracted whole with the stand-in; returns what a later check needs."""
    tokens, starts, lengths = corpus(lengths)
    total = int(lengths.sum())
    plan = ecf.plan_batches(lengths, max_tokens=max_tokens, max_chains=max_chains)
    contract = contract_of(
        tokens, lengths, plan, ecf.settings_of(max_tokens, max_chains)
    )
    with rows(tmp_path, total) as out:
        done = ecf.extract(
            stand_in,
            tokens,
            starts,
            lengths,
            out,
            plan,
            tmp_path / "progress.json",
            contract,
            log=lambda s: None,
        )
    assert done == len(plan)
    return tokens, starts, lengths, plan, contract


def test_plan_is_a_partition_longest_first_within_the_cap():
    lengths = [60, 300, 176, 300, 90, 120, 120, 61]
    plan = ecf.plan_batches(lengths, max_tokens=700, max_chains=3)
    flat = [i for batch in plan for i in batch]
    assert sorted(flat) == list(range(len(lengths)))
    assert [lengths[i] for i in flat] == sorted(lengths, reverse=True)
    for batch in plan:
        assert len(batch) <= 3
        assert lengths[batch[0]] * len(batch) <= 700 or len(batch) == 1
    # ties are by index, so the plan is a function of the lengths only
    assert plan == ecf.plan_batches(list(lengths), max_tokens=700, max_chains=3)
    assert plan[0][:2] == [1, 3]


def test_a_chain_longer_than_the_cap_is_a_batch_of_its_own():
    plan = ecf.plan_batches([500, 100, 100], max_tokens=300, max_chains=8)
    assert plan[0] == [0]
    assert plan[1] == [1, 2]


def test_batch_arrays_pad_on_the_right_and_mask_the_residues():
    tokens, starts, lengths = corpus([3, 5])
    tok, mask = ecf.batch_arrays(tokens, starts, lengths, [1, 0])
    assert tok.shape == mask.shape == (2, 5)
    assert mask.tolist() == [[1] * 5, [1, 1, 1, 0, 0]]
    assert tok[0].tolist() == tokens[3:8].tolist()
    assert tok[1, 3:].tolist() == [0, 0]


def test_each_chain_lands_in_its_own_slice_whatever_the_batching(tmp_path):
    tokens, starts, lengths = corpus([4, 7, 2, 7, 5])
    ecf.check_layout(starts, lengths, int(lengths.sum()))
    finished(tmp_path, [4, 7, 2, 7, 5])
    assert np.array_equal(np.load(tmp_path / "a.npy"), expected(tokens))


def test_the_rows_are_written_by_seek_and_read_back_as_an_ordinary_npy(tmp_path):
    total = 10
    with rows(tmp_path, total) as out:
        block = np.arange(3 * DIM, dtype=np.float16).reshape(3, DIM) % 7
        out[4:7] = block
        with pytest.raises(ValueError, match="take"):
            out[0:2] = block
        with pytest.raises(TypeError, match="slice"):
            out[::2] = block
        with pytest.raises(TypeError, match="slice"):
            out[3] = block[0]
    loaded = np.load(tmp_path / "a.npy")
    assert loaded.shape == (total, DIM) and loaded.dtype == np.float16
    assert np.array_equal(loaded[4:7], block)
    assert not loaded[:4].any() and not loaded[7:].any()
    # the file is what `np.load(mmap_mode="r")` maps, with the rows where the header says
    view = np.load(tmp_path / "a.npy", mmap_mode="r")
    assert ecf.npy_offset(tmp_path / "a.npy", total) == view.offset


def test_an_array_of_another_corpus_or_a_short_file_is_refused(tmp_path):
    with rows(tmp_path, 6):
        pass
    assert ecf.npy_offset(tmp_path / "a.npy", 6) > 0
    with pytest.raises(ValueError, match="not the array of this corpus"):
        ecf.npy_offset(tmp_path / "a.npy", 7)
    short = tmp_path / "short.npy"
    short.write_bytes((tmp_path / "a.npy").read_bytes()[:-DIM])
    with pytest.raises(ValueError):
        ecf.npy_offset(short, 6)


def test_a_budget_stops_between_batches_and_a_second_run_resumes(tmp_path):
    tokens, starts, lengths = corpus([6, 6, 6, 6, 6, 6])
    total = int(lengths.sum())
    plan = ecf.plan_batches(lengths, max_tokens=12, max_chains=2)
    assert len(plan) == 3
    contract = contract_of(tokens, lengths, plan)
    ticks = iter(
        [0.0, 0.0, 5.0, 5.0, 5.0, 5.0]
    )  # the first batch starts at once, nothing after the budget
    with rows(tmp_path, total) as out:
        first = ecf.extract(
            stand_in,
            tokens,
            starts,
            lengths,
            out,
            plan,
            tmp_path / "p.json",
            contract,
            budget_s=1.0,
            clock=lambda: next(ticks),
            log=lambda s: None,
        )
    assert first == 1
    record = json.loads((tmp_path / "p.json").read_text())
    assert record["done"] == 1 and record["n_batches"] == 3
    assert record["plan_sha256"] == ecf.plan_digest(plan)
    assert record["corpus_sha256"] == ecf.digest_of(tokens, lengths)
    assert record["settings"] == SETTINGS
    with ecf.RowFile.open(tmp_path / "a.npy", total) as out:
        second = ecf.extract(
            stand_in,
            tokens,
            starts,
            lengths,
            out,
            plan,
            tmp_path / "p.json",
            contract,
            log=lambda s: None,
        )
    assert second == 3
    assert np.array_equal(np.load(tmp_path / "a.npy"), expected(tokens))


def test_the_plan_digest_follows_the_batches_and_where_each_ends():
    assert ecf.plan_digest([[1, 2], [3]]) == ecf.plan_digest([[1, 2], [3]])
    assert ecf.plan_digest([[1, 2], [3]]) != ecf.plan_digest([[1], [2, 3]])
    assert ecf.plan_digest([[1, 2], [3]]) != ecf.plan_digest([[2, 1], [3]])
    assert ecf.plan_digest([[1, 2], [3]]) != ecf.plan_digest([[1, 2, 3]])


def test_a_record_of_another_plan_of_the_same_size_is_refused(tmp_path):
    # the same chains cut two ways with two batches each: a count of batches alone could not tell them apart
    tokens, _, lengths = corpus([6, 5, 4, 3])
    first = ecf.plan_batches(lengths, max_tokens=12, max_chains=2)
    second = ecf.plan_batches(lengths, max_tokens=100, max_chains=3)
    assert first == [[0, 1], [2, 3]] and second == [[0, 1, 2], [3]]
    assert len(first) == len(second)
    ecf.write_progress(tmp_path / "p.json", 1, contract_of(tokens, lengths, first))
    with pytest.raises(ValueError, match="another plan"):
        ecf.read_progress(tmp_path / "p.json", contract_of(tokens, lengths, second))
    assert (
        ecf.read_progress(tmp_path / "p.json", contract_of(tokens, lengths, first)) == 1
    )


def test_a_record_of_another_corpus_or_other_settings_is_refused(tmp_path):
    tokens, _, lengths = corpus([6, 6, 6, 6])
    plan = ecf.plan_batches(lengths, max_tokens=12, max_chains=2)
    ecf.write_progress(tmp_path / "p.json", 1, contract_of(tokens, lengths, plan))
    other = tokens.copy()
    other[3] = (other[3] + 1) % 20
    with pytest.raises(ValueError, match="another corpus"):
        ecf.read_progress(tmp_path / "p.json", contract_of(other, lengths, plan))
    changed = dict(SETTINGS, state="ESMC-6B pre-norm state")
    with pytest.raises(ValueError, match="another settings"):
        ecf.read_progress(
            tmp_path / "p.json", contract_of(tokens, lengths, plan, changed)
        )


def test_a_run_under_another_contract_does_not_touch_the_array(tmp_path):
    tokens, starts, lengths = corpus([6, 6, 6, 6, 6, 6])
    total = int(lengths.sum())
    plan = ecf.plan_batches(lengths, max_tokens=12, max_chains=2)
    contract = contract_of(tokens, lengths, plan)
    ticks = iter([0.0, 0.0, 5.0, 5.0])
    with rows(tmp_path, total) as out:
        ecf.extract(
            stand_in,
            tokens,
            starts,
            lengths,
            out,
            plan,
            tmp_path / "p.json",
            contract,
            budget_s=1.0,
            clock=lambda: next(ticks),
            log=lambda s: None,
        )
    before = (tmp_path / "a.npy").read_bytes()
    other_plan = ecf.plan_batches(lengths, max_tokens=18, max_chains=3)
    with (
        ecf.RowFile.open(tmp_path / "a.npy", total) as out,
        pytest.raises(ValueError, match="another plan"),
    ):
        ecf.extract(
            stand_in,
            tokens,
            starts,
            lengths,
            out,
            other_plan,
            tmp_path / "p.json",
            contract_of(tokens, lengths, other_plan),
            log=lambda s: None,
        )
    assert (tmp_path / "a.npy").read_bytes() == before


def test_a_record_that_counts_batches_only_is_continued_only_when_complete_and_confirmed(
    tmp_path,
):
    tokens, _, lengths = corpus([6, 6, 6, 6])
    plan = ecf.plan_batches(lengths, max_tokens=12, max_chains=2)
    contract = contract_of(tokens, lengths, plan)
    legacy = tmp_path / "p.json"
    meta = {
        "complete": True,
        "corpus_sha256": contract["corpus_sha256"],
        "max_tokens": SETTINGS["max_tokens"],
        "max_chains": SETTINGS["max_chains"],
        "state": ecf.STATE,
    }
    # a finished run that meta.json confirms: nothing remains to be computed under a wrong plan
    legacy.write_text(json.dumps({"done": 2, "n_batches": 2}))
    assert ecf.read_progress(legacy, contract, meta) == 2
    # the same record without meta.json, or with another corpus, or another batching, is not enough
    with pytest.raises(ValueError, match="does not confirm"):
        ecf.read_progress(legacy, contract, None)
    with pytest.raises(ValueError, match="does not confirm"):
        ecf.read_progress(legacy, contract, dict(meta, corpus_sha256="0" * 64))
    with pytest.raises(ValueError, match="does not confirm"):
        ecf.read_progress(legacy, contract, dict(meta, max_chains=3))
    with pytest.raises(ValueError, match="does not confirm"):
        ecf.read_progress(legacy, contract, dict(meta, complete=False))
    # an unfinished record of this form cannot say which plan it counted in
    legacy.write_text(json.dumps({"done": 1, "n_batches": 2}))
    with pytest.raises(ValueError, match="not complete"):
        ecf.read_progress(legacy, contract, meta)
    # a finished record of a plan of another size is another plan
    legacy.write_text(json.dumps({"done": 3, "n_batches": 3}))
    with pytest.raises(ValueError, match="not complete"):
        ecf.read_progress(legacy, contract, meta)


def test_a_finished_run_of_the_first_form_is_upgraded_to_a_record_with_the_digests(
    tmp_path,
):
    tokens, starts, lengths, plan, contract = finished(tmp_path, [4, 7, 2, 7, 5])
    (tmp_path / "progress.json").write_text(
        json.dumps({"done": len(plan), "n_batches": len(plan)})
    )
    meta = {
        "complete": True,
        "corpus_sha256": contract["corpus_sha256"],
        "max_tokens": 14,
        "max_chains": 2,
        "state": ecf.STATE,
    }
    with ecf.RowFile.open(tmp_path / "a.npy", int(lengths.sum())) as out:
        done = ecf.extract(
            stand_in,
            tokens,
            starts,
            lengths,
            out,
            plan,
            tmp_path / "progress.json",
            contract,
            meta=meta,
            log=lambda s: None,
        )
    assert done == len(plan)
    record = json.loads((tmp_path / "progress.json").read_text())
    assert record["plan_sha256"] == contract["plan_sha256"] and record["version"] == 2


def test_layout_must_be_contiguous_and_complete():
    with pytest.raises(ValueError, match="contiguous"):
        ecf.check_layout(np.array([0, 5]), np.array([4, 3]), 7)
    with pytest.raises(ValueError, match="add up"):
        ecf.check_layout(np.array([0, 4]), np.array([4, 3]), 8)


def test_a_state_that_half_precision_cannot_hold_is_an_error_not_a_clip():
    with pytest.raises(ValueError, match="finite"):
        ecf.to_half(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="half precision"):
        ecf.to_half(np.array([[7.0e4]]))
    assert ecf.to_half(np.array([[1.5]])).dtype == np.float16


def test_the_check_finds_a_corrupted_row(tmp_path):
    tokens, starts, lengths, _, _ = finished(tmp_path, [4, 7, 2, 7, 5])
    out = np.load(tmp_path / "a.npy", mmap_mode="r")
    clean = ecf.self_check(stand_in, tokens, starts, lengths, out, count=5)
    assert clean["chains"] == 5 and clean["max_abs_diff"] == 0.0
    assert clean["rel_l2"]["max"] == 0.0 and clean["cosine"]["min"] > 0.999999
    assert clean["zero_rows"] == 0 and clean["residues"] == int(lengths.sum())
    damaged = np.load(tmp_path / "a.npy")
    damaged[int(starts[2])] += np.float16(1.0)
    dirty = ecf.self_check(stand_in, tokens, starts, lengths, damaged, count=5)
    assert dirty["max_abs_diff"] > 0.5 and dirty["rel_l2"]["max"] > 0.0


def test_the_check_reads_a_difference_next_to_the_size_of_the_vector(tmp_path):
    tokens, starts, lengths, _, _ = finished(tmp_path, [4, 7, 2, 7, 5])
    stored = np.load(tmp_path / "a.npy")
    # every stored vector 10 % too long: relative L2 = 0.1 / 1.1, the direction (cosine) untouched
    scaled = (stored.astype(np.float32) * 1.1).astype(np.float16)
    scaled_check = ecf.self_check(stand_in, tokens, starts, lengths, scaled, count=5)
    assert scaled_check["rel_l2"]["median"] == pytest.approx(0.1 / 1.1, abs=2e-3)
    assert scaled_check["cosine"]["min"] > 0.9999
    # the vectors of the stand-in have a norm near sqrt(2560) = 50, which is what a mean absolute difference has to be read against
    assert 45 < scaled_check["stored_norm"]["median"] < 60
    assert (
        0.05 < scaled_check["mean_abs_diff"] < 0.12
    )  # 0.1 * E|x| = 0.08 for unit-variance entries
    # rows swapped for those of another token: nothing like the vector, the cosine says so
    wrong = np.roll(stored, 1, axis=0)
    wrong_check = ecf.self_check(stand_in, tokens, starts, lengths, wrong, count=5)
    assert wrong_check["cosine"]["median"] < 0.5
    assert wrong_check["rel_l2"]["median"] > 1.0
    # a chain that was never written has zero rows: counted by name, and kept out of the quantiles (no relative distance to a zero vector)
    gap = stored.copy()
    gap[int(starts[1]) : int(starts[1]) + int(lengths[1])] = 0
    gap_check = ecf.self_check(stand_in, tokens, starts, lengths, gap, count=5)
    assert gap_check["zero_rows"] == int(lengths[1])
    assert gap_check["residues"] == int(lengths.sum())
    assert gap_check["rel_l2"]["max"] == 0.0
    nothing = ecf.self_check(
        stand_in, tokens, starts, lengths, np.zeros_like(stored), count=5
    )
    assert nothing["zero_rows"] == int(lengths.sum())
    assert nothing["rel_l2"]["median"] is None and nothing["cosine"]["min"] is None


def test_a_check_of_a_finished_array_writes_check_json_and_refuses_what_is_not_its_own(
    tmp_path,
):
    tokens, starts, lengths, _, contract = finished(tmp_path, [4, 7, 2, 7, 5])
    meta = {
        "array": "a.npy",
        "complete": True,
        "corpus_sha256": contract["corpus_sha256"],
        "state": ecf.STATE,
    }
    with pytest.raises(ValueError, match="no finished array"):
        ecf.run_check(tmp_path, tokens, starts, lengths, stand_in, 3)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result = ecf.run_check(
        tmp_path, tokens, starts, lengths, stand_in, 3, stack_commit="abc"
    )
    written = json.loads((tmp_path / "check.json").read_text())
    assert written["check"]["chains"] == 3 and written["stack_commit"] == "abc"
    assert result["check"]["max_abs_diff"] == 0.0
    other = tokens.copy()
    other[0] = (other[0] + 1) % 20
    with pytest.raises(ValueError, match="another corpus"):
        ecf.run_check(tmp_path, other, starts, lengths, stand_in, 3)
    (tmp_path / "meta.json").write_text(json.dumps(dict(meta, complete=False)))
    with pytest.raises(ValueError, match="complete"):
        ecf.run_check(tmp_path, tokens, starts, lengths, stand_in, 3)
    (tmp_path / "meta.json").write_text(json.dumps(dict(meta, state="pre-norm")))
    with pytest.raises(ValueError, match="another state"):
        ecf.run_check(tmp_path, tokens, starts, lengths, stand_in, 3)


def test_the_digest_follows_the_sequences_and_their_boundaries():
    tokens, _, lengths = corpus([4, 7])
    base = ecf.digest_of(tokens, lengths)
    assert base == ecf.digest_of(tokens.copy(), lengths.copy())
    assert base != ecf.digest_of(tokens, np.array([5, 6]))
    other = tokens.copy()
    other[0] = (other[0] + 1) % 20
    assert base != ecf.digest_of(other, lengths)
