"""A design in the flow stream (`src/foldspace/rfd4/flow_data.py`): the release's conditions for the known residues only, the ESMC rows of the masked chain, and the slots that carry them."""

from __future__ import annotations

import json

import foldspace.rfd4.structure_condition as fst
import numpy as np
import pytest
import foldspace.rfd4.flow_data as fd

pytestmark = pytest.mark.offline

E = 6  # the toy's ESMC width


def item(n: int = 12) -> dict:
    rng = np.random.default_rng(1)
    return {
        "coords": rng.normal(size=(n, 37, 3)).astype(np.float32),
        "atom37_mask": np.ones((n, 37), dtype=bool),
        "residue_type": rng.integers(0, 20, size=n).astype(np.int64),
        "token_type": np.zeros(n, dtype=np.int32),
        "chain_iid": np.zeros(n, dtype=np.int16),
        "charge": np.zeros((n, 37), dtype=np.int8),
        "ground_truth_element": np.zeros((n, 37), dtype=np.int8),
        "incident_bond_type_counts": np.zeros((n, 37, 4), dtype=np.int8),
        "supports_local_latent": np.ones(n, dtype=bool),
        "is_in_context_conditioning": np.zeros(n, dtype=bool),
        "is_resolved_mask": np.ones(n, dtype=bool),
    }


def test_the_release_conditions_name_the_known_residues_only() -> None:
    pytest.importorskip("rfproteina")
    from atomworks.ml.conditions import C_SEQ
    from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS

    types = item(12)["residue_type"]
    known = np.array([True] * 4 + [False] * 5 + [True] * 3)
    given = fd.sequence_conditions(types, known)
    c_seq = given[C_SEQ.full_name]
    assert c_seq.dtype == np.int64
    assert np.array_equal(c_seq[known], types[known])
    assert (c_seq[~known] == fd.mask_token_index()).all()
    probabilities = np.stack(
        [given[c.full_name] for c in SEQUENCE_PROBABILITY_CONDITIONS], axis=-1
    )
    assert probabilities.dtype == np.float32
    assert np.isnan(probabilities[~known]).all()
    assert not np.isnan(probabilities[known]).any()
    assert (probabilities[known].sum(-1) == 1).all()
    # all known is the whole sequence of the release's folding task, bit for bit
    whole = fd.sequence_conditions(types)
    all_known = fd.sequence_conditions(types, np.ones(12, dtype=bool))
    assert whole.keys() == all_known.keys()
    assert all(np.array_equal(whole[k], all_known[k]) for k in whole)


def test_a_design_conditions_do_not_depend_on_what_is_under_the_mask() -> None:
    """The contract at the release's keys: two chains that differ only at the residues to generate give the denoiser the same conditions."""
    pytest.importorskip("rfproteina")
    types = item(12)["residue_type"]
    other = types.copy()
    other[4:9] = (other[4:9] + 7) % 20
    known = np.array([True] * 4 + [False] * 5 + [True] * 3)
    a = fd.sequence_conditions(types, known)
    b = fd.sequence_conditions(other, known)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_known_must_be_one_flag_a_residue() -> None:
    pytest.importorskip("rfproteina")
    with pytest.raises(ValueError, match="known has shape"):
        fd.sequence_conditions(item(12)["residue_type"], np.ones(11, dtype=bool))


def test_a_design_chain_reads_the_masked_rows_and_its_crop_keeps_them_aligned() -> None:
    pytest.importorskip("rfproteina")
    from atomworks.ml.conditions import C_SEQ

    it = item(20)
    whole = np.random.default_rng(2).normal(size=(20, E)).astype(np.float16)
    masked = np.random.default_rng(3).normal(size=(20, E)).astype(np.float16)
    known = np.ones(20, dtype=bool)
    known[6:11] = False
    feats = fd.chain_features(
        it,
        np.random.default_rng(5),
        crop=12,
        esmc=whole,
        condition_sequence=True,
        context_rows=True,
        known=known,
        masked_context=masked,
    )
    plain = fd.chain_features(
        it,
        np.random.default_rng(5),
        crop=12,
        esmc=whole,
        condition_sequence=True,
        context_rows=True,
    )
    # the same window and rotation as the example given its whole sequence: the mask draws nothing from the augmentation stream
    assert np.array_equal(feats["coords"], plain["coords"])
    # the adapter reads the masked chain's rows, not the whole chain's, over the same residues
    assert not np.array_equal(feats["esmc_context"], plain["esmc_context"])
    start = next(
        s for s in range(9) if np.array_equal(feats["esmc"], whole[s : s + 12])
    )
    assert np.array_equal(feats["esmc_context"], masked[start : start + 12])
    c_seq = feats[C_SEQ.full_name]
    assert (c_seq[~known[start : start + 12]] == fd.mask_token_index()).all()
    with pytest.raises(ValueError, match="needs condition_sequence"):
        fd.chain_features(it, np.random.default_rng(5), esmc=whole, known=known)
    with pytest.raises(ValueError, match="rows for a chain of"):
        fd.chain_features(
            it,
            np.random.default_rng(5),
            esmc=whole,
            condition_sequence=True,
            context_rows=True,
            masked_context=masked[:-1],
        )


def masked_rows(tmp_path, names=("a", "b")) -> fd.MaskedRows:
    """Two chains' masked rows in one array, and the index for slots 0:0 and 0:1."""
    lengths = {"a": 10, "b": 7}
    spans = {"a": [[2, 5]], "b": [[0, 2], [4, 6]]}
    rng = np.random.default_rng(9)
    array = rng.normal(size=(17, E)).astype(np.float16)
    np.save(tmp_path / "rows.npy", array)
    slots, start = {}, 0
    for j, name in enumerate(names):
        slots[f"0:{j}"] = {
            "name": name,
            "start": start,
            "length": lengths[name],
            "spans": spans[name],
        }
        start += lengths[name]
    (tmp_path / "index.json").write_text(json.dumps({"slots": slots}))
    return fd.MaskedRows(tmp_path / "rows.npy", tmp_path / "index.json")


def test_the_masked_rows_of_a_slot_come_with_its_mask(tmp_path) -> None:
    rows = masked_rows(tmp_path)
    got, known = rows.get(0, 1, "b")
    assert got.shape == (7, E) and got.dtype == np.float16
    assert known.tolist() == [False, False, True, True, False, False, True]
    got_a, known_a = rows.get(0, 0, "a")
    assert got_a.shape == (10, E) and (~known_a).sum() == 3


def test_a_slot_that_is_not_the_one_the_rows_were_made_for_is_refused(tmp_path) -> None:
    rows = masked_rows(tmp_path)
    with pytest.raises(ValueError, match="are of b"):
        rows.get(0, 1, "c")
    with pytest.raises(KeyError, match="another stream"):
        rows.get(3, 0, "a")


def stream(fraction: float, mask: float, tmp_path=None, structure: float = 0.0):
    rows = None if mask <= 0 else masked_rows(tmp_path)
    return fd.FlowStream(
        "shards",
        ["a", "b", "c", "d"],
        batch=2,
        seed=7,
        condition_fraction=fraction,
        mask_fraction=mask,
        masked_rows=rows,
        structure_fraction=structure,
    )


def test_the_masked_coin_is_its_own_stream_and_only_among_the_conditioned(
    tmp_path,
) -> None:
    base = stream(0.75, 0.0)
    masked = stream(0.75, 0.6, tmp_path)
    for s in range(100):
        for j in range(2):
            # which chains are conditioned at all is the stream's without the mask coin
            assert masked.conditioned(s, j) == base.conditioned(s, j)
            if not masked.conditioned(s, j):
                assert not masked.masked(s, j)
    draws = [masked.masked(s, j) for s in range(600) for j in range(2)]
    conditioned = [masked.conditioned(s, j) for s in range(600) for j in range(2)]
    share = np.mean([m for m, c in zip(draws, conditioned, strict=True) if c])
    assert 0.5 < share < 0.7
    assert draws == [masked.masked(s, j) for s in range(600) for j in range(2)]
    assert not any(stream(0.75, 0.0).masked(s, j) for s in range(50) for j in range(2))
    assert all(
        stream(1.0, 1.0, tmp_path).masked(s, j) for s in range(50) for j in range(2)
    )


def test_the_slots_that_need_rows_are_listed_in_step_order(tmp_path) -> None:
    s = stream(1.0, 1.0, tmp_path)
    slots = s.masked_slots(range(3, 6))
    assert [(a, b) for a, b, _ in slots] == [
        (st, j) for st in range(3, 6) for j in range(2)
    ]
    assert [name for _, _, name in slots] == [
        n for st in range(3, 6) for n in s.chains_of_step(st)
    ]


def test_a_mask_fraction_without_rows_is_refused_when_a_batch_is_asked_for() -> None:
    s = fd.FlowStream(
        "shards", ["a"], batch=1, condition_fraction=1.0, mask_fraction=1.0
    )
    # the slots can be listed without rows (that is how the rows are made); a batch cannot be made without them
    assert s.masked_slots(range(3))
    with pytest.raises(ValueError, match="needs the masked rows"):
        s[0]


def test_the_stream_with_its_masked_rows_survives_a_spawned_worker(tmp_path) -> None:
    """The trainer's data loader spawns its workers: the stream is pickled to a fresh interpreter, with its rows still unread, and masks the same slots there."""
    import pickle

    s = stream(0.75, 0.6, tmp_path)
    again = pickle.loads(pickle.dumps(s))
    assert [again.masked(st, j) for st in range(80) for j in range(2)] == [
        s.masked(st, j) for st in range(80) for j in range(2)
    ]
    rows, known = again.masked_rows.get(0, 0, "a")
    assert rows.shape == (10, E) and (~known).sum() == 3


def known_with_a_span(n: int = 20) -> np.ndarray:
    known = np.ones(n, dtype=bool)
    known[6:11] = False
    return known


def test_the_structure_condition_is_the_known_backbone_in_the_conditioned_frame() -> (
    None
):
    it = item(20)
    known = known_with_a_span()
    rotation = fd.random_rotation(np.random.default_rng(5))
    mask = it["atom37_mask"].astype(bool)
    target, keys = fd.structure_condition(it["coords"], mask, known, rotation)
    coords = keys[fst.KEY_COORDINATES]
    finite = ~np.isnan(coords).any(axis=-1)
    # the atoms N, CA, C and O of the known residues, and no other
    want = np.zeros((20, 37), dtype=bool)
    want[np.ix_(known, list(fst.BACKBONE_SLOTS))] = True
    assert np.array_equal(finite, want)
    assert keys[fst.KEY_GROUP].dtype == np.int16
    assert np.array_equal(keys[fst.KEY_GROUP] == 0, finite)
    assert (keys[fst.KEY_GROUP][~finite] == -1).all()
    assert np.array_equal(keys[fst.KEY_MASK], want[:, 0])
    # centred on the conditioned atoms, as the release centres its own coordinate-conditioned examples
    assert np.allclose(coords[finite].mean(axis=0), 0.0, atol=1e-5)
    # the target is in that frame for every atom, so that at a conditioned atom the condition and the target are one number
    assert target.dtype == np.float32
    assert np.array_equal(target[finite], coords[finite])
    # the example without the structure is centred on all its atoms: the same rotated chain, shifted
    plain = fd.place(it["coords"], mask, rotation)
    shift = target - plain
    assert np.allclose(shift, shift[0, 0], atol=1e-5)


def test_the_structure_condition_does_not_depend_on_the_residues_to_generate() -> None:
    """The contract at the release's keys: two chains that differ only in the coordinates of the residues under the mask give the denoiser the same conditions, and the same frame."""
    it = item(20)
    known = known_with_a_span()
    moved = {**it, "coords": it["coords"].copy()}
    moved["coords"][~known] += (
        np.random.default_rng(4).normal(size=(5, 37, 3)).astype(np.float32) * 5
    )
    rotation = fd.random_rotation(np.random.default_rng(5))
    mask = it["atom37_mask"].astype(bool)
    a_target, a = fd.structure_condition(it["coords"], mask, known, rotation)
    b_target, b = fd.structure_condition(moved["coords"], mask, known, rotation)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    # the target of the known residues is in the same frame; only the residues to generate differ
    np.testing.assert_array_equal(a_target[known], b_target[known])
    assert not np.array_equal(a_target[~known], b_target[~known])
    # the control: a centre taken over the whole chain does depend on where the residues to generate are
    a_whole = fd.place(it["coords"], mask, rotation)
    b_whole = fd.place(moved["coords"], mask, rotation)
    assert not np.array_equal(a_whole[known], b_whole[known])


def test_a_chain_with_no_known_backbone_atom_has_nothing_to_give() -> None:
    it = item(12)
    rotation = np.eye(3, dtype=np.float32)
    mask = it["atom37_mask"].astype(bool)
    assert (
        fd.structure_condition(it["coords"], mask, np.zeros(12, dtype=bool), rotation)
        is None
    )
    # known residues whose backbone atoms are absent from the structure give none either
    bare = mask.copy()
    bare[:, list(fst.BACKBONE_SLOTS)] = False
    assert (
        fd.structure_condition(it["coords"], bare, np.ones(12, dtype=bool), rotation)
        is None
    )


def test_a_design_chain_given_the_structure_keeps_its_crop_and_draws() -> None:
    pytest.importorskip("rfproteina")
    it = item(20)
    known = known_with_a_span()
    rows = np.random.default_rng(2).normal(size=(20, E)).astype(np.float16)

    def made(structure: bool, known=known) -> dict:
        return fd.chain_features(
            it,
            np.random.default_rng(5),
            crop=12,
            esmc=rows,
            condition_sequence=True,
            context_rows=True,
            known=known,
            masked_context=rows,
            structure=structure,
        )

    plain, given = made(False), made(True)
    # the same window and rotation as the example without the structure: the structure draws nothing from the augmentation stream
    assert np.array_equal(plain["residue_type"], given["residue_type"])
    start = next(
        s
        for s in range(9)
        if np.array_equal(given["residue_type"], it["residue_type"][s : s + 12])
    )
    finite = ~np.isnan(given[fst.KEY_COORDINATES]).any(axis=-1)
    assert np.array_equal(finite[:, 1], known[start : start + 12])
    assert not finite[~known[start : start + 12]].any()
    assert fst.KEY_COORDINATES not in plain
    assert np.array_equal(given["coords"][finite], given[fst.KEY_COORDINATES][finite])
    # a crop that holds no known residue is the masked sequence alone
    nothing = np.zeros(20, dtype=bool)
    alone, again = made(True, nothing), made(False, nothing)
    assert fst.KEY_COORDINATES not in alone
    assert np.array_equal(alone["coords"], again["coords"])
    with pytest.raises(ValueError, match="needs known"):
        fd.chain_features(it, np.random.default_rng(5), structure=True)


def test_the_structure_coin_is_its_own_stream_and_only_among_the_masked(
    tmp_path,
) -> None:
    base = stream(0.75, 0.6, tmp_path)
    half = stream(0.75, 0.6, tmp_path, structure=0.5)
    for s in range(400):
        for j in range(2):
            # which chains are conditioned and which masked are the stream's without the structure coin
            assert half.conditioned(s, j) == base.conditioned(s, j)
            assert half.masked(s, j) == base.masked(s, j)
            assert not base.structured(s, j)
            if not half.masked(s, j):
                assert not half.structured(s, j)
    masked = [(s, j) for s in range(800) for j in range(2) if half.masked(s, j)]
    draws = [half.structured(s, j) for s, j in masked]
    assert 0.4 < np.mean(draws) < 0.6
    assert draws == [half.structured(s, j) for s, j in masked]
    everything = stream(0.75, 0.6, tmp_path, structure=1.0)
    assert all(
        everything.structured(s, j) == everything.masked(s, j) for s, j in masked
    )
    import pickle

    again = pickle.loads(pickle.dumps(half))
    assert [again.structured(s, j) for s, j in masked] == draws
