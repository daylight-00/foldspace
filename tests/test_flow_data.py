"""The flow batches of `src/foldspace/rfd4/flow_data.py`: the pair features of an unbroken chain, the rotation, the crop and
the centring, the ESMC rows of a crop, and the stream a run reads (the same chains and augmentations for the same seed, resumable
at any step). The comparison with the release pipeline's own output is `src/foldspace/rfd4/flow_check.py`, which needs the stack."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import foldspace.rfd4.flow_data as fd

pytestmark = pytest.mark.offline


def item(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = rng.random((n, 37)) < 0.5
    mask[:, :4] = True
    coords = np.where(mask[..., None], rng.normal(size=(n, 37, 3)) * 10, 0.0)
    out = {
        "coords": coords.astype(np.float32),
        "atom37_mask": mask,
        "residue_type": rng.integers(0, 20, size=n).astype(np.int64),
        "token_type": np.zeros(n, dtype=np.int32),
        "chain_iid": np.zeros(n, dtype=np.int16),
        "charge": np.zeros((n, 37), dtype=np.int16),
        "ground_truth_element": np.ones((n, 37), dtype=np.int16),
        "incident_bond_type_counts": np.zeros((n, 37, 10), dtype=np.int16),
        "supports_local_latent": np.ones(n, dtype=bool),
        "is_in_context_conditioning": np.zeros(n, dtype=bool),
        "is_resolved_mask": np.ones(n, dtype=bool),
    }
    return out


def test_path_length_is_the_residue_distance_with_the_cutoff_past_it() -> None:
    path, _ = fd.pair_features(80)
    assert path.dtype == np.int16
    assert (np.diag(path) == 0).all()
    assert path[3, 7] == 4 and path[7, 3] == 4
    assert path[0, 64] == 64 and path[0, 65] == 65 and path[0, 79] == 65


def test_canonical_ordering_orients_the_backbone_linkage() -> None:
    _, order = fd.pair_features(5)
    assert order.dtype == np.int8
    assert order[1, 0] == 1 and order[0, 1] == -1
    assert order[4, 3] == 1 and order[3, 4] == -1
    assert order[2, 0] == 0 and (order == -order.T).all()
    assert int((order != 0).sum()) == 2 * 4


def test_rotation_is_proper_and_orthogonal() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = fd.random_rotation(rng)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-5)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-5)


def test_rotations_are_spread_over_the_sphere() -> None:
    rng = np.random.default_rng(2)
    axis = np.stack(
        [fd.random_rotation(rng) @ np.array([0.0, 0.0, 1.0]) for _ in range(4000)]
    )
    assert np.abs(axis.mean(0)).max() < 0.05
    assert abs(axis[:, 2].std() - 1 / np.sqrt(3)) < 0.02


def test_crop_window_is_all_or_a_window_of_the_crop() -> None:
    rng = np.random.default_rng(3)
    assert fd.crop_window(100, 256, rng) == (0, 100)
    assert fd.crop_window(256, 256, rng) == (0, 256)
    starts = {fd.crop_window(300, 256, rng)[0] for _ in range(200)}
    assert min(starts) >= 0 and max(starts) <= 44 and len(starts) > 20
    assert all(b - a == 256 for a, b in [fd.crop_window(300, 256, rng)])


def test_place_keeps_distances_centres_present_atoms_and_zeroes_absent_ones() -> None:
    it = item(30)
    mask = it["atom37_mask"]
    rot = fd.random_rotation(np.random.default_rng(4))
    out = fd.place(it["coords"], mask, rot)
    assert np.allclose(out[mask].mean(0), 0.0, atol=1e-4)
    assert (out[~mask] == 0).all()
    a, b = it["coords"][mask][:50], out[mask][:50]
    da = np.linalg.norm(a[:, None] - a[None], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None], axis=-1)
    assert np.allclose(da, db, atol=1e-3)


def test_chain_features_without_crop_or_rotation_keep_the_shard() -> None:
    it = item(40)
    f = fd.chain_features(it, np.random.default_rng(0), rotate=False)
    for key in fd.TOKEN_KEYS:
        assert np.array_equal(f[key], it[key])
    assert np.allclose(
        f["coords"][it["atom37_mask"]],
        it["coords"][it["atom37_mask"]] - it["coords"][it["atom37_mask"]].mean(0),
        atol=1e-4,
    )
    assert f["token_bond_path_length"].shape == (40, 40)
    assert f["molecule_iid"].dtype == np.int16 and not f["molecule_iid"].any()
    assert (
        f["molecule_entity_id"].dtype == np.int16
        and (f["molecule_entity_id"] == 1).all()
    )
    assert not f["condition_index-group_1_token"].any()
    assert np.array_equal(f["tok_idx"], np.arange(40))
    assert "esmc" not in f


def test_a_crop_cuts_every_tensor_and_the_esmc_rows_to_the_same_residues() -> None:
    n = 300
    it = item(n, seed=5)
    rows = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 8), dtype=np.float32)
    f = fd.chain_features(it, np.random.default_rng(6), esmc=rows)
    assert f["residue_type"].shape == (fd.CROP,)
    assert f["token_bond_path_length"].shape == (fd.CROP, fd.CROP)
    first = int(f["esmc"][0, 0])
    assert np.array_equal(
        f["esmc"][:, 0].astype(int), np.arange(first, first + fd.CROP)
    )
    assert np.array_equal(
        f["residue_type"], it["residue_type"][first : first + fd.CROP]
    )
    assert f["esmc"].dtype == np.float16
    with pytest.raises(ValueError, match="rows"):
        fd.chain_features(it, np.random.default_rng(0), esmc=rows[:-1])


def stream(
    ids: list[str], batch: int = 4, seed: int = 7, offset: int = 0
) -> fd.FlowStream:
    return fd.FlowStream(
        "unused", ids, batch=batch, seed=seed, steps=100, offset=offset
    )


def test_the_stream_is_a_function_of_the_seed_and_the_step() -> None:
    ids = [f"c{i}" for i in range(10)]
    a, b = stream(ids), stream(ids)
    assert [a.chains_of_step(s) for s in range(6)] == [
        b.chains_of_step(s) for s in (0, 1, 2, 3, 4, 5)
    ]
    assert stream(ids, seed=8).chains_of_step(0) != a.chains_of_step(0)
    # a resumed stream continues where the first one was: its step 0 is the first one's step 3
    c = stream(ids, offset=3)
    assert c.chains_of_step(3) == a.chains_of_step(3)


def test_every_chain_is_read_once_in_each_epoch() -> None:
    ids = [f"c{i}" for i in range(12)]
    s = stream(ids, batch=4)
    first = list(itertools.chain.from_iterable(s.chains_of_step(k) for k in range(3)))
    second = list(
        itertools.chain.from_iterable(s.chains_of_step(k) for k in range(3, 6))
    )
    assert sorted(first) == sorted(ids) and sorted(second) == sorted(ids)
    assert first != second


def test_a_batch_across_an_epoch_boundary_takes_the_end_of_one_order_and_the_start_of_the_next() -> (
    None
):
    ids = [f"c{i}" for i in range(10)]
    s = stream(ids, batch=4)
    # step 2 takes positions 8..11: the last two chains of epoch 0's order and the first two of epoch 1's
    names = s.chains_of_step(2)
    order0, order1 = s._order(0).copy(), s._order(1).copy()
    assert names == [ids[int(i)] for i in (*order0[8:], *order1[:2])]
