"""The arithmetic of the data-path check (`src/foldspace/rfd4/datapath_stats.py`): the choice of chains, the scores of a reconstruction and the
summary of a latent, each on inputs whose answer is known."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.datapath_stats as ds

pytestmark = pytest.mark.offline


def table(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows, lengths = [], {}
    for i in range(n):
        name = f"{1000 + i}_1"
        length = int(rng.integers(60, 301))
        lengths[f"{1000 + i}_1"] = length
        rows.append(
            {
                "entry": str(1000 + i),
                "entity": "1",
                "nonstandard_in_sequence": str(int(rng.random() < 0.06)),
                "frame_breaks": str(int(rng.random() < 0.15)),
                "polymer_chains": str(1 + int(rng.random() < 0.4)),
                "coverage": str(round(float(rng.uniform(0.85, 1.0)), 3)),
            }
        )
        assert name in lengths
    return rows, lengths


def test_the_choice_is_a_function_of_the_table_and_the_seed_and_has_no_repeats():
    rows, lengths = table()
    first = ds.choose_chains(rows, lengths, per_bin=8, stress=4, seed=1)
    again = ds.choose_chains(rows, lengths, per_bin=8, stress=4, seed=1)
    other = ds.choose_chains(rows, lengths, per_bin=8, stress=4, seed=2)
    assert first == again and first["ids"] != other["ids"]
    assert len(first["ids"]) == len(set(first["ids"])) == 4 * 8 + 4 * 4
    order = [f"{r['entry']}_{r['entity']}" for r in rows]
    assert first["ids"] == sorted(first["ids"], key=order.index)


def test_each_stratum_and_each_pool_holds_what_it_is_named_for():
    rows, lengths = table()
    chosen = ds.choose_chains(rows, lengths, per_bin=8, stress=4, seed=3)
    by_id = {f"{r['entry']}_{r['entity']}": r for r in rows}
    seen = {}
    for name, why in chosen["why"].items():
        seen.setdefault(why, []).append(name)
        if why.startswith("length"):
            low, high = (int(x) for x in why.removeprefix("length ").split("-"))
            assert low <= lengths[name] <= high
        else:
            assert ds.STRESS[why](by_id[name])
    assert {k: len(v) for k, v in seen.items() if k.startswith("length")} == {
        "length 60-109": 8,
        "length 110-175": 8,
        "length 176-244": 8,
        "length 245-300": 8,
    }
    assert {k: len(v) for k, v in seen.items() if not k.startswith("length")} == {
        "nonstandard": 4,
        "breaks": 4,
        "copies": 4,
        "unobserved": 4,
    }


def test_a_pool_smaller_than_asked_gives_what_it_has():
    rows, lengths = table(n=120)
    for row in rows:
        row["nonstandard_in_sequence"] = "0"
    rows[5]["nonstandard_in_sequence"] = "2"
    chosen = ds.choose_chains(rows, lengths, per_bin=3, stress=4, seed=4)
    assert sum(why == "nonstandard" for why in chosen["why"].values()) <= 1


def test_kabsch_rmsd_is_zero_for_a_rigid_copy_and_exact_for_a_shift_of_one_atom():
    rng = np.random.default_rng(0)
    p = rng.standard_normal((40, 3)) * 10
    theta = 0.7
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    q = p @ rot.T + np.array([3.0, -2.0, 5.0])
    assert ds.kabsch_rmsd(p, q) == pytest.approx(0.0, abs=1e-6)
    # a mirror image is not a rotation: it cannot be superposed exactly
    assert ds.kabsch_rmsd(p * np.array([1, 1, -1]), q) > 1.0
    with pytest.raises(ValueError):
        ds.kabsch_rmsd(p, q[:-1])


def test_residue_scores_split_backbone_and_side_chain_and_ignore_what_the_target_lacks():
    n = 4
    target = np.zeros((n, 37, 3))
    predicted = target.copy()
    mask = np.zeros((n, 37), dtype=bool)
    mask[:, [0, 1, 2, 4]] = True  # backbone everywhere
    mask[:, 5] = True  # one side-chain atom
    mask[3, 5] = False  # glycine-like: no side-chain atom in the last residue
    predicted[:, 5, 0] = 1.0  # every side-chain atom is 1.0 off
    predicted[:, 6, 0] = 50.0  # a slot the target does not have: must not count
    target[:, 7] = np.nan  # and the target of an atom it lacks may be NaN
    residues = np.array([True, True, True, True])
    s = ds.residue_scores(target, predicted, mask, residues)
    assert s["backbone_rms"] == 0.0 and s["side_chain_rms"] == pytest.approx(1.0)
    assert s["side_chain_atoms"] == 3 and s["atoms"] == 4 * 4 + 3
    assert s["all_atom_rms"] == pytest.approx(np.sqrt(3 / 19))
    # per residue: five atoms, one off by 1.0 -> sqrt(1/5); the last has no side chain -> 0
    assert s["per_residue_mean"] == pytest.approx((3 * np.sqrt(1 / 5) + 0.0) / 4)
    only_last = ds.residue_scores(target, predicted, mask, np.array([0, 0, 0, 1], bool))
    assert only_last["all_atom_rms"] == 0.0 and only_last["residues"] == 1
    assert ds.residue_scores(target, predicted, mask, np.zeros(n, bool)) == {
        "residues": 0
    }


def test_presence_f1_counts_only_legal_slots_of_scored_residues():
    legal = np.zeros((3, 37), dtype=bool)
    legal[:, :6] = True
    target = np.zeros((3, 37), dtype=bool)
    target[:, [0, 1, 2]] = True
    predicted = np.zeros((3, 37), dtype=bool)
    predicted[:, [0, 1]] = True  # misses slot 2 everywhere
    predicted[:, 20] = True  # an illegal slot: ignored
    residues = np.array([True, True, False])
    s = ds.presence_scores(target, predicted, legal, residues)
    assert s["precision"] == 1.0 and s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(0.8) and s["target_atoms"] == 6


def test_sequence_scores_are_over_the_residues_scored_with_recall_by_type():
    true = np.array([0, 0, 1, 2, 2, 2])
    predicted = np.array([0, 1, 1, 2, 2, 0])
    s = ds.sequence_scores(
        true, predicted, np.array([1, 1, 1, 1, 1, 0], bool), n_types=4
    )
    assert s["residues"] == 5 and s["accuracy"] == pytest.approx(4 / 5)
    assert s["recall_by_type"] == {0: 0.5, 1: 1.0, 2: 1.0}


def test_latent_summary_on_a_latent_with_known_structure():
    rng = np.random.default_rng(0)
    n, d = 4000, 4
    types = rng.integers(0, 5, size=n)
    centres = rng.standard_normal((5, d)) * 2.0
    mean = centres[types] + 0.1 * rng.standard_normal((n, d))
    mean[:, 3] = mean[:, 2]  # two channels carry the same thing
    log_scale = np.full((n, d), np.log(0.5))
    log_scale[:, 0] = 0.0  # one channel with the prior's own scale
    s = ds.latent_summary(
        mean, log_scale, types, sample=mean + 0.5 * rng.standard_normal((n, d))
    )
    assert s["residues"] == n and s["channels"] == d
    assert np.allclose(s["scale_mean"], [1.0, 0.5, 0.5, 0.5])
    assert s["overall_std_of_mean"] > 1.0
    # the type alone fixes nearly all of the posterior mean here
    assert s["variance_between_residue_types_overall"] > 0.95
    assert all(v > 0.9 for v in s["variance_between_residue_types"])
    # channels 2 and 3 are one direction: correlation 1, and the covariance has at most three directions
    assert s["max_abs_correlation"] == pytest.approx(1.0, abs=1e-6)
    assert s["covariance_eigenvalues"][-1] == pytest.approx(0.0, abs=1e-6)
    assert s["participation_ratio"] <= 3.0 + 1e-6
    # KL of a channel with scale 1 and a mean spread of ~2 is about m^2/2; of the narrow ones the scale term adds
    assert s["active_channels"] == 4
    assert s["kl_per_channel"][0] == pytest.approx(
        0.5 * float((mean[:, 0] ** 2).mean()), rel=1e-6
    )
    assert len(s["std_of_sample"]) == d


def test_a_latent_that_is_noise_has_no_residue_identity_and_no_active_channels():
    rng = np.random.default_rng(1)
    n, d = 5000, 3
    types = rng.integers(0, 20, size=n)
    s = ds.latent_summary(rng.standard_normal((n, d)) * 0.01, np.zeros((n, d)), types)
    assert s["variance_between_residue_types_overall"] < 0.02
    assert s["active_channels"] == 0
