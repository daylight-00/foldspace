"""The parts of the data-path check that need no model (`src/foldspace/rfd4/ae_datapath.py`): the corpus's rule for the run of residues, and
the summaries over a set of chains, which count a chain that fails before the networks as a chain that did not map."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.ae_datapath as dp

pytestmark = pytest.mark.offline


def test_the_run_is_the_longest_consecutive_complete_one_and_the_earlier_of_equals():
    ids = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 12])
    ok = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    # complete: 1-3, 5, 7-9, 12  ->  the runs 1-3 and 7-9 have three residues each; the first is kept
    assert dp.longest_complete_run(ids, ok) == [1, 2, 3]
    ok[0] = False  # now 2-3 (two) against 7-9 (three)
    assert dp.longest_complete_run(ids, ok) == [7, 8, 9]
    assert dp.longest_complete_run(ids, np.zeros(10, bool)) == []
    # a gap in the numbering ends a run even where both neighbours are complete
    assert dp.longest_complete_run(np.array([1, 2, 4, 5, 6]), np.ones(5, bool)) == [
        4,
        5,
        6,
    ]


def record(why, ok=True, one_to_one=True, mapping_ok=True, given=0, **extra):
    base = {
        "id": "x",
        "why": why,
        "ok": ok,
        "one_to_one": one_to_one,
        "mapping_ok": mapping_ok,
    }
    if ok:
        base.update(
            {
                "same_length": True,
                "same_sequence": given == 0,
                "units_ok": True,
                "residue_types_in_corpus_alphabet": True,
                "backbone_rmsd_to_corpus": 0.001,
                "c_seq_given_by_the_pipeline": given,
                "atoms_without_coordinates": 0,
                "masks": {"padding_mask": True, "supports_local_latent": True},
                "residues_complete": 0.99,
                "missing_legal_by_slot": [0] * 37,
            }
        )
    base.update(extra)
    return base


def test_only_the_length_strata_speak_for_the_corpus():
    assert dp.is_random_stratum({"why": "length 60-109"})
    assert not dp.is_random_stratum({"why": "copies"})


def test_a_chain_that_failed_before_the_networks_counts_as_unmapped():
    chains = [
        record("length 60-109"),
        record("length 60-109", ok=False, one_to_one=False, mapping_ok=False),
    ]
    block = dp.mapping_block(chains)
    assert block["chains"] == 2 and block["mapping_ok"] == 0.5
    assert block["failed_before_the_autoencoder"] == 1
    summary = dp.mapping_summary(chains + [record("copies", given=2)])
    assert summary["random_strata"]["chains"] == 2 and summary["all"]["chains"] == 3
    assert (
        summary["by_stratum"]["copies"]["chains_with_a_condition_given_by_the_pipeline"]
        == 1.0
    )
    assert summary["legal_slots_without_an_atom_by_slot"] == [0] * 37


def structure(residues, per_residue, backbone, side, total):
    return {
        "residues": residues,
        "per_residue_mean": per_residue,
        "per_residue_median": per_residue,
        "backbone_rms": backbone,
        "side_chain_rms": side,
        "all_atom_rms": total,
    }


def test_reconstruction_is_pooled_by_residues_and_split_by_completeness():
    def scored(residues, rmsd, acc, f1):
        return {
            "mean_structure": structure(residues, rmsd, rmsd, rmsd, rmsd),
            "mean_sequence": {"residues": residues, "accuracy": acc},
            "mean_presence": {"f1": f1, "target_atoms": 10},
            "leak_sequence": {"residues": residues, "accuracy": 0.99},
            "leak_decode_sequence": {"residues": residues, "accuracy": 1.0},
            "clean_decode_sequence": {"residues": residues, "accuracy": acc},
        }

    a = record("length 60-109", **scored(100, 0.10, 1.0, 1.0))
    b = record("length 110-175", **scored(300, 0.20, 0.9, 0.8))
    summary = dp.reconstruction_summary([a, b])["random_strata"]
    pooled = summary["mean/all"]
    assert pooled["residues"] == 400
    # weighted by residues: (100 * 0.10 + 300 * 0.20) / 400
    assert pooled["per_residue_rmsd_mean"] == pytest.approx(0.175)
    assert pooled["sequence_accuracy"] == pytest.approx((100 * 1.0 + 300 * 0.9) / 400)
    # no residue was scored as `complete` or `truncated` in these records
    assert summary["mean/complete"] is None and summary["mean/truncated"] is None
    assert summary["leak_control"]["decode_given_the_true_sequence"] == 1.0
    assert summary["leak_control"]["decode_given_nothing"] == pytest.approx(0.95)
