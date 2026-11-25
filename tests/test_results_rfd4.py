"""The reported results of the RFD4 experiments, read from the records committed under `experiments/rfd4/`.

Each test states a number of the experiment READMEs and reads it from the record that holds it. Nothing is recomputed from
weights; the records are the readings that were made, and these tests say that they are the ones the text quotes. Standard
library only.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

ROOT = Path(__file__).resolve().parents[1]
ADAPT =ROOT / "experiments/rfd4/latent_adaptation"
HYBRID = ROOT / "experiments/rfd4/hybrid_latents"
REPLACE = ROOT / "experiments/rfd4/latent_replacement"
RETARGET = ADAPT / "flow_retarget"

NATIVE, ESMC, SCRATCH, INPLACE, TYPE_ONLY = (
    "native_continuation", "esmc_residual8", "esmc_residual8_scratch", "inplace_hybrid", "type_only_residual8",
)
ARMS = (NATIVE, ESMC, SCRATCH, INPLACE, TYPE_ONLY)
PRIMARY, REPLICATION = "design_panel_primary", "design_panel_replication"


def load(path: Path):
    return json.loads(path.read_text())


def decoder(name: str) -> dict:
    """The final dev reading of an adapted decoder, on residues with every atom."""
    return load(ADAPT / "decoder_adaptation" / name / "summary.json")["end"]["complete"]


# ----------------------------------------------------------------------------------------- latent adaptation
def test_the_esmc_only_decoder_reads_at_0_707_angstrom_and_recovers_99_85_percent_of_the_types():
    first, second = decoder("esmc_only_seed1"), decoder("esmc_only_seed2")
    assert (first["per_residue_rmsd"], second["per_residue_rmsd"]) == pytest.approx((0.7118, 0.7015), abs=5e-5)
    assert (first["per_residue_rmsd"] + second["per_residue_rmsd"]) / 2 == pytest.approx(0.707, abs=5e-4)
    assert (first["sequence_accuracy"] + second["sequence_accuracy"]) / 2 == pytest.approx(0.9985, abs=5e-5)


def test_a_fixed_code_of_the_residue_types_recovers_every_type_and_esmc_context_is_worth_0_48_angstrom():
    fixed = decoder("type_only")
    assert fixed["per_residue_rmsd"] == pytest.approx(1.183, abs=5e-4)
    assert fixed["sequence_accuracy"] == 1.0
    esmc = (decoder("esmc_only_seed1")["per_residue_rmsd"] + decoder("esmc_only_seed2")["per_residue_rmsd"]) / 2
    assert fixed["per_residue_rmsd"] - esmc == pytest.approx(0.48, abs=5e-3)


def test_with_eight_residual_channels_the_type_code_reconstructs_as_well_as_esmc_code():
    assert decoder("esmc_residual8")["per_residue_rmsd"] == pytest.approx(0.193, abs=5e-4)
    assert decoder("type_only_residual8")["per_residue_rmsd"] == pytest.approx(0.170, abs=5e-4)
    assert decoder("esmc_residual4")["per_residue_rmsd"] == pytest.approx(0.292, abs=5e-4)
    assert decoder("type_only_residual8")["per_residue_rmsd"] < decoder("esmc_residual8")["per_residue_rmsd"]


def test_the_inplace_layout_needs_its_code_channels_for_the_sequence():
    layout, ablated = decoder("inplace_hybrid"), decoder("inplace_hybrid_code_ablated")
    assert layout["per_residue_rmsd"] == pytest.approx(0.198, abs=5e-4)
    assert layout["sequence_accuracy"] == pytest.approx(0.9995, abs=5e-5)
    assert ablated["per_residue_rmsd"] == pytest.approx(0.357, abs=5e-4)
    assert ablated["sequence_accuracy"] == pytest.approx(0.795, abs=5e-4)


def test_the_decoders_were_trained_alike():
    for name in (
        "esmc_only_seed1", "esmc_only_seed2", "esmc_residual4", "esmc_residual8", "type_only", "type_only_residual8",
        "inplace_hybrid", "inplace_hybrid_code_ablated",
    ):
        summary = load(ADAPT / "decoder_adaptation" / name / "summary.json")
        assert (summary["steps"], summary["epochs"], summary["train_chains"], summary["dev_chains"], summary["batch"]) == (
            6144, 24, 2048, 256, 8,
        )
        assert (summary["lr"], summary["lr_new"], summary["noise"]) == (1e-4, 1e-3, 0.2)


def test_reading_adapters_in_front_of_the_frozen_decoder_do_not_reconstruct():
    reading = {
        name: load(ADAPT / "reading_adapters" / name / "summary.json")["end"]["complete"]
        for name in ("backbone_only", "residue_type", "esmc")
    }
    assert reading["backbone_only"]["per_residue_rmsd"] == pytest.approx(1.116, abs=5e-4)
    assert reading["residue_type"]["per_residue_rmsd"] == pytest.approx(1.322, abs=5e-4)
    assert reading["esmc"]["per_residue_rmsd"] == pytest.approx(1.229, abs=5e-4)
    assert reading["esmc"]["sequence_accuracy"] == pytest.approx(0.576, abs=5e-4)
    native = load(ADAPT / "reading_adapters/native_encoder/summary.json")["dev"]["complete"]
    assert native["per_residue_rmsd"] == pytest.approx(0.069, abs=5e-4)


def test_the_release_decoder_reads_16_channels_and_the_hybrid_layout_keeps_14_of_them():
    audit = load(ADAPT / "channel_audit.json")
    active = audit["active_channels"]
    assert len(active) == 16 and len(set(range(32)) - set(active)) == 16
    assert sorted(set(active) - {0, 10}) == [1, 4, 5, 8, 13, 14, 15, 16, 17, 22, 25, 27, 28, 31]


def test_the_flow_batch_equals_the_release_pipelines_on_all_but_an_entity_numbering():
    chains = load(ADAPT / "flow_batch_check.json")["chains"]
    assert len(chains) == 8
    for chain in chains:
        compared = chain["compared"]
        assert chain["differing"] == ["molecule_entity_id"] and len(compared) == 16
        assert sum(v == "0 entries differ" for v in compared.values()) == 14
        assert compared["coords"].startswith("max abs difference")
        assert float(compared["coords"].rsplit(" ", 1)[1]) < 4e-5
        assert compared["molecule_entity_id"].endswith("(built here: [0], the pipeline's: [1])")


def test_the_data_path_reproduces_the_native_autoencoder_at_0_07_angstrom():
    summary = load(ADAPT / "data_path/summary.json")
    assert summary["ok"] == summary["chains"] == 272 and summary["failed"] == {}
    mean = summary["reconstruction"]["random_strata"]["mean/complete"]["per_residue_rmsd_mean"]
    assert round(mean, 2) == 0.07


def test_the_five_flow_arms_have_one_budget_and_differ_in_latent_and_initialisation():
    summaries = {name: load(RETARGET / name / "summary.json") for name in ARMS}
    budget = {
        (s["updates_total"], s["example_slots"], s["distinct_presentations"], s["residues"], s["residues_squared"])
        for s in summaries.values()
    }
    assert budget == {(800, 25600, 6400, 1172823, 235901619)}
    assert {s["args"]["seed"] for s in summaries.values()} == {20261111}
    assert {name: s["args"]["init"] for name, s in summaries.items()} == {
        "native_continuation": "pretrained", "esmc_residual8": "pretrained", "esmc_residual8_scratch": "scratch",
        "inplace_hybrid": "pretrained", "type_only_residual8": "pretrained",
    }
    assert {name: s["args"]["target"] for name, s in summaries.items()} == {
        "native_continuation": "native", "esmc_residual8": "esmc", "esmc_residual8_scratch": "esmc",
        "inplace_hybrid": "esmc", "type_only_residual8": "fixed",
    }
    assert all(2.4 < s["gpu_hours"] < 2.7 for s in summaries.values())


def test_the_pretrained_backbone_prior_carries_over_and_a_fresh_flow_does_not():
    comparison = load(RETARGET / "comparison_800.json")
    assert comparison["transfer"]["observed"] is True
    skill = comparison["after_updates"]  # the Cα skill over the noisy-coordinate predictor, by arm and noise level
    for t in ("0.3", "0.5", "0.7", "0.9"):
        for arm in (ESMC, INPLACE, TYPE_ONLY):
            assert abs(skill[arm][t]["bb_ca"] - skill[NATIVE][t]["bb_ca"]) <= 0.05
    assert skill[SCRATCH]["0.5"]["bb_ca"] == pytest.approx(0.781, abs=5e-4)
    assert min(skill[arm]["0.5"]["bb_ca"] for arm in (NATIVE, ESMC, INPLACE, TYPE_ONLY)) > 0.96


def test_only_the_native_continuation_samples_at_800_updates():
    samples = load(RETARGET / "comparison_800.json")["samples"]
    both = {arm: samples[f"{arm}/samples"]["chains_and_sequence"] for arm in ARMS}
    assert both == pytest.approx({NATIVE: 0.875, ESMC: 0.0625, SCRATCH: 0.0, INPLACE: 0.1875, TYPE_ONLY: 0.0})
    assert {arm: samples[f"{arm}/samples"]["samples"] for arm in ARMS} == dict.fromkeys(ARMS, 32)
    assert {arm: samples[f"{arm}/samples"]["feasible"] for arm in ARMS} == {
        NATIVE: True, ESMC: False, SCRATCH: False, INPLACE: False, TYPE_ONLY: False,
    }


def test_the_share_that_passes_both_rules_does_not_rise_at_1600_updates():
    continued = (ESMC, INPLACE, TYPE_ONLY)
    samples = load(RETARGET / "continued_1600/comparison.json")["samples"]
    both = {arm: samples[f"{arm}/samples"]["chains_and_sequence"] for arm in continued}
    assert both == pytest.approx({ESMC: 1 / 96, INPLACE: 13 / 96, TYPE_ONLY: 0.0})
    assert all(samples[f"{arm}/samples"]["samples"] == 96 for arm in continued)
    assert not any(samples[f"{arm}/samples"]["feasible_joint"] for arm in continued)
    budget = load(RETARGET / "continued_1600/budget.json")
    assert {arm: v["total"]["updates"] for arm, v in budget.items()} == dict.fromkeys(continued, 1600)


# ------------------------------------------------------------------------------------------------ hybrid latents
def test_the_control_that_removes_context_is_valid_before_it_is_trained():
    checks = load(HYBRID / "type_shuffled_control/checks.json")
    assert checks["passes"] is True
    assert all(abs(x) < checks["no_context"]["limit"] for x in checks["no_context"]["context_r2_both"])
    assert all(x > 0.09 for x in checks["no_context"]["reference_context_r2_both"])
    assert checks["type_structure_kept"]["alignment_gap"] < checks["type_structure_kept"]["alignment_limit"]
    assert all(x < checks["type_structure_kept"]["share_limit"] for x in checks["type_structure_kept"]["within_type_share_gaps"])


def test_designable_and_P_counts_on_the_two_design_panels():
    counts = load(HYBRID / "base_vs_release.json")["counts"]
    expected = {
        PRIMARY: {
            "native_continuation": (74, 44), "native_identity_control": (79, 41),
            "contextual_hybrid": (80, 65), "type_shuffled_hybrid": (79, 54),
            "contextual_hybrid_seed2": (67, 44), "type_shuffled_hybrid_seed2": (71, 63),
        },
        REPLICATION: {
            "native_continuation": (74, 48), "native_identity_control": (71, 44),
            "contextual_hybrid": (71, 57), "type_shuffled_hybrid": (71, 53),
            "contextual_hybrid_seed2": (68, 41), "type_shuffled_hybrid_seed2": (69, 60),
        },
    }
    for panel, arms in expected.items():
        for arm, (designable, p) in arms.items():
            assert (counts[panel][arm]["designable"], counts[panel][arm]["p"]) == (designable, p), (panel, arm)
    assert (counts[PRIMARY]["ProteinMPNN"]["p"], counts[REPLICATION]["ProteinMPNN"]["p"]) == (59, 54)


def test_the_sign_of_the_context_effect_follows_the_training_seed():
    cells = load(HYBRID / "design_panels.json")["rules"]["net_by_seed_and_panel"]["cells"]
    net = {cell: v["net_contextual_hybrid_minus_type_shuffled_hybrid"] for cell, v in cells.items()}
    assert net == {
        f"s1,{PRIMARY}": 11, f"s1,{REPLICATION}": 4, f"s2,{PRIMARY}": -19, f"s2,{REPLICATION}": -19,
    }
    assert all(v["pairs"] == 96 for v in cells.values())
    assert cells[f"s1,{PRIMARY}"]["net_interval_95"] == [2.2, 15.7]
    assert cells[f"s2,{PRIMARY}"]["net_interval_95"][1] < 0 and cells[f"s2,{REPLICATION}"]["net_interval_95"][1] < 0
    pooled = load(HYBRID / "design_panels.json")["rules"]["net_by_seed_and_panel"]["pooled_seed1_primary_and_seed2_replication"]
    assert pooled["net_contextual_hybrid_minus_type_shuffled_hybrid"] == -8
    assert pooled["net_interval_95"][0] < 0 < pooled["net_interval_95"][1]
    assert pooled["p"] == pytest.approx(0.30, abs=5e-3)


def test_in_free_generation_the_contextual_hybrid_is_below_its_control_and_the_continuation_at_both_seeds():
    usable = load(HYBRID / "unconditional_generation.json")["usable_of_192_at_scale_1"]
    assert usable == {
        "native_continuation": 142, "contextual_hybrid": 72, "native_identity_control": 133,
        "type_shuffled_hybrid": 117, "contextual_hybrid_seed2": 43, "type_shuffled_hybrid_seed2": 106,
    }
    assert usable["contextual_hybrid"] < usable["type_shuffled_hybrid"] < usable["native_continuation"]
    assert usable["contextual_hybrid_seed2"] < usable["type_shuffled_hybrid_seed2"]


# ------------------------------------------------------------------------------------------ latent replacement
def test_esmc_only_code_has_context_that_the_native_latent_lacks_and_type_only_has_none():
    latents = load(REPLACE / "latent_statistics.json")["latents"]
    assert latents["native"]["context"]["both"] == pytest.approx(0.011, abs=5e-4)
    assert latents["esmc_only_seed1"]["context"]["both"] == pytest.approx(0.349, abs=5e-4)
    assert latents["esmc_only_seed2"]["context"]["both"] == pytest.approx(0.336, abs=5e-4)
    assert latents["type_only_seed1"]["context"] is None and latents["type_only_seed2"]["context"] is None
    assert latents["native"]["participation_ratio"] == pytest.approx(10.6, abs=0.05)
    assert latents["esmc_only_seed1"]["participation_ratio"] == pytest.approx(22.4, abs=0.05)
    assert latents["type_only_seed1"]["participation_ratio"] == pytest.approx(11.6, abs=0.05)


def test_the_decoders_of_the_flow_experiments_reconstruct_esmc_only_better_than_the_type_only_control():
    latents = load(REPLACE / "latent_statistics.json")["latents"]
    reading = {name: latents[name]["recorded_reading"] for name in latents if name != "native"}
    assert reading["esmc_only_seed1"]["per_residue_rmsd"] == pytest.approx(0.712, abs=5e-4)
    assert reading["esmc_only_seed2"]["per_residue_rmsd"] == pytest.approx(0.702, abs=5e-4)
    assert reading["type_only_seed1"]["per_residue_rmsd"] == pytest.approx(1.202, abs=5e-4)
    assert reading["type_only_seed2"]["per_residue_rmsd"] == pytest.approx(1.166, abs=5e-4)
    assert reading["type_only_seed1"]["sequence_accuracy"] == reading["type_only_seed2"]["sequence_accuracy"] == 1.0
    for seed in ("seed1", "seed2"):
        assert reading[f"esmc_only_{seed}"]["per_residue_rmsd"] < reading[f"type_only_{seed}"]["per_residue_rmsd"]
    esmc = (reading["esmc_only_seed1"]["per_residue_rmsd"] + reading["esmc_only_seed2"]["per_residue_rmsd"]) / 2
    type_only = (reading["type_only_seed1"]["per_residue_rmsd"] + reading["type_only_seed2"]["per_residue_rmsd"]) / 2
    assert type_only - esmc == pytest.approx(0.48, abs=5e-3)


def test_the_type_only_decoder_renders_bond_lengths_that_are_not_native():
    readings = load(REPLACE / "decoder_geometry.json")["readings"]
    assert readings["esmc_only_seed1/mean"]["n_ca"] == pytest.approx(1.46, abs=5e-3)
    assert readings["esmc_only_seed2/mean"]["n_ca"] == pytest.approx(1.455, abs=5e-3)
    assert readings["type_only_seed1/mean"]["n_ca"] == pytest.approx(1.27, abs=5e-3)
    assert readings["type_only_seed2/mean"]["n_ca"] == pytest.approx(1.31, abs=5e-3)


def test_neither_the_esmc_only_latent_nor_its_type_only_control_is_generated():
    pilot = load(REPLACE / "generation_1600.json")
    assert pilot["native"]["unconditional_usable_at_1"] == 142
    assert (pilot["seeds"]["s1"]["esmc_only_unconditional_usable_at_1"], pilot["seeds"]["s2"]["esmc_only_unconditional_usable_at_1"]) == (2, 43)
    assert (pilot["seeds"]["s1"]["type_only_unconditional_usable_at_1"], pilot["seeds"]["s2"]["type_only_unconditional_usable_at_1"]) == (0, 0)
    assert pilot["outcome"] == "neither_generates"


def test_more_updates_do_not_raise_the_esmc_only_flow():
    extended = load(REPLACE / "generation_extended.json")
    assert extended["seeds"]["s1"]["unconditional_usable_by_update"] == {"1600": 2, "2400": 11, "3200": 0}
    assert extended["seeds"]["s2"]["unconditional_usable_by_update"] == {"1600": 43, "2400": 17, "3200": 1}
    assert extended["seeds"]["s1"]["by_rule"]["3200"]["samples"] == 192


def test_the_generated_code_loses_variance_and_calibration_restores_the_sequences_and_not_the_folds():
    calibration = load(REPLACE / "calibration.json")
    traces = calibration["traces"]
    native = traces["native_continuation s1/1600 scale 1"]
    assert native["final_state_variance"] == pytest.approx(native["v1"], rel=1e-9)
    assert traces["esmc_only s2/1600 scale 1"]["v1"] == pytest.approx(33.72, abs=5e-3)
    assert traces["esmc_only s2/1600 scale 1"]["final_state_variance"] == pytest.approx(13.45, abs=5e-3)
    assert traces["esmc_only s2/3200 scale 1"]["final_state_variance"] == pytest.approx(2.28, abs=5e-3)
    assert traces["esmc_only s1/3200 scale 1"]["final_state_variance"] == pytest.approx(1.96, abs=5e-3)
    late = calibration["calibrated"]["s2/3200"]
    assert (late["by_rule_calibrated"]["sequence"], late["by_rule_uncalibrated"]["sequence"]) == (114, 1)
    assert (late["by_rule_calibrated"]["designable"], late["by_rule_uncalibrated"]["designable"]) == (6, 4)
    assert (late["usable_calibrated"], late["usable_uncalibrated"]) == (3, 1)
    early = calibration["calibrated"]["s2/1600"]
    assert (early["usable_calibrated"], early["usable_uncalibrated"]) == (51, 43)
