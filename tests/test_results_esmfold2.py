"""The central results of the ESMFold2 experiments, read from the records they are published as.

Every number stated in `experiments/esmfold2/*/README.md` is asserted here from a record of the same directory: the audit and
the parity and contribution readings of the interface, the accuracy of the sequence heads and what a decoded sequence and a
perturbed single do to the fold, and the folds of the generated sequences, their shuffles and the natural reference chains. The
summary of each generated set is recomputed from its per-sample records by `foldspace.esmfold2.seqeval summarize` and must equal
the published one. Standard library and numpy only; nothing is folded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

ROOT = Path(__file__).resolve().parents[1]
INTERFACE = ROOT / "experiments/esmfold2/interface_validation/results"
DECODING = ROOT / "experiments/esmfold2/sequence_decoding/results"
GENERATION = ROOT / "experiments/esmfold2/latent_generation/results"
ARMS = ("standardized", "whitened", "whitened_with_sequence_loss")


def read(path: Path):
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------------------------- the interface
def test_the_audit_of_the_training_singles() -> None:
    audit = read(INTERFACE / "target_audit.json")
    assert (audit["chains"], audit["rows"], audit["dim"]) == (17063, 3232149, 256)
    assert audit["scale"]["dead_channels"] == 0
    assert round(audit["scale"]["std"]["min"], 1) == 8.2
    assert round(audit["scale"]["std"]["max"], 1) == 25.9
    skew = audit["shape"]["skewness"]
    assert max(abs(skew["min"]), abs(skew["max"])) < 0.254
    assert {k: v for k, v in audit["spectrum"]["covariance"]["components_for"].items()} == {
        "0.90": 78,
        "0.95": 105,
        "0.99": 162,
    }


@pytest.mark.parametrize("card", ["rtx6000ada", "blackwell"])
def test_injecting_the_single_is_the_same_computation_in_deterministic_mode(card: str) -> None:
    record = read(INTERFACE / "parity" / card / "parity_det.json")
    assert record["settings"]["deterministic"] is True
    verdict = record["verdict"]
    assert verdict["all_routes_within"] and verdict["all_tensors_within"]
    assert verdict["all_roundtrips_exact"] and verdict["all_observed_exact"]


def test_the_fold_follows_the_single_of_another_protein() -> None:
    audit = read(INTERFACE / "contribution" / "audit.json")
    assert (audit["pairs"], audit["swaps"]) == (24, 48)
    counts = audit["classification_by_threshold"]["global_kabsch"]["0.5"]
    assert (counts["follow_the_single"], counts["keep_the_sequences_fold"], counts["neither"], counts["both"]) == (
        23,
        3,
        22,
        0,
    )
    assert round(audit["ambiguity"]["tm_between_the_two_deposited_structures"]["median"], 2) == 0.13
    first = read(INTERFACE / "contribution" / "contribution_det.json")
    replicate = read(INTERFACE / "contribution" / "contribution_det_replicate.json")
    assert (first["summary"]["pairs"], replicate["summary"]["pairs"]) == (8, 16)


# ------------------------------------------------------------------------------------------ sequence decoding
def test_the_accuracy_of_the_sequence_heads_on_clusters_they_never_saw() -> None:
    summary = read(DECODING / "decoders" / "summary.json")
    test = summary["test"]

    def cluster_weighted(name: str) -> float:
        return test[name]["summary"]["cluster_weighted"]["acc"]

    assert round(cluster_weighted("contextual-s0"), 4) == 0.9912
    assert round(cluster_weighted("linear-s0"), 4) == 0.9850
    assert round(cluster_weighted("lda"), 4) == 0.9750
    scored = test["contextual-s0"]["summary"]
    assert (scored["chains"], scored["clusters"], scored["residues"]) == (1578, 400, 298386)
    # only the head carried forward is scored on the test split, once
    assert set(test) == {"contextual-s0", "linear-s0", "lda"}


def test_the_contextual_head_is_carried_on_dev_at_both_seeds() -> None:
    summary = read(DECODING / "decoders" / "summary.json")
    dev = {name: arm["full"]["best"]["cw_acc"] for name, arm in summary["arms"].items()}
    assert round(dev["contextual-s0"], 4) == 0.9909
    assert round(dev["contextual-s1"], 4) == 0.9913
    assert round(dev["linear-s0"], 4) == 0.9844
    gains = summary["chosen"]["gains_over_linear"]
    assert round(gains["contextual-s0"] * 100, 3) == 0.655 and round(gains["contextual-s1"] * 100, 3) == 0.688
    assert all(g >= 0.002 for g in gains.values()) and summary["chosen"]["contextual_adopted"] is True
    assert summary["arms"]["contextual-s0"]["full"]["parameters"] == 3246868
    assert summary["arms"]["linear-s0"]["full"]["parameters"] == 5140


def test_a_decoded_sequence_folds_like_the_original() -> None:
    panel = read(DECODING / "structure_check" / "panel_summary.json")
    assert panel["passes"] is True and panel["chains"] == 24 and panel["stable_chains"] == 20
    assert panel["reencoded"]["share_keeping_stable"] == 1.0
    assert round(panel["reencoded"]["tm_teacher_median_stable"], 4) == 0.9989
    assert round(panel["reencoded"]["tm_teacher_median_stable_with_errors"], 4) == 0.9977
    assert round(panel["twin"]["reencoded_tm_teacher_median_stable_with_errors"], 4) == 0.9961
    assert panel["atlasfold"]["sctm_decoded_median"] == pytest.approx(0.96505, abs=1e-9)
    assert panel["atlasfold"]["sctm_original_median"] == pytest.approx(0.9707, abs=1e-9)
    assert panel["exactly_recovered"] == 10


def test_the_fold_tolerates_errors_that_the_head_cannot_read_through() -> None:
    standard = read(DECODING / "robustness" / "standard_levels" / "robustness_summary.json")
    folds = standard["folds"]["conditions"]
    assert len(folds) == 18 and standard["folds"]["stable_chains"] == 20
    assert min(c["keeps"] for c in folds.values()) >= 0.95  # at least 19 of 20 chains keep their fold
    decode = standard["decode"]["conditions"]
    assert round(decode["noise_0.1"]["contextual-s0"], 3) == 0.987
    assert round(decode["noise_0.4"]["contextual-s0"], 3) == 0.832
    assert round(decode["noise_0.8"]["contextual-s0"], 3) == 0.467
    assert round(decode["rank_128"]["contextual-s0"], 3) == 0.898
    assert round(decode["rank_16"]["contextual-s0"], 3) == 0.200
    assert round(decode["dropout_0.3"]["contextual-s0"], 3) == 0.715
    coarse = read(DECODING / "robustness" / "coarse_levels" / "robustness_summary.json")["folds"]["conditions"]
    assert coarse["shrink_1"]["keeps"] == 0.0 and coarse["dropout_0.9"]["keeps"] == 0.0
    assert coarse["rank_1"]["keeps"] == 0.05 and coarse["rank_2"]["keeps"] == 0.1


# ------------------------------------------------------------------------------------------ latent generation
def _summarize(fold_dir: Path, tmp_path: Path, novelty: Path | None = None, novelty_set: str | None = None, direct: Path | None = None) -> dict:
    out = tmp_path / "summary.json"
    command = [sys.executable, "-m", "foldspace.esmfold2.seqeval", "summarize", "--fold-dir", str(fold_dir), "--out", str(out)]
    if novelty is not None:
        command += ["--novelty", str(novelty), "--novelty-set", str(novelty_set)]
    if direct is not None:
        command += ["--direct", str(direct)]
    done = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return read(out)


@pytest.mark.parametrize("arm", ARMS)
def test_the_summary_of_each_arm_is_what_its_records_give(arm: str, tmp_path: Path) -> None:
    base = GENERATION / "pilot_seed0"
    got = _summarize(
        base / arm / "reencoded_fold",
        tmp_path,
        novelty=base / "reencoded_novelty.json",
        novelty_set=arm,
        direct=base / arm / "direct_fold",
    )
    assert got == read(base / arm / "reencoded_fold" / "summary.json")


def test_the_summary_of_the_second_seed_is_what_its_records_give(tmp_path: Path) -> None:
    base = GENERATION / "pilot_seed1"
    got = _summarize(
        base / "standardized" / "reencoded_fold",
        tmp_path,
        novelty=base / "reencoded_novelty.json",
        novelty_set="standardized",
    )
    assert got == read(base / "standardized" / "reencoded_fold" / "summary.json")


def test_the_summaries_of_the_controls_are_what_their_records_give(tmp_path: Path) -> None:
    controls = GENERATION / "controls"
    got = _summarize(controls / "protgpt2", tmp_path)
    assert got == read(controls / "protgpt2" / "summary.json")
    novelty = controls / "sequence_only" / "novelty.json"
    for name in ("seed0_temperature0.8", "seed0_temperature1.0", "seed1_temperature0.8", "seed1_temperature1.0"):
        got = _summarize(controls / "sequence_only" / name, tmp_path, novelty=novelty, novelty_set=name)
        assert got == read(controls / "sequence_only" / name / "summary.json")
    got = _summarize(controls / "esmc_unmasking", tmp_path, novelty=novelty, novelty_set="esmc")
    assert got == read(controls / "esmc_unmasking" / "summary.json")


def test_no_sample_of_the_pilot_is_a_success_on_either_route() -> None:
    direct = read(GENERATION / "pilot_seed0" / "direct_summary.json")["arms"]
    total = 0
    for arm in ARMS:
        reencoded = read(GENERATION / "pilot_seed0" / arm / "reencoded_fold" / "summary.json")
        assert reencoded["samples"] == direct[arm]["samples"] == 48
        assert reencoded["success"]["count"] == 0 and reencoded["credited"]["count"] == 0
        assert direct[arm]["success_share"] == 0.0 and direct[arm]["self_consistency"]["samples_share_consistent"] == 0.0
        assert reencoded["direct_against_reencoded"] == {
            "direct_and_reencoded": 0,
            "direct_only": 0,
            "neither": 48,
            "reencoded_only": 0,
        }
        # the floor (each sample's shuffle) and the natural chains: the ruler can be passed
        assert reencoded["floor_success_share"] == 0.0 == direct[arm]["floor"]["success_share"]
        assert reencoded["references"] == 24 and round(reencoded["reference_success_share"] * 24) == 22
        total += reencoded["samples"]
    assert total == 144
    assert direct["standardized"]["direct_fold"]["share_confident"] == 0.0
    assert direct["whitened"]["direct_fold"]["share_confident"] == 0.0
    assert round(direct["whitened_with_sequence_loss"]["direct_fold"]["share_confident"] * 48) == 1
    assert round(direct["standardized"]["direct_fold"]["plddt_median"], 2) == 0.47
    assert round(direct["whitened"]["direct_fold"]["plddt_median"], 2) == 0.46


def test_the_second_seed_of_the_standardized_arm_is_no_better() -> None:
    base = GENERATION / "pilot_seed1"
    direct = read(base / "direct_summary.json")["arms"]["standardized"]
    reencoded = read(base / "standardized" / "reencoded_fold" / "summary.json")
    assert direct["samples"] == reencoded["samples"] == 48
    assert direct["success_share"] == 0.0 and direct["direct_fold"]["share_confident"] == 0.0
    assert reencoded["success"]["count"] == 0 and reencoded["credited"]["count"] == 0


def test_a_generated_single_is_far_from_the_single_of_its_own_sequence() -> None:
    for arm in ARMS:
        consistency = read(GENERATION / "pilot_seed0" / arm / "reencoded_fold" / "summary.json")["consistency"]
        assert 1.34 < consistency["paired"]["relative_l2"] < 1.37 and consistency["paired"]["cosine"] < 0.2
        assert 1.49 < consistency["unpaired"]["relative_l2"] < 1.51 and consistency["unpaired"]["cosine"] < 0.03
        assert consistency["round_trip"]["n"] == 24
        assert round(consistency["round_trip"]["relative_l2"], 3) == 0.045
        assert round(consistency["round_trip"]["cosine"], 3) == 0.999
    # and the frozen head reads a generated sequence back from the single the model derives for it
    decoded = read(GENERATION / "pilot_seed0" / "reencoded_decoding.json")
    assert set(decoded) == set(ARMS)
    assert all(d["accuracy_median"] == 1.0 and d["accuracy_min"] > 0.988 and d["samples"] == 48 for d in decoded.values())


def test_a_published_generator_passes_the_ruler_in_part() -> None:
    summary = read(GENERATION / "controls" / "protgpt2" / "summary.json")
    assert summary["samples"] == 48 and summary["success"]["count"] == 7
    assert round(summary["credited"]["share"], 3) == 0.146 and summary["floor_success_share"] == 0.0
    assert round(summary["sequence"]["degenerate_share"] * 48) == 5
    novelty = read(GENERATION / "controls" / "protgpt2" / "novelty.json")["sets"]["protgpt2"]
    assert novelty["copies"] == 0 and novelty["identity"]["max"] < 0.41


def test_a_sequence_only_model_on_the_same_corpus_does_the_same_as_the_pilot() -> None:
    base = GENERATION / "controls" / "sequence_only"
    for name in ("seed0_temperature0.8", "seed0_temperature1.0", "seed1_temperature0.8", "seed1_temperature1.0"):
        summary = read(base / name / "summary.json")
        assert summary["samples"] == 48 and summary["success"]["count"] == 0 and summary["credited"]["count"] == 0
    classes = read(base / "classes.json")
    assert classes["overall"] == "at the pilot's level"
    assert classes["pilot"]["n"] == 144 and classes["pilot"]["successes"] == 0
    assert all(chosen["class"] == "at the pilot's level" for chosen in classes["chosen"].values())


def test_every_sequence_of_the_unmasking_control_is_degenerate_and_none_is_credited() -> None:
    summary = read(GENERATION / "controls" / "esmc_unmasking" / "summary.json")
    assert summary["samples"] == 48 and summary["success"]["count"] == 23
    assert summary["credited"]["count"] == 0 and summary["sequence"]["degenerate_share"] == 1.0
