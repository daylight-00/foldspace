"""The reading of robustness on folds and decoders whose answer is known. Numpy only.

Synthetic Cα traces play the folds: a condition's fold is the teacher's with an error that grows with its severity, so
that the level at which the fold is lost, and so the tolerance of a family, is set by the test and read back.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldspace.esmfold2 import evaluate_robustness
from foldspace.esmfold2 import perturb as P

pytestmark = pytest.mark.offline

#: the error, in angstrom, of the fold at each severity rank of a family (mild to severe)
ERROR = {
    "noise": [0.1, 0.2, 6.0, 12.0],
    "anoise": [0.1, 0.1, 0.1, 9.0],  # lost only at the severest level
    "shrink": [0.1, 0.1, 0.1],  # nothing is lost: every level keeps the fold
    "rank": [8.0, 9.0, 10.0, 11.0],  # the mildest level already loses it
    "dropout": [0.1, 0.2, 7.0],
}


def _trace(rng, length=80):
    steps = rng.standard_normal((length, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(steps * 3.8, axis=0)


def _noisy(ca, amplitude, rng):
    return ca + rng.standard_normal(ca.shape) * amplitude


def _world(rng, n=10, unstable=(), which="standard"):
    chains, coords = [], {}
    for j in range(n):
        i = f"c{j}"
        ca = _trace(rng)
        teacher = _noisy(ca, 0.1, rng)
        coords[f"{i}__deposited"] = ca
        coords[f"{i}__teacher"] = teacher
        coords[f"{i}__teacher_other_seed"] = (
            _trace(rng) if j in unstable else _noisy(teacher, 0.1, rng)
        )
        coords[f"{i}__no_prior"] = _trace(rng)
        plddt, ptm, size = {}, {}, {}
        for family, level in P.conditions(which):
            name = P.name_of(family, level)
            rank = P.severity_order(family, which).index(level)
            error = (
                ERROR[family][rank]
                if which == "standard"
                else (0.1 if rank == 0 else 9.0)
            )
            coords[f"{i}__{name}"] = _noisy(teacher, error, rng)
            plddt[name], ptm[name], size[name] = 0.8, 0.7, 0.1 * (rank + 1)
        for name in evaluate_robustness.SPECIAL:
            plddt[name] = ptm[name] = 0.8
        chains.append(
            {"id": i, "length": 80, "plddt": plddt, "ptm": ptm, "relative_size": size}
        )
    return {"chains": chains, "level_set": which}, coords


def test_the_tolerance_of_a_family_is_the_last_level_before_the_fold_is_lost():
    rng = np.random.default_rng(0)
    result, coords = _world(rng)
    s = evaluate_robustness.summarize_folds(result, coords)
    assert s["stable_chains"] == 10 and s["unstable_chains"] == 0
    noise = s["tolerance"]["noise"]
    assert noise["levels_mild_to_severe"] == [0.1, 0.2, 0.4, 0.8]
    assert noise["shares"] == [1.0, 1.0, 0.0, 0.0]
    assert noise["tolerance"] == 0.2 and not noise["censored"]
    # nothing was lost anywhere in this family: the tolerance is at least the most severe level tried
    assert s["tolerance"]["shrink"]["tolerance"] == 0.5
    assert s["tolerance"]["shrink"]["censored"]
    # the mildest level was already too much
    assert s["tolerance"]["rank"]["tolerance"] is None
    assert not s["tolerance"]["rank"]["censored"]
    # a smaller rank is the more severe level, so the levels run 128, 64, 32, 16
    assert s["tolerance"]["rank"]["levels_mild_to_severe"] == [128, 64, 32, 16]
    assert s["tolerance"]["dropout"]["tolerance"] == 0.15
    # the data-shaped noise is lost only at its severest level
    assert s["tolerance"]["anoise"]["tolerance"] == 0.2
    assert s["tolerance"]["anoise"]["levels_mild_to_severe"] == [0.05, 0.1, 0.2, 0.4]
    # the medians sit where the errors put them
    assert s["conditions"]["noise_0.1"]["median_tm_teacher"] > 0.95
    assert s["conditions"]["noise_0.8"]["median_tm_teacher"] < 0.5
    assert s["conditions"]["noise_0.4"]["median_relative_size"] == pytest.approx(0.3)
    assert s["floor"]["median_tm_teacher"] < 0.5


def test_an_unstable_teacher_cannot_show_what_a_perturbation_does():
    rng = np.random.default_rng(1)
    result, coords = _world(rng, unstable=(0, 1, 2, 3))
    s = evaluate_robustness.summarize_folds(result, coords)
    assert s["stable_chains"] == 6 and s["unstable_chains"] == 4
    assert [r["stable"] for r in s["per_chain"]][:5] == [False] * 4 + [True]
    # the readings are on the stable chains, and the others are reported apart
    assert s["conditions"]["noise_0.1"]["stable_chains"] == 6
    assert s["conditions"]["noise_0.1"]["keeps"] == 1.0
    assert s["conditions"]["noise_0.1"]["unstable_median_tm_teacher"] > 0.9


def test_a_share_that_is_not_a_number_is_not_a_pass():
    t = evaluate_robustness._tolerance("noise", lambda level: float("nan"))
    assert t["tolerance"] is None and not t["censored"]
    t = evaluate_robustness._tolerance("noise", lambda level: 1.0)
    assert t["tolerance"] == 0.8 and t["censored"]


def _decode(rng, ids, readable_until=2, which="standard"):
    """Accuracies that fall with severity; the panel's chains are readable under the first levels only."""
    arms = ["linear-s0", "contextual-s0"]
    conditions = {"none": {a: {"cw_acc": 0.99, "micro_acc": 0.995} for a in arms}}
    panel = {i: {} for i in ids}
    for family, level in P.conditions(which):
        name = P.name_of(family, level)
        rank = P.severity_order(family, which).index(level)
        acc = 0.99 if rank < readable_until else 0.6
        conditions[name] = {
            a: {"cw_acc": acc - 0.001 * k, "micro_acc": acc} for k, a in enumerate(arms)
        }
        conditions[name]["median_relative_size"] = 0.1 * (rank + 1)
        for i in ids:
            panel[i][name] = dict.fromkeys(arms, acc)
    return {
        "level_set": which,
        "arms": arms,
        "carried": "contextual-s0",
        "conditions": conditions,
        "panel": panel,
    }


def test_the_decoder_has_its_own_tolerance_and_the_panel_is_tabulated_against_the_folds():
    rng = np.random.default_rng(2)
    result, coords = _world(rng)
    folds = evaluate_robustness.summarize_folds(result, coords)
    decode = _decode(rng, [c["id"] for c in result["chains"]], readable_until=3)
    s = evaluate_robustness.summarize_decode(decode, folds)
    # readable through the third level of every family; the fold is lost after the second in noise
    assert s["tolerance"]["contextual-s0"]["noise"]["tolerance"] == 0.4
    assert s["tolerance"]["contextual-s0"]["rank"]["tolerance"] == 32
    assert s["conditions"]["noise_0.4"]["linear-s0"] == pytest.approx(0.99)
    table = s["panel_cross_tab"]
    assert table["decoder"] == "contextual-s0"
    # at noise 0.4 every chain reads and none keeps its fold: the two scales part there
    assert table["noise_0.4"] == {
        "keeps_and_readable": 0,
        "keeps_not_readable": 0,
        "readable_not_kept": 10,
        "neither": 0,
    }
    # at noise 0.1 both hold
    assert table["noise_0.1"]["keeps_and_readable"] == 10
    # at noise 0.8 neither holds
    assert table["noise_0.8"]["neither"] == 10
    for name, cells in table.items():
        if name in ("decoder", "readable_at"):
            continue
        assert sum(cells.values()) == 10


def test_the_cross_tab_counts_only_stable_chains_and_is_absent_without_the_folds():
    rng = np.random.default_rng(3)
    result, coords = _world(rng, unstable=(0, 1))
    folds = evaluate_robustness.summarize_folds(result, coords)
    decode = _decode(rng, [c["id"] for c in result["chains"]])
    table = evaluate_robustness.summarize_decode(decode, folds)["panel_cross_tab"]
    assert sum(table["noise_0.1"].values()) == 8
    assert "panel_cross_tab" not in evaluate_robustness.summarize_decode(decode, None)


def test_the_whole_summary_holds_both_readings():
    rng = np.random.default_rng(4)
    result, coords = _world(rng)
    decode = _decode(rng, [c["id"] for c in result["chains"]])
    s = evaluate_robustness.summarize(result, coords, decode)
    assert set(s) == {"folds", "decode"}
    assert set(evaluate_robustness.summarize(result, coords)) == {"folds"}


def test_the_coarse_levels_are_read_as_their_own_set_and_the_fold_breaks_where_the_test_breaks_it():
    rng = np.random.default_rng(6)
    result, coords = _world(rng, which="coarse")
    s = evaluate_robustness.summarize_folds(result, coords)
    assert set(s["conditions"]) == {
        P.name_of(f, level) for f, level in P.conditions("coarse")
    }
    # the first coarse level of every family keeps the fold and the next loses it
    assert s["tolerance"]["noise"]["levels_mild_to_severe"] == [1.6, 3.2]
    assert (
        s["tolerance"]["noise"]["tolerance"] == 1.6
        and not s["tolerance"]["noise"]["censored"]
    )
    assert s["tolerance"]["shrink"]["tolerance"] == 0.75
    assert s["tolerance"]["rank"]["levels_mild_to_severe"] == [8, 4, 2, 1]
    assert s["tolerance"]["rank"]["tolerance"] == 8
    # a summary of the standard levels of the same record is not confused with it
    assert "noise_0.1" not in s["conditions"]


def test_the_coarse_decoding_is_summarised_under_the_coarse_names():
    rng = np.random.default_rng(7)
    result, coords = _world(rng, which="coarse")
    folds = evaluate_robustness.summarize_folds(result, coords)
    decode = _decode(rng, [c["id"] for c in result["chains"]], which="coarse")
    s = evaluate_robustness.summarize_decode(decode, folds)
    assert "shrink_1" in s["conditions"] and "rank_1" in s["conditions"]
    assert s["tolerance"]["contextual-s0"]["shrink"]["levels_mild_to_severe"] == [
        0.75,
        0.9,
        1.0,
    ]
    assert "panel_cross_tab" in s and "rank_8" in s["panel_cross_tab"]
