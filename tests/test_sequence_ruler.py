"""The ruler for a set of sequences, on sequences, folds and tables whose answer is known. Numpy only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from foldspace.esmfold2 import latent_generation as latent_generation, seqeval
from foldspace.esmfold2 import perturb as P

pytestmark = pytest.mark.offline

D = 16


def _protein(seed, n=120):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list(latent_generation.LETTERS), n))


# ------------------------------------------------------------ the sequences


def test_entropy_is_zero_for_a_homopolymer_and_log2_twenty_for_a_uniform_composition():
    assert seqeval.composition_entropy("A" * 50) == 0.0
    assert seqeval.composition_entropy(latent_generation.LETTERS * 5) == pytest.approx(math.log2(20))
    assert 4.0 < seqeval.composition_entropy(_protein(0, 200)) <= math.log2(20)
    assert seqeval.composition_entropy("") == 0.0


def test_a_repeat_has_few_distinct_three_mers_and_a_protein_has_nearly_all():
    assert seqeval.distinct_share("A" * 10) == pytest.approx(1 / 8)
    assert seqeval.distinct_share("EAAAK" * 20) < 0.1
    assert seqeval.distinct_share(_protein(1, 150)) > 0.95
    assert seqeval.distinct_share("AC") == 0.0  # no 3-mer at all


def test_degenerate_catches_a_repeat_a_poor_composition_and_spares_a_protein():
    assert seqeval.degenerate("A" * 100)
    assert seqeval.degenerate("EAAAK" * 24)  # entropy 1.9 bits and few 3-mers
    eight = "ACDEFGHI" * 15  # uniform over eight letters: 3 bits, and 8 distinct 3-mers
    assert seqeval.degenerate(eight)
    assert not seqeval.degenerate(_protein(2, 130))
    # a chain whose composition is poor but whose order is not a repeat is read on its entropy alone
    rng = np.random.default_rng(3)
    poor = "".join(rng.choice(list("ACDEFGHI"), 150))
    assert seqeval.composition_entropy(poor) < seqeval.ENTROPY_MIN
    assert seqeval.degenerate(poor)


def test_the_floor_keeps_the_composition_and_the_length_and_loses_the_order():
    s = _protein(4, 140)
    f = seqeval.floor_sequence(s, 7, "s001")
    assert sorted(f) == sorted(s) and f != s and len(f) == len(s)
    assert f == seqeval.floor_sequence(s, 7, "s001")  # deterministic
    assert f != seqeval.floor_sequence(s, 7, "s002")  # another sample, another shuffle
    assert f != seqeval.floor_sequence(s, 8, "s001")  # another seed


def test_masks_are_a_share_of_the_chain_without_repeats_and_at_least_one_position():
    rng = np.random.default_rng(0)
    masks = seqeval.mask_positions(200, rng)
    assert len(masks) == seqeval.MASKS
    for m in masks:
        assert len(m) == 30 and len(set(m.tolist())) == 30 and np.all(np.diff(m) > 0)
        assert m.min() >= 0 and m.max() < 200
    assert all(
        len(m) == 1 for m in seqeval.mask_positions(3, rng)
    )  # 0.45 rounds to 0, and one is the least
    assert all(len(m) == 2 for m in seqeval.mask_positions(2, rng, rate=1.0))
    # different draws differ
    assert any(not np.array_equal(masks[0], m) for m in masks[1:])


# ---------------------------------------------------------------- the tests


def test_fisher_is_the_hypergeometric_tail():
    # five of ten against none of ten: C(10,5) / C(20,5)
    assert seqeval.fisher_greater(5, 10, 0, 10) == pytest.approx(252 / 15504)
    assert seqeval.fisher_greater(0, 10, 0, 10) == pytest.approx(1.0)
    assert (
        seqeval.fisher_greater(3, 10, 3, 10) > 0.5
    )  # no difference: more than half the mass
    # the first share smaller than the second: nearly all the mass
    assert seqeval.fisher_greater(0, 48, 20, 48) > 0.999
    # the pilot's: 5 of 48 against none of 144 is well under 0.05
    assert seqeval.fisher_greater(5, 48, 0, 144) < 0.01
    assert seqeval.fisher_greater(2, 48, 0, 144) > 0.05


def test_mann_whitney_is_one_sided_and_handles_ties():
    x = np.arange(11, 21, dtype=float)
    y = np.arange(1, 11, dtype=float)
    assert seqeval.mannwhitney_greater(x, y) < 1e-3
    assert seqeval.mannwhitney_greater(y, x) > 0.999
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=60), rng.normal(size=60)
    assert 0.05 < seqeval.mannwhitney_greater(a, b) < 0.95
    assert seqeval.mannwhitney_greater([1.0] * 5, [1.0] * 5) == 0.5  # all tied
    assert math.isnan(seqeval.mannwhitney_greater([], [1.0]))
    # a shift of one standard deviation on 48 against 144 is called at p < 0.01
    big = rng.normal(1.0, 1.0, 48)
    ref = rng.normal(0.0, 1.0, 144)
    assert seqeval.mannwhitney_greater(big, ref) < 0.01


def test_wilson_interval_is_inside_zero_and_one_and_wider_for_fewer():
    lo, hi = seqeval.wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    lo, hi = seqeval.wilson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.8
    w48 = seqeval.wilson(24, 48)
    w480 = seqeval.wilson(240, 480)
    assert (w48[1] - w48[0]) > (w480[1] - w480[0])
    assert all(math.isnan(v) for v in seqeval.wilson(0, 0))


# ------------------------------------------------------------- consistency


def _spectrum():
    return P.Spectrum(np.eye(D), np.ones(D))


def test_a_single_is_consistent_with_itself_and_far_from_an_unrelated_one():
    rng = np.random.default_rng(0)
    mean, std = np.zeros(D), np.ones(D)
    h = rng.standard_normal((80, D))
    same = seqeval.consistency(h, h, mean=mean, std=std, spectrum=_spectrum())
    assert same["relative_l2"] == 0.0 and same["cosine"] == pytest.approx(1.0)
    assert same["anoise_level"] == 0.0
    other = rng.standard_normal((80, D))
    far = seqeval.consistency(other, h, mean=mean, std=std, spectrum=_spectrum())
    assert far["relative_l2"] == pytest.approx(math.sqrt(2), rel=0.15)
    assert abs(far["cosine"]) < 0.1


def test_the_anoise_level_is_the_level_of_the_noise_that_made_the_difference():
    rng = np.random.default_rng(1)
    mean, std = np.zeros(D), np.ones(D)
    h = rng.standard_normal((600, D))
    noisy = h + 0.5 * rng.standard_normal(h.shape)
    got = seqeval.consistency(noisy, h, mean=mean, std=std, spectrum=_spectrum())
    assert got["anoise_level"] == pytest.approx(0.5, rel=0.05)
    assert got["relative_l2"] == pytest.approx(
        0.5, rel=0.05
    )  # |noise| / |h| with unit variance everywhere


def test_standardisation_is_applied_before_the_comparison():
    rng = np.random.default_rng(2)
    mean, std = np.full(D, 3.0), np.full(D, 2.0)
    z = rng.standard_normal((50, D))
    raw = z * std + mean
    got = seqeval.consistency(raw, raw, mean=mean, std=std, spectrum=_spectrum())
    assert got["relative_l2"] == 0.0
    shifted = seqeval.consistency(
        raw + 1.0, raw, mean=mean, std=std, spectrum=_spectrum()
    )
    assert shifted["relative_l2"] == pytest.approx(
        0.5 * math.sqrt(50 * D) / np.linalg.norm(z), rel=1e-6
    )
    with pytest.raises(ValueError):
        seqeval.consistency(raw[:10], raw, mean=mean, std=std, spectrum=_spectrum())


def test_the_unpaired_comparison_is_far_and_the_paired_one_near():
    rng = np.random.default_rng(3)
    mean, std = np.zeros(D), np.ones(D)
    derived = {f"s{i}": rng.standard_normal((60 + 5 * i, D)) for i in range(5)}
    generated = {k: v + 0.2 * rng.standard_normal(v.shape) for k, v in derived.items()}
    far = seqeval.unpaired_consistency(
        generated, derived, mean=mean, std=std, spectrum=_spectrum()
    )
    assert set(far) == set(derived)
    for k, truth in derived.items():
        near = seqeval.consistency(
            generated[k], truth, mean=mean, std=std, spectrum=_spectrum()
        )
        assert near["relative_l2"] < 0.3 < far[k]["relative_l2"]
        assert abs(far[k]["cosine"]) < 0.15


# -------------------------------------------------------------- the reading


def _chain(kind, i, plddt, *, nll=None, consistency=None, sequence=None):
    c = {
        "id": i,
        "kind": kind,
        "length": 120,
        "sequence": sequence
        if sequence is not None
        else _protein(latent_generation.zlib_crc(f"{kind}:{i}") % 10_000),
        "plddt": {seqeval.ROUTE: plddt},
    }
    if nll is not None:
        c["nll"] = nll
    if consistency is not None:
        c["consistency"] = consistency
    return c


def _tables(sctm, plddt=None):
    return {
        i: {"scTM": s, "plddt": (plddt or {}).get(i, 50.0)} for i, s in sctm.items()
    }


def _fold(n=10, n_ref=4):
    chains = []
    for k in range(n):
        chains.append(_chain("sample", f"s{k:02d}", 0.8 if k < 4 else 0.5, nll=2.5))
        chains.append(_chain("floor", f"s{k:02d}", 0.4, nll=3.0))
    for k in range(n_ref):
        chains.append(_chain("reference", f"r{k}", 0.9, nll=1.8))
    return {"chains": chains}


def _atlas(n=10, n_ref=4, sample_sctm=0.7, ref_sctm=0.95):
    samples = {f"s{k:02d}": (sample_sctm if k < 4 else 0.2) for k in range(n)}
    floor = {f"s{k:02d}": 0.2 for k in range(n)}
    reference = {f"r{k}": ref_sctm for k in range(n_ref)}
    return {
        "samples": _tables(
            samples, {k: 70.0 if v > 0.5 else 45.0 for k, v in samples.items()}
        ),
        "floor": _tables(floor, {k: 45.0 for k in floor}),
        "reference": _tables(reference, {k: 96.0 for k in reference}),
    }


def test_a_sample_succeeds_when_confident_and_consistent_and_the_shares_follow():
    s = seqeval.summarize_set(_fold(), _atlas())
    assert s["samples"] == 10 and s["success"]["count"] == 4
    assert s["reference_success_share"] == 1.0 and s["floor_success_share"] == 0.0
    v = s["verdict"]
    assert v["success_share"] == pytest.approx(0.4)
    # 0.4 is at least half of 1.0? no: 0.5 is the bar
    assert v["supported_success"] is False
    s2 = seqeval.summarize_set(_fold(), _atlas(), None)
    assert s2["credited"]["novelty_known"] is False and v["supported_credited"] is None


def test_confident_but_not_consistent_is_not_a_success_and_the_other_way_round():
    fold = _fold()
    atlas = _atlas(sample_sctm=0.3)  # the first four are confident and inconsistent
    assert seqeval.summarize_set(fold, atlas)["success"]["count"] == 0
    atlas = _atlas()
    for c in fold["chains"]:
        if c["kind"] == "sample" and c["id"] == "s00":
            c["plddt"][seqeval.ROUTE] = 0.6  # consistent, not confident
    assert seqeval.summarize_set(fold, atlas)["success"]["count"] == 3


def test_a_missing_refold_makes_a_sample_fail_and_the_verdict_unmade_without_the_reference():
    s = seqeval.summarize_set(_fold(), {"samples": _atlas()["samples"]})
    assert (
        s["verdict"]["supported_success"] is None
    )  # no reference and no floor refolded: not made, not false
    assert math.isnan(s["reference_success_share"])


def test_the_verdict_is_entry_fours_rule():
    fold = _fold()
    atlas = _atlas()
    for k in range(4, 6):  # six of ten succeed: 0.6 >= 0.5 * 1.0 and >= 0 + 0.25
        atlas["samples"][f"s{k:02d}"]["scTM"] = 0.7
        for c in fold["chains"]:
            if c["kind"] == "sample" and c["id"] == f"s{k:02d}":
                c["plddt"][seqeval.ROUTE] = 0.8
    s = seqeval.summarize_set(fold, atlas)
    assert s["success"]["count"] == 6 and s["verdict"]["supported_success"] is True
    # the floor succeeding as often takes the margin away
    for k in range(10):
        atlas["floor"][f"s{k:02d}"]["scTM"] = 0.7
    for c in fold["chains"]:
        if c["kind"] == "floor":
            c["plddt"][seqeval.ROUTE] = 0.8
    s = seqeval.summarize_set(fold, atlas)
    assert (
        s["floor_success_share"] == 1.0 and s["verdict"]["supported_success"] is False
    )


def test_credited_leaves_out_copies_and_degenerate_sequences_and_needs_the_novelty_to_say_so():
    fold = _fold()
    atlas = _atlas()
    for c in fold["chains"]:
        if c["kind"] == "sample" and c["id"] == "s01":
            c["sequence"] = "EAAAK" * 24  # a success, and a repeat
    hits = {f"s{k:02d}": {"copy": False, "identity": 0.2} for k in range(10)}
    hits["s02"] = {
        "copy": True,
        "identity": 0.97,
    }  # a success, and a copy of a train chain
    novelty = {"hits": hits, "identity": {"n": 10}, "clusters_30": 7}
    s = seqeval.summarize_set(fold, atlas, novelty)
    assert s["success"]["count"] == 4 and s["credited"]["count"] == 2
    assert s["sequence"]["copy_share"] == pytest.approx(0.1)
    assert s["sequence"]["clusters_30"] == 7 and s["sequence"][
        "clusters_30_over_n"
    ] == pytest.approx(0.7)
    assert s["credited"]["novelty_known"] is True
    assert (
        s["verdict"]["supported_credited"] is False
    )  # 0.2 of ten is below half the reference's share
    rows = {r["id"]: r for r in s["table"]}
    assert rows["s01"]["degenerate"] and not rows["s01"]["credited"]
    assert rows["s02"]["copy"] and not rows["s02"]["credited"]
    assert rows["s00"]["credited"] and rows["s00"]["best_identity"] == 0.2


def test_the_continuous_readings_close_a_share_of_the_gap_between_floor_and_reference():
    s = seqeval.summarize_set(_fold(), _atlas())
    c = s["continuous"]["atlas_plddt"]
    # samples: 4 at 70 and 6 at 45 -> median 45; floor 45; reference 96: nothing of the gap covered
    assert c["samples"] == 45.0 and c["floor"] == 45.0 and c["reference"] == 96.0
    assert c["gap_closed"] == pytest.approx(0.0)
    sctm = s["continuous"]["sctm"]
    assert sctm["floor"] == 0.2 and sctm["reference"] == 0.95
    assert sctm["gap_closed"] == pytest.approx((sctm["samples"] - 0.2) / (0.95 - 0.2))
    nll = s["continuous"]["esmc_nll"]
    assert nll["samples"] == 2.5 and nll["floor"] == 3.0 and nll["reference"] == 1.8
    assert nll["gap_closed"] == pytest.approx((2.5 - 3.0) / (1.8 - 3.0))


def test_the_consistency_block_reads_the_paired_the_unpaired_and_the_round_trip():
    fold = _fold(n=4, n_ref=2)
    for c in fold["chains"]:
        if c["kind"] == "sample":
            c["consistency"] = {"relative_l2": 0.9, "cosine": 0.4, "anoise_level": 1.1}
            c["consistency_unpaired"] = {
                "relative_l2": 1.4,
                "cosine": 0.02,
                "anoise_level": 2.0,
            }
            c["pair_bias_relative_l2"] = 0.8
        if c["kind"] == "reference":
            c["consistency"] = {
                "relative_l2": 0.05,
                "cosine": 0.99,
                "anoise_level": 0.04,
            }
            c["pair_bias_relative_l2"] = 0.03
    s = seqeval.summarize_set(fold, _atlas(4, 2))
    k = s["consistency"]
    assert k["paired"]["relative_l2"] == 0.9 and k["paired"]["n"] == 4
    assert k["unpaired"]["relative_l2"] == 1.4
    assert k["round_trip"]["relative_l2"] == 0.05 and k["round_trip"]["n"] == 2
    assert k["pair_bias"] == {"paired_median": 0.8, "round_trip_median": 0.03}
    assert "consistency" not in seqeval.summarize_set(
        _fold(), _atlas()
    )  # nothing generated, nothing to compare


def test_direct_against_reencoded_is_a_two_by_two_of_the_same_samples():
    fold = _fold()
    atlas = _atlas()  # s00..s03 succeed on the re-encoded route
    direct_fold = {
        "chains": [
            {
                "id": f"s{k:02d}",
                "kind": "sample",
                "plddt": {"direct": 0.9 if k in (0, 5) else 0.3},
            }
            for k in range(10)
        ]
    }
    direct_atlas = {f"s{k:02d}": {"scTM": 0.8} for k in range(10)}
    s = seqeval.summarize_set(
        fold, atlas, None, {"fold": direct_fold, "atlas": direct_atlas}
    )
    t = s["direct_against_reencoded"]
    # s00: both; s05: direct only; s01..s03: re-encoded only; the rest: neither
    assert t == {
        "direct_and_reencoded": 1,
        "direct_only": 1,
        "reencoded_only": 3,
        "neither": 5,
    }
    assert sum(t.values()) == 10


# ----------------------------------------------------------- the classes


def _summary(credited, n=48, plddt=None, ref=0.9, floor=0.0, supported=None):
    rows = [
        {
            "id": f"s{i:02d}",
            "atlas_plddt": (plddt[i] if plddt is not None else 50.0 + (i % 5)),
        }
        for i in range(n)
    ]
    share = credited / n
    rule = share >= 0.5 * ref and share >= floor + 0.25
    return {
        "samples": n,
        "credited": {"count": credited},
        "verdict": {"supported_credited": rule if supported is None else supported},
        "table": rows,
    }


def _pilot(successes=0, n=144, plddt_level=54.0):
    rng = np.random.default_rng(0)
    return {
        "successes": successes,
        "n": n,
        "atlas_plddt": (plddt_level + rng.normal(0, 4, n)).tolist(),
    }


def test_supported_needs_the_rule_and_the_control_to_pass_it():
    got = seqeval.classify(_summary(30), _pilot(), control_supported=True)
    assert got["class"] == "supported" and not got["supported_withdrawn"]
    withdrawn = seqeval.classify(_summary(30), _pilot(), control_supported=False)
    assert withdrawn["class"] == "above the pilot" and withdrawn["supported_withdrawn"]
    assert withdrawn["above_by_share"]


def test_above_the_pilot_by_share_needs_both_the_margin_and_the_test():
    pilot = _pilot()
    five = seqeval.classify(
        _summary(5), pilot, control_supported=True
    )  # 0.104 over zero, p ~ 0.0005
    assert five["above_by_share"] and five["class"] == "above the pilot"
    two = seqeval.classify(
        _summary(2), pilot, control_supported=True
    )  # 0.042: under the margin
    assert not two["above_by_share"]


def test_above_the_pilot_by_plddt_needs_five_points_and_the_test():
    pilot = _pilot(plddt_level=54.0)
    plddt = (np.random.default_rng(1).normal(61.0, 4.0, 48)).tolist()
    got = seqeval.classify(_summary(0, plddt=plddt), pilot, control_supported=True)
    assert got["above_by_plddt"] and got["class"] == "above the pilot"
    small = (
        np.random.default_rng(1).normal(57.0, 4.0, 48)
    ).tolist()  # three points: under the margin
    assert not seqeval.classify(
        _summary(0, plddt=small), pilot, control_supported=True
    )["above_by_plddt"]
    none = seqeval.classify(_summary(0), pilot, control_supported=True)
    assert none["class"] == "at the pilot's level"


def test_the_pilots_own_successes_raise_the_bar_for_the_share_reading():
    pilot = _pilot(successes=20, n=144)  # 0.139
    got = seqeval.classify(
        _summary(8), pilot, control_supported=True
    )  # 0.167: not 0.10 over it
    assert not got["above_by_share"]


def _write(tmp_path, name, obj):
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return str(path)


def _control(tmp_path, supported):
    return _write(
        tmp_path,
        "control.json",
        {"verdict": {"supported_credited": supported, "supported_success": supported}},
    )


def _pilot_arm(tmp_path, name):
    return _write(
        tmp_path,
        name,
        {
            "samples": 48,
            "success": {"count": 0},
            "table": [
                {"id": f"s{i}", "atlas_plddt": 54.0 + (i % 7)} for i in range(48)
            ],
        },
    )


def test_classes_choose_the_better_temperature_per_seed_and_need_the_seeds_to_agree(
    tmp_path,
):
    pilot = [_pilot_arm(tmp_path, f"p{k}.json") for k in range(3)]
    control = _control(tmp_path, True)
    sets = []
    for seed in (0, 1):
        sets.append(f"{seed}:1.0={_write(tmp_path, f's{seed}_1.json', _summary(2))}")
        sets.append(f"{seed}:0.8={_write(tmp_path, f's{seed}_08.json', _summary(30))}")
    out = tmp_path / "classes.json"

    class A:
        pass

    a = A()
    a.pilot, a.control, a.set, a.out = pilot, control, sets, str(out)
    result = seqeval.run_classes(a)
    assert result["chosen"]["seed0"]["temperature"] == "0.8"
    assert result["overall"] == "supported"
    assert json.loads(out.read_text())["overall"] == "supported"

    # the seeds disagree: one supported, one at the pilot's level
    a.set = [
        f"0:1.0={_write(tmp_path, 'a.json', _summary(30))}",
        f"1:1.0={_write(tmp_path, 'b.json', _summary(0))}",
    ]
    result = seqeval.run_classes(a)
    assert result["overall"].startswith("undecided")

    # three seeds: the class of the median
    a.set = [
        f"0:1.0={_write(tmp_path, 'c0.json', _summary(30))}",
        f"1:1.0={_write(tmp_path, 'c1.json', _summary(0))}",
        f"2:1.0={_write(tmp_path, 'c2.json', _summary(6))}",  # above the pilot (6/48 over 0/144)
    ]
    assert seqeval.run_classes(a)["overall"] == "above the pilot"


def test_a_control_that_does_not_pass_the_rule_withdraws_supported_everywhere(tmp_path):
    a = type("A", (), {})()
    a.pilot = [_pilot_arm(tmp_path, "p.json")]
    a.control = _control(tmp_path, False)
    a.set = [f"0:1.0={_write(tmp_path, 's.json', _summary(30))}"]
    a.out = str(tmp_path / "classes.json")
    result = seqeval.run_classes(a)
    assert (
        result["control_supported"] is False and result["overall"] == "above the pilot"
    )
    assert result["per_set"]["seed0_temperature1.0"]["supported_withdrawn"] is True


# ----------------------------------------------------- the margin over the shuffles


def _margin_world(n=40, degenerate=(0, 1, 2)):
    """``n`` samples whose scTM is 0.4 and whose shuffles' is 0.25 (a clear margin), except the ``degenerate`` ones, whose scTM is 0.9 against a shuffle's 0.1."""
    table, samples, floor = [], {}, {}
    rng = np.random.default_rng(0)
    for k in range(n):
        i = f"s{k:03d}"
        bad = k in degenerate
        table.append({"id": i, "degenerate": bad})
        samples[i] = {
            "scTM": 0.9 if bad else 0.4 + 0.02 * rng.normal(),
            "plddt": 80.0 if bad else 55.0 + 8.0 * rng.normal(),
        }
        floor[i] = {
            "scTM": 0.1 if bad else 0.25 + 0.02 * rng.normal(),
            "plddt": 40.0 if bad else 54.0 + 8.0 * rng.normal(),
        }
    return table, {"samples": samples, "floor": floor}


def test_the_margin_is_the_difference_of_medians_with_a_rank_test_and_leaves_the_degenerate_out():
    table, atlas = _margin_world()
    kept = seqeval.margin_over_shuffles(table, atlas, "scTM")
    assert kept["sequences"] == 37 and kept["non_degenerate"] is True
    assert kept["margin"] == pytest.approx(0.15, abs=0.02) and kept["p"] < 1e-6
    every = seqeval.margin_over_shuffles(table, atlas, "scTM", non_degenerate=False)
    assert every["sequences"] == 40 and every["margin"] > kept["margin"] - 0.01
    # a pLDDT margin of one point under a spread of eight is not one that 37 sequences can show
    weak = seqeval.margin_over_shuffles(table, atlas, "plddt")
    assert abs(weak["margin"]) < 6 and weak["p"] > 0.05


def test_a_degenerate_sequence_can_make_a_margin_that_the_others_do_not_have():
    table, atlas = _margin_world(
        n=40, degenerate=tuple(range(0, 40, 2))
    )  # half of them
    for i in atlas["samples"]:
        if not any(r["id"] == i and r["degenerate"] for r in table):
            atlas["samples"][i]["scTM"] = atlas["floor"][i][
                "scTM"
            ]  # no margin of its own
    every = seqeval.margin_over_shuffles(table, atlas, "scTM", non_degenerate=False)
    kept = seqeval.margin_over_shuffles(table, atlas, "scTM")
    assert every["p"] < 0.05 < kept["p"]
    assert abs(kept["margin"]) < 1e-9


def test_a_sample_without_a_shuffle_or_a_value_is_not_counted_and_an_empty_set_is_nan():
    table, atlas = _margin_world(n=6, degenerate=())
    del atlas["floor"]["s003"]
    atlas["samples"]["s004"]["scTM"] = None
    got = seqeval.margin_over_shuffles(table, atlas, "scTM")
    assert got["sequences"] == 4  # s003 has no shuffle, s004 no value
    empty = seqeval.margin_over_shuffles([], {"samples": {}, "floor": {}}, "scTM")
    assert empty["sequences"] == 0 and math.isnan(empty["margin"])
    assert math.isnan(empty["p"])


def test_the_margin_command_reads_a_sets_directory(tmp_path):
    table, atlas = _margin_world(n=20, degenerate=(0,))
    (tmp_path / "summary.json").write_text(json.dumps({"table": table}))
    for kind in ("samples", "floor"):
        record = {"per_target": [{"id": i, **v} for i, v in atlas[kind].items()]}
        (tmp_path / f"atlas_{kind}.json").write_text(json.dumps(record))
    args = type("A", (), {})()
    args.fold_dir, args.out = tmp_path, tmp_path / "margin.json"
    result = seqeval.run_margin(args)
    assert set(result) == {"scTM", "scTM_all", "plddt", "plddt_all"}
    assert json.loads((tmp_path / "margin.json").read_text())["scTM"]["sequences"] == 19


# ------------------------------------------- the paired margin and its intervals


def test_the_paired_differences_are_each_sequences_own_shuffle_and_leave_the_unpaired_and_degenerate_out():
    table, atlas = _margin_world(n=20, degenerate=(0, 1))
    delta = seqeval.paired_deltas(table, atlas, "scTM")
    assert len(delta) == 18 and abs(np.median(delta) - 0.15) < 0.03
    every = seqeval.paired_deltas(table, atlas, "scTM", non_degenerate=False)
    assert (
        len(every) == 20 and (every > 0.5).sum() == 2
    )  # the two degenerate ones: +0.8
    del atlas["floor"]["s005"]
    atlas["samples"]["s006"]["scTM"] = None
    assert (
        len(seqeval.paired_deltas(table, atlas, "scTM")) == 16
    )  # a pair needs both sides


def test_a_clear_paired_shift_has_an_interval_above_zero_and_the_smallest_sign_flip_p():
    rng = np.random.default_rng(1)
    delta = 0.15 + 0.03 * rng.normal(size=40)
    got = seqeval.paired_summary(delta, draws=2000, resamples=2000)
    assert got["pairs"] == 40 and got["share_above_zero"] == 1.0
    assert got["ci_median"][0] > 0.12 and got["ci_mean"][1] < 0.19
    assert got["ci_median"][0] < got["median"] < got["ci_median"][1]
    assert got["p"] == pytest.approx(1 / 2001)
    again = seqeval.paired_summary(delta, draws=2000, resamples=2000)
    assert again == got  # the seed fixes the draws


def test_differences_that_scatter_around_zero_are_not_a_margin():
    rng = np.random.default_rng(2)
    delta = 0.05 * rng.normal(size=60)
    got = seqeval.paired_summary(delta, draws=4000, resamples=2000)
    assert got["ci_mean"][0] < 0 < got["ci_mean"][1] and got["p"] > 0.05
    empty = seqeval.paired_summary([])
    assert (
        empty["pairs"] == 0 and math.isnan(empty["p"]) and math.isnan(empty["median"])
    )


def test_pairing_shows_a_small_consistent_shift_that_the_unpaired_rank_test_cannot():
    """The sequences of a set differ from one another far more than a sequence from its shuffle: the pooled values hide a shift of 0.03 that every pair has."""
    rng = np.random.default_rng(3)
    base = rng.uniform(0.2, 0.9, size=48)
    samples = base + 0.03 + 0.01 * rng.normal(size=48)
    table = [{"id": f"s{k:03d}", "degenerate": False} for k in range(48)]
    atlas = {
        "samples": {f"s{k:03d}": {"scTM": float(samples[k])} for k in range(48)},
        "floor": {f"s{k:03d}": {"scTM": float(base[k])} for k in range(48)},
    }
    unpaired = seqeval.margin_over_shuffles(table, atlas, "scTM")
    paired = seqeval.paired_summary(
        seqeval.paired_deltas(table, atlas, "scTM"), draws=2000, resamples=2000
    )
    assert unpaired["p"] > 0.05  # the rank test does not see it
    assert paired["p"] == pytest.approx(1 / 2001) and paired["ci_median"][0] > 0.02


def test_two_groups_of_differences_are_compared_by_their_medians_with_an_interval():
    rng = np.random.default_rng(4)
    a = 0.05 + 0.03 * rng.normal(size=48)
    b = -0.01 + 0.03 * rng.normal(size=120)
    got = seqeval.paired_comparison(a, b, resamples=2000)
    assert got["difference_of_medians"] == pytest.approx(0.06, abs=0.02)
    assert got["ci"][0] > 0.03 and got["p"] < 0.001
    same = seqeval.paired_comparison(b, b, resamples=2000)  # the same group twice
    assert same["difference_of_medians"] == 0.0
    assert same["ci"][0] < 0 < same["ci"][1] and same["p"] > 0.3
    empty = seqeval.paired_comparison([], b)
    assert math.isnan(empty["difference_of_medians"]) and math.isnan(empty["p"])


def test_the_paired_command_reads_sets_directories_and_compares_the_first_with_the_rest(
    tmp_path,
):
    sets = []
    for name, shift in (("a", 0.2), ("b", 0.0), ("c", 0.0)):
        table, atlas = _margin_world(n=30, degenerate=())
        for i in atlas["samples"]:
            atlas["samples"][i]["scTM"] = atlas["floor"][i]["scTM"] + shift
        directory = tmp_path / name
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps({"table": table}))
        for kind in ("samples", "floor"):
            record = {"per_target": [{"id": i, **v} for i, v in atlas[kind].items()]}
            (directory / f"atlas_{kind}.json").write_text(json.dumps(record))
        sets.append(directory)
    args = type("A", (), {})()
    args.fold_dir, args.out = sets, tmp_path / "paired.json"
    result = seqeval.run_paired(args)
    assert set(result["sets"]) == {"a", "b", "c"}
    assert result["sets"]["a"]["scTM"]["median"] == pytest.approx(0.2, abs=1e-9)
    assert result["first_against_the_rest"]["difference_of_medians"] == pytest.approx(
        0.2, abs=1e-9
    )
    assert (
        json.loads((tmp_path / "paired.json").read_text())["sets"]["b"]["scTM"]["pairs"]
        == 30
    )


def test_two_sets_with_the_same_directory_name_are_both_in_the_record(tmp_path):
    """Two seeds of one arm are both ``z``: the second must not overwrite the first (its margin was printed as the first's once)."""
    sets = []
    for parent, shift in (("seed1", 0.1), ("seed0", 0.3)):
        table, atlas = _margin_world(n=30, degenerate=())
        for i in atlas["samples"]:
            atlas["samples"][i]["scTM"] = atlas["floor"][i]["scTM"] + shift
        directory = tmp_path / parent / "standardized"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps({"table": table}))
        for kind in ("samples", "floor"):
            record = {"per_target": [{"id": i, **v} for i, v in atlas[kind].items()]}
            (directory / f"atlas_{kind}.json").write_text(json.dumps(record))
        sets.append(directory)
    args = type("A", (), {})()
    args.fold_dir, args.out = sets, None
    result = seqeval.run_paired(args)
    assert len(result["sets"]) == 2 and "standardized" in result["sets"]
    medians = sorted(v["scTM"]["median"] for v in result["sets"].values())
    assert medians == pytest.approx([0.1, 0.3], abs=1e-9)
    assert result["sets"]["standardized"]["scTM"]["median"] == pytest.approx(0.1, abs=1e-9)
    assert result["first_against_the_rest"]["difference_of_medians"] == pytest.approx(
        -0.2, abs=1e-9
    )


def test_the_repeat_of_atlasfold_measures_how_far_each_sequences_scTM_moves_between_two_runs():
    first = {f"s{k:03d}": {"scTM": 0.3 + 0.01 * k, "plddt": 60.0} for k in range(30)}
    rng = np.random.default_rng(5)
    second = {
        i: {"scTM": v["scTM"] + 0.02 * rng.normal(), "plddt": v["plddt"]}
        for i, v in first.items()
    }
    got = seqeval.atlas_repeat(first, second, "scTM")
    assert got["sequences"] == 30
    assert 0.005 < got["mean_abs_difference"] < 0.03
    assert got["sd_of_difference"] == pytest.approx(0.02, abs=0.008)
    assert got["pearson_r"] > 0.9
    same = seqeval.atlas_repeat(first, first, "plddt")
    assert same["mean_abs_difference"] == 0.0 and same["sd_of_difference"] == 0.0
    second["s000"]["scTM"] = None
    del second["s001"]
    assert seqeval.atlas_repeat(first, second, "scTM")["sequences"] == 28
    assert seqeval.atlas_repeat({}, {}, "scTM")["sequences"] == 0


def test_the_repeat_command_reads_a_set_against_its_second_evaluation(tmp_path):
    table, atlas = _margin_world(n=24, degenerate=())
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "summary.json").write_text(json.dumps({"table": table}))
    rng = np.random.default_rng(6)
    for kind in ("samples", "floor"):
        for directory, wobble in ((first, 0.0), (second, 0.01)):
            record = {
                "per_target": [
                    {
                        "id": i,
                        "scTM": v["scTM"] + wobble * rng.normal(),
                        "plddt": v["plddt"],
                    }
                    for i, v in atlas[kind].items()
                ]
            }
            (directory / f"atlas_{kind}.json").write_text(json.dumps(record))
    reference = {"per_target": [{"id": "r0", "scTM": 0.9, "plddt": 90.0}]}
    for directory in (first, second):
        (directory / "atlas_reference.json").write_text(json.dumps(reference))
    args = type("A", (), {})()
    args.fold_dir, args.repeat_dir, args.out = first, second, tmp_path / "repeat.json"
    result = seqeval.run_repeat(args)
    assert result["agreement"]["samples"]["scTM"]["sequences"] == 24
    assert result["agreement"]["reference"]["scTM"]["mean_abs_difference"] == 0.0
    assert (
        result["margin"]["first"]["pairs"] == result["margin"]["second"]["pairs"] == 24
    )
    assert abs(result["margin"]["first"]["median"] - 0.15) < 0.03
    assert (tmp_path / "repeat.json").exists()
