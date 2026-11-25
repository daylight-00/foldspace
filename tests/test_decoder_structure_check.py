"""The reading of the structure panel, on folds whose answer is known. Numpy only.

Synthetic Cα traces play the folds of each route; the panel is read by the criteria fixed in the log: on the chains the
teacher folds stably, the re-encoded decoded sequences keep the fold (TM at least 0.9, for 90 % of them), their errors
cost no more than random ones, and AtlasFold sees no cost (median scTM within 0.02 of the original's).
"""

from __future__ import annotations

import numpy as np
import pytest

from foldspace.esmfold2 import decoder_structure_check as structure

pytestmark = pytest.mark.offline


def _trace(rng, length=80):
    steps = rng.standard_normal((length, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(steps * 3.8, axis=0)


def _noisy(ca, noise, rng):
    return ca + rng.standard_normal(ca.shape) * noise


def _panel(
    rng, n=10, *, decoded_noise=0.15, twin_noise=4.0, unstable=(), wrong_decoder=()
):
    """Routes and Cα for ``n`` chains. ``unstable`` chains have a teacher that moves with the seed."""
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
        coords[f"{i}__teacher_from_cache"] = teacher.copy()
        bad = j in wrong_decoder
        coords[f"{i}__reconstructed"] = _noisy(teacher, 0.1, rng)
        coords[f"{i}__reencoded"] = (
            _trace(rng) if bad else _noisy(teacher, decoded_noise, rng)
        )
        coords[f"{i}__twin_reconstructed"] = _noisy(teacher, 0.1, rng)
        coords[f"{i}__twin_reencoded"] = _noisy(teacher, twin_noise, rng)
        conf = dict.fromkeys(structure.ROUTES, 0.8)
        chains.append(
            {
                "id": i,
                "length": 80,
                "substitutions": 2,
                "ptm": conf,
                "plddt": conf,
                "sequence": "A" * 80,
            }
        )
    return {"chains": chains}, coords


def _atlas(ids, original, decoded, twin=None):
    def rows(value):
        return [{"id": i, "scTM": v} for i, v in zip(ids, value, strict=True)]

    out = {"S": rows(original), "D": rows(decoded)}
    if twin is not None:
        out["R"] = rows(twin)
    return out


def test_a_decoder_whose_errors_are_mild_passes_every_reading():
    rng = np.random.default_rng(0)
    routes, coords = _panel(rng)
    ids = [c["id"] for c in routes["chains"]]
    atlas = _atlas(ids, [0.9] * 10, [0.895] * 10, [0.7] * 10)
    s = structure.summarize(routes, coords, atlas)
    assert s["stable_chains"] == 10
    assert s["gates"]["a_reencoded_keeps_the_fold"]["met"]
    assert s["gates"]["b_decoder_errors_cost_no_more_than_random_ones"]["met"]
    assert s["gates"]["c_atlasfold_does_not_see_a_cost"]["met"]
    assert s["passes"] is True
    assert s["atlasfold"]["difference_median"] == pytest.approx(-0.005)
    assert s["cache_parity"]["max_rmsd_a"] < 1e-6


def test_a_decoder_that_loses_the_fold_on_many_chains_fails_the_first_reading():
    rng = np.random.default_rng(1)
    routes, coords = _panel(rng, wrong_decoder=(0, 1, 2, 3))
    s = structure.summarize(routes, coords, None)
    assert s["reencoded"]["share_keeping_stable"] == pytest.approx(0.6)
    assert not s["gates"]["a_reencoded_keeps_the_fold"]["met"]
    # AtlasFold was not run: the panel is not passed, however the other two read
    assert s["gates"]["c_atlasfold_does_not_see_a_cost"]["met"] is None
    assert s["passes"] is False


def test_errors_that_are_worse_than_random_ones_fail_the_second_reading():
    rng = np.random.default_rng(2)
    routes, coords = _panel(rng, decoded_noise=3.0, twin_noise=0.2)
    s = structure.summarize(
        routes,
        coords,
        _atlas([c["id"] for c in routes["chains"]], [0.9] * 10, [0.9] * 10),
    )
    assert not s["gates"]["b_decoder_errors_cost_no_more_than_random_ones"]["met"]
    assert s["passes"] is False


def test_an_unstable_teacher_cannot_indict_the_decoder():
    rng = np.random.default_rng(3)
    # chains 0 to 3 have a teacher that does not hold across seeds; the decoder loses the fold on exactly those
    routes, coords = _panel(rng, unstable=(0, 1, 2, 3), wrong_decoder=(0, 1, 2, 3))
    ids = [c["id"] for c in routes["chains"]]
    s = structure.summarize(routes, coords, _atlas(ids, [0.9] * 10, [0.9] * 10))
    assert s["stable_chains"] == 6 and s["unstable_teachers"]["chains"] == 4
    assert s["gates"]["a_reencoded_keeps_the_fold"]["met"]
    assert s["unstable_teachers"]["reencoded_tm_teacher_median"] < 0.5


def test_atlasfold_seeing_a_cost_fails_the_third_reading():
    rng = np.random.default_rng(4)
    routes, coords = _panel(rng)
    ids = [c["id"] for c in routes["chains"]]
    s = structure.summarize(routes, coords, _atlas(ids, [0.9] * 10, [0.8] * 10))
    assert s["atlasfold"]["difference_median"] == pytest.approx(-0.1)
    assert not s["gates"]["c_atlasfold_does_not_see_a_cost"]["met"]
    assert s["passes"] is False


def test_a_cache_that_is_not_the_models_is_visible():
    rng = np.random.default_rng(5)
    routes, coords = _panel(rng)
    coords["c3__teacher_from_cache"] = _noisy(coords["c3__teacher"], 1.0, rng)
    s = structure.summarize(routes, coords, None)
    assert s["cache_parity"]["max_rmsd_a"] > 0.1


def test_a_decoded_panel_is_read_and_a_mismatched_one_is_refused(tmp_path):
    good = tmp_path / "good.tsv"
    good.write_text(
        "seq_id\tsequence\tdecoded\tsubstitutions\nA_1\tMKV\tMKA\t1\nB_1\tGGS\tGGS\t0\n"
    )
    rows = structure.read_panel(good)
    assert [r["substitutions"] for r in rows] == [1, 0] and rows[0]["decoded"] == "MKA"
    bad = tmp_path / "bad.tsv"
    bad.write_text("seq_id\tsequence\tdecoded\tsubstitutions\nA_1\tMKV\tMK\t1\n")
    with pytest.raises(ValueError, match="decoded as"):
        structure.read_panel(bad)
    other = tmp_path / "other.tsv"
    other.write_text("seq_id\tsequence\nA_1\tMKV\n")
    with pytest.raises(ValueError, match="not a decoded panel"):
        structure.read_panel(other)
