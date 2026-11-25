"""The contribution audit counts what it says it counts, on folds whose answer is known.

Synthetic Cα traces: a swap is built to land on the single's protein, to keep the sequence's own
fold, or on neither, and the classification, ``f`` and the pair-level resampling have to say so.
Numpy only; US-align is used where the binary exists.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from foldspace.esmfold2 import contribution_audit as audit

pytestmark = pytest.mark.offline


def _trace(rng, length):
    steps = rng.standard_normal((length, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(steps * 3.8, axis=0)


def _noisy(ca, noise, rng):
    return ca + rng.standard_normal(ca.shape) * noise


def _pair(index, outcome, rng, length=80):
    """One pair, with both swaps of the stated outcome: follow, keep or neither."""
    a, b = _trace(rng, length), _trace(rng, length)
    ref = {"AA": _noisy(a, 0.1, rng), "BB": _noisy(b, 0.1, rng)}
    cas = {
        "AA": ref["AA"],
        "BB": ref["BB"],
        "AA_seed": _noisy(ref["AA"], 0.1, rng),
        "native_A": a,
        "native_B": b,
        "A0": _trace(rng, length),
        "B0": _trace(rng, length),
        "Am": _trace(rng, length),
        "Bm": _trace(rng, length),
    }
    if outcome == "follow":
        cas["AB"], cas["BA"] = _noisy(b, 0.2, rng), _noisy(a, 0.2, rng)
    elif outcome == "keep":
        cas["AB"], cas["BA"] = _noisy(a, 0.2, rng), _noisy(b, 0.2, rng)
    else:
        cas["AB"], cas["BA"] = _trace(rng, length), _trace(rng, length)
    conf = {name: 0.5 for name in cas}
    return {
        "pair": index,
        "run": "synthetic",
        "length": length,
        "A": f"a{index}",
        "B": f"b{index}",
        "sequence_identity": 0.05,
        "ptm": conf,
        "plddt": conf,
        "cas": cas,
    }


@pytest.fixture
def pairs():
    rng = np.random.default_rng(0)
    outcomes = ["follow"] * 4 + ["keep"] * 2 + ["neither"] * 3
    return [_pair(i, o, rng) for i, o in enumerate(outcomes)]


@pytest.fixture
def backends():
    return {"global_kabsch": audit.tm_backends(binary=None)["global_kabsch"]}


def test_a_swap_is_classified_by_where_its_fold_went(pairs, backends):
    rows = audit.swap_rows(pairs)
    assert len(rows) == 18
    scores = audit.score_rows(rows, backends)["global_kabsch"]
    counts = audit.classify(scores)
    assert counts["swaps"] == 18
    assert counts["follow_the_single"] == 8
    assert counts["keep_the_sequences_fold"] == 4
    assert counts["neither"] == 6
    assert counts["both"] == 0


def test_counts_do_not_grow_as_the_threshold_rises(pairs, backends):
    scores = audit.score_rows(audit.swap_rows(pairs), backends)["global_kabsch"]
    follows = [audit.classify(scores, t)["follow_the_single"] for t in audit.THRESHOLDS]
    assert follows == sorted(follows, reverse=True)
    neither = [audit.classify(scores, t)["neither"] for t in audit.THRESHOLDS]
    assert neither == sorted(neither)


def test_f_is_a_ratio_of_losses_and_undefined_when_the_prior_does_not_matter(
    pairs, backends
):
    rows = audit.swap_rows(pairs)
    scores = audit.score_rows(rows, backends)["global_kabsch"]
    values = audit.f_values(scores)
    f = values["f"]
    kept = np.array([r["swap"] in ("AB", "BA") and r["pair"] in (4, 5) for r in rows])
    # a swap that keeps the sequence's fold recovers almost all of what no prior loses
    assert np.nanmax(f[kept]) < 0.2
    # ... and one that goes elsewhere loses what no prior loses (the ratio is near 1)
    elsewhere = np.array([r["pair"] in (0, 1, 2, 3, 6, 7, 8) for r in rows])
    assert np.nanmedian(f[elsewhere]) > 0.7
    # the difference of TMs is the same fact without the ratio
    assert np.nanmedian(values["delta_tm"][kept]) > 0.5

    # no prior at all that folds as the reference: nothing to recover, so f is undefined
    scores = {k: v.copy() for k, v in scores.items()}
    scores["none_ref"][0] = 0.995
    assert np.isnan(audit.f_values(scores)["f"][0])
    assert np.isfinite(audit.f_values(scores)["f"][1])


def test_the_pair_is_the_unit_of_the_interval(pairs, backends):
    rows = audit.swap_rows(pairs)
    scores = audit.score_rows(rows, backends)["global_kabsch"]
    pair_of_row = np.array([r["pair"] for r in rows])

    def follow_share(idx):
        c = audit.classify({k: v[idx] for k, v in scores.items()})
        return c["follow_the_single"] / c["swaps"]

    a = audit.bootstrap_pairs(pair_of_row, follow_share, n=500, seed=3)
    b = audit.bootstrap_pairs(pair_of_row, follow_share, n=500, seed=3)
    assert a == b
    assert a["low"] <= 8 / 18 <= a["high"]
    # nine pairs resampled: the interval is wide, wider than it would be for eighteen free swaps
    assert a["high"] - a["low"] > 0.2


def test_a_run_is_read_back_with_its_folds(tmp_path, pairs):
    record = {
        "pairs": [
            {k: v for k, v in p.items() if k not in ("cas", "run")} for p in pairs
        ]
    }
    (tmp_path / "contribution_det.json").write_text(json.dumps(record))
    arrays = {}
    for p in pairs:
        for name, ca in p["cas"].items():
            arrays[f"pair{p['pair']}_{name}"] = ca
    np.savez(tmp_path / "contribution_det_coords.npz", **arrays)

    loaded = audit.load_run(tmp_path)
    assert len(loaded) == len(pairs)
    assert np.array_equal(loaded[0]["cas"]["AB"], pairs[0]["cas"]["AB"])
    assert loaded[0]["run"] == "contribution_det"
    with pytest.raises(FileNotFoundError):
        audit.load_run(tmp_path / "missing")


def test_the_audit_reports_what_it_was_built_to_show(tmp_path, pairs):
    record = {
        "pairs": [
            {k: v for k, v in p.items() if k not in ("cas", "run")} for p in pairs
        ]
    }
    (tmp_path / "contribution_det.json").write_text(json.dumps(record))
    arrays = {f"pair{p['pair']}_{n}": ca for p in pairs for n, ca in p["cas"].items()}
    np.savez(tmp_path / "contribution_det_coords.npz", **arrays)

    report = audit.audit([tmp_path], binary=None, targets=None, stats=None)

    assert report["pairs"] == 9 and report["swaps"] == 18
    assert report["usalign_available"] is False
    assert (
        report["classification_by_threshold"]["global_kabsch"]["0.5"][
            "follow_the_single"
        ]
        == 8
    )
    assert report["f"]["defined"] == 18
    assert report["pair_level_outcomes"] == {
        "follow+follow": 4,
        "keep+keep": 2,
        "neither+neither": 3,
    }
    # unrelated random walks of one length are not alike: following the single is not ambiguous
    assert report["ambiguity"]["swaps_with_reference_folds_alike_at_0.5"] == 0
    assert report["confidence_by_outcome"]["follow"]["n"] == 8


@pytest.mark.skipif(shutil.which("USalign") is None, reason="needs the US-align binary")
def test_usalign_agrees_that_a_fold_is_itself():
    rng = np.random.default_rng(1)
    ca = _trace(rng, 90)
    backends = audit.tm_backends("USalign")
    assert backends["usalign_identity"](ca, ca) == pytest.approx(1.0, abs=1e-3)
    assert backends["usalign_free"](ca, ca) == pytest.approx(1.0, abs=1e-3)
    other = _trace(rng, 90)
    assert backends["usalign_identity"](ca, other) < 0.4
