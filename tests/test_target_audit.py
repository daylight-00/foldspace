"""The audit reads known structure back out of synthetic shards.

Each test builds rows whose distribution is fixed by construction -- a channel
scale, an autocorrelation, a share of variance that belongs to the chain -- and
checks that the reading returns it. Numpy only: no torch, no weights.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from foldspace.esmfold2 import target_audit as audit

pytestmark = pytest.mark.offline

D = 6


def _chains(rng, n_chains, *, rho=0.0, chain_sd=0.0, scale=None, length=(60, 90)):
    """AR(1) rows along each chain, plus a per-chain offset; returns (rows, lengths)."""
    scale = np.ones(D) if scale is None else np.asarray(scale, dtype=float)
    lengths = rng.integers(length[0], length[1], size=n_chains)
    rows = []
    for n in lengths:
        eps = rng.standard_normal((n, D))
        x = np.empty((n, D))
        x[0] = eps[0]
        for i in range(1, n):
            x[i] = rho * x[i - 1] + np.sqrt(1 - rho**2) * eps[i]
        x += chain_sd * rng.standard_normal(D)
        rows.append(x * scale)
    return rows, lengths


def _write(path, rows, lengths, ids, offset=0.0, provenance=None):
    z = np.concatenate(rows).astype(np.float32) + np.float32(offset)
    held = {
        "ids": np.array(ids),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "starts": np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64),
        "z": z,
    }
    if provenance is not None:
        held["provenance"] = json.dumps(provenance)
    np.savez(path, **held)


def test_channel_scale_matches_numpy_when_the_mean_dwarfs_the_spread():
    rng = np.random.default_rng(0)
    x = 1e6 + rng.standard_normal((5000, D)) * np.arange(1, D + 1)
    scale = audit.channel_scale(iter([x[:2000], x[2000:]]))
    assert np.allclose(scale.mean, x.mean(axis=0), rtol=1e-12)
    assert np.allclose(scale.std, x.std(axis=0), rtol=1e-6)


def test_the_reading_returns_the_autocorrelation_and_scales_it_was_built_with(tmp_path):
    rng = np.random.default_rng(1)
    scales = np.array([1.0, 2.0, 0.5, 4.0, 1.0, 3.0])
    shard_ids = []
    for s in range(2):
        rows, lengths = _chains(rng, 60, rho=0.6, scale=scales)
        ids = [f"c{s}_{i}" for i in range(len(lengths))]
        shard_ids.append(ids)
        _write(tmp_path / f"z_shard{s}.npz", rows, lengths, ids, offset=5.0)

    reading = audit.read(tmp_path)
    summary = reading.summary

    assert summary["chains"] == 120
    assert np.allclose(reading.std, scales, rtol=0.1)
    assert np.allclose(reading.mean, 5.0, atol=0.3)
    lag1 = summary["autocorrelation"]["raw"]["1"]["mean"]
    lag2 = summary["autocorrelation"]["raw"]["2"]["mean"]
    assert lag1 == pytest.approx(0.6, abs=0.05)
    assert lag2 == pytest.approx(0.36, abs=0.06)
    # Gaussian rows: no excess tail, no skew.
    assert abs(np.median([summary["shape"]["excess_kurtosis"]["median"]])) < 0.3
    assert summary["shape"]["tails"]["6"]["channels_above_100x_gaussian"] == 0
    assert summary["scale"]["std_max_over_median"] == pytest.approx(
        scales.max() / np.median(scales), rel=0.2
    )


def test_a_per_chain_offset_is_counted_as_between_chain_variance(tmp_path):
    rng = np.random.default_rng(2)
    rows, lengths = _chains(rng, 200, rho=0.0, chain_sd=1.0, length=(80, 120))
    _write(tmp_path / "z_shard0.npz", rows, lengths, [f"c{i}" for i in range(200)])
    share = audit.read(tmp_path).summary["between_chains_share_of_variance"]
    # unit offset variance against unit within variance: about one half
    assert share == pytest.approx(0.5, abs=0.06)

    rows, lengths = _chains(rng, 200, rho=0.0, chain_sd=0.0, length=(80, 120))
    _write(tmp_path / "z_shard0.npz", rows, lengths, [f"c{i}" for i in range(200)])
    assert audit.read(tmp_path).summary["between_chains_share_of_variance"] < 0.05


def test_chain_mean_removal_separates_composition_from_local_dependence(tmp_path):
    """A slow per-chain offset is autocorrelation in the raw rows and none in the centred ones."""
    rng = np.random.default_rng(3)
    rows, lengths = _chains(rng, 200, rho=0.0, chain_sd=2.0, length=(80, 120))
    _write(tmp_path / "z_shard0.npz", rows, lengths, [f"c{i}" for i in range(200)])
    ac = audit.read(tmp_path).summary["autocorrelation"]
    assert ac["raw"]["1"]["mean"] > 0.5
    assert abs(ac["chain_mean_removed"]["1"]["mean"]) < 0.05


def test_terminal_means_recover_a_position_fixed_component(tmp_path):
    rng = np.random.default_rng(6)
    n_signature = np.linspace(4.0, 1.0, audit.EDGE)[:, None] * np.ones(D)
    c_signature = -np.linspace(3.0, 0.5, audit.EDGE)[:, None] * np.ones(D)
    rows, lengths = _chains(rng, 150, rho=0.0, length=(70, 110))
    for r in rows:
        r[: audit.EDGE] += n_signature
        r[len(r) - audit.EDGE :] += c_signature[::-1]
    _write(tmp_path / "z_shard0.npz", rows, lengths, [f"c{i}" for i in range(150)])

    terminal = audit.read(tmp_path).terminal
    assert np.allclose(terminal.n_terminus, n_signature, atol=0.35)
    assert np.allclose(terminal.c_terminus, c_signature, atol=0.35)
    assert np.allclose(terminal.interior, 0.0, atol=0.1)

    built = audit.positional_mean(100, terminal)
    assert built.shape == (100, D)
    assert np.allclose(built[0], terminal.n_terminus[0])
    assert np.allclose(built[-1], terminal.c_terminus[0])
    assert np.allclose(built[50], terminal.interior)
    with pytest.raises(ValueError, match="twice the edge"):
        audit.positional_mean(2 * audit.EDGE, terminal)


def test_a_heavy_tailed_channel_is_reported(tmp_path):
    rng = np.random.default_rng(4)
    rows, lengths = _chains(rng, 100, length=(90, 110))
    for r in rows:
        r[:, 0] = rng.standard_t(df=3, size=len(r))
    _write(tmp_path / "z_shard0.npz", rows, lengths, [f"c{i}" for i in range(100)])
    shape = audit.read(tmp_path).summary["shape"]
    assert shape["excess_kurtosis"]["max"] > 3.0
    assert shape["tails"]["6"]["channels_above_100x_gaussian"] >= 1


def test_spectrum_summary_counts_directions():
    flat = audit.spectrum_summary(np.ones(8))
    assert flat["participation_ratio"] == pytest.approx(8.0)
    assert flat["entropy_rank"] == pytest.approx(8.0)
    assert flat["condition"] == pytest.approx(1.0)
    assert flat["components_for"]["0.90"] == 8

    one = audit.spectrum_summary(np.array([1.0, 0.0, 0.0, 0.0]))
    assert one["participation_ratio"] == pytest.approx(1.0)
    assert one["components_for"]["0.99"] == 1
    assert one["condition"] == float("inf")

    with pytest.raises(ValueError):
        audit.spectrum_summary(np.zeros(3))


def test_cluster_overlap_flags_chains_whose_cluster_is_in_another_shard():
    corpus_ids = np.array(["a", "b", "c", "d", "e", "f"])
    corpus_clusters = np.array([0, 0, 1, 2, 3, 3])
    splits = np.array(["train"] * 6)
    out = audit.cluster_overlap(
        [np.array(["a", "c"]), np.array(["b", "d"]), np.array(["e", "f", "zz"])],
        corpus_ids,
        corpus_clusters,
        splits,
    )
    by_shard = {r["shard"]: r for r in out["per_shard"]}
    # shard 0: a shares cluster 0 with b (shard 1); c (cluster 1) is alone
    assert by_shard[0]["chains_sharing_a_cluster_with_other_shards"] == pytest.approx(
        0.5
    )
    assert by_shard[1]["chains_sharing_a_cluster_with_other_shards"] == pytest.approx(
        0.5
    )
    # shard 2 holds both members of cluster 3 and nobody else does
    assert by_shard[2]["chains_sharing_a_cluster_with_other_shards"] == 0.0
    assert out["ids_not_in_corpus"] == 1
    assert out["splits_of_chains"] == {"train": 6}


def test_identity_share_is_the_between_type_part_of_the_variance():
    rng = np.random.default_rng(5)
    n_types, per, d = 4, 5000, 3
    type_means = np.array([[3.0, 0, 0], [-3.0, 0, 0], [0, 3.0, 0], [0, -3.0, 0]])
    x = np.concatenate([m + rng.standard_normal((per, d)) for m in type_means])
    pooled = {
        "n_residues": np.int64(len(x)),
        "sum_z": x.sum(axis=0),
        "sum_zz": x.T @ x,
        "sum_by_aa": np.stack(
            [x[i * per : (i + 1) * per].sum(axis=0) for i in range(n_types)]
        ),
        "count_by_aa": np.full(n_types, per),
    }
    got = audit.identity_share(pooled)["between_type_share"]
    # between-type variance 9 (two channels at 4.5), within-type 3 (unit per channel)
    assert got == pytest.approx(9.0 / 12.0, abs=0.02)
    assert audit.identity_share({}) is None


def test_provenance_says_whether_one_checkpoint_made_every_shard():
    same = [{"checkpoint": {"weights_sha256": "abc"}, "esm": "43b4548"}] * 4
    assert audit.summarize_provenance(same)["one_checkpoint"] is True
    mixed = same[:3] + [{"checkpoint": {"weights_sha256": "xyz"}, "esm": "43b4548"}]
    assert audit.summarize_provenance(mixed)["one_checkpoint"] is False
    assert audit.summarize_provenance([{}])["one_checkpoint"] is False
