"""The arithmetic of the full-width latent's audit (`src/foldspace/rfd4/esmc_only_audit.py`): the moments, dimension and type structure of a latent and the pooled held-out context R² of its within-type
residual. Numpy only; the dev-chain pass that feeds it needs torch and the release's autoencoder."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.esmc_only_audit as audit

pytestmark = pytest.mark.offline

TYPES, D = 6, 8


def latent(noise: float, n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((TYPES, D)) * 3.0
    t = rng.integers(0, TYPES, n)
    return centroids[t] + noise * rng.standard_normal((n, D)), t


def test_a_latent_that_is_its_type_and_a_little_noise_is_fixed_by_the_type() -> None:
    z, t = latent(0.1)
    s = audit.latent_summary(z, t)
    assert s["residues"] == 4000 and s["channels"] == D
    assert s["type_share_of_variance"] > 0.99
    assert s["types_with_residues"] == TYPES
    assert (
        s["centroid_distance_min"] > 0.5
        and s["centroid_distance_median"] >= s["centroid_distance_min"]
    )
    # the type centroids lie many spreads apart, and the participation ratio of the whole is that of the six centroids (at most 5 directions: six points)
    assert s["centroid_distance_min_in_spreads"] > 5
    assert s["participation_ratio"] <= TYPES - 1 + 0.5
    assert s["eigenvalues_for_90_percent"] <= TYPES - 1
    assert (
        0.0 <= s["within_type_share_range"][0] <= s["within_type_share_range"][1] < 0.05
    )


def test_the_within_type_residual_of_isotropic_noise_spreads_over_every_channel() -> (
    None
):
    z, t = latent(1.0, n=20000, seed=1)
    s = audit.latent_summary(z, t)
    assert s["within_type_participation_ratio"] == pytest.approx(D, rel=0.1)
    assert s["within_type_spread"] == pytest.approx(np.sqrt(D), rel=0.05)


def test_a_code_with_nothing_within_its_types_has_no_within_type_dimension_and_no_context() -> (
    None
):
    z, t = latent(0.0)
    s = audit.latent_summary(z, t)
    assert s["type_share_of_variance"] == pytest.approx(1.0)
    assert s["within_type_participation_ratio"] is None
    assert s["centroid_distance_min_in_spreads"] is None
    n = len(t)
    seq = np.zeros((n, 3))
    structure = np.zeros((n, 3))
    usable = np.ones(n, dtype=bool)
    chain = np.repeat(np.arange(n // 100 + 1), 100)[:n]
    assert audit.context_summary(z, t, seq, structure, usable, chain, 0) is None


def test_the_moments_of_a_standardized_latent_are_zero_and_one() -> None:
    rng = np.random.default_rng(2)
    z = rng.standard_normal((30000, D))
    t = rng.integers(0, TYPES, 30000)
    s = audit.latent_summary(z, t)
    assert s["channel_mean_abs_max"] < 0.05
    assert 0.95 < s["channel_std_range"][0] <= s["channel_std_range"][1] < 1.05
    assert s["type_share_of_variance"] < 0.01  # the types say nothing of pure noise
    assert s["participation_ratio"] == pytest.approx(D, rel=0.1)


def test_the_summary_keeps_the_mean_and_the_covariance_whole() -> None:
    rng = np.random.default_rng(3)
    z = rng.standard_normal((2000, D)) @ np.diag(np.arange(1.0, D + 1)) + 5.0
    t = rng.integers(0, TYPES, 2000)
    s = audit.latent_summary(z, t)
    assert np.allclose(s["mean"], z.mean(0))
    assert np.allclose(np.array(s["covariance"]), np.cov(z.T))
    assert np.array(s["covariance"]).shape == (D, D)


def test_pooled_r2_weighs_each_channel_by_the_variance_it_has_to_explain() -> None:
    assert audit.pooled_r2(np.array([1.0, 0.0]), np.array([1.0, 3.0])) == pytest.approx(
        0.25
    )
    assert audit.pooled_r2(np.array([0.5, 0.5]), np.array([2.0, 7.0])) == pytest.approx(
        0.5
    )


def test_a_context_that_the_residual_depends_on_is_read_beyond_what_shuffled_features_reach() -> (
    None
):
    rng = np.random.default_rng(3)
    chains, length = 40, 60
    n = chains * length
    chain = np.repeat(np.arange(chains), length)
    t = rng.integers(0, TYPES, n)
    centroids = rng.standard_normal((TYPES, D)) * 3.0
    seq = rng.standard_normal((n, 5))  # the neighbours' identities as features
    structure = rng.standard_normal((n, 4))
    weights = rng.standard_normal((5, D))
    z = centroids[t] + seq @ weights * 0.5 + 0.2 * rng.standard_normal((n, D))
    usable = np.ones(n, dtype=bool)
    c = audit.context_summary(z, t, seq, structure, usable, chain, 0)
    assert c is not None and c["residues"] == n
    assert c["sequence"] > 0.7 and c["both"] >= c["sequence"] - 0.02
    assert c["structure"] < 0.05  # unrelated features give nothing
    assert c["sequence_shuffled"] < 0.05 and c["both_shuffled"] < 0.05
    assert len(c["per_channel_both"]) == D
