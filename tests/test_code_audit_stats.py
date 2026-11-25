"""The statistics of the representation audit (`src/foldspace/rfd4/code_audit_stats.py`), on data whose answer is known: centroids and the variance within types, a similarity alignment that is exact for a similarity
transform and does not exceed its permutation null for unrelated codes, the participation ratio, the context features of an ideal helix and of an extended strand, a grouped regression that learns what is in the features
and cannot learn what is particular to a chain, and the regularizers of a learned posterior against the closed form of the KL of two Gaussians."""

from __future__ import annotations

import foldspace.rfd4.code_audit_stats as cs
import numpy as np
import pytest

pytestmark = pytest.mark.offline


def coded(n_per: int = 200, d: int = 2, noise: float = 0.3, seed: int = 0):
    """Residues of 20 types whose codes are a centroid per type plus noise."""
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((cs.N_TYPES, d)) * 2.0
    t = np.repeat(np.arange(cs.N_TYPES), n_per)
    x = centroids[t] + noise * rng.standard_normal((len(t), d))
    return x, t, centroids


def test_centroids_residuals_and_the_share_of_variance_within_types() -> None:
    x, t, centroids = coded()
    found, counts = cs.type_centroids(x, t)
    assert counts.tolist() == [200.0] * 20
    assert np.abs(found - centroids).max() < 0.15
    resid = cs.within_type_residual(x, t)
    assert np.abs(cs.type_centroids(resid, t)[0]).max() < 1e-9
    share = cs.within_type_share(x, t)
    # the type fixes almost all of the variance: noise 0.3 against centroids spread 2.0 gives a within-type share near 0.02
    assert share.shape == (2,) and (share < 0.05).all()
    # with no type structure at all the whole variance is within types
    rng = np.random.default_rng(1)
    flat = cs.within_type_share(
        rng.standard_normal((4000, 2)), rng.integers(0, 20, 4000)
    )
    assert (flat > 0.95).all()
    # a type with no residues is a row of NaN, not a zero
    missing = cs.type_centroids(x[t != 3], t[t != 3])[0]
    assert np.isnan(missing[3]).all()


def test_a_similarity_transform_is_aligned_exactly_and_unrelated_codes_are_not() -> (
    None
):
    rng = np.random.default_rng(2)
    source = rng.standard_normal((20, 2))
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    target = 3.0 * source @ rotation + np.array([5.0, -2.0])
    assert cs.similarity_r2(target, source) == pytest.approx(1.0, abs=1e-12)
    # a reflection is allowed: the channels of a code have no handedness
    reflected = target * np.array([1.0, -1.0])
    assert cs.similarity_r2(reflected, source) == pytest.approx(1.0, abs=1e-12)
    noisy = target + 0.05 * rng.standard_normal(target.shape)
    assert 0.95 < cs.similarity_r2(noisy, source) < 1.0
    unrelated = rng.standard_normal((20, 2))
    null = cs.permutation_null(target, source, n=300, seed=3)
    assert null.shape == (300,) and ((0.0 <= null) & (null <= 1.0)).all()
    assert cs.similarity_r2(target, source) > np.quantile(null, 0.99)
    # unrelated codes sit inside their own null, which is well above zero for twenty points in two dimensions
    own_null = cs.permutation_null(target, unrelated, n=300, seed=4)
    assert cs.similarity_r2(target, unrelated) < np.quantile(own_null, 0.99)
    assert np.median(null) > 0.05
    # the affine fit is never below the similarity fit
    assert cs.affine_r2(noisy, source) >= cs.similarity_r2(noisy, source) - 1e-12


def test_the_ceiling_of_two_halves_of_the_same_chains_is_high_for_a_code_that_has_type_structure() -> (
    None
):
    x, t, _ = coded(n_per=300, noise=0.5)
    chain = np.arange(len(t)) % 60
    ceiling = cs.split_half_ceiling(x, t, chain, repeats=6)
    assert ceiling.shape == (6,) and ceiling.min() > 0.97
    rng = np.random.default_rng(5)
    flat = cs.split_half_ceiling(
        rng.standard_normal((6000, 2)),
        rng.integers(0, 20, 6000),
        np.arange(6000) % 60,
        repeats=6,
    )
    assert (
        flat.max() < 0.8
    )  # centroids of a code with no type structure do not agree between halves


def test_the_participation_ratio_counts_the_directions_a_covariance_spreads_over() -> (
    None
):
    assert cs.participation_ratio(np.eye(5)) == pytest.approx(5.0)
    assert cs.participation_ratio(np.diag([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert 1.0 < cs.participation_ratio(np.diag([4.0, 1.0, 1.0])) < 3.0


def helix(n: int) -> np.ndarray:
    k = np.arange(n)
    angle = np.deg2rad(100.0) * k
    return np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * k], axis=1)


def strand(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.stack([3.3 * k, 0.9 * (k % 2), np.zeros(n)], axis=1)


def test_the_context_features_of_an_ideal_helix_and_an_extended_strand() -> None:
    n = 30
    types = np.arange(n) % 20
    valid = np.ones(n, dtype=bool)
    seq, structure, usable = cs.chain_features(types, helix(n), valid)
    assert seq.shape == (n, 120) and structure.shape == (n, 12)
    assert usable.tolist() == [False] * 3 + [True] * (n - 6) + [False] * 3
    inside = np.flatnonzero(usable)
    d3 = structure[inside, 2]
    assert ((d3 > 4.8) & (d3 < 5.8)).all()  # an alpha helix: Ca(i) to Ca(i+3) near 5 A
    assert (structure[inside, 6] == 1.0).all() and (structure[inside, 7] == 0.0).all()
    _, extended, _ = cs.chain_features(types, strand(n), valid)
    assert (extended[inside, 7] == 1.0).all() and (extended[inside, 6] == 0.0).all()
    # the neighbours: at offset +1 of residue 10 the type of residue 11, at -3 the type of residue 7 (slots run -3, -2, -1, +1, +2, +3)
    assert np.argmax(seq[10, 3 * 20 : 4 * 20]) == types[11]
    assert np.argmax(seq[10, 0:20]) == types[7]
    # a residue near the end has zeros where the chain stops; a position without a Cα is not usable and its neighbours are not either
    holes = valid.copy()
    holes[15] = False
    _, _, usable_h = cs.chain_features(types, helix(n), holes)
    assert not usable_h[12:19].any() and usable_h[11] and usable_h[19]


def test_a_grouped_regression_learns_what_is_in_the_features_and_not_what_is_particular_to_a_chain() -> (
    None
):
    rng = np.random.default_rng(6)
    n, groups = 4000, np.repeat(np.arange(80), 50)
    x = rng.standard_normal((n, 6))
    w = np.array([1.0, -2.0, 0.0, 0.5, 0.0, 0.0])
    y = x @ w + 0.5 * rng.standard_normal(n)
    r2 = cs.grouped_ridge_r2(x, y, groups)
    expected = 1.0 - 0.25 / (w @ w + 0.25)
    assert r2.shape == (1,) and float(r2[0]) == pytest.approx(expected, abs=0.02)
    # a target that is not in the features: held-out R² is zero or below
    noise = rng.standard_normal(n)
    assert float(cs.grouped_ridge_r2(x, noise, groups)[0]) < 0.01
    # a chain-particular target and a feature that identifies the chain: learnable inside a fold, not across folds
    signature = np.zeros((n, 80))
    signature[np.arange(n), groups] = 1.0
    offset = rng.standard_normal(80)[groups]
    assert float(cs.grouped_ridge_r2(signature, offset, groups, alpha=0.1)[0]) < 0.05
    # several targets at once
    both = cs.grouped_ridge_r2(x, np.stack([x @ w, noise], axis=1), groups)
    assert both.shape == (2,) and both[0] > 0.95 and both[1] < 0.01


def test_the_regularizers_are_the_closed_form_of_the_kl_of_two_gaussians_and_the_rate_term_is_its_width_part() -> (
    None
):
    rng = np.random.default_rng(7)
    mu = rng.standard_normal((500, 18))
    sigma = rng.uniform(0.1, 0.99, size=(500, 18))
    # q = N(sqrt(1 - sigma^2) mu, sigma^2) against N(0, 1): 1/2 (m^2 + s^2 - 1 - ln s^2), summed over channels, averaged over residues
    m = np.sqrt(1 - sigma**2) * mu
    closed = float((0.5 * (m**2 + sigma**2 - 1 - np.log(sigma**2))).sum(-1).mean())
    assert cs.kl_exact(mu, sigma) == pytest.approx(closed, rel=1e-12)
    assert cs.kl_exact(mu, sigma) - cs.kl_rate(sigma) == pytest.approx(
        cs.kl_omitted(mu, sigma), rel=1e-9
    )
    # where the code's second moment is one for every residue the two are the same number
    unit = np.where(rng.random((500, 18)) < 0.5, 1.0, -1.0)
    assert cs.kl_exact(unit, sigma) == pytest.approx(cs.kl_rate(sigma), rel=1e-12)
    assert cs.kl_omitted(unit, sigma) == pytest.approx(0.0, abs=1e-12)
    # and the rate term alone is what a width at its ceiling costs least: -ln 0.99 per channel
    assert cs.kl_rate(np.full((3, 18), 0.99)) == pytest.approx(-18 * np.log(0.99))
