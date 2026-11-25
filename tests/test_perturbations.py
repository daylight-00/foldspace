"""The perturbations of robustness, on arrays whose answer is known. Numpy only."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from foldspace.esmfold2 import perturb as P

pytestmark = pytest.mark.offline

D = 32


def _chain(seed=0, length=60):
    return np.random.default_rng(seed).standard_normal((length, D))


def _spectrum(seed=1, decay=0.8):
    """A covariance with a few large directions and many small ones, as the singles have."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    values = decay ** np.arange(D)
    covariance = (q * values) @ q.T
    return P.spectrum_from_covariance(covariance), covariance


def test_the_conditions_are_eighteen_distinct_ones_in_a_fixed_order():
    conditions = P.conditions()
    assert len(conditions) == 18 and len(set(conditions)) == 18
    assert [f for f, _ in conditions][:4] == ["noise"] * 4
    # the family added after the first four were read stays last, so that their realisations are the ones that were read
    assert [f for f, _ in conditions][-4:] == ["anoise"] * 4
    assert P.FAMILIES[:4] == ("noise", "shrink", "rank", "dropout")
    assert P.name_of("rank", 64) == "rank_64" and P.name_of("noise", 0.4) == "noise_0.4"
    assert P.name_of("anoise", 0.05) == "anoise_0.05"
    assert P.name_of("dropout", 0.30) == "dropout_0.3"
    assert len({P.name_of(f, level) for f, level in conditions}) == 18


def test_noise_has_the_level_as_its_standard_deviation_and_is_float32():
    z = _chain(length=4000)
    out = P.perturb(z, "noise", 0.4, np.random.default_rng(0))
    assert out.dtype == np.float32 and out.shape == z.shape
    assert np.std(out - z) == pytest.approx(0.4, rel=0.02)
    assert abs(np.mean(out - z)) < 0.01


def test_anoise_has_the_data_covariance_times_the_level_squared():
    spectrum, covariance = _spectrum()
    z = _chain(length=60000)
    out = P.perturb(z, "anoise", 0.5, np.random.default_rng(0), spectrum=spectrum)
    error = (out - z).astype(np.float64)
    sample = error.T @ error / len(error)
    assert np.allclose(sample, 0.25 * covariance, atol=0.02)
    # in every direction the error is `level` times that direction's own standard deviation
    along = np.einsum("ij,jk,ki->i", spectrum.vectors.T, sample, spectrum.vectors)
    ratio = along[:8] / (0.25 * spectrum.values[:8])
    assert np.allclose(ratio, 1.0, atol=0.05)
    # the same error in a small direction that isotropic noise of the same level would swamp
    iso = P.perturb(z, "noise", 0.5, np.random.default_rng(0)) - z
    iso_along = np.einsum(
        "ij,jk,ki->i",
        spectrum.vectors.T,
        iso.astype(np.float64).T @ iso / len(iso),
        spectrum.vectors,
    )
    assert iso_along[-1] > 100 * along[-1]


def test_the_same_generator_gives_the_same_perturbation_and_conditions_are_independent():
    z = _chain()
    a = P.perturb(z, "noise", 0.2, P.rng_for(7, 3, "noise", 0.2))
    b = P.perturb(z, "noise", 0.2, P.rng_for(7, 3, "noise", 0.2))
    c = P.perturb(z, "noise", 0.4, P.rng_for(7, 3, "noise", 0.4))
    d = P.perturb(z, "noise", 0.2, P.rng_for(7, 4, "noise", 0.2))
    assert np.array_equal(a, b)
    # another level or another chain is another stream, not the same draw scaled
    assert not np.allclose((c - z) / 0.4, (a - z) / 0.2)
    assert not np.allclose(a, d)


def test_a_chain_is_keyed_by_its_id_and_levels_run_from_mild_to_severe():
    a = P.rng_for(7, "1ABC_1", "noise", 0.2).standard_normal(5)
    b = P.rng_for(7, "1ABC_1", "noise", 0.2).standard_normal(5)
    c = P.rng_for(7, "1ABD_1", "noise", 0.2).standard_normal(5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert P.severity_order("noise") == [0.1, 0.2, 0.4, 0.8]
    assert P.severity_order("anoise") == [0.05, 0.1, 0.2, 0.4]
    assert P.severity_order("dropout") == [0.05, 0.15, 0.3]
    assert P.severity_order("rank") == [128, 64, 32, 16]  # fewer directions is worse


def test_shrink_removes_that_fraction_of_the_deviation_and_its_size_is_the_fraction():
    z = _chain()
    out = P.perturb(z, "shrink", 0.25, np.random.default_rng(0))
    assert np.allclose(out, 0.75 * z, atol=1e-6)
    assert P.relative_size(z, out) == pytest.approx(0.25, abs=1e-6)


def test_the_spectrum_is_orthonormal_ordered_and_never_negative():
    spectrum, covariance = _spectrum()
    assert np.allclose(spectrum.vectors.T @ spectrum.vectors, np.eye(D), atol=1e-9)
    assert np.all(np.diff(spectrum.values) <= 1e-12)
    along = np.einsum("ij,jk,ki->i", spectrum.vectors.T, covariance, spectrum.vectors)
    assert np.allclose(along, spectrum.values, atol=1e-9)
    # a covariance that is singular up to rounding does not give an imaginary noise
    singular = np.outer(np.ones(D), np.ones(D))
    assert P.spectrum_from_covariance(singular).values.min() >= 0.0
    assert np.array_equal(P.basis_from_covariance(covariance), spectrum.vectors)


def test_rank_is_a_projection_onto_the_leading_directions():
    spectrum, _ = _spectrum()
    z = _chain()
    rng = np.random.default_rng(0)
    out = P.perturb(z, "rank", 8, rng, spectrum=spectrum)
    again = P.perturb(out.astype(np.float64), "rank", 8, rng, spectrum=spectrum)
    assert np.allclose(out, again, atol=1e-5)  # idempotent
    # what is removed is orthogonal to what is kept
    assert np.abs((z - out) @ spectrum.vectors[:, :8]).max() < 1e-4
    # all the directions give the single back, and the cost falls as more are kept
    full = P.perturb(z, "rank", D, rng, spectrum=spectrum)
    assert np.allclose(full, z, atol=1e-5)
    sizes = [
        P.relative_size(z, P.perturb(z, "rank", k, rng, spectrum=spectrum))
        for k in (4, 8, 16, 24)
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_dropout_sets_exactly_that_many_positions_to_the_mean():
    z = _chain(length=80) + 3.0  # no row is zero by accident
    out = P.perturb(z, "dropout", 0.15, np.random.default_rng(0))
    zeroed = np.flatnonzero(~out.any(axis=1))
    assert len(zeroed) == round(0.15 * 80) == 12
    kept = np.setdiff1d(np.arange(80), zeroed)
    assert np.allclose(out[kept], z[kept], atol=1e-6)
    other = P.perturb(z, "dropout", 0.15, np.random.default_rng(1))
    assert not np.array_equal(np.flatnonzero(~other.any(axis=1)), zeroed)
    # the count does not depend on the seed
    assert (~other.any(axis=1)).sum() == 12


def test_what_is_not_a_condition_is_refused():
    z = _chain()
    rng = np.random.default_rng(0)
    spectrum = _spectrum()[0]
    with pytest.raises(ValueError, match="unknown family"):
        P.perturb(z, "rotate", 1.0, rng)
    with pytest.raises(ValueError, match="a fraction"):
        P.perturb(z, "shrink", 1.5, rng)
    with pytest.raises(ValueError, match="a share"):
        P.perturb(z, "dropout", 2.0, rng)
    with pytest.raises(ValueError, match="needs the spectrum"):
        P.perturb(z, "rank", 8, rng)
    with pytest.raises(ValueError, match="needs the spectrum"):
        P.perturb(z, "anoise", 0.1, rng)
    with pytest.raises(ValueError, match="directions"):
        P.perturb(z, "rank", D + 1, rng, spectrum=spectrum)
    with pytest.raises(ValueError, match=r"\[L, D\]"):
        P.perturb(z[0], "noise", 0.1, rng)
    with pytest.raises(ValueError, match="no scale"):
        P.relative_size(np.zeros((3, D)), np.ones((3, D)))
    with pytest.raises(ValueError, match="square"):
        P.spectrum_from_covariance(np.zeros((3, 4)))


def test_the_coarse_levels_follow_the_standard_ones_and_keep_their_realisations():
    coarse = P.conditions("coarse")
    assert len(coarse) == 13 and len(set(coarse)) == 13
    assert {f for f, _ in coarse} == set(P.FAMILIES)
    # no coarse level is a standard one, and each is more severe than every standard level of its family
    for family in P.FAMILIES:
        standard = P.severity_order(family)
        harsher = P.severity_order(family, "coarse")
        assert not set(standard) & set(harsher)
        assert (
            (harsher[0] < standard[-1])
            if family == "rank"
            else (harsher[0] > standard[-1])
        )
    # a standard condition's generator is the one it had before the coarse levels were added
    a = P.rng_for(7, "1ABC_1", "noise", 0.2).standard_normal(3)
    b = np.random.default_rng(
        np.random.SeedSequence([7, zlib.crc32(b"1ABC_1"), 0, 1])
    ).standard_normal(3)
    assert np.array_equal(a, b)
    # a coarse level has a generator of its own
    c = P.rng_for(7, "1ABC_1", "noise", 1.6).standard_normal(3)
    assert not np.array_equal(a, c)
    with pytest.raises(ValueError, match="standard or coarse"):
        P.levels_of("noise", "finer")


def test_the_coarsest_shrink_is_the_mean_single_and_the_coarsest_rank_is_one_direction():
    spectrum, _ = _spectrum()
    z = _chain()
    assert np.array_equal(
        P.perturb(z, "shrink", 1.0, np.random.default_rng(0)),
        np.zeros_like(z, dtype=np.float32),
    )
    one = P.perturb(z, "rank", 1, np.random.default_rng(0), spectrum=spectrum)
    assert np.linalg.matrix_rank(one.astype(np.float64), tol=1e-5) == 1
