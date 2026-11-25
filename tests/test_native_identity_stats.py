"""The release encoder's statistics of an in-place code's channels (`src/foldspace/rfd4/native_identity_stats.py`): the mean and the covariance of each residue type from sums, their Cholesky factors, and the pooled
statistics for a type with too few residues. The reading of the release's encoder itself needs torch and the checkpoint and is made on the stored artifacts."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.native_identity_stats as ns

pytestmark = pytest.mark.offline


def accumulate(groups: dict[int, np.ndarray], rows: int):
    d = next(iter(groups.values())).shape[1]
    counts, sums, sumsq = np.zeros(rows), np.zeros((rows, d)), np.zeros((rows, d, d))
    for r, x in groups.items():
        counts[r] = len(x)
        sums[r] = x.sum(0)
        sumsq[r] = x.T @ x
    return counts, sums, sumsq


def test_each_types_mean_and_covariance_come_back_from_the_sums_and_the_factor_reproduces_the_covariance() -> (
    None
):
    rng = np.random.default_rng(0)
    truth = {
        0: (
            np.array([1.0, -2.0, 0.5]),
            np.array([[1.0, 0.3, 0.0], [0.3, 0.5, 0.1], [0.0, 0.1, 2.0]]),
        ),
        1: (
            np.array([-1.0, 0.0, 3.0]),
            np.array([[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]]),
        ),
    }
    groups = {
        r: rng.multivariate_normal(m, c, size=20000) for r, (m, c) in truth.items()
    }
    stats = ns.finalize(*accumulate(groups, rows=3), min_count=500)
    for r, (m, c) in truth.items():
        assert np.abs(stats["mean"][r] - m).max() < 0.05
        assert np.abs(stats["cov"][r] - c).max() < 0.06
        assert (
            np.abs(stats["chol"][r] @ stats["chol"][r].T - stats["cov"][r]).max() < 1e-5
        )  # the ridge is 1e-6
        assert np.allclose(np.triu(stats["chol"][r], 1), 0.0)
    assert stats["pooled"].tolist() == [False, False, True]


def test_a_type_with_too_few_residues_gets_the_statistics_of_all_of_them_together() -> (
    None
):
    rng = np.random.default_rng(1)
    groups = {
        0: rng.normal(2.0, 1.0, size=(3000, 2)),
        1: rng.normal(-2.0, 1.0, size=(3000, 2)),
        2: rng.normal(0.0, 1.0, size=(10, 2)),
    }
    counts, sums, sumsq = accumulate(groups, rows=3)
    stats = ns.finalize(counts, sums, sumsq, min_count=500)
    everything = np.concatenate(list(groups.values()))
    assert stats["pooled"].tolist() == [False, False, True]
    assert np.allclose(stats["mean"][2], everything.mean(0), atol=1e-9)
    assert np.allclose(stats["cov"][2], np.cov(everything.T), atol=1e-9)
    assert np.all(np.linalg.eigvalsh(stats["cov"][2]) > 0)
    # the pooled covariance of two groups 4 apart is wide where the types' own are not
    assert stats["cov"][2][0, 0] > 3 * stats["cov"][0][0, 0]


def test_a_read_of_fewer_than_two_residues_is_refused() -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        ns.finalize(np.array([1.0, 0.0]), np.ones((2, 2)), np.ones((2, 2, 2)))
