"""The full-width ESMC latent and its type-only control: all 32 channels from ESMC's rows and nothing from the release encoder, and a fixed
code of 32 channels moment-matched to the marginals of the latent it controls. The arithmetic and the refused argument combinations need numpy only; the encoder that gives a flow its target needs torch and the
release's model-mask module and is skipped where they are missing (it runs in the stack's environment)."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.train_autoencoder as autoencoder

pytestmark = pytest.mark.offline


def test_a_standardized_table_has_zero_mean_and_unit_variance_in_every_channel_over_the_residues() -> (
    None
):
    rng = np.random.default_rng(0)
    table = rng.standard_normal((24, 32)) * rng.uniform(0.5, 2.0, 32) + rng.uniform(
        -1.0, 1.0, 32
    )
    weight = rng.uniform(0.0, 1.0, 24)
    weight[[3, 7]] = 0.0  # types that no residue has
    out = autoencoder.standardize_table(table, weight)
    assert out.dtype == np.float32 and out.shape == table.shape
    w = weight / weight.sum()
    mean = (w[:, None] * out.astype(np.float64)).sum(0)
    var = (w[:, None] * (out.astype(np.float64) - mean) ** 2).sum(0)
    assert np.abs(mean).max() < 1e-5
    assert np.abs(var - 1.0).max() < 1e-5


def test_a_standardized_table_keeps_the_order_of_the_rows_in_each_channel() -> None:
    rng = np.random.default_rng(1)
    table = rng.standard_normal((10, 4))
    out = autoencoder.standardize_table(table, np.ones(10))
    for channel in range(4):
        assert np.array_equal(
            np.argsort(table[:, channel]), np.argsort(out[:, channel])
        )  # a per-channel shift and scale: no row changes place


def test_a_standardized_table_refuses_what_has_nothing_to_match() -> None:
    table = np.zeros((5, 3))
    table[:, 0] = np.arange(5)
    with pytest.raises(ValueError, match="no variance"):
        autoencoder.standardize_table(table, np.ones(5))  # channels 1 and 2 are constant
    with pytest.raises(ValueError, match="non-negative"):
        autoencoder.standardize_table(np.eye(5), np.zeros(5))
    with pytest.raises(ValueError, match="non-negative"):
        autoencoder.standardize_table(np.eye(5), -np.ones(5))
    with pytest.raises(ValueError, match="non-negative"):
        autoencoder.standardize_table(np.eye(5), np.ones(4))


def run_refused(*arguments: str, capsys) -> str:
    base = ["--shards", "s", "--ae", "a", "--out", "o"]
    with pytest.raises(SystemExit) as stopped:
        autoencoder.main([*base, *arguments])
    assert stopped.value.code == 2
    return capsys.readouterr().err


def test_the_type_only_standardization_is_refused_where_it_has_no_meaning(
    capsys,
) -> None:
    message = "--standardize-code is for --source fixed without --inplace"
    assert message in run_refused(
        "--source", "esmc", "--standardize-code", capsys=capsys
    )
    assert message in run_refused(
        "--source", "fixed", "--inplace", "--standardize-code", capsys=capsys
    )
    assert message in run_refused(
        "--source", "fixed", "--residual", "8", "--standardize-code", capsys=capsys
    )
