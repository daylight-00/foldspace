"""The in-place latents' identity control and posterior (`src/foldspace/rfd4/train_autoencoder.py`, `src/foldspace/rfd4/flow_model.py`): a fixed code whose types lie far apart, a code that carries Gaussian noise of a fixed
scale when it is sampled, and an encoder that gives a flow the release encoder's *samples* in the channels kept in place. These need numpy only; the tests that need torch and the release's model-mask module are in `test_inplace_posterior_torch.py`."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.train_autoencoder as autoencoder

pytestmark = pytest.mark.offline

STANDARD = tuple(range(1, 21))


def test_separation_is_the_smallest_distance_between_two_rows_of_those_asked_for() -> (
    None
):
    code = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    assert autoencoder.code_separation(code) == pytest.approx(1.0)
    assert autoencoder.code_separation(code, (0, 1)) == pytest.approx(5.0)


def test_the_fixed_code_is_a_function_of_the_seed_and_the_best_of_its_draws() -> None:
    a = autoencoder.separated_code(40, 18, STANDARD)
    b = autoencoder.separated_code(40, 18, STANDARD)
    assert a.shape == (40, 18) and a.dtype == np.float32 and np.array_equal(a, b)
    first = np.random.default_rng(20261112).standard_normal((40, 18)).astype(np.float32)
    # the best of 64 draws is at least the first one, and the types are kept apart by more than a code that a decoder could not tell (0.04 to 0.7)
    assert autoencoder.code_separation(a, STANDARD) >= autoencoder.code_separation(first, STANDARD)
    assert autoencoder.code_separation(a, STANDARD) > 2.5
    other = autoencoder.separated_code(40, 18, STANDARD, seed=7)
    assert not np.array_equal(a, other)


def test_the_in_place_channels_are_the_14_kept_and_the_18_of_the_code() -> None:
    assert len(autoencoder.INPLACE_KEEP) == 14 and len(set(autoencoder.INPLACE_KEEP)) == 14
    assert all(0 <= c < 32 for c in autoencoder.INPLACE_KEEP)


def run_refused(*arguments: str, capsys) -> str:
    """`main` with the required arguments and these, which it must refuse before it loads anything."""
    base = ["--shards", "s", "--ae", "a", "--out", "o", "--inplace"]
    with pytest.raises(SystemExit) as stopped:
        autoencoder.main([*base, *arguments])
    assert stopped.value.code == 2
    return capsys.readouterr().err


def test_a_learned_posterior_is_refused_where_it_has_no_meaning(capsys) -> None:
    ok = ["--source", "esmc", "--posterior-native", "--kl", "0.001"]
    assert "--stochastic-code is for --source esmc" in run_refused(
        "--stochastic-code", "--source", "esmc", "--kl", "0.001", capsys=capsys
    )  # without the release encoder's samples in the channels kept
    assert "--stochastic-code is for --source esmc" in run_refused(
        "--stochastic-code",
        "--source",
        "fixed",
        "--posterior-native",
        "--kl",
        "0.001",
        capsys=capsys,
    )
    assert "--stochastic-code is for --source esmc" in run_refused(
        "--stochastic-code", "--source", "esmc", "--posterior-native", capsys=capsys
    )  # no KL: the width would only fall
    assert "learns the width that --code-sigma fixes" in run_refused(
        "--stochastic-code", *ok, "--code-sigma", "0.2", capsys=capsys
    )
    assert "--kl is the weight of the learned posterior" in run_refused(
        "--source", "esmc", "--posterior-native", "--kl", "0.001", capsys=capsys
    )
    assert "two noises on the same channels" in run_refused(
        "--stochastic-code", *ok, "--noise", "0.2", capsys=capsys
    )
    assert "--kl-exact is for the learned posterior" in run_refused(
        "--source", "esmc", "--posterior-native", "--kl-exact", capsys=capsys
    )


def test_a_code_bank_is_refused_where_it_has_no_meaning(capsys) -> None:
    assert "--code-bank is for --source fixed --inplace" in run_refused(
        "--code-bank", "b.npz", "--source", "esmc", "--posterior-native", capsys=capsys
    )
    assert "--code-bank is for --source fixed --inplace" in run_refused(
        "--code-bank", "b.npz", "--source", "fixed", capsys=capsys
    )
    assert "two tables for the same code" in run_refused(
        "--code-bank",
        "b.npz",
        "--native-code",
        "n.npz",
        "--source",
        "fixed",
        "--posterior-native",
        capsys=capsys,
    )
