"""The trace of a sampling trajectory (`src/foldspace/rfd4/flow_trace.py`) and the calibration of generated codes (`src/foldspace/rfd4/code_calibration.py`): the pooled moments equal the moments of the pooled rows, the ideal variance
path and the contraction time are what they say, and the map gives codes the moments of their target. Numpy only, and one contract test of the upstream sampler's arguments that needs the RFD4 source."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.offline

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ft = load("flow_trace", "src/foldspace/rfd4/flow_trace.py")
cc = load("code_calibration", "src/foldspace/rfd4/code_calibration.py")
W = 5


def snapshot(step: int, x1: np.ndarray, xt: np.ndarray, ca: np.ndarray) -> dict:
    return {
        "step": step,
        "x_1": {"local_latents": x1, "bb_ca": ca},
        "x_t": {"local_latents": xt, "bb_ca": ca},
    }


def test_the_pooled_moments_are_those_of_the_pooled_rows() -> None:
    rng = np.random.default_rng(0)
    batches = [
        (
            rng.standard_normal((4, 9, W)) * [1, 2, 3, 4, 5] + 1.0,
            rng.standard_normal((4, 9, 3)),
        ),
        (
            rng.standard_normal((3, 12, W)) * [5, 4, 3, 2, 1] - 2.0,
            rng.standard_normal((3, 12, 3)),
        ),
    ]
    trace = ft.Trace(W)
    for x1, ca in batches:
        trace.add([snapshot(0, x1, x1 * 0.5, ca), snapshot(9, x1 * 2.0, x1, ca)])
    a = trace.arrays(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    pooled = np.concatenate([b[0].reshape(-1, W) for b in batches])
    assert np.allclose(a["mean_x1"][0], pooled.mean(0))
    assert np.allclose(a["cov_x1"][0], np.cov(pooled.T, bias=True))
    assert np.allclose(a["cov_x1"][1], 4.0 * np.cov(pooled.T, bias=True))
    assert np.allclose(a["cov_xt"][0], 0.25 * np.cov(pooled.T, bias=True))
    assert a["tokens"].tolist() == [4 * 9 + 3 * 12] * 2
    assert a["step"].tolist() == [0, 9]


def test_the_ca_step_is_the_mean_distance_of_consecutive_rows_of_the_clean_prediction() -> (
    None
):
    ca = np.zeros((2, 5, 3))
    ca[..., 0] = 0.4 * np.arange(5)  # 0.4 between neighbours
    x = np.zeros((2, 5, W))
    trace = ft.Trace(W)
    trace.add([snapshot(0, x, x, ca)])
    assert trace.arrays(np.linspace(0, 1, 3), np.linspace(0, 1, 3))[
        "ca_step_mean"
    ] == pytest.approx([0.4])


def test_the_times_of_a_snapshot_are_the_predictions_and_the_states_of_the_schedule() -> (
    None
):
    x, ca = np.zeros((1, 3, W)), np.zeros((1, 3, 3))
    trace = ft.Trace(W)
    trace.add([snapshot(2, x, x, ca), snapshot(7, x, x, ca)])
    ts = np.arange(11) / 10.0
    a = trace.arrays(ts, ts + 0.01)
    assert a["t_pred_latent"].tolist() == pytest.approx([0.2, 0.7])
    assert a["t_state_latent"].tolist() == pytest.approx([0.3, 0.8])
    assert a["t_pred_ca"].tolist() == pytest.approx([0.21, 0.71])


def test_a_snapshot_has_one_step_across_batches() -> None:
    x, ca = np.zeros((1, 3, W)), np.zeros((1, 3, 3))
    trace = ft.Trace(W)
    trace.add([snapshot(0, x, x, ca)])
    with pytest.raises(ValueError, match="snapshot 0"):
        trace.add([snapshot(5, x, x, ca)])


def test_the_collapse_time_is_the_first_time_a_set_is_under_half_of_the_reference() -> (
    None
):
    t = np.array([0.001, 0.05, 0.18, 0.38, 0.65, 0.98])
    reference = np.array([0.01, 1.5, 3.0, 4.0, 6.0, 6.9])
    assert ft.collapse_time(t, reference, 0.9 * reference) is None
    mid = np.array([0.0, 1.5, 3.0, 1.9, 2.0, 2.0])  # 1.9 is under half of 4.0
    assert ft.collapse_time(t, reference, mid) == 0.38
    assert (
        ft.collapse_time(t, reference, np.array([0.01, 1.5, 3.0, 4.0, 6.0, 3.0]))
        == 0.98
    )
    # exactly half is not under it, and the first snapshot (the prediction of pure noise) is left out
    assert ft.collapse_time(t, reference, 0.5 * reference) is None
    assert (
        ft.collapse_time(t, reference, np.array([0.0, 1.5, 3.0, 4.0, 6.0, 6.9])) is None
    )


def test_the_map_gives_codes_the_mean_and_covariance_of_the_target() -> None:
    rng = np.random.default_rng(1)
    mix_from, mix_to = rng.standard_normal((W, W)), rng.standard_normal((W, W))
    mean_from, mean_to = rng.standard_normal(W), rng.standard_normal(W)
    z = rng.standard_normal((60000, W)) @ mix_from + mean_from
    cov_to = mix_to.T @ mix_to + np.eye(W)
    mean, cov = cc.moments_of([z[:30000], z[30000:]])
    a, b = cc.calibration_map(mean, cov, mean_to, cov_to, ridge=0.0)
    mapped = cc.apply(z, a, b)
    assert np.allclose(mapped.mean(0), mean_to, atol=1e-6)
    assert np.allclose(np.cov(mapped.T), cov_to, atol=1e-6)
    report = cc.check([z[:30000], z[30000:]], a, b, mean_to, cov_to)
    assert report["mean_abs_max_difference"] < 1e-6
    assert report["covariance_relative_difference"] < 1e-6


def test_the_map_is_the_identity_where_the_moments_already_agree() -> None:
    rng = np.random.default_rng(2)
    m = rng.standard_normal((W, W))
    cov = m.T @ m + np.eye(W)
    mean = rng.standard_normal(W)
    a, b = cc.calibration_map(mean, cov, mean, cov, ridge=0.0)
    assert np.allclose(a, np.eye(W), atol=1e-8) and np.allclose(b, 0.0, atol=1e-8)


def test_a_ridge_keeps_a_direction_of_almost_no_variance_from_being_blown_up() -> None:
    cov_from = np.diag([1.0, 1.0, 1.0, 1.0, 1e-10])
    target = np.eye(W)
    a_raw, _ = cc.calibration_map(np.zeros(W), cov_from, np.zeros(W), target, ridge=0.0)
    a_ridge, _ = cc.calibration_map(np.zeros(W), cov_from, np.zeros(W), target)
    assert np.linalg.svd(a_raw, compute_uv=False)[0] == pytest.approx(1e5, rel=1e-6)
    assert np.linalg.svd(a_ridge, compute_uv=False)[0] < 20.0


def test_the_upstream_sampler_takes_the_arguments_the_trace_asks_for() -> None:
    """`--trace-every` goes through `full_simulation`'s own keyword arguments and reads the schedule by `sample_schedule`: a rename upstream fails here, not in a draw."""
    matcher = pytest.importorskip(
        "rfproteina.flow_matching.product_space_flow_matcher",
        reason="needs the RFD4-Proteina source",
    )
    params = inspect.signature(
        matcher.ProductSpaceFlowMatcher.full_simulation
    ).parameters
    assert "save_trajectory_every" in params and "dump_noisy_trajectories" in params
    assert {"nsteps", "sampling_model_args"} <= set(
        inspect.signature(matcher.ProductSpaceFlowMatcher.sample_schedule).parameters
    )
    source = inspect.getsource(matcher.ProductSpaceFlowMatcher.full_simulation)
    for key in ('"x_1"', '"x_t"', '"step"', "trajectory_steps"):
        assert key in source, f"full_simulation no longer names {key}"
