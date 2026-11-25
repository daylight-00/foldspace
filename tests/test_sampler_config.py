"""The sampler's noise scales and the record of the configuration it integrated with (`src/foldspace/rfd4/sample_flow.py`)."""

from __future__ import annotations

import pytest
import foldspace.rfd4.sample_flow as sm

pytestmark = pytest.mark.offline


def checkpoint_config() -> dict:
    """The part of a flow checkpoint's generation configuration that the noise scales live in: the release's flows carry 0.3 for the Cα and 0.1 for the latent."""
    return {
        modality: {
            "simulation_step_params": {
                "sampling_mode": "vf_tsr",
                "sc_scale_noise": noise,
                "tsr_k": 16.0,
            }
        }
        for modality, noise in (("bb_ca", 0.3), ("local_latents", 0.1))
    }


def test_the_checkpoints_own_noise_scales_are_kept_unless_others_are_asked_for() -> (
    None
):
    cfg = checkpoint_config()
    sm.apply_noise_scales(cfg, "")
    assert sm.sampler_record(cfg) == {
        "bb_ca": {"sampling_mode": "vf_tsr", "sc_scale_noise": 0.3},
        "local_latents": {"sampling_mode": "vf_tsr", "sc_scale_noise": 0.1},
    }
    sm.apply_noise_scales(cfg, "1.0,1.0")
    assert sm.sampler_record(cfg) == {
        "bb_ca": {"sampling_mode": "vf_tsr", "sc_scale_noise": 1.0},
        "local_latents": {"sampling_mode": "vf_tsr", "sc_scale_noise": 1.0},
    }
    # nothing else in the configuration was touched
    assert cfg["bb_ca"]["simulation_step_params"]["tsr_k"] == 16.0


def test_the_two_scales_are_the_cas_and_then_the_latents() -> None:
    cfg = checkpoint_config()
    sm.apply_noise_scales(cfg, "0.5,2")
    record = sm.sampler_record(cfg)
    assert record["bb_ca"]["sc_scale_noise"] == 0.5
    assert record["local_latents"]["sc_scale_noise"] == 2.0


@pytest.mark.parametrize("bad", ["1.0", "1.0,1.0,1.0", "-1,1", "a,b"])
def test_noise_scales_are_two_numbers_that_are_not_negative(bad: str) -> None:
    with pytest.raises(ValueError):
        sm.apply_noise_scales(checkpoint_config(), bad)


def test_the_record_is_read_back_from_the_configuration_and_a_missing_setting_is_not_defaulted() -> (
    None
):
    cfg = checkpoint_config()
    del cfg["local_latents"]["simulation_step_params"]["sc_scale_noise"]
    with pytest.raises(KeyError):
        sm.sampler_record(cfg)
