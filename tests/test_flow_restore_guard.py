"""A reading refuses a checkpoint that is not at the update it was asked for (`src/foldspace/rfd4/sample_flow.py`, `restore`). A run continued from a copy of
an earlier checkpoint left that copy under the name the readings open, and the reading of the wrong weights looked like a finding."""

from __future__ import annotations

import pytest
import foldspace.rfd4.sample_flow as fs

pytestmark = pytest.mark.offline


def test_a_checkpoint_at_another_update_is_refused_and_one_at_the_asked_for_update_is_restored(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "last.ckpt"
    torch.save(
        {
            "global_step": 800,
            "epoch": 0,
            "nsamples_processed": 6400,
            "state_dict": state,
            "callbacks": {"SimpleEMA": {"ema_state_dict": state}},
        },
        path,
    )
    with pytest.raises(
        SystemExit, match="is at update 800, and update 1600 was asked for"
    ):
        fs.restore(model, path, "ema", expect_step=1600)
    info = fs.restore(model, path, "ema", expect_step=800)
    assert info["global_step"] == 800 and info["state"] == "ema"
    # without an expectation the checkpoint is read as it always was
    assert fs.restore(model, path, "raw")["global_step"] == 800
