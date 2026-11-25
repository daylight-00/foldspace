"""The release bundle's denoiser tensors (`src/foldspace/rfd4/flow_model.py`, `nn_state_from_bundle`): the release's flow checkpoint holds its adapter
unmerged and its bundle holds it merged, and the denoiser built here has plain layers, so the weights of the release are the bundle's `nn.` tensors."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.flow_model as fm

pytestmark = pytest.mark.offline


def test_the_bundles_denoiser_tensors_are_read_without_their_prefix_and_the_autoencoders_are_left(
    tmp_path,
) -> None:
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    path = tmp_path / "model.safetensors"
    tensors = {
        "nn.ca_linear.0.weight": np.ones((3, 2), dtype=np.float32),
        "nn.transformer_layers.0.mhba.mha.to_qkv.weight": np.arange(
            6, dtype=np.float32
        ).reshape(2, 3),
        "autoencoder.decoder.atom_presence_linear.1.weight": np.zeros(
            (37, 4), dtype=np.float32
        ),
    }
    safetensors_numpy.save_file(tensors, str(path))
    state = fm.nn_state_from_bundle(path)
    assert set(state) == {
        "ca_linear.0.weight",
        "transformer_layers.0.mhba.mha.to_qkv.weight",
    }
    assert np.array_equal(
        state["transformer_layers.0.mhba.mha.to_qkv.weight"],
        tensors["nn.transformer_layers.0.mhba.mha.to_qkv.weight"],
    )
