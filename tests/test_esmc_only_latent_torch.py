"""The full-width ESMC latent and its type-only control at torch level: the encoder that gives a flow its target reads nothing from the
release encoder, is deterministic, and the type-only code is a function of the residue type alone. They need torch and the release's model-mask module and are skipped where those are missing (they run in the
stack's environment). The arithmetic and the refused argument combinations are in `test_esmc_only_latent.py`."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.train_autoencoder as autoencoder

pytestmark = pytest.mark.offline

torch = pytest.importorskip("torch")
pytest.importorskip("rfproteina.model_masks")
import foldspace.rfd4.flow_model as fm

BATCH, LENGTH, TYPES, ROWS = 3, 120, 30, 64


class Boom(torch.nn.Module):
    """The release encoder where nothing may call it: a latent without native channels never reads it."""

    def forward(self, batch):
        raise AssertionError(
            "the release encoder is on the path of a full-width latent"
        )


def tiny_batch() -> dict:
    return {
        "padding_mask": torch.ones(BATCH, LENGTH, dtype=torch.bool),
        "is_in_context_conditioning": torch.zeros(BATCH, LENGTH, dtype=torch.bool),
        "supports_local_latent": torch.ones(BATCH, LENGTH, dtype=torch.bool),
        "residue_type": torch.randint(1, 21, (BATCH, LENGTH)),
        "esmc": torch.randn(BATCH, LENGTH, ROWS),
    }


def encoder(latent):
    return fm.esm_encoder_class()(Boom(), latent, residual=0)


def test_the_full_width_latent_is_32_channels_of_the_rows_and_never_reads_the_release_encoder() -> (
    None
):
    torch.manual_seed(0)
    latent = autoencoder.anchored_class()("esmc", TYPES, in_dim=ROWS)
    enc = encoder(latent)
    batch = tiny_batch()
    out = enc(batch)  # Boom would raise
    assert out["z_latent"].shape == (BATCH, LENGTH, 32)
    assert torch.equal(out["z_latent"], out["mean"])  # deterministic: no posterior
    assert torch.equal(out["z_latent"], enc(batch)["z_latent"])
    # the whole width is a function of the rows: other rows, another latent
    changed = dict(batch, esmc=batch["esmc"] + 1.0)
    assert not torch.allclose(enc(changed)["z_latent"], out["z_latent"])
    # and of nothing else about the chain: the residue types are not read
    retyped = dict(batch, residue_type=(batch["residue_type"] % 19) + 1)
    assert torch.equal(enc(retyped)["z_latent"], out["z_latent"])


def test_the_type_only_code_is_a_function_of_the_residue_type_alone() -> None:
    latent = autoencoder.anchored_class()("fixed", TYPES)
    table = autoencoder.standardize_table(latent.code.numpy(), np.ones(latent.code.shape[0]))
    with torch.no_grad():
        latent.code.copy_(torch.from_numpy(table))
    enc = encoder(latent)
    batch = tiny_batch()
    out = enc(batch)  # Boom would raise
    assert out["z_latent"].shape == (BATCH, LENGTH, 32)
    assert torch.equal(out["z_latent"], out["mean"])
    assert torch.equal(
        out["z_latent"], latent.code[batch["residue_type"]].to(out["z_latent"])
    )
    # the rows are not read, and equal types have equal codes
    other = dict(batch, esmc=torch.randn_like(batch["esmc"]))
    assert torch.equal(enc(other)["z_latent"], out["z_latent"])
    same = batch["residue_type"] == batch["residue_type"][0, 0]
    codes = out["z_latent"][same]
    assert torch.equal(codes, codes[:1].expand_as(codes))
    # the standardized table has the marginals of a moment-matched latent: zero mean and unit variance over a uniform residue distribution
    z = out["z_latent"].reshape(-1, 32)
    assert z.shape[1] == 32
    flat = latent.code.numpy()
    assert np.abs(flat.mean(0)).max() < 1e-5 and np.abs(flat.std(0) - 1).max() < 1e-4
