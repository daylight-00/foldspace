"""The banked code of the control `type_shuffled_hybrid` at torch level (`src/foldspace/rfd4/train_autoencoder.py`, `src/foldspace/rfd4/flow_model.py`): a fixed in-place code whose draws are the codes of other residues of the same type from a bank, with the noise of
the arm it matches, and an encoder that gives a flow a draw of it. They need torch and the release's model-mask module: they are skipped where those are not installed and run in the stack's environment. The layout of the bank and the
verdict on the control are in `test_code_bank.py`, the arguments the trainer refuses in `test_inplace_posterior.py`."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.code_bank as bk
import foldspace.rfd4.train_autoencoder as autoencoder

pytestmark = pytest.mark.offline

torch = pytest.importorskip("torch")
pytest.importorskip("rfproteina.model_masks")
import foldspace.rfd4.flow_model as fm

BATCH, LENGTH = 4, 400


def batch() -> dict:
    return {
        "padding_mask": torch.ones(BATCH, LENGTH, dtype=torch.bool),
        "is_in_context_conditioning": torch.zeros(BATCH, LENGTH, dtype=torch.bool),
        "supports_local_latent": torch.ones(BATCH, LENGTH, dtype=torch.bool),
        "residue_type": torch.randint(1, 21, (BATCH, LENGTH)),
        "esmc": torch.randn(BATCH, LENGTH, 64),
    }


class StubNative(torch.nn.Module):
    """A release encoder whose sample differs from its mean by exactly one in every channel."""

    def forward(self, b: dict) -> dict:
        mean = torch.zeros(BATCH, LENGTH, 32)
        return {"mean": mean, "z_latent": mean + 1.0}


def banked_latent(tmp_path, sigma: float = 0.0, seed: int = 0):
    """A fixed in-place code whose draws are donors of the residue's type from a bank of synthetic codes (the real bank is `src/foldspace/rfd4/code_bank.py`'s): twenty types with 700 donors each."""
    rng = np.random.default_rng(seed)
    types = np.repeat(np.arange(1, 21), 700)
    centres = rng.normal(size=(31, 18)) * 3.0
    codes = centres[types] + rng.normal(size=(len(types), 18)) * 0.5
    layout = bk.bank_layout(types, codes, 31, min_count=500)
    latent = autoencoder.anchored_class()("fixed", 30, inplace=True, banked=len(types))
    path = tmp_path / "bank.npz"
    np.savez(path, **layout, channels=np.array(latent.code_idx))
    autoencoder.load_code_bank(latent, str(path))
    latent.code_sigma = sigma
    latent.eval()
    return latent, layout, path


def test_a_banked_code_is_the_code_of_another_residue_of_its_type_and_its_mean_is_the_centroid(
    tmp_path,
) -> None:
    latent, layout, _ = banked_latent(tmp_path)
    b = batch()
    native = torch.randn(BATCH, LENGTH, 32)
    keep, code = latent.keep, latent.code_idx
    # without a sample the code is the type's centroid, and the channels kept are the release's
    z = latent(b, native=native)
    assert torch.equal(
        z[..., code], torch.from_numpy(layout["mean"])[b["residue_type"]]
    )
    assert torch.equal(z[..., keep], native[..., keep])
    # sampled, every residue's code is a row of the bank that belongs to its own type
    torch.manual_seed(1)
    drawn = latent(b, native=native, sample=True)[..., code].reshape(-1, 18)
    bank, bank_types = (
        torch.from_numpy(layout["code_bank"]),
        torch.from_numpy(layout["bank_types"]),
    )
    # the exact mode: the default expands the square and is only good to a few thousandths on rows this long
    distance = torch.cdist(drawn, bank, compute_mode="donot_use_mm_for_euclid_dist")
    assert float(distance.min(1).values.max()) == 0.0
    assert torch.equal(bank_types[distance.argmin(1)], b["residue_type"].reshape(-1))
    assert torch.equal(
        latent(b, native=native, sample=True)[..., keep], native[..., keep]
    )


def test_banked_draws_are_independent_between_residues_and_between_calls(
    tmp_path,
) -> None:
    latent, layout, _ = banked_latent(tmp_path)
    b = batch()
    b["residue_type"] = torch.full((BATCH, LENGTH), 5)
    native = torch.randn(BATCH, LENGTH, 32)
    torch.manual_seed(2)
    first = latent(b, native=native, sample=True)[..., latent.code_idx].reshape(-1, 18)
    second = latent(b, native=native, sample=True)[..., latent.code_idx].reshape(-1, 18)
    assert not torch.equal(first, second)
    # 1,600 residues of one type draw their donors among its 700: most of the donors are different residues, so nothing of a residue's own position decides which code it has
    assert len({tuple(row.tolist()) for row in first}) > 500
    # the draws are uniform over the donors: their mean is the type's centroid
    many = torch.cat(
        [
            latent(b, native=native, sample=True)[..., latent.code_idx].reshape(-1, 18)
            for _ in range(20)
        ]
    )
    assert float((many.mean(0) - torch.from_numpy(layout["mean"])[5]).abs().max()) < 0.1


def test_the_noise_of_a_banked_code_is_added_to_the_donor_and_nothing_else_changes(
    tmp_path,
) -> None:
    quiet, _, _ = banked_latent(tmp_path)
    noisy, _, _ = banked_latent(tmp_path, sigma=0.6)
    b = batch()
    b["residue_type"] = torch.full((BATCH, LENGTH), 5)
    native = torch.randn(BATCH, LENGTH, 32)
    torch.manual_seed(3)
    plain = torch.cat(
        [
            quiet(b, native=native, sample=True)[..., quiet.code_idx].reshape(-1, 18)
            for _ in range(20)
        ]
    )
    torch.manual_seed(3)
    with_noise = torch.cat(
        [
            noisy(b, native=native, sample=True)[..., noisy.code_idx].reshape(-1, 18)
            for _ in range(20)
        ]
    )
    # a donor's own spread plus the noise's variance of 0.36, by channel
    assert float(((with_noise.var(0) - plain.var(0)) - 0.36).abs().max()) < 0.06
    # no sample, no noise: the centroid
    assert torch.equal(
        noisy(b, native=native)[..., noisy.code_idx],
        quiet(b, native=native)[..., quiet.code_idx],
    )


def test_a_banked_latent_says_so_in_its_state_loads_only_a_bank_of_its_own_channels_and_is_rebuilt_from_it(
    tmp_path,
) -> None:
    latent, layout, path = banked_latent(tmp_path)
    state = latent.state_dict()
    assert {"code_bank", "bank_start", "bank_count", "code", "inplace_flag"} <= set(
        state
    )
    assert "code_chol" not in state
    assert (
        "code_bank" not in autoencoder.anchored_class()("fixed", 30, inplace=True).state_dict()
    )
    rebuilt = autoencoder.anchored_class()(
        "fixed", 30, inplace=True, banked=int(state["code_bank"].shape[0])
    )
    rebuilt.load_state_dict(state)
    assert torch.equal(rebuilt.code_bank, latent.code_bank) and torch.equal(
        rebuilt.bank_count, latent.bank_count
    )
    with pytest.raises(ValueError, match="a fixed code in place"):
        autoencoder.anchored_class()("esmc", 30, inplace=True, in_dim=64, banked=10)
    with pytest.raises(ValueError, match="one table or the other"):
        autoencoder.anchored_class()("fixed", 30, inplace=True, banked=10, calibrated=True)
    other = autoencoder.anchored_class()(
        "fixed", 30, inplace=True, banked=len(layout["code_bank"])
    )
    contents = dict(np.load(path))
    np.savez(path, **{**contents, "channels": np.arange(18)})
    with pytest.raises(ValueError, match="not of this latent's code channels"):
        autoencoder.load_code_bank(other, str(path))
    np.savez(path, **{**contents, "mean": contents["mean"][:10]})
    with pytest.raises(ValueError, match="mean has shape"):
        autoencoder.load_code_bank(other, str(path))
    broken = np.array(contents["bank_count"])
    broken[1] = len(layout["code_bank"]) + 1
    np.savez(path, **{**contents, "bank_count": broken})
    with pytest.raises(ValueError, match="are not rows of the bank"):
        autoencoder.load_code_bank(other, str(path))


def test_the_encoder_gives_the_flow_a_draw_of_the_banked_code_and_its_centroid_beside_it(
    tmp_path,
) -> None:
    latent, layout, _ = banked_latent(tmp_path, sigma=0.6)
    b = batch()
    code = latent.code_idx
    out = fm.esm_encoder_class()(StubNative(), latent, 0, native_sample=True)(b)
    assert torch.equal(
        out["mean"][..., code], torch.from_numpy(layout["mean"])[b["residue_type"]]
    )
    assert not torch.equal(out["z_latent"][..., code], out["mean"][..., code])
    assert torch.all(
        out["z_latent"][..., latent.keep] == 1.0
    )  # the release encoder's sample in the channels kept
