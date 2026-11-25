"""The latent generation driver on the synthetic world: the readings on known errors, the stages, training and resuming, samples, a whole run.

Runs on a CPU with tiny models. What is checked is what a run cannot check about itself: that the error is read in the units
of robustness and a prediction of the mean reads as 1, that a stage that cannot pass fails, that an interrupted run resumes at the step it
stopped at, that a sample's sequence is as long as its single, that the cross-entropy term reaches the generator and not the
frozen decoder, and that a whole run writes what the next step reads.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esmfold2_world import build

from foldspace.esmfold2 import latent_generation as latent_generation, train_decoder
from foldspace.esmfold2 import perturb as P
from foldspace.esmfold2.decoder import build_decoder

CPU = torch.device("cpu")
TINY = "d_model=32,n_layers=2,n_heads=2,d_ff=64,steps=12,eval_every=6,warmup_steps=3,max_tokens=1024,n_samples=4,sample_steps=3"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build(tmp_path_factory.mktemp("world"))


@pytest.fixture(scope="module")
def prepared(world, tmp_path_factory):
    """The covariance file and the frozen decoder that latent generation takes from the robustness and decoding experiments, made from the world."""
    from foldspace.esmfold2.decoder_data import Normaliser, load_split

    cache = world["cache"]
    norm = Normaliser.load(
        cache / "esmfold2" / "model_single_train_normalisation.npz", CPU
    )
    train = load_split(
        "train",
        cache=cache,
        chains=world["chains"],
        train_table=world["train_table"],
        device=CPU,
    )
    x = ((train.z.float() - norm.mean) / norm.std).double().numpy()
    covariance = x.T @ x / len(x)
    directory = tmp_path_factory.mktemp("prepared")
    np.savez(directory / "covariance.npz", covariance=covariance)
    torch.manual_seed(0)
    model = build_decoder({"arch": "linear"})
    train_decoder.save_checkpoint(
        directory / "decoder.pt",
        model,
        train_decoder.Config(arch="linear"),
        epoch=0,
        dev={},
        norm_digest=norm.digest,
    )
    return directory


def _harness(world, prepared, **kw):
    spectrum = P.spectrum_from_covariance(
        np.load(prepared / "covariance.npz")["covariance"]
    )
    return latent_generation.Harness(
        cache=world["cache"],
        chains=world["chains"],
        train_table=world["train_table"],
        device="cpu",
        spectrum=spectrum,
        decoder_path=prepared / "decoder.pt",
        **kw,
    )


def _config(**kw):
    return latent_generation.replace(latent_generation.Config(), **{**latent_generation.parse_overrides(TINY), **kw})


# ----------------------------------------------------------------- readings


def test_overrides_take_the_type_of_the_field_and_refuse_a_name_that_is_not_one():
    got = latent_generation.parse_overrides("steps=5,lr=0.5,coordinate=whitened")
    assert got == {"steps": 5, "lr": 0.5, "coordinate": "whitened"}
    with pytest.raises(ValueError, match="not a field"):
        latent_generation.parse_overrides("depth=3")


def test_an_error_proportional_to_the_data_reads_as_its_level_and_the_mean_reads_as_one():
    values = np.concatenate([np.full(64, 4.0), np.full(128, 0.25), np.full(64, 0.01)])
    count = 1000
    # an error that is `level` times each direction's own standard deviation is the anoise of that level
    reading = latent_generation.eigen_error(count * 0.3**2 * values, values, count)
    assert reading["relative"] == pytest.approx(0.3)
    assert reading["relative_size"] == pytest.approx(0.3)
    assert all(v == pytest.approx(0.3) for v in reading["by_group"].values())
    # predicting the mean is an error equal to the variance in every direction
    assert latent_generation.eigen_error(count * values, values, count)["relative"] == pytest.approx(
        1.0
    )
    # an error of the same absolute size everywhere is small against the large directions and large against the small
    flat = latent_generation.eigen_error(np.full(256, 0.01) * count, values, count)
    assert flat["by_group"]["leading_64"] == pytest.approx(0.05)
    assert flat["by_group"]["trailing_64"] == pytest.approx(1.0)


def test_sample_lengths_are_spread_over_what_dev_has_between_the_bounds():
    lengths = np.concatenate([np.arange(20, 400), np.arange(70, 220)])
    got = latent_generation.sample_lengths(lengths, 12, 70, 220)
    assert len(got) == 12 and got.min() == 70 and got.max() == 220
    assert np.all(np.diff(got) >= 0) and np.all((got >= 70) & (got <= 220))
    with pytest.raises(ValueError, match="fewer than"):
        latent_generation.sample_lengths(np.array([100, 120]), 12, 70, 220)


# ------------------------------------------------------------------ harness


def test_the_harness_reads_one_dev_chain_of_each_cluster_and_freezes_the_decoder(
    world, prepared
):
    h = _harness(world, prepared)
    assert len(h.eval_chains) == len(np.unique(h.dev.clusters))
    assert len({int(h.dev.clusters[c]) for c in h.eval_chains}) == len(h.eval_chains)
    assert all(not p.requires_grad for p in h.decoder.parameters())
    assert h.whitening(_config()).vectors is None
    assert h.whitening(_config(coordinate="whitened")).vectors.shape == (256, 256)
    with pytest.raises(ValueError, match="standardized or whitened"):
        h.whitening(_config(coordinate="pca"))


def test_the_cross_entropy_term_reaches_the_generator_and_never_the_frozen_decoder(
    world, prepared
):
    from foldspace.esmfold2 import batching

    h = _harness(world, prepared)
    config = _config(coordinate="whitened", ce_weight=0.5)
    whiten = h.whitening(config)
    model = h.model(config)
    gen = torch.Generator().manual_seed(0)
    batch = next(
        iter(batching.pack(np.arange(h.train.n_chains), h.train.lengths, 1024))
    )
    loss, mse, ce = latent_generation._step_loss(model, h, whiten, config, batch, gen)
    loss.backward()
    assert ce > 0 and loss > mse
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert all(p.grad is None for p in h.decoder.parameters())
    # without the term the loss is the flow loss alone
    plain, _, ce2 = latent_generation._step_loss(
        model,
        h,
        whiten,
        _config(coordinate="whitened"),
        batch,
        torch.Generator().manual_seed(0),
    )
    # the same draw, so the two losses differ by the term and by nothing else
    assert ce2 == 0 and float(loss.detach()) - float(plain.detach()) == pytest.approx(
        0.5 * float(ce.detach()), rel=1e-3
    )


# ------------------------------------------------------------------- stages


def test_the_smoke_stage_passes_on_a_tiny_model(world, prepared):
    h = _harness(world, prepared)
    config = _config()
    result = latent_generation.stage_smoke(h, config, h.whitening(config))
    assert result["ok"] and result["gradient_finite"] and result["parameter_moved"]


def test_the_overfit_stage_fails_when_it_cannot_pass(world, prepared, monkeypatch):
    h = _harness(world, prepared)
    config = _config()
    monkeypatch.setattr(latent_generation, "OVERFIT_STEPS", 25)
    monkeypatch.setattr(latent_generation, "OVERFIT_RATIO", 1e-6)  # nothing gets there
    assert not latent_generation.stage_overfit(h, config, h.whitening(config))["ok"]
    monkeypatch.setattr(latent_generation, "OVERFIT_RATIO", 1.2)  # anything does
    assert latent_generation.stage_overfit(h, config, h.whitening(config))["ok"]


# ----------------------------------------------------------------- training


def test_training_writes_its_readings_and_an_interrupted_run_resumes_where_it_stopped(
    world, prepared, tmp_path
):
    h = _harness(world, prepared)
    config = _config()
    whiten = h.whitening(config)
    log = []
    first = latent_generation.train_steps(
        h, config, whiten, h.model(config), steps=6, out=tmp_path, log=log.append
    )
    assert first["steps"] == 6 and [r["step"] for r in first["history"]] == [6]
    reading = first["history"][0]["denoise"]
    assert set(reading) == {"t=0.5", "t=0.8", "t=0.95"}
    assert set(reading["t=0.8"]["by_group"]) == {
        "leading_64",
        "middle_128",
        "trailing_64",
    }
    assert 0.0 <= reading["t=0.8"]["decode_acc_per_chain"] <= 1.0
    assert (tmp_path / "last.pt").exists()
    # asked for twelve, it picks up at six rather than starting over
    second = latent_generation.train_steps(
        h, config, whiten, h.model(config), steps=12, out=tmp_path, log=log.append
    )
    assert second["steps"] == 12 and [r["step"] for r in second["history"]] == [6, 12]
    assert any("resumed" in line for line in log)
    assert second["digest"] != first["digest"]


def test_the_error_of_the_estimate_falls_with_training_in_either_coordinate(
    world, prepared, tmp_path
):
    for coordinate in ("standardized", "whitened"):
        h = _harness(world, prepared)
        config = _config(
            coordinate=coordinate, steps=250, eval_every=10**9, lr=3e-3, warmup_steps=5
        )
        whiten = h.whitening(config)
        model = h.model(config)
        before = latent_generation._error_at(model, h, whiten, h.dev, h.eval_chains)
        latent_generation.train_steps(
            h,
            config,
            whiten,
            model,
            steps=250,
            out=tmp_path / coordinate,
            log=lambda *_: None,
            resume=False,
        )
        assert latent_generation._error_at(model, h, whiten, h.dev, h.eval_chains) < 0.9 * before


# ------------------------------------------------------------------ samples


def test_a_sample_is_a_single_and_the_sequence_read_off_it_is_as_long(
    world, prepared, tmp_path, monkeypatch
):
    h = _harness(world, prepared)
    config = _config()
    whiten = h.whitening(config)
    model = h.model(config)
    monkeypatch.setattr(latent_generation, "SAMPLE_LENGTHS", (1, 1000))
    info = latent_generation.export_samples(model, h, whiten, tmp_path, n=4, steps=3, seed=7)
    held = np.load(tmp_path / "samples.npz")
    rows = (tmp_path / "samples.tsv").read_text().splitlines()
    assert rows[0] == "sample_id\tsequence\tmean_confidence" and len(rows) == 5
    for i, row in enumerate(rows[1:]):
        sid, sequence, confidence = row.split("\t")
        single = held[sid]
        assert single.shape == (info["lengths"][i], 256) and single.dtype == np.float32
        assert len(sequence) == len(single) and 0.0 <= float(confidence) <= 1.0
    again = latent_generation.export_samples(
        model, h, whiten, tmp_path / "again", n=4, steps=3, seed=7
    )
    assert np.array_equal(
        np.load(tmp_path / "again" / "samples.npz")["s000"], held["s000"]
    )
    assert again["lengths"] == info["lengths"]


# -------------------------------------------------------------- a whole run


def test_a_whole_run_writes_what_the_next_step_reads(
    world, prepared, tmp_path, monkeypatch
):
    monkeypatch.setattr(latent_generation, "OVERFIT_STEPS", 20)
    monkeypatch.setattr(latent_generation, "OVERFIT_RATIO", 1.2)
    monkeypatch.setattr(latent_generation, "SMALL_STEPS", 12)
    monkeypatch.setattr(latent_generation, "SMALL_RATIO", 1.2)
    monkeypatch.setattr(latent_generation, "SAMPLE_LENGTHS", (1, 1000))
    args = argparse.Namespace(
        out=tmp_path / "pilot",
        device="cpu",
        cache=world["cache"],
        chains=world["chains"],
        train_table=world["train_table"],
        covariance=prepared / "covariance.npz",
        decoder=prepared / "decoder.pt",
        arms="standardized,whitened_with_sequence_loss",
        stages="smoke,overfit,small,full",
        set=TINY,
        limit_chains=0,
        allow_dirty=True,
    )
    assert latent_generation.run_train(args) == 0
    summary = json.loads((tmp_path / "pilot" / "generation_summary.json").read_text())
    assert set(summary["arms"]) == {"standardized", "whitened_with_sequence_loss"}
    assert (
        summary["provenance"]["device"] == "cpu"
        and "reproducible_tree" in summary["provenance"]
    )
    for name, arm in summary["arms"].items():
        assert arm["failed"] is None and arm["full"]["steps"] == 12
        assert (tmp_path / "pilot" / name / "samples" / "samples.tsv").exists()
        assert (tmp_path / "pilot" / name / "full" / "ema.pt").exists()
    assert summary["arms"]["whitened_with_sequence_loss"]["config"]["ce_weight"] == 0.1


def test_a_real_single_is_completed_from_a_partial_noising_and_read(
    world, prepared, tmp_path, monkeypatch
):
    from foldspace.esmfold2 import validate_interface

    h = _harness(world, prepared)
    config = _config()
    arm = tmp_path / "arm"
    (arm / "full").mkdir(parents=True)
    torch.save(
        {"model": h.model(config).state_dict(), "config": latent_generation.asdict(config)},
        arm / "full" / "ema.pt",
    )
    ids = h.dev.ids[:3]
    panel = tmp_path / "panel.tsv"
    panel.write_text(
        "seq_id\tsequence\tdecoded\tsubstitutions\n"
        + "".join(
            f"{i}\t{h.dev.sequence(k)}\t{h.dev.sequence(k)}\t0\n"
            for k, i in enumerate(ids)
        )
    )
    monkeypatch.setattr(
        validate_interface,
        "corpus_chains",
        lambda path, split: [
            {"id": i, "ca": np.zeros((int(h.dev.lengths[k]), 3))}
            for k, i in enumerate(h.dev.ids)
        ],
    )
    args = argparse.Namespace(
        arm_dir=arm,
        panel=panel,
        out=tmp_path / "partial",
        device="cpu",
        cache=world["cache"],
        chains=world["chains"],
        train_table=world["train_table"],
        covariance=prepared / "covariance.npz",
        decoder=prepared / "decoder.pt",
        t0="0.5,0.9",
        steps=3,
        seed=1,
        allow_dirty=True,
    )
    result = latent_generation.run_partial(args)
    assert set(result["t0"]) == {"0.5", "0.9"}
    for table in result["t0"].values():
        assert sorted(r["id"] for r in table) == sorted(ids)
        assert all(
            0.0 <= r["identity"] <= 1.0 and len(r["decoded"]) == r["length"]
            for r in table
        )
    records = json.loads((tmp_path / "partial" / "records_t0.9.json").read_text())
    assert len(records) == 3 and len(records[0]["coordinates"]) == len(
        records[0]["sequence"]
    )
    assert (tmp_path / "partial" / "partial.json").exists()


def test_samples_are_made_from_a_trained_arm_at_a_chosen_number_of_steps_and_shift(
    world, prepared, tmp_path, monkeypatch
):
    h = _harness(world, prepared)
    config = _config()
    arm = tmp_path / "arm"
    (arm / "full").mkdir(parents=True)
    torch.save(
        {"model": h.model(config).state_dict(), "config": latent_generation.asdict(config)},
        arm / "full" / "ema.pt",
    )
    monkeypatch.setattr(latent_generation, "SAMPLE_LENGTHS", (1, 1000))
    args = argparse.Namespace(
        arm_dir=arm,
        out=tmp_path / "again",
        device="cpu",
        cache=world["cache"],
        chains=world["chains"],
        train_table=world["train_table"],
        covariance=prepared / "covariance.npz",
        decoder=prepared / "decoder.pt",
        n=4,
        steps=3,
        shift=3.0,
        seed=7,
    )
    info = latent_generation.run_sample(args)
    assert info["steps"] == 3 and info["shift"] == 3.0 and info["n"] == 4
    rows = (tmp_path / "again" / "samples.tsv").read_text().splitlines()
    assert rows[0] == "sample_id\tsequence\tmean_confidence" and len(rows) == 5
    # the same lengths as a run at another shift: only how the sample was made differs
    other = latent_generation.run_sample(
        argparse.Namespace(**{**vars(args), "out": tmp_path / "even", "shift": 1.0})
    )
    assert other["lengths"] == info["lengths"]
    assert not np.array_equal(
        np.load(tmp_path / "again" / "samples.npz")["s000"],
        np.load(tmp_path / "even" / "samples.npz")["s000"],
    )
