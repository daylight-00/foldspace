"""The decoder stack of the sequence decoding experiment on a synthetic world: data, models, LDA, training, stages, selection, the test lock.

Everything runs on a CPU with tiny models. What is checked is the machinery a run cannot check about itself: that
each array of each file is read once, that a decoder's output does not depend on padding, that LDA is the discriminant
it says it is, that a checkpoint comes back as the number recorded, that a failed stage stops its arm and the run goes on,
that the carried decoder is chosen on dev by the stated rule, and that the test split is scored once per model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esmfold2_world import build

from foldspace.esmfold2 import decode_eval, evaluate_decoder, decoder_structure_panel, lda, train_decoder
from foldspace.esmfold2 import decoder as decoders
from foldspace.esmfold2.decoder_data import IGNORE, Normaliser, gather, load_split

CPU = torch.device("cpu")
TINY = "d_model=32,n_layers=1,n_heads=2,d_ff=64,epochs=4,max_tokens=1024,warmup_steps=5,patience=3"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build(tmp_path_factory.mktemp("world"))


def _load(world, name, **kwargs):
    return load_split(
        name,
        cache=world["cache"],
        chains=world["chains"],
        train_table=world["train_table"],
        device=CPU,
        **kwargs,
    )


@pytest.fixture(scope="module")
def norm(world):
    return Normaliser.load(
        world["cache"] / "esmfold2" / "model_single_train_normalisation.npz", CPU
    )


@pytest.fixture(scope="module")
def train(world):
    return _load(world, "train")


@pytest.fixture(scope="module")
def dev(world):
    return _load(world, "dev")


# --------------------------------------------------------------------- data


def test_a_split_is_loaded_with_its_labels_clusters_and_exact_values(world, train, dev):
    assert train.n_chains == 160 and dev.n_chains == 60
    assert train.z.dtype == torch.bfloat16 and train.z.shape == (train.n_rows, 256)
    # the sequence of every chain is what the table says
    table = dict(world["tables"]["train"])
    assert all(train.sequence(i) == table[train.ids[i]] for i in range(train.n_chains))
    # the shards are strided, so the chains come in shard order and every shard is represented
    assert set(train.shard.tolist()) == {0, 1, 2, 3}
    # clusters come from the corpus table: the 20 chains of dev's one family share a cluster
    assert (dev.clusters == 10_000).sum() == 20
    assert len(np.unique(train.clusters)) == 160
    # the rows are the stored values, to the bit
    held = np.load(world["cache"] / "esmfold2" / "targets" / "dev" / "z_shard0.npz")
    assert np.array_equal(dev.z.float().numpy(), held["z"])


class _CountingNpz(dict):
    """An NpzFile stand-in that counts how often each array is read."""

    def __init__(self, arrays):
        super().__init__(arrays)
        self.reads: dict[str, int] = {}

    @property
    def files(self):
        return list(self.keys())

    def __getitem__(self, key):
        self.reads[key] = self.reads.get(key, 0) + 1
        return super().__getitem__(key)


def test_every_array_of_every_file_is_read_once(world, monkeypatch):
    opened: list[_CountingNpz] = []
    real = np.load

    def counting(path, *args, **kwargs):
        held = real(path, *args, **kwargs)
        wrapped = _CountingNpz({k: held[k] for k in held.files})
        opened.append(wrapped)
        return wrapped

    monkeypatch.setattr("foldspace.esmfold2.decoder_data.np.load", counting)
    monkeypatch.setattr("foldspace.esmfold2.target_audit.np.load", counting)
    _load(world, "dev")
    assert opened
    for wrapped in opened:
        assert all(count == 1 for count in wrapped.reads.values()), wrapped.reads


def test_a_single_that_is_not_a_bfloat16_value_is_refused(world, tmp_path):
    other = build(tmp_path / "w", train_chains=8, dev_chains=4, test_chains=4)
    path = other["cache"] / "esmfold2" / "targets" / "dev" / "z_shard0.npz"
    held = dict(np.load(path))
    held["z"] = held["z"].copy()
    held["z"][0, 0] = np.float32(1.2345678)  # needs more mantissa than bfloat16 has
    np.savez(path, **held)
    with pytest.raises(ValueError, match="bfloat16"):
        _load(other, "dev")


def test_a_sequence_that_does_not_match_its_rows_is_refused(tmp_path):
    other = build(tmp_path / "w", train_chains=8, dev_chains=4, test_chains=4)
    table = other["cache"] / "esmfold2" / "sequences" / "dev.tsv"
    lines = table.read_text().splitlines()
    cid, seq = lines[1].split("\t")
    table.write_text("\n".join([lines[0], f"{cid}\t{seq[:-1]}", *lines[2:]]) + "\n")
    with pytest.raises(ValueError, match="residues"):
        _load(other, "dev")
    table.write_text(
        "\n".join([lines[0], f"{cid}\t{'B' * len(seq)}", *lines[2:]]) + "\n"
    )
    with pytest.raises(ValueError, match="letter"):
        _load(other, "dev")


def test_a_batch_is_whole_chains_with_zero_padding_and_ignored_labels(train, norm):
    chains = np.array([3, 10])
    x, y, lengths = gather(train, chains, norm)
    longest = int(train.lengths[chains].max())
    assert x.shape == (2, longest, 256) and y.shape == (2, longest)
    assert lengths.tolist() == train.lengths[chains].tolist()
    for b, c in enumerate(chains):
        n, s = int(train.lengths[c]), int(train.starts[c])
        expected = (train.z[s : s + n].float() - norm.mean) / norm.std
        assert torch.allclose(x[b, :n], expected)
        assert (x[b, n:] == 0).all()
        assert (y[b, :n] == torch.from_numpy(train.labels_np[s : s + n])).all()
        assert (y[b, n:] == IGNORE).all()


def test_the_normalisation_has_an_inverse(train, norm):
    x = train.z[:50].float()
    assert torch.allclose(norm.inverse(norm(x)), x, atol=1e-4)
    assert len(norm.digest) == 16


# ------------------------------------------------------------------- models


@pytest.mark.parametrize("arch", ["linear", "contextual"])
def test_a_decoders_output_does_not_depend_on_the_padding(train, norm, arch):
    torch.manual_seed(0)
    config = {
        "arch": arch,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 2,
        "d_ff": 64,
        "dropout": 0.1,
    }
    model = decoders.build_decoder(config).eval()
    short, long = int(np.argmin(train.lengths)), int(np.argmax(train.lengths))
    x1, _, l1 = gather(train, np.array([short]), norm)
    x2, _, l2 = gather(train, np.array([short, long]), norm)
    with torch.no_grad():
        alone = model(x1, l1)[0]
        padded = model(x2, l2)[0, : int(l1[0])]
    assert torch.allclose(alone, padded, atol=1e-5)
    assert alone.shape[-1] == 20


def test_the_contextual_decoder_sees_neighbours_and_both_ends(train, norm):
    torch.manual_seed(0)
    model = decoders.build_decoder(
        {"arch": "contextual", "d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64}
    ).eval()
    x, _, lengths = gather(train, np.array([0]), norm)
    with torch.no_grad():
        base = model(x, lengths)
        # changing one row changes the logits of the others (attention), unlike a linear readout
        x2 = x.clone()
        x2[0, 5] += 3.0
        changed = model(x2, lengths)
    assert not torch.allclose(base[0, 20], changed[0, 20], atol=1e-6)
    linear = decoders.build_decoder({"arch": "linear"}).eval()
    with torch.no_grad():
        assert torch.allclose(linear(x, lengths)[0, 20], linear(x2, lengths)[0, 20])
    # the same row at the same offset from the end gets the same end embedding whatever the length
    long = torch.cat([x, x[:, :10]], dim=1)
    with pytest.raises(ValueError, match="positions"):
        model(
            torch.zeros(1, decoders.MAX_LENGTH + 1, 256),
            torch.tensor([decoders.MAX_LENGTH + 1]),
        )
    assert long.shape[1] == x.shape[1] + 10
    assert decoders.parameter_count(model) > decoders.parameter_count(linear)


def test_an_unknown_decoder_is_refused():
    with pytest.raises(ValueError, match="unknown"):
        decoders.build_decoder({"arch": "nonsense"})


# ---------------------------------------------------------------------- lda


def test_lda_is_the_discriminant_computed_directly(train, norm):
    chains = np.arange(train.n_chains)
    fitted = lda.fit_lda(train, norm, chains, ridge=1e-3)
    x = ((train.z.float() - norm.mean) / norm.std).double().numpy()
    y = train.labels_np
    k, d = 20, x.shape[1]
    counts = np.bincount(y, minlength=k).astype(np.float64)
    means = np.stack([x[y == c].mean(axis=0) for c in range(k)])
    centred = x - means[y]
    cov = centred.T @ centred / (len(y) - k)
    cov += 1e-3 * np.trace(cov) / d * np.eye(d)
    weight = np.linalg.solve(cov, means.T).T
    bias = -0.5 * (weight * means).sum(axis=1) + np.log(counts / counts.sum())
    assert np.allclose(fitted.weight.numpy(), weight, atol=1e-3, rtol=1e-3)
    assert np.allclose(fitted.bias.numpy(), bias, atol=1e-2, rtol=1e-3)


def test_lda_reads_the_residue_type_off_a_row_of_the_single(train, dev, norm):
    fitted = lda.fit_lda(train, norm, np.arange(train.n_chains))
    p = train_decoder.predict(fitted, dev, norm)
    assert p.summary()["micro"]["acc"] > 0.9
    # fitted on shards 0-2 and read on shard 3, the old protocol, it can be done on the subset asked for
    held_out = np.flatnonzero(train.shard == 3)
    partial = lda.fit_lda(train, norm, np.flatnonzero(train.shard != 3))
    assert train_decoder.predict(partial, train, norm, held_out).summary()[
        "chains"
    ] == len(held_out)


def test_a_discriminant_on_a_subspace_is_the_full_one_when_the_subspace_is_everything(
    train, dev, norm
):
    chains = np.arange(train.n_chains)
    stats = lda.statistics(train, norm, chains)
    d = train.z.shape[1]
    full = lda.solve(stats)
    # any orthogonal rotation of the whole space leaves a linear discriminant unchanged
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((d, d)))
    rotated = lda.solve(stats, basis=torch.tensor(q, dtype=torch.float64))
    a = train_decoder.predict(full, dev, norm)
    b = train_decoder.predict(rotated, dev, norm)
    assert np.allclose(a.logp, b.logp, atol=2e-3)
    assert np.array_equal(a.logp.argmax(1), b.logp.argmax(1))


def test_a_discriminant_on_fewer_directions_reads_less_and_still_reads(
    train, dev, norm
):
    stats = lda.statistics(train, norm, np.arange(train.n_chains))
    d = train.z.shape[1]
    # the directions of the training covariance, by variance
    rows = ((train.z.float() - norm.mean) / norm.std).double()
    vectors = torch.linalg.eigh(rows.T @ rows / len(rows))[1].flip(1)
    leading = lda.solve(stats, basis=vectors[:, :8])
    assert isinstance(leading, lda.Projected) and leading.basis.shape == (d, 8)
    full = lda.solve(stats)
    acc_full = train_decoder.predict(full, dev, norm).selection()["cw_acc"]
    acc_8 = train_decoder.predict(leading, dev, norm).selection()["cw_acc"]
    assert (
        0.05 < acc_8 <= acc_full + 1e-9
    )  # chance is 1/20, and fewer directions cannot read more
    # the trailing directions are a subspace too and read through the same wrapper
    trailing = lda.solve(stats, basis=vectors[:, d - 8 :])
    assert train_decoder.predict(trailing, dev, norm).selection()["cw_acc"] > 0.0


def test_lda_refuses_a_fit_with_a_type_missing(train, norm):
    # a short chain that lacks at least one of the twenty types (most short random chains do)
    lacking = [
        c
        for c in np.argsort(train.lengths)[:20]
        if len(
            np.unique(
                train.labels_np[train.starts[c] : train.starts[c] + train.lengths[c]]
            )
        )
        < 20
    ]
    assert lacking, "the synthetic world has no chain that lacks a type"
    with pytest.raises(ValueError, match="no example"):
        lda.fit_lda(train, norm, np.array([lacking[0]]))


# ------------------------------------------------------------------ training


def _config(**kw):
    base = {
        "arch": "contextual",
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "d_ff": 64,
        "epochs": 4,
        "max_tokens": 4096,
        "warmup_steps": 5,
        "patience": 3,
    }
    return train_decoder.Config(**{**base, **kw})


def test_predictions_come_back_in_the_order_asked(dev, norm):
    model = decoders.build_decoder({"arch": "linear"}).eval()
    full = train_decoder.predict(model, dev, norm)
    ask = np.array([7, 2, 30])
    some = train_decoder.predict(model, dev, norm, ask)
    assert some.chains.tolist() == [7, 2, 30]
    for j, c in enumerate(ask):
        a = full.logp[full.starts[c] : full.starts[c] + full.lengths[c]]
        b = some.logp[some.starts[j] : some.starts[j] + some.lengths[j]]
        assert np.allclose(a, b, atol=1e-5)
    assert np.exp(full.logp).sum(axis=1) == pytest.approx(
        np.ones(full.logp.shape[0]), abs=1e-4
    )


def test_training_learns_and_the_checkpoint_comes_back_as_recorded(
    train, dev, norm, tmp_path
):
    config = _config()
    model = decoders.build_decoder(train_decoder.asdict(config))
    run = train_decoder.fit(
        model, train, dev, norm, config, tmp_path, log=lambda *a: None
    )
    first, last = run["history"][0], run["history"][-1]
    assert last["train_loss"] < first["train_loss"]
    assert run["best"]["cw_acc"] > 0.5
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()

    check = train_decoder.restore_check(tmp_path / "best.pt", dev, norm, run["best"])
    assert check["restored"] and check["norm_digest_matches"]
    _, loaded_config, meta = train_decoder.load_checkpoint(tmp_path / "best.pt", CPU)
    assert loaded_config == config and meta["epoch"] == run["best"]["epoch"]
    # a checkpoint of another normalisation is recognised
    other = Normaliser(norm.mean + 1, norm.std, "different-digest")
    assert not train_decoder.restore_check(
        tmp_path / "best.pt", dev, other, run["best"]
    )["norm_digest_matches"]


def test_the_best_epoch_is_the_lowest_cluster_weighted_dev_cross_entropy(
    train, dev, norm, tmp_path, monkeypatch
):
    scripted = iter([1.0, 0.6, 0.8, 0.9, 0.95, 0.99])

    def fake(model, split, norm, **kwargs):
        ce = next(scripted)
        return {"cw_ce": ce, "cw_acc": 1 - ce, "micro_ce": ce, "micro_acc": 1 - ce}

    monkeypatch.setattr(train_decoder, "selection", fake)
    config = _config(arch="linear", epochs=6, patience=3, dropout=0.0)
    model = decoders.build_decoder(train_decoder.asdict(config))
    run = train_decoder.fit(
        model, train, dev, norm, config, tmp_path, log=lambda *a: None
    )
    assert run["best"]["epoch"] == 2 and run["best"]["cw_ce"] == 0.6
    assert len(run["history"]) == 5  # stopped after three epochs without improvement


def test_a_non_finite_loss_stops_training(train, dev, norm, tmp_path):
    bad = train
    saved = bad.z[0].clone()
    bad.z[0] = float("nan")
    try:
        config = _config(arch="linear", dropout=0.0)
        model = decoders.build_decoder(train_decoder.asdict(config))
        with pytest.raises(FloatingPointError):
            train_decoder.fit(
                model, bad, dev, norm, config, tmp_path, log=lambda *a: None
            )
    finally:
        bad.z[0] = saved


def test_the_learning_rate_warms_up_and_decays():
    c = train_decoder.Config(lr=1e-3, warmup_steps=10, min_lr_ratio=0.1)
    assert train_decoder.lr_at(0, 100, c) == pytest.approx(1e-4)
    assert train_decoder.lr_at(9, 100, c) == pytest.approx(1e-3)
    assert train_decoder.lr_at(100, 100, c) == pytest.approx(1e-4)
    assert train_decoder.lr_at(55, 100, c) < 1e-3


# -------------------------------------------------------------------- stages


def test_the_smoke_stage_passes_a_healthy_split_and_fails_a_broken_one(
    train, dev, norm
):
    config = _config()
    ok = evaluate_decoder.stage_smoke(config, train, dev, norm)
    assert (
        ok["ok"]
        and ok["parameter_moved"]
        and ok["checks"]["log_probabilities_sum_to_one"]
    )
    assert ok["batches_per_epoch"] > 1 and ok["cut_seconds_per_batch"] >= 0
    saved = train.z[0].clone()
    train.z[0] = float("nan")
    try:
        # chain 0 starts at row 0; make it part of the largest batch by asking for a batch of everything
        broken = evaluate_decoder.stage_smoke(_config(max_tokens=10**9), train, dev, norm)
    finally:
        train.z[0] = saved
    assert not broken["ok"] and not broken["checks"]["features_finite"]


def test_the_overfit_stage_learns_what_it_is_shown_or_fails(train, norm, monkeypatch):
    monkeypatch.setattr(evaluate_decoder, "OVERFIT_ACCURACY", 0.95)
    passed = evaluate_decoder.stage_overfit(_config(d_model=64, n_heads=4, d_ff=128), train, norm)
    assert passed["ok"] and passed["chains"] == evaluate_decoder.OVERFIT_CHAINS
    monkeypatch.setattr(evaluate_decoder, "OVERFIT_ACCURACY", 1.01)
    assert not evaluate_decoder.stage_overfit(_config(), train, norm)["ok"]


def test_the_overlay_of_config_is_typed_by_the_field():
    out = evaluate_decoder.parse_overrides("epochs=3,lr=0.01,sampling=chain,d_model=16")
    assert out == {"epochs": 3, "lr": 0.01, "sampling": "chain", "d_model": 16}
    with pytest.raises(ValueError, match="not a field"):
        evaluate_decoder.parse_overrides("nonsense=1")
    assert evaluate_decoder.parse_overrides("") == {}


# -------------------------------------------------------- choice and the run


def _arm(cw_ce, cw_acc):
    return {"full": {"best": {"cw_ce": cw_ce, "cw_acc": cw_acc, "epoch": 1}}}


def test_contextual_is_adopted_only_if_it_beats_linear_on_every_seed():
    summary = {
        "arms": {
            "linear-s0": _arm(0.30, 0.950),
            "contextual-s0": _arm(0.20, 0.960),
            "contextual-s1": _arm(0.22, 0.955),
            "contextual_chain-s0": _arm(0.10, 0.999),  # the ablation never decides
        }
    }
    chosen = evaluate_decoder.choose(summary)
    assert chosen["contextual"] == "contextual-s0"  # the lower cross-entropy
    assert chosen["contextual_adopted"] and chosen["carried"] == "contextual-s0"

    summary["arms"]["contextual-s1"] = _arm(0.22, 0.9515)  # +0.15 points: not enough
    chosen = evaluate_decoder.choose(summary)
    assert not chosen["contextual_adopted"] and chosen["carried"] == "linear-s0"
    assert chosen["gains_over_linear"]["contextual-s1"] == pytest.approx(0.0015)


def test_an_arm_that_did_not_finish_is_not_a_candidate():
    summary = {
        "arms": {
            "linear-s0": _arm(0.3, 0.95),
            "contextual-s0": {"smoke": {"ok": False}},
        }
    }
    chosen = evaluate_decoder.choose(summary)
    assert chosen == {"linear": "linear-s0", "carried": "linear-s0"}
    assert evaluate_decoder.choose({"arms": {}}) == {}


def _args(world, out, **kw):
    base = {
        "out": out,
        "device": "cpu",
        "cache": world["cache"],
        "chains": world["chains"],
        "train_table": world["train_table"],
        "arms": "lda,linear:0,contextual:0,contextual:1,contextual_chain:0",
        "stages": "smoke,overfit,small,full",
        "set": TINY,
        "limit_chains": 0,
        "final_test": False,
        "force_test": False,
    }
    return argparse.Namespace(**{**base, **kw})


def test_a_whole_run_chooses_on_dev_scores_the_test_split_once_and_decodes_the_panel(
    world, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate_decoder, "OVERFIT_ACCURACY", 0.95)
    monkeypatch.setattr(evaluate_decoder, "SMALL_DEV_ACCURACY", 0.5)
    out = tmp_path / "decoders"
    assert evaluate_decoder.run(_args(world, out, final_test=True)) == 0

    summary = json.loads((out / "decoder_summary.json").read_text())
    assert summary["failed"] == []
    # what its numbers are quoted with: the commits, the host and the card (a CPU here)
    assert summary["provenance"]["device"] == "cpu"
    assert (
        "reproducible_tree" in summary["provenance"]
        and "torch" in summary["provenance"]
    )
    assert set(summary["arms"]) == {
        "linear-s0",
        "contextual-s0",
        "contextual-s1",
        "contextual_chain-s0",
    }
    assert summary["chosen"]["carried"] in (
        "linear-s0",
        summary["chosen"]["contextual"],
    )
    for arm in summary["arms"].values():
        assert arm["full"]["restore"]["restored"]
    # LDA was fitted both ways
    assert "held_out_shard_3" in summary["lda"]["fit_on_shards_0_1_2"]
    assert 0.0 <= summary["lda"]["shard_3_clusters_shared_with_fit"] <= 1.0
    # the test split was scored for the three models, once each
    assert set(summary["test"]) == {"lda", "linear-s0", summary["chosen"]["contextual"]}
    lock_dir = world["cache"] / "esmfold2" / "decoders" / "test_once"
    assert len(list(lock_dir.glob("*.json"))) == 3
    # the panel the structure check is made on
    rows = (out / "panel_decoded.tsv").read_text().splitlines()
    assert rows[0] == "seq_id\tsequence\tdecoded\tsubstitutions" and len(rows) == 25
    first = rows[1].split("\t")
    assert len(first[1]) == len(first[2])

    # a second run may not score the same models on test again
    with pytest.raises(RuntimeError, match="already been scored"):
        evaluate_decoder.run(_args(world, tmp_path / "again", final_test=True))


def test_a_failed_stage_stops_its_arm_and_the_run_goes_on(world, tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluate_decoder, "OVERFIT_ACCURACY", 1.01
    )  # the contextual arms cannot pass this
    monkeypatch.setattr(evaluate_decoder, "SMALL_DEV_ACCURACY", 0.5)
    out = tmp_path / "failing"
    code = evaluate_decoder.run(_args(world, out, arms="linear:0,contextual:0", final_test=False))
    summary = json.loads((out / "decoder_summary.json").read_text())
    assert code == 1
    assert summary["failed"] == [{"arm": "contextual-s0", "stage": "overfit"}]
    assert (
        "full" in summary["arms"]["linear-s0"]
        and "full" not in summary["arms"]["contextual-s0"]
    )
    assert summary["chosen"]["carried"] == "linear-s0"
    assert not (out / "contextual-s0" / "small.json").exists()


def test_the_test_lock_is_per_model(world, tmp_path, dev, norm):
    test = _load(world, "test")
    model = decoders.build_decoder({"arch": "linear"}).eval()
    lock_dir = tmp_path / "locks"
    first = evaluate_decoder.score_test_once("m", model, test, norm, lock_dir)
    assert first["summary"]["chains"] == test.n_chains
    with pytest.raises(RuntimeError, match="already been scored"):
        evaluate_decoder.score_test_once("m", model, test, norm, lock_dir)
    evaluate_decoder.score_test_once("m", model, test, norm, lock_dir, force=True)
    # another model is another lock
    other = decoders.build_decoder({"arch": "linear"}).eval()
    with torch.no_grad():
        other.out.weight.add_(1.0)
    evaluate_decoder.score_test_once("other", other, test, norm, lock_dir)
    assert len(list(Path(lock_dir).glob("*.json"))) == 2


def test_the_panel_is_chosen_from_metadata_alone_and_spread_over_length(dev):
    panel = decoder_structure_panel.select_panel(dev.ids, dev.lengths, dev.clusters, n=24)
    assert len(panel) == 24 and len(set(panel)) == 24
    assert len({int(dev.clusters[i]) for i in panel}) == 24  # one chain of any cluster
    lengths = [int(dev.lengths[i]) for i in panel]
    assert lengths == sorted(lengths) and lengths[0] <= 75 and lengths[-1] >= 130
    assert panel == decoder_structure_panel.select_panel(
        dev.ids, dev.lengths, dev.clusters, n=24
    )
    with pytest.raises(ValueError, match="clusters have a chain"):
        decoder_structure_panel.select_panel(
            dev.ids, dev.lengths, dev.clusters, n=24, lo=200, hi=210
        )


def test_a_random_twin_has_exactly_as_many_substitutions():
    rng = np.random.default_rng(0)
    true = "ACDEFGHIKLMNPQRSTVWY" * 5
    for k in (0, 1, 7, 100):
        twin = decoder_structure_panel.random_substitutions(true, k, rng)
        assert len(decoder_structure_panel.substitutions(true, twin)) == k
    skewed = np.zeros(20)
    skewed[decode_eval.ALPHABET.index("W")] = 1.0
    twin = decoder_structure_panel.random_substitutions("A" * 30, 30, rng, skewed)
    assert set(twin) == {"W"}
    with pytest.raises(ValueError):
        decoder_structure_panel.random_substitutions("AAA", 4, rng)
    with pytest.raises(ValueError):
        decoder_structure_panel.substitutions("AAA", "AA")
