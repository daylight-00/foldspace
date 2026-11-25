"""The sequence-only generator on batches, losses and chains whose answer is known.

The arithmetic (the schedule, the batches, the cluster weighting, the budget) is numpy only. The model runs on a CPU with tiny
sizes: what is checked is what a run cannot check about itself, that the model is causal and ignores the padding, that the length
token and the one-place shift of the targets are the convention the sampler uses, that the best checkpoint and the stop do what the docstring says, and that a model trained on a pattern draws the pattern.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from foldspace.esmfold2 import latent_generation as latent_generation, sequence_only

pytestmark = pytest.mark.offline

TINY = {
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "d_ff": 64,
    "dropout": 0.0,
    "lr": 3e-3,
    "warmup_steps": 5,
    "max_tokens": 512,
    "eval_every": 20,
    "max_len": 64,
    "seed": 0,
}


def _cfg(**kw):
    return sequence_only.LMConfig(**{**TINY, **kw})


def _corpus(sequences, clusters=None):
    clusters = clusters if clusters is not None else range(len(sequences))
    return sequence_only.corpus_from(
        [
            {"id": f"c{i}", "sequence": s, "cluster": int(c)}
            for i, (s, c) in enumerate(zip(sequences, clusters, strict=True))
        ]
    )


def _random_sequences(n, seed, lo=20, hi=30):
    rng = np.random.default_rng(seed)
    return [
        "".join(rng.choice(list(latent_generation.LETTERS), int(rng.integers(lo, hi))))
        for _ in range(n)
    ]


# ------------------------------------------------------------------ numpy


def test_the_schedule_warms_up_peaks_and_falls_to_a_tenth():
    cfg = sequence_only.LMConfig()
    assert sequence_only.lr_at(0, 20000, cfg) == pytest.approx(cfg.lr / cfg.warmup_steps)
    assert sequence_only.lr_at(cfg.warmup_steps - 1, 20000, cfg) == pytest.approx(cfg.lr)
    assert sequence_only.lr_at(cfg.warmup_steps, 20000, cfg) == pytest.approx(cfg.lr)
    assert sequence_only.lr_at(20000, 20000, cfg) == pytest.approx(cfg.lr * cfg.min_lr_ratio)
    mid = sequence_only.lr_at(10250, 20000, cfg)
    assert cfg.lr * cfg.min_lr_ratio < mid < cfg.lr
    assert sequence_only.lr_at(30000, 20000, cfg) == pytest.approx(
        cfg.lr * cfg.min_lr_ratio
    )  # it does not go round again


def test_overrides_are_typed_by_the_field_and_a_misspelling_is_refused():
    got = sequence_only.parse_overrides("dropout=0.3,patience=5,steps=1000")
    assert got == {"dropout": 0.3, "patience": 5, "steps": 1000}
    assert isinstance(got["patience"], int) and isinstance(got["dropout"], float)
    assert sequence_only.parse_overrides("") == {}
    with pytest.raises(ValueError, match="dropuot"):
        sequence_only.parse_overrides("dropuot=0.3")


def test_a_batch_is_the_length_token_the_residues_shifted_and_padding_ignored():
    corpus = _corpus(["ACD", "EFGHI"])
    inputs, targets = sequence_only.make_batch(corpus, np.array([0, 1]))
    pad, base = sequence_only.PAD, sequence_only.LEN_BASE
    # lengths 3 and 5 are both in the first bin
    # the alphabet is ACDEFGHIKLMNPQRSTVWY: A 0, C 1, D 2, E 3, F 4, G 5, H 6, I 7
    assert inputs.tolist() == [
        [
            base,
            0,
            1,
            2,
            pad,
        ],  # length token, A, C, and D, which only precedes a padded target
        [base, 3, 4, 5, 6],  # length token, E, F, G, H
    ]
    assert targets.tolist() == [
        [0, 1, 2, sequence_only.IGNORE, sequence_only.IGNORE],  # A, C, D
        [3, 4, 5, 6, 7],  # E, F, G, H, I
    ]
    long = _corpus(["A" * 129, "C" * 60])
    inputs, _ = sequence_only.make_batch(long, np.array([0, 1]))
    assert inputs[0, 0] == base + 12 and inputs[1, 0] == base + 6  # bins of ten
    huge = _corpus(["A" * 500])
    assert (
        sequence_only.make_batch(huge, np.array([0]))[0][0, 0] == base + sequence_only.LEN_BINS - 1
    )  # the last bin takes the rest


def test_the_batch_does_not_depend_on_the_order_of_the_corpus():
    seqs = _random_sequences(6, 0)
    a, b = _corpus(seqs), _corpus(seqs[::-1])
    ia, ta = sequence_only.make_batch(a, np.array([0, 3]))
    ib, tb = sequence_only.make_batch(
        b, np.array([5, 2])
    )  # the same two chains, in the same order
    assert np.array_equal(ia, ib) and np.array_equal(ta, tb)


def test_cluster_weighting_is_the_mean_of_cluster_means():
    chain_mean = np.array([1.0, 3.0, 5.0])
    clusters = np.array([0, 0, 1])
    assert sequence_only.cluster_weighted(chain_mean, clusters) == pytest.approx(
        3.5
    )  # not the chain mean, 3.0
    assert sequence_only.cluster_weighted(chain_mean, np.array([7, 8, 9])) == pytest.approx(3.0)


def test_the_budget_counts_slots_unique_presentations_residues_and_squares():
    lengths = np.array([10, 20, 30, 40])
    b = sequence_only.Budget(4)
    b.add(np.array([0, 1]), lengths)
    b.add(np.array([1, 3]), lengths)
    snap = b.snapshot()
    assert snap["updates"] == 2 and snap["example_slots"] == 4
    assert snap["presentations_unique"] == 3  # chain 1 was drawn twice
    assert snap["valid_residues"] == 10 + 20 + 20 + 40
    assert snap["padded_tokens"] == 20 * 2 + 40 * 2
    assert snap["sum_L2"] == 100 + 400 + 400 + 1600
    b.seconds = 7200.0
    assert b.snapshot()["gpu_hours"] == pytest.approx(2.0)


def test_a_corpus_table_has_a_header_and_three_columns(tmp_path):
    path = tmp_path / "corpus.tsv"
    path.write_text(
        "seq_id\tsequence\tcluster\nx1\tACDE\t4\nx2\tFGHI\t4\nx3\tKLMN\t9\n"
    )
    corpus = sequence_only.load_table(path)
    assert corpus.ids == ["x1", "x2", "x3"] and corpus.clusters.tolist() == [4, 4, 9]
    assert corpus.sequence(1) == "FGHI" and corpus.lengths.tolist() == [4, 4, 4]
    path.write_text("seq_id\tsequence\nx1\tACDE\n")
    with pytest.raises(ValueError, match="not a corpus table"):
        sequence_only.load_table(path)


def test_samples_are_written_in_the_pilots_format(tmp_path):
    sequence_only.write_samples(tmp_path / "s", ["ACDE", "FGHIK"], [0.5, 0.25], {"n": 2})
    rows = latent_generation.read_samples(tmp_path / "s")
    assert [(r["id"], r["sequence"], r["confidence"]) for r in rows] == [
        ("s000", "ACDE", 0.5),
        ("s001", "FGHIK", 0.25),
    ]
    assert json.loads((tmp_path / "s" / "samples.json").read_text()) == {"n": 2}


# ------------------------------------------------------------------- torch


def test_the_default_model_is_the_pilots_size():
    pytest.importorskip("torch")
    n = sequence_only.parameters(sequence_only.build_model(sequence_only.LMConfig()))
    assert 39.0e6 < n < 40.5e6  # the pilot's denoiser has 39.0 M


def _model(**kw):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return sequence_only.build_model(_cfg(**kw)).eval(), torch


def test_a_position_does_not_see_what_comes_after_it():
    model, torch = _model()
    ids = torch.as_tensor(np.random.default_rng(0).integers(0, 20, (3, 12)))
    ids[:, 0] = sequence_only.LEN_BASE + 2
    other = ids.clone()
    other[:, 8:] = (other[:, 8:] + 5) % 20
    with torch.no_grad():
        a, b = model(ids), model(other)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)
    assert not torch.allclose(a[:, 8:], b[:, 8:], atol=1e-3)


def test_right_padding_changes_no_real_position():
    model, torch = _model()
    ids = torch.as_tensor(np.random.default_rng(1).integers(0, 20, (2, 10)))
    ids[:, 0] = sequence_only.LEN_BASE
    padded = torch.cat([ids, torch.full((2, 7), sequence_only.PAD)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(ids), model(padded)[:, :10], atol=1e-5)


def test_the_length_token_changes_what_the_model_says_about_the_first_residue():
    model, torch = _model()
    a = torch.as_tensor([[sequence_only.LEN_BASE + 3]])
    b = torch.as_tensor([[sequence_only.LEN_BASE + 20]])
    with torch.no_grad():
        assert not torch.allclose(model(a), model(b), atol=1e-4)


def test_evaluation_is_the_mean_cross_entropy_of_a_uniform_model_and_is_weighted_by_cluster():
    model, torch = _model()
    for p in model.parameters():
        p.data.zero_()  # every logit is 0: the model says 1/20 for every residue
    corpus = _corpus(_random_sequences(6, 2), clusters=[0, 0, 0, 0, 1, 2])
    got = sequence_only.evaluate(model, corpus, 256, torch.device("cpu"))
    assert got["cw_nll"] == pytest.approx(math.log(20), abs=1e-5)
    assert got["micro_nll"] == pytest.approx(math.log(20), abs=1e-5)
    assert got["cw_perplexity"] == pytest.approx(20.0, rel=1e-4)
    assert got["chains"] == 6


def test_training_lowers_the_loss_keeps_the_best_checkpoint_and_records_the_budget(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    seqs = _random_sequences(8, 3)
    corpus = _corpus(seqs)
    cfg = _cfg(steps=300, patience=1000, lr=5e-3)
    result = sequence_only.train(
        corpus, corpus, cfg, tmp_path, device=torch.device("cpu"), log=lambda *a: None
    )
    first, last = result["history"][0], result["history"][-1]
    assert (
        last["dev"]["cw_nll"] < first["dev"]["cw_nll"] - 0.3
        and first["dev"]["cw_nll"] < math.log(20) + 0.5
    )
    # the best reading is the lowest of them, and it is the one kept
    assert result["best"]["cw_nll"] == min(
        h["dev"]["cw_nll"] for h in result["history"]
    )
    assert (tmp_path / "best.pt").exists() and (tmp_path / "final.pt").exists()
    assert result["stopped"] == "the step ceiling" and result["steps"] == 300
    b = result["run_budget"]
    assert (
        b["updates"] == 300
        and b["example_slots"] >= 300
        and b["presentations_unique"] <= 8
    )
    assert b["valid_residues"] > 0 and b["gpu_hours"] > 0
    assert result["best"]["budget"]["updates"] == result["best"]["step"]
    _, _, state = sequence_only.load_checkpoint(tmp_path / "best.pt", torch.device("cpu"))
    assert state["step"] == result["best"]["step"] and state["dev"][
        "cw_nll"
    ] == pytest.approx(result["best"]["cw_nll"])


def test_the_run_stops_after_the_patience_without_a_better_dev_loss(tmp_path):
    torch = pytest.importorskip("torch")
    train_corpus = _corpus(_random_sequences(8, 4))
    dev_corpus = _corpus(
        _random_sequences(8, 5)
    )  # other random chains: what is learnt of the first does not carry over
    cfg = _cfg(steps=400, patience=2, lr=1e-2)
    result = sequence_only.train(
        train_corpus,
        dev_corpus,
        cfg,
        tmp_path,
        device=torch.device("cpu"),
        log=lambda *a: None,
    )
    assert result["stopped"].startswith("no better dev loss")
    assert result["steps"] < 400
    assert result["best"]["step"] <= result["steps"] - 2 * cfg.eval_every
    _, _, state = sequence_only.load_checkpoint(tmp_path / "best.pt", torch.device("cpu"))
    assert (
        state["step"] == result["best"]["step"]
    )  # the checkpoint kept is the best one, not the last


def test_a_run_resumes_where_it_stopped(tmp_path):
    torch = pytest.importorskip("torch")
    corpus = _corpus(_random_sequences(8, 6))
    device = torch.device("cpu")
    sequence_only.train(
        corpus,
        corpus,
        _cfg(steps=40, patience=1000),
        tmp_path,
        device=device,
        log=lambda *a: None,
    )
    resumed = sequence_only.train(
        corpus,
        corpus,
        _cfg(steps=80, patience=1000),
        tmp_path,
        device=device,
        log=lambda *a: None,
    )
    assert resumed["steps"] == 80
    assert [r["step"] for r in resumed["history"]] == [20, 40, 60, 80]
    assert (
        resumed["run_budget"]["updates"] == 80
    )  # the 40 before are counted, not repeated


def test_a_model_trained_on_a_pattern_draws_the_pattern_at_the_lengths_it_is_asked_for(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    pattern = "ACDEFGHIKL" * 3
    corpus = _corpus([pattern] * 16, clusters=[i % 4 for i in range(16)])
    cfg = _cfg(steps=200, patience=1000)
    sequence_only.train(
        corpus, corpus, cfg, tmp_path, device=torch.device("cpu"), log=lambda *a: None
    )
    model, _, _ = sequence_only.load_checkpoint(tmp_path / "best.pt", torch.device("cpu"))
    gen = torch.Generator().manual_seed(0)
    sequences, confidence = sequence_only.sample(
        model, [30] * 6, temperature=1.0, generator=gen, device=torch.device("cpu")
    )
    assert [len(s) for s in sequences] == [30] * 6
    agree = np.mean(
        [a == b for s in sequences for a, b in zip(s, pattern, strict=True)]
    )
    assert (
        agree > 0.95
    )  # the length token and the one-place shift are the convention the sampler uses
    assert all(0.0 < c <= 1.0 for c in confidence) and np.mean(confidence) > 0.9
    # a shorter chain is a shorter draw, not a cut of a longer one
    short, _ = sequence_only.sample(
        model,
        [12, 20],
        temperature=1.0,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    assert [len(s) for s in short] == [12, 20]


def test_sampling_is_deterministic_in_the_generator_and_sharpens_with_the_temperature():
    model, torch = _model()
    device = torch.device("cpu")
    a, _ = sequence_only.sample(
        model,
        [25, 25, 25],
        temperature=1.0,
        generator=torch.Generator().manual_seed(3),
        device=device,
    )
    b, _ = sequence_only.sample(
        model,
        [25, 25, 25],
        temperature=1.0,
        generator=torch.Generator().manual_seed(3),
        device=device,
    )
    c, _ = sequence_only.sample(
        model,
        [25, 25, 25],
        temperature=1.0,
        generator=torch.Generator().manual_seed(4),
        device=device,
    )
    assert a == b and a != c
    cold1, _ = sequence_only.sample(
        model,
        [25, 25],
        temperature=1e-4,
        generator=torch.Generator().manual_seed(5),
        device=device,
    )
    cold2, _ = sequence_only.sample(
        model,
        [25, 25],
        temperature=1e-4,
        generator=torch.Generator().manual_seed(6),
        device=device,
    )
    assert (
        cold1 == cold2
    )  # at a near-zero temperature the seed does not matter: the argmax is drawn
    assert set("".join(a)) <= set(latent_generation.LETTERS)


# ----------------------------------------------- the floor a sequence model is read against


def _iid(n_chains, length, probs, seed, letters="ACDE"):
    rng = np.random.default_rng(seed)
    return _corpus(
        ["".join(rng.choice(list(letters), length, p=probs)) for _ in range(n_chains)]
    )


def test_the_unigram_floor_is_the_entropy_of_the_composition():
    probs = [0.5, 0.25, 0.125, 0.125]
    entropy = -sum(p * math.log(p) for p in probs)  # 1.2130 nats
    got = sequence_only.ngram_nll(_iid(150, 200, probs, 0), _iid(60, 200, probs, 1), 0)
    assert got["cw_nll"] == pytest.approx(entropy, abs=0.02)
    assert got["micro_nll"] == pytest.approx(entropy, abs=0.02)
    assert got["cw_perplexity"] == pytest.approx(math.exp(entropy), rel=0.03)


def _markov(n_chains, length, stay, seed):
    rng = np.random.default_rng(seed)
    chains = []
    for _ in range(n_chains):
        state, out = int(rng.integers(2)), []
        for _ in range(length):
            out.append("AC"[state])
            state = state if rng.random() < stay else 1 - state
        chains.append("".join(out))
    return _corpus(chains)


def test_the_previous_residue_is_read_where_there_is_context_and_not_where_there_is_none():
    markov_train, markov_dev = _markov(120, 200, 0.9, 0), _markov(50, 200, 0.9, 1)
    h = -(
        0.9 * math.log(0.9) + 0.1 * math.log(0.1)
    )  # 0.3251 nats once the previous residue is known
    uni = sequence_only.ngram_nll(markov_train, markov_dev, 0)
    bi = sequence_only.ngram_nll(markov_train, markov_dev, 1)
    assert uni["cw_nll"] == pytest.approx(math.log(2), abs=0.03)
    assert bi["cw_nll"] == pytest.approx(h, abs=0.03)
    assert bi["cw_nll"] < uni["cw_nll"] - 0.3
    # independent residues: the previous one says nothing, and the bigram gains nothing
    probs = [0.4, 0.3, 0.2, 0.1]
    a, b = _iid(150, 200, probs, 2), _iid(60, 200, probs, 3)
    assert sequence_only.ngram_nll(a, b, 1)["cw_nll"] == pytest.approx(
        sequence_only.ngram_nll(a, b, 0)["cw_nll"], abs=0.02
    )


def test_the_floor_is_weighted_by_cluster_and_refuses_an_order_it_does_not_have():
    train = _iid(40, 100, [0.9, 0.1], 4, letters="AC")
    easy = ["A" * 50] * 5
    hard = ["C" * 50]
    dev = _corpus(easy + hard, clusters=[0, 0, 0, 0, 0, 1])
    got = sequence_only.ngram_nll(train, dev, 0)
    assert (
        got["cw_nll"] > got["micro_nll"]
    )  # the one hard chain is half the clusters and a sixth of the residues
    with pytest.raises(ValueError, match="order"):
        sequence_only.ngram_nll(train, dev, 2)


# ------------------------------------------------ a slice of the corpus itself held out


def test_whole_clusters_are_held_out_and_the_rest_shares_none_with_them():
    seqs = _random_sequences(30, 7)
    clusters = [i // 2 for i in range(30)]  # fifteen clusters of two
    corpus = _corpus(seqs, clusters=clusters)
    train, held = sequence_only.split_heldout(corpus, 4)
    assert len(np.unique(held.clusters)) == 4 and held.n_chains == 8
    assert train.n_chains == 22 and not set(train.clusters) & set(held.clusters)
    assert sorted(train.ids + held.ids) == sorted(
        corpus.ids
    )  # nothing lost, nothing twice
    for part in (train, held):  # the chains are the same chains, in the same residues
        for k, name in enumerate(part.ids):
            assert part.sequence(k) == seqs[int(name[1:])]
    again, held_again = sequence_only.split_heldout(corpus, 4)
    assert (
        held_again.ids == held.ids and again.ids == train.ids
    )  # the same slice every time
    _, other_held = sequence_only.split_heldout(corpus, 4, seed=1)
    assert other_held.ids != held.ids
    with pytest.raises(ValueError, match="held out"):
        sequence_only.split_heldout(corpus, 0)
    with pytest.raises(ValueError, match="held out"):
        sequence_only.split_heldout(corpus, 15)


def test_the_checkpoint_can_be_chosen_by_the_held_out_slice_and_says_what_chose_it(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    train = _corpus(_random_sequences(8, 8))
    dev = _corpus(
        _random_sequences(8, 9)
    )  # unrelated chains: their loss does not improve
    held = train  # a held-out slice that the model does improve on, so that the two readings part
    cfg = _cfg(steps=200, patience=1000, lr=5e-3, select="heldout")
    result = sequence_only.train(
        train,
        dev,
        cfg,
        tmp_path,
        device=torch.device("cpu"),
        log=lambda *a: None,
        heldout=held,
    )
    best = result["best"]
    assert best["selected_by"] == "heldout" and best["heldout_nll"] is not None
    assert best["cw_nll"] == best["heldout_nll"]
    assert best["cw_nll"] == min(h["heldout"]["cw_nll"] for h in result["history"])
    assert (
        best["dev_nll"] > best["heldout_nll"]
    )  # the dev chains would have chosen an earlier checkpoint
    assert result["heldout"] == {"chains": 8, "clusters": 8}
    _, _, state = sequence_only.load_checkpoint(tmp_path / "best.pt", torch.device("cpu"))
    assert state["heldout"]["cw_nll"] == pytest.approx(best["heldout_nll"])


def test_choosing_by_a_held_out_slice_needs_one_and_a_misspelling_is_refused(tmp_path):
    torch = pytest.importorskip("torch")
    corpus = _corpus(_random_sequences(6, 10))
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="held-out"):
        sequence_only.train(
            corpus,
            corpus,
            _cfg(select="heldout"),
            tmp_path,
            device=device,
            log=lambda *a: None,
        )
    with pytest.raises(ValueError, match="dev or heldout"):
        sequence_only.train(
            corpus,
            corpus,
            _cfg(select="helodut"),
            tmp_path,
            device=device,
            log=lambda *a: None,
        )


def test_the_floors_can_be_read_on_the_slice_held_out_of_a_corpus(tmp_path):
    import argparse

    rng = np.random.default_rng(11)
    lines = ["seq_id\tsequence\tcluster"]
    for i in range(40):
        lines.append(
            f"c{i}\t{''.join(rng.choice(list('ACDE'), 80, p=[0.5, 0.25, 0.125, 0.125]))}\t{i // 2}"
        )
    table = tmp_path / "corpus.tsv"
    table.write_text("\n".join(lines) + "\n")
    out = tmp_path / "ngram.json"
    args = argparse.Namespace(
        corpus_table=table, heldout_clusters=5, chains=None, out=str(out)
    )
    assert sequence_only.run_ngram(args) == 0
    got = json.loads(out.read_text())
    assert got["read_on"] == "the held-out slice of the corpus"
    assert got["dev_chains"] == 10 and got["train_chains"] == 30
    entropy = -sum(p * math.log(p) for p in (0.5, 0.25, 0.125, 0.125))
    assert got["models"]["unigram"]["cw_nll"] == pytest.approx(entropy, abs=0.12)
    assert got["uniform_nll"] == pytest.approx(math.log(20))
