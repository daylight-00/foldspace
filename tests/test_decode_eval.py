"""Batching and the decoder's metrics, on inputs whose answers are known. Numpy only."""

from __future__ import annotations

import numpy as np
import pytest

from foldspace.esmfold2 import batching, decode_eval

pytestmark = pytest.mark.offline

K = len(decode_eval.ALPHABET)


# ----------------------------------------------------------------- batching


def test_a_cluster_is_drawn_uniformly_and_then_a_chain_of_it():
    clusters = np.array([0, 0, 0, 0, 1, 2])  # one family of four, two singletons
    p = batching.cluster_probabilities(clusters)
    assert p.sum() == pytest.approx(1.0)
    # each of the three clusters has one third, split among its chains
    assert p[:4] == pytest.approx([1 / 12] * 4)
    assert p[4] == pytest.approx(1 / 3) and p[5] == pytest.approx(1 / 3)

    rng = np.random.default_rng(0)
    drawn = batching.draw(6, 60_000, rng, p)
    shares = np.bincount(clusters[drawn], minlength=3) / len(drawn)
    assert shares == pytest.approx([1 / 3] * 3, abs=0.01)
    uniform = batching.draw(6, 60_000, rng)
    assert np.bincount(clusters[uniform], minlength=3)[0] / len(
        uniform
    ) == pytest.approx(4 / 6, abs=0.01)
    with pytest.raises(ValueError):
        batching.draw(6, 10, rng, p[:5])


def test_packing_respects_the_padded_token_budget_and_keeps_every_chain():
    rng = np.random.default_rng(1)
    # the size of an epoch's draw: about 17,000 chains over lengths 60 to 300
    lengths = rng.integers(60, 301, size=17_000)
    indices = np.arange(17_000)
    batches = batching.pack(indices, lengths, 20_000, rng)
    seen = np.concatenate(batches)
    assert sorted(seen.tolist()) == indices.tolist()
    for b in batches:
        assert lengths[b].max() * len(b) <= 20_000 or len(b) == 1
    # chains of alike length share a batch: little of the budget is padding
    padded = sum(lengths[b].max() * len(b) for b in batches)
    assert lengths.sum() / padded > 0.97


def test_without_a_generator_packing_is_by_length_and_repeatable():
    lengths = np.array([300, 60, 150, 90, 200, 120])
    a = batching.pack(np.arange(6), lengths, 400)
    b = batching.pack(np.arange(6), lengths, 400)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    flat = np.concatenate(a)
    assert lengths[flat].tolist() == sorted(lengths.tolist())


def test_a_chain_longer_than_the_budget_is_a_batch_of_its_own():
    lengths = np.array([300, 70, 80])
    batches = batching.pack(np.arange(3), lengths, 200)
    assert [len(b) for b in batches] == [2, 1] or [len(b) for b in batches] == [1, 2]
    assert any(len(b) == 1 and lengths[b[0]] == 300 for b in batches)
    assert batching.pack(np.array([], dtype=np.int64), lengths, 200) == []


# ------------------------------------------------------------------ metrics


def _logp(labels, wrong, rng, sure=0.95):
    """Log-probabilities that call the true type, except at ``wrong`` rows where they call another."""
    n = len(labels)
    called = labels.copy()
    for i in wrong:
        called[i] = (labels[i] + 1) % K
    logp = np.full((n, K), np.log((1 - sure) / (K - 1)))
    logp[np.arange(n), called] = np.log(sure)
    return logp


@pytest.fixture
def split():
    """Three chains of 20, 30 and 50 residues; clusters 0, 0, 1 (the first two are one family)."""
    rng = np.random.default_rng(0)
    lengths = np.array([20, 30, 50])
    starts = np.array([0, 20, 50])
    labels = rng.integers(0, K, size=100)
    clusters = np.array([0, 0, 1])
    return labels, lengths, starts, clusters, rng


def test_a_perfect_decoder_scores_one_everywhere(split):
    labels, lengths, starts, clusters, rng = split
    logp = _logp(labels, [], rng)
    s = decode_eval.summarize(logp, labels, starts, lengths, clusters)
    assert s["micro"]["acc"] == 1.0
    assert s["cluster_weighted"]["acc"] == 1.0
    assert s["chain_weighted"]["chain_exact_rate"] == 1.0
    assert s["substitutions_per_chain"]["max"] == 0
    assert s["top_confusions"] == []
    assert s["micro"]["ce"] == pytest.approx(-np.log(0.95))


def test_each_cluster_counts_once_in_the_primary_reading(split):
    labels, lengths, starts, clusters, rng = split
    # the 50-residue chain (its own cluster) gets 5 wrong; the family of two chains gets none wrong
    wrong = [50 + i for i in range(5)]
    logp = _logp(labels, wrong, rng)
    s = decode_eval.summarize(logp, labels, starts, lengths, clusters)
    assert s["micro"]["acc"] == pytest.approx(95 / 100)
    # cluster 0: 50 of 50 right; cluster 1: 45 of 50 -> the mean of 1.0 and 0.9
    assert s["cluster_weighted"]["acc"] == pytest.approx(0.95)
    # chain exact: the family has two exact chains of two, the singleton none: (1.0 + 0.0) / 2
    assert s["cluster_weighted"]["chain_exact_rate"] == pytest.approx(0.5)
    assert s["chain_weighted"]["chain_exact_rate"] == pytest.approx(2 / 3)
    # the largest cluster is the family of two chains
    assert s["largest_cluster"]["chains"] == 2
    assert s["largest_cluster"]["acc"] == 1.0
    assert s["without_largest_cluster"]["acc"] == pytest.approx(0.9)
    assert s["substitutions_per_chain"]["max"] == 5
    stats = decode_eval.selection_stats(logp, labels, starts, lengths, clusters)
    assert stats["cw_acc"] == pytest.approx(s["cluster_weighted"]["acc"])
    assert stats["cw_ce"] == pytest.approx(s["cluster_weighted"]["ce"])
    assert stats["micro_acc"] == pytest.approx(0.95)


def test_the_cluster_weighted_cross_entropy_is_not_the_micro_one(split):
    labels, lengths, starts, clusters, rng = split
    # make the single-chain cluster confidently wrong everywhere: micro CE moves by its share (50 %), cw by 1/2
    wrong = list(range(50, 100))
    logp = _logp(labels, wrong, rng)
    s = decode_eval.selection_stats(logp, labels, starts, lengths, clusters)
    good, bad = -np.log(0.95), -np.log(0.05 / (K - 1))
    assert s["micro_ce"] == pytest.approx(0.5 * good + 0.5 * bad)
    assert s["cw_ce"] == pytest.approx(
        0.5 * good + 0.5 * bad
    )  # both halves are 50 residues here
    # with unequal clusters the two part ways: make cluster 0 the wrong one instead (30 + 20 residues vs 50)
    logp2 = _logp(labels, list(range(50)), rng)
    s2 = decode_eval.selection_stats(logp2, labels, starts, lengths, clusters)
    assert s2["cw_acc"] == pytest.approx(0.5)


def test_the_confusions_that_cost_most_are_listed_with_their_share(split):
    labels, lengths, starts, clusters, rng = split
    wrong = [0, 1, 2, 3]
    logp = _logp(labels, wrong, rng)
    s = decode_eval.summarize(logp, labels, starts, lengths, clusters)
    assert sum(c["count"] for c in s["top_confusions"]) == 4
    first = s["top_confusions"][0]
    assert (
        first["true"] in decode_eval.ALPHABET
        and first["predicted"] in decode_eval.ALPHABET
    )
    assert 0 < first["share_of_true"] <= 1
    recalls = [v["recall"] for v in s["per_residue"].values() if v["n"]]
    assert min(recalls) < 1.0


def test_accuracy_by_distance_from_the_ends(split):
    labels, lengths, starts, clusters, rng = split
    # errors on the first residue of every chain only
    wrong = [int(x) for x in starts]
    logp = _logp(labels, wrong, rng)
    s = decode_eval.summarize(logp, labels, starts, lengths, clusters)
    assert s["by_terminal_offset"]["N"][0] == 0.0
    assert s["by_terminal_offset"]["N"][1] == 1.0
    assert s["by_terminal_offset"]["C"][0] == 1.0
    assert s["by_terminal_offset"]["interior"] == 1.0


def test_calibration_sees_a_decoder_that_is_sure_and_wrong(split):
    labels, lengths, starts, clusters, rng = split
    sure_and_wrong = _logp(labels, list(range(0, 100, 2)), rng, sure=0.99)
    s = decode_eval.summarize(sure_and_wrong, labels, starts, lengths, clusters)
    assert s["calibration"]["ece"] > 0.4
    assert s["calibration"]["accuracy_when_at_least_0.9"] == pytest.approx(0.5)
    modest = _logp(labels, list(range(0, 100, 2)), rng, sure=0.5)
    assert (
        decode_eval.summarize(modest, labels, starts, lengths, clusters)["calibration"][
            "ece"
        ]
        < 0.05
    )


def test_inputs_that_do_not_agree_are_refused(split):
    labels, lengths, starts, clusters, rng = split
    logp = _logp(labels, [], rng)
    with pytest.raises(ValueError, match="logp"):
        decode_eval.summarize(logp[:, :5], labels, starts, lengths, clusters)
    with pytest.raises(ValueError, match="lengths"):
        decode_eval.summarize(logp, labels, starts, lengths + 1, clusters)
