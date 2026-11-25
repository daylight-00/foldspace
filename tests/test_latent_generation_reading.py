"""The reading of latent generation's samples on sequences and folds whose answer is known. Numpy only."""

from __future__ import annotations

import numpy as np
import pytest

from foldspace.esmfold2 import latent_generation

pytestmark = pytest.mark.offline


def _sequence(seed, n=120):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list(latent_generation.LETTERS), n))


def test_a_kmer_is_a_base_twenty_number_and_a_short_sequence_has_none():
    # window [A, C]: A is 0 and C is 1, the first position counts for 1 and the second for 20
    assert list(latent_generation.kmer_codes("AC", 2)) == [20]
    assert list(latent_generation.kmer_codes("ACA", 2)) == [20, 1]
    assert len(latent_generation.kmer_codes("ACD", 8)) == 0
    codes = latent_generation.kmer_codes(_sequence(0), 8)
    assert len(codes) == 120 - 8 + 1 and codes.dtype == np.int64


def test_containment_is_one_for_a_copy_about_a_half_for_a_chimera_and_near_zero_for_another_protein():
    a, b = _sequence(1), _sequence(2)
    reference = latent_generation.kmer_set([a])
    assert latent_generation.containment(a, reference) == 1.0
    assert latent_generation.containment(b, reference) < 0.02
    chimera = a[:60] + b[:60]
    assert 0.4 < latent_generation.containment(chimera, reference) < 0.6
    assert np.isnan(latent_generation.containment("ACD", reference))


def test_composition_and_its_divergence():
    p = latent_generation.composition(["AAAA", "CCCC"])
    assert p[0] == 0.5 and p[1] == 0.5 and p.sum() == pytest.approx(1.0)
    assert latent_generation.jensen_shannon(p, p) == 0.0
    q = latent_generation.composition(["DDDD"])
    assert latent_generation.jensen_shannon(p, q) == pytest.approx(1.0)  # disjoint: one bit
    assert latent_generation.jensen_shannon(p, q) == pytest.approx(latent_generation.jensen_shannon(q, p))
    mix = latent_generation.composition(["AAAA", "CCCC", "AAAA"])
    assert 0.0 < latent_generation.jensen_shannon(p, mix) < 0.1


def _fold(
    n_samples=6,
    n_refs=3,
    *,
    direct=0.8,
    mean=0.4,
    shuffled=0.5,
    floor=0.3,
    ref=0.9,
    confident_up_to=None,
):
    chains = []
    for i in range(n_samples):
        p = direct if confident_up_to is None or i < confident_up_to else 0.4
        chains.append(
            {
                "id": f"s{i:03d}",
                "kind": "sample",
                "length": 100,
                "sequence": _sequence(10 + i),
                "floor_sequence": _sequence(70 + i),
                "plddt": {
                    "direct": p,
                    "mean": mean,
                    "shuffled": shuffled,
                    "floor": floor,
                },
                "ptm": {"direct": 0.6, "mean": 0.3, "shuffled": 0.4, "floor": 0.2},
            }
        )
    for j in range(n_refs):
        chains.append(
            {
                "id": f"r{j}",
                "kind": "reference",
                "length": 100,
                "sequence": _sequence(50 + j),
                "plddt": {"direct": ref},
                "ptm": {"direct": 0.8},
            }
        )
    return {"chains": chains}


def _tables(sample_sctm, ref_sctm=0.9, floor_sctm=0.2, n_refs=3):
    return {
        "samples": [{"id": f"s{i:03d}", "scTM": v} for i, v in enumerate(sample_sctm)],
        "reference": [{"id": f"r{j}", "scTM": ref_sctm} for j in range(n_refs)],
        "floor": [
            {"id": f"s{i:03d}", "scTM": floor_sctm} for i in range(len(sample_sctm))
        ],
    }


def _samples(fold):
    return [
        {"id": c["id"], "sequence": c["sequence"], "confidence": 0.9}
        for c in fold["chains"]
        if c["kind"] == "sample"
    ]


def _reference_data():
    train = [_sequence(900 + i) for i in range(5)]
    return latent_generation.kmer_set(train), latent_generation.composition(train)


def test_a_success_is_a_confident_fold_that_refolds_to_itself_and_the_arm_is_read_against_the_reference_and_the_floor():
    fold = _fold(confident_up_to=5)  # five of six samples are confident
    atlas = _tables(
        [0.9, 0.9, 0.9, 0.9, 0.2, 0.9]
    )  # one of the confident ones does not refold to itself
    kmers, comp = _reference_data()
    s = latent_generation.summarize_arm(fold, _samples(fold), atlas, kmers, comp)
    assert s["direct_fold"]["share_confident"] == pytest.approx(5 / 6)
    assert s["success_share"] == pytest.approx(
        4 / 6
    )  # confident and consistent: 0, 1, 2, 3
    assert s["reference"]["success_share"] == 1.0
    assert s["floor"]["share_confident"] == 0.0 and s["floor"]["success_share"] == 0.0
    assert s["verdict"]["supported"] is True
    # a sample is not a copy of train
    assert s["sequence"]["share_memorised"] == 0.0
    assert s["sequence"]["mean_pairwise_eight_mer_jaccard"] < 0.05
    assert s["self_consistency"]["samples_share_consistent"] == pytest.approx(5 / 6)


def test_a_floor_that_folds_as_confidently_and_consistently_leaves_an_arm_unsupported():
    fold = _fold(floor=0.9)  # shuffled sequences fold confidently
    atlas = _tables([0.9] * 6, floor_sctm=0.9)  # and refold to themselves
    kmers, comp = _reference_data()
    s = latent_generation.summarize_arm(fold, _samples(fold), atlas, kmers, comp)
    assert s["success_share"] == 1.0 and s["floor"]["success_share"] == 1.0
    assert s["verdict"]["supported"] is False


def test_the_mean_and_shuffled_controls_are_read_but_do_not_decide_the_verdict():
    # given the sequence the model folds from almost any prior: the controls fold as confidently as the sample does
    fold = _fold(mean=0.9, shuffled=0.9)
    atlas = _tables([0.9] * 6)
    kmers, comp = _reference_data()
    s = latent_generation.summarize_arm(fold, _samples(fold), atlas, kmers, comp)
    assert s["controls"]["mean"]["share_confident"] == 1.0
    assert s["controls"]["shuffled"]["share_confident"] == 1.0
    assert s["verdict"]["supported"] is True


def test_an_arm_that_succeeds_on_few_samples_is_not_supported_however_it_compares_to_the_floor():
    fold = _fold(confident_up_to=2)
    atlas = _tables([0.9] * 6)
    kmers, comp = _reference_data()
    s = latent_generation.summarize_arm(fold, _samples(fold), atlas, kmers, comp)
    assert s["success_share"] == pytest.approx(2 / 6)
    assert s["verdict"]["supported"] is False  # under half of what the real chains do


def test_a_verdict_needs_the_references_and_the_floor_to_have_been_refolded():
    fold = _fold()
    kmers, comp = _reference_data()
    atlas = _tables([0.9] * 6)
    del atlas["floor"]
    s = latent_generation.summarize_arm(fold, _samples(fold), atlas, kmers, comp)
    assert s["verdict"]["supported"] is None  # not False: it cannot be made
    # an old fold record that has no floor route is read, not refused
    old = _fold()
    for c in old["chains"]:
        c["plddt"].pop("floor", None)
    t = latent_generation.summarize_arm(old, _samples(old), None, kmers, comp)
    assert t["floor"]["share_confident"] == 0.0


def test_without_atlasfold_the_arm_has_no_verdict_and_a_copy_is_seen_as_one():
    fold = _fold()
    train = [c["sequence"] for c in fold["chains"] if c["kind"] == "sample"][
        :2
    ]  # two samples are train chains
    s = latent_generation.summarize_arm(
        fold, _samples(fold), None, latent_generation.kmer_set(train), latent_generation.composition(train)
    )
    assert "verdict" not in s and "success_share" not in s
    assert s["sequence"]["share_memorised"] == pytest.approx(2 / 6)


def test_a_samples_table_is_read_and_a_different_table_is_refused(tmp_path):
    (tmp_path / "samples.tsv").write_text(
        "sample_id\tsequence\tmean_confidence\ns000\tMKV\t0.91\n"
    )
    assert latent_generation.read_samples(tmp_path) == [
        {"id": "s000", "sequence": "MKV", "confidence": 0.91}
    ]
    (tmp_path / "samples.tsv").write_text("seq_id\tsequence\nA\tMKV\n")
    with pytest.raises(ValueError, match="not a samples table"):
        latent_generation.read_samples(tmp_path)


def test_a_partial_completion_is_read_as_identity_and_whether_it_folds():
    partial = {
        "t0": {
            "0.5": [
                {"id": "a", "identity": 0.95, "mean_confidence": 0.99},
                {"id": "b", "identity": 0.85, "mean_confidence": 0.97},
            ],
            "0.9": [
                {"id": "a", "identity": 1.0, "mean_confidence": 1.0},
                {"id": "b", "identity": 0.99, "mean_confidence": 0.99},
            ],
        }
    }
    atlas = {"0.5": {"a": 0.9, "b": 0.3}, "0.9": {"a": 0.97, "b": 0.95}}
    s = latent_generation.summarize_partial(partial, atlas, {"a": 0.97, "b": 0.96})
    assert s["0.5"]["identity_median"] == pytest.approx(0.9)
    assert s["0.5"]["identity_share_over_0.9"] == pytest.approx(0.5)
    assert s["0.5"]["sctm_share_over_0.5"] == pytest.approx(0.5)
    assert s["0.9"]["sctm_median"] == pytest.approx(0.96)
    assert s["real_sequences"]["sctm_median"] == pytest.approx(0.965)
    # without AtlasFold only the identity is read
    assert "sctm_median" not in latent_generation.summarize_partial(partial)["0.5"]
