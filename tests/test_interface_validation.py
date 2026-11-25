"""The selection, gates and analysis of the interface validation.

The GPU runs are the experiment; these tests pin the parts a run cannot check
about itself -- that pairs are what they claim to be, that a gate passes and fails
where it says it does, and that the analysis recovers a known ordering of
conditions. Numpy only. The injection paths at tensor level are in
`test_injection.py`, which needs torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldspace.esmfold2 import validate_interface
from foldspace.esmfold2 import interface as I

offline = pytest.mark.offline


def _chain(i, length, cluster, rng):
    walk = np.cumsum(rng.standard_normal((length, 3)) * 1.2, axis=0) + 3.8 * np.arange(
        length
    )[:, None] * [1, 0, 0]
    return {
        "id": f"c{i}",
        "length": length,
        "cluster": cluster,
        "sequence": "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=length)),
        "ca": walk,
    }


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    chains, i = [], 0
    for length in range(80, 140, 4):
        for cluster in (i % 3, (i + 1) % 3 + 10, i % 3):
            chains.append(_chain(i, length, cluster + 100 * (length % 2), rng))
            i += 1
    return chains


class _CountingNpz(dict):
    """An ``NpzFile`` stand-in that counts how often each array is read."""

    def __init__(self, arrays):
        super().__init__(arrays)
        self.reads = {}

    def __getitem__(self, key):
        self.reads[key] = self.reads.get(key, 0) + 1
        return super().__getitem__(key)


def _corpus_arrays():
    lengths = np.array([5, 7, 6])
    n = int(lengths.sum())
    xyz = np.arange(n * 9, dtype=np.float32).reshape(n, 3, 3)
    return {
        "ids": np.array(["a", "b", "c"]),
        "lengths": lengths,
        "starts": np.array([0, 5, 12]),
        "clusters": np.array([1, 2, 3]),
        "splits": np.array(["train", "dev", "dev"]),
        "tokens": (np.arange(n) % 20).astype(np.int8),
        "xyz": xyz,
    }


@offline
def test_corpus_chains_reads_the_split_with_its_sequence_and_calpha(
    tmp_path, monkeypatch
):
    np.savez(tmp_path / "chains.npz", **_corpus_arrays())
    dev = validate_interface.corpus_chains(tmp_path / "chains.npz", "dev")
    assert [c["id"] for c in dev] == ["b", "c"]
    assert [c["length"] for c in dev] == [7, 6]
    assert [c["cluster"] for c in dev] == [2, 3]
    arrays = _corpus_arrays()
    # the second atom of each residue is Cα
    assert np.array_equal(dev[0]["ca"], arrays["xyz"][5:12, 1, :])
    assert dev[0]["sequence"] == validate_interface.sequence_of(arrays["tokens"][5:12])
    assert len(dev[1]["sequence"]) == 6


@offline
def test_corpus_chains_reads_each_array_once(monkeypatch):
    """Indexing an NpzFile inside the loop re-reads the 150 MB ``xyz`` for every chain."""
    held = _CountingNpz(_corpus_arrays())
    monkeypatch.setattr(validate_interface.np, "load", lambda *a, **k: held)
    validate_interface.corpus_chains("ignored.npz", "dev")
    assert held.reads
    assert all(count == 1 for count in held.reads.values()), held.reads


@offline
def test_cached_rows_reads_each_array_of_a_shard_once(tmp_path):
    z = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    np.savez(
        tmp_path / "z_shard0.npz",
        ids=np.array(["a", "b", "c"]),
        lengths=np.array([5, 7, 0]),
        starts=np.array([0, 5, 12]),
        z=z,
    )
    rows = validate_interface._cached_rows(tmp_path, ["b", "a"])
    assert np.array_equal(rows["a"], z[0:5])
    assert np.array_equal(rows["b"], z[5:12])
    with pytest.raises(KeyError, match="not in the cache"):
        validate_interface._cached_rows(tmp_path, ["zz"])


@offline
def test_sequence_of_reads_the_corpus_alphabet():
    from foldspace.evaluation import residues

    assert validate_interface.sequence_of([0, 1, 2]) == residues.RESIDUES[:3]
    assert len(validate_interface.sequence_of(range(20))) == 20


@offline
def test_pairs_are_length_matched_cross_cluster_and_deterministic(corpus):
    pairs = validate_interface.select_pairs(corpus, n_pairs=5, lo=80, hi=140, seed=1)
    assert len(pairs) == 5
    assert len({a["length"] for a, _ in pairs}) == 5
    for a, b in pairs:
        assert a["length"] == b["length"]
        assert a["cluster"] != b["cluster"]
        assert a["id"] != b["id"]
    again = validate_interface.select_pairs(corpus, n_pairs=5, lo=80, hi=140, seed=1)
    assert [(a["id"], b["id"]) for a, b in pairs] == [
        (a["id"], b["id"]) for a, b in again
    ]
    other = validate_interface.select_pairs(corpus, n_pairs=5, lo=80, hi=140, seed=2)
    assert [(a["id"], b["id"]) for a, b in pairs] != [
        (a["id"], b["id"]) for a, b in other
    ]


@offline
def test_a_length_with_one_cluster_admits_no_pair():
    rng = np.random.default_rng(0)
    same = [_chain(i, 100, 7, rng) for i in range(4)]
    with pytest.raises(ValueError, match="cross-cluster"):
        validate_interface.select_pairs(same, n_pairs=1, lo=90, hi=110, seed=0)


@offline
def test_spread_by_length_covers_the_range(corpus):
    chosen = validate_interface.spread_by_length(corpus, 4, 80, 140)
    lengths = [c["length"] for c in chosen]
    assert lengths == sorted(lengths)
    assert lengths[0] <= 84 and lengths[-1] >= 132
    with pytest.raises(ValueError):
        validate_interface.spread_by_length(corpus, 4, 500, 600)


@offline
def test_identity_at_equal_length():
    assert validate_interface.identity_at_equal_length("AAAA", "AAAA") == 1.0
    assert validate_interface.identity_at_equal_length("AAAA", "AABB") == 0.5
    with pytest.raises(ValueError):
        validate_interface.identity_at_equal_length("AA", "AAA")


def _record(shifts, repeat=0.0):
    structure = {"repeat": {"max_atom_shift": repeat}}
    for name, s in zip(validate_interface.ROUTES, shifts, strict=True):
        structure[name] = {"max_atom_shift": s}
    tensor = {
        "single_vs_stack": {"max_abs_over_max": 0.0},
        "single_lite_vs_stack": {"max_abs_over_max": 0.0},
    }
    return {
        "id": "x", "length": 100, "structure": structure, "tensor": tensor,
        "single_roundtrip_exact": True,
        "observed_single": {"exact": True},
    }  # fmt: skip


@offline
def test_gate_is_absolute_when_the_repeat_floor_is_zero():
    ok = validate_interface.parity_verdict([_record([0.0, 0.0, 5e-4])], tensor_tol=1e-6)
    assert ok["all_routes_within"] and ok["all_tensors_within"]
    bad = validate_interface.parity_verdict([_record([0.0, 0.0, 0.01])], tensor_tol=1e-6)
    assert not bad["all_routes_within"]


@offline
def test_gate_scales_with_the_repeat_floor():
    # two runs of native differ by 0.5 A, so a route may differ by 1.5 A
    ok = validate_interface.parity_verdict([_record([1.0, 1.4, 0.2], repeat=0.5)], tensor_tol=1e-6)
    assert ok["all_routes_within"]
    bad = validate_interface.parity_verdict([_record([1.0, 1.6, 0.2], repeat=0.5)], tensor_tol=1e-6)
    assert not bad["all_routes_within"]


@offline
def test_tensor_and_roundtrip_failures_are_reported_separately():
    rec = _record([0.0, 0.0, 0.0])
    rec["tensor"]["single_vs_stack"]["max_abs_over_max"] = 1e-2
    rec["single_roundtrip_exact"] = False
    out = validate_interface.parity_verdict([rec], tensor_tol=1e-6)
    assert out["all_routes_within"]
    assert not out["all_tensors_within"]
    assert not out["all_roundtrips_exact"]


@offline
def test_the_single_must_equal_the_one_the_model_derived_to_the_last_bit():
    """The tensor tolerance is not enough here: the definition of H is the model's own."""
    ok = validate_interface.parity_verdict([_record([0.0, 0.0, 0.0])], tensor_tol=1e-6)
    assert ok["all_observed_exact"]

    near = _record([0.0, 0.0, 0.0])
    near["observed_single"] = {"exact": False, "max_abs_over_max": 1e-9}
    assert not validate_interface.parity_verdict([near], tensor_tol=1e-6)["all_observed_exact"]

    # a record from before the observation existed does not pass it by omission
    old = _record([0.0, 0.0, 0.0])
    del old["observed_single"]
    assert not validate_interface.parity_verdict([old], tensor_tol=1e-6)["all_observed_exact"]


def _fold(ca, noise, rng):
    return ca + rng.standard_normal(ca.shape) * noise


@offline
def test_analysis_orders_conditions_by_how_far_they_were_moved():
    rng = np.random.default_rng(3)
    a, b = _chain(0, 120, 1, rng)["ca"], _chain(1, 120, 2, rng)["ca"]
    cas = {"AA": _fold(a, 0.2, rng), "BB": _fold(b, 0.2, rng)}
    cas["AA_seed"] = _fold(cas["AA"], 0.3, rng)
    cas["AB"], cas["BA"] = _fold(cas["AA"], 1.5, rng), _fold(cas["BB"], 1.5, rng)
    cas["Am"], cas["Bm"] = _fold(cas["AA"], 3.0, rng), _fold(cas["BB"], 3.0, rng)
    cas["A0"], cas["B0"] = _fold(cas["AA"], 8.0, rng), _fold(cas["BB"], 8.0, rng)

    pair = {"conditions": validate_interface.analyse_pair(cas, a, b)}
    c = pair["conditions"]
    assert c["AA"]["tm_to_ref"] == pytest.approx(1.0)
    assert (
        c["AA_seed"]["tm_to_ref"]
        > c["AB"]["tm_to_ref"]
        > c["Am"]["tm_to_ref"]
        > c["A0"]["tm_to_ref"]
    )
    assert c["AA"]["tm_to_native"] > 0.9

    summary = validate_interface.summarize_contribution([pair, pair])
    groups = summary["groups"]
    assert (
        groups["seed_floor"]["tm_to_ref"]["median"]
        > groups["swap"]["tm_to_ref"]["median"]
    )
    assert groups["swap"]["tm_to_ref"]["median"] > groups["none"]["tm_to_ref"]["median"]
    fraction = summary["prior_effect_fraction"]["median"]
    assert 0.0 < fraction < 1.0


def _swap_pair(follows_the_single: bool, rng):
    """A pair whose swaps either land on the single's protein or keep the sequence's own."""
    a, b = _chain(0, 120, 1, rng)["ca"], _chain(1, 120, 2, rng)["ca"]
    cas = {"AA": _fold(a, 0.1, rng), "BB": _fold(b, 0.1, rng)}
    cas["AA_seed"] = _fold(cas["AA"], 0.1, rng)
    if follows_the_single:
        cas["AB"], cas["BA"] = _fold(cas["BB"], 0.2, rng), _fold(cas["AA"], 0.2, rng)
    else:
        cas["AB"], cas["BA"] = _fold(cas["AA"], 0.2, rng), _fold(cas["BB"], 0.2, rng)
    cas["Am"], cas["Bm"] = _fold(cas["AA"], 9.0, rng), _fold(cas["BB"], 9.0, rng)
    cas["A0"], cas["B0"] = _fold(cas["AA"], 9.0, rng), _fold(cas["BB"], 9.0, rng)
    return {"conditions": validate_interface.analyse_pair(cas, a, b)}


@offline
def test_a_swap_that_lands_on_the_singles_protein_is_counted_as_following_it():
    rng = np.random.default_rng(7)
    follows = validate_interface.summarize_contribution([_swap_pair(True, rng) for _ in range(3)])
    kept = validate_interface.summarize_contribution([_swap_pair(False, rng) for _ in range(3)])

    assert follows["swap_follows"]["swaps"] == 6
    assert follows["swap_follows"]["follow_the_single"] == 6
    assert follows["swap_follows"]["keep_the_sequences_fold"] == 0
    assert kept["swap_follows"]["follow_the_single"] == 0
    assert kept["swap_follows"]["keep_the_sequences_fold"] == 6
    # a swap that follows the single loses its own sequence's fold about as much as removing
    # the prior does; one that keeps it loses almost nothing. (The value depends on how far
    # the synthetic no-prior fold is from the reference, so the relation is what is checked.)
    assert follows["prior_effect_fraction"]["median"] > 0.5
    assert kept["prior_effect_fraction"]["median"] < 0.1
    assert (
        follows["prior_effect_fraction"]["median"]
        > 5 * kept["prior_effect_fraction"]["median"]
    )


@offline
def test_prior_effect_fraction_is_undefined_when_removing_the_prior_moves_nothing():
    rng = np.random.default_rng(4)
    a, b = _chain(0, 100, 1, rng)["ca"], _chain(1, 100, 2, rng)["ca"]
    cas = {n: a.copy() for n in ("AA", "AB", "Am", "A0", "AA_seed")}
    cas.update({n: b.copy() for n in ("BB", "BA", "Bm", "B0")})
    pair = {"conditions": validate_interface.analyse_pair(cas, a, b)}
    fraction = validate_interface.summarize_contribution([pair])["prior_effect_fraction"]
    assert fraction["n"] == 0


@offline
def test_tensor_and_coordinate_diffs():
    x = np.arange(12, dtype=float).reshape(3, 4)
    assert I.tensor_diff(x, x)["exact"] is True
    d = I.tensor_diff(x + 1e-3, x)
    assert d["max_abs"] == pytest.approx(1e-3)
    assert d["exact"] is False
    with pytest.raises(ValueError):
        I.tensor_diff(x, x[:2])

    rng = np.random.default_rng(5)
    ca = np.cumsum(rng.standard_normal((50, 3)), axis=0) * 3
    coord = np.concatenate([ca, ca + 1.0])
    f = I.Folded(coord=coord, ca=ca, metrics={})
    same = I.coord_diff(f, f)
    assert same["identical"] and same["ca_tm"] == pytest.approx(1.0)
    shifted = I.coord_diff(
        f, I.Folded(coord=coord + [0.0, 0.0, 0.5], ca=ca + [0.0, 0.0, 0.5], metrics={})
    )
    assert shifted["max_atom_shift"] == pytest.approx(0.5)
    assert shifted["ca_rmsd"] == pytest.approx(0.0, abs=1e-9)
