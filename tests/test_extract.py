"""The sequence tables and the shard writer the extraction uses. Numpy only."""

from __future__ import annotations

import numpy as np
import pytest

from foldspace.esmfold2 import target_audit as audit, extract, sequences

pytestmark = pytest.mark.offline


def test_a_table_round_trips_in_the_corpus_order(tmp_path):
    chains = [{"id": "1ABC_1", "sequence": "MKV"}, {"id": "2XYZ_1", "sequence": "GGSA"}]
    path = tmp_path / "split.tsv"
    assert sequences.write_table(path, chains) == 2
    assert path.read_text().splitlines()[0] == "seq_id\tsequence"
    assert sequences.read_table(path) == [("1ABC_1", "MKV"), ("2XYZ_1", "GGSA")]


def test_a_chain_listed_twice_is_refused(tmp_path):
    chains = [{"id": "1ABC_1", "sequence": "MKV"}, {"id": "1ABC_1", "sequence": "MKV"}]
    with pytest.raises(ValueError, match="twice"):
        sequences.write_table(tmp_path / "split.tsv", chains)


def test_a_shard_is_read_back_by_the_audit_unchanged(tmp_path):
    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal((n, 8)).astype(np.float32) for n in (5, 7, 6)]
    path = tmp_path / "z_shard0.npz"
    rows = extract.write_shard(
        path,
        ["a", "b", "c"],
        blocks,
        [("d", 9, "TokenCountMismatch")],
        checkpoint="ESMFold2",
        provenance={"checkpoint": {"weights_sha256": "abc"}},
    )
    assert rows == 18

    shard = audit.load_shard(path)
    assert list(shard.ids) == ["a", "b", "c"]
    assert list(shard.lengths) == [5, 7, 6]
    assert list(shard.starts) == [0, 5, 12]
    assert np.array_equal(shard.z, np.concatenate(blocks))
    assert shard.z.dtype == np.float32
    assert shard.provenance["checkpoint"]["weights_sha256"] == "abc"
    held = np.load(path, allow_pickle=False)
    assert list(held["failed_ids"]) == ["d"]


def test_a_shard_with_no_target_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no sequence produced a target"):
        extract.write_shard(
            tmp_path / "z.npz", [], [], [], checkpoint="ESMFold2", provenance={}
        )


def test_a_bfloat16_value_survives_the_float32_it_is_stored_in():
    """The model's single is bfloat16; its float32 copy loses nothing."""
    torch = pytest.importorskip("torch")
    x = torch.randn(1000) * 10
    b = x.to(torch.bfloat16)
    assert torch.equal(b.to(torch.float32).to(torch.bfloat16), b)
