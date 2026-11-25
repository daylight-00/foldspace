"""RFD4's designs as the evaluator's records: what is accepted, and every refusal named."""

from __future__ import annotations

import json
import math

import pytest
from foldspace.evaluation import rfd4_records

pytestmark = pytest.mark.offline


def chain(n=6, step=3.8, names=None):
    """`n` residues on a straight line, `step` Å apart, indexed from 1."""
    names = names or ["ALA"] * n
    return [(names[i], i + 1, (i * step, 0.0, 0.0)) for i in range(n)]


def test_a_chain_of_standard_residues_in_a_peptide_trace_has_no_reason_to_be_refused():
    assert rfd4_records.check(chain(), expected=6) == []


@pytest.mark.parametrize(
    "residues, reason",
    [
        (
            chain(names=["ALA", "ALA", "UNK", "ALA", "ALA", "ALA"]),
            "non_standard_residue",
        ),
        (
            chain(names=["ALA", "DA", "ALA", "ALA", "ALA", "ALA"]),
            "non_standard_residue",
        ),
        (
            [*chain()[:3], ("ALA", 3, (11.4, 0.0, 0.0)), *chain()[4:]],
            "duplicate_residue_index",
        ),
        ([*chain()[:3], *chain()[4:]], "residue_indices_not_consecutive"),
        ([*chain()[:2], ("ALA", 3, None), *chain()[3:]], "missing_CA"),
        (
            [*chain()[:2], ("ALA", 3, (math.nan, 0.0, 0.0)), *chain()[3:]],
            "non_finite_coordinate",
        ),
        ([*chain()[:3], ("ALA", 4, (20.0, 0.0, 0.0)), *chain()[4:]], "chain_break"),
        (chain(step=1.0), "CA_overlap"),
        ([], "no_protein_residues"),
    ],
)
def test_each_way_a_chain_can_mislead_the_evaluator_is_named(residues, reason):
    assert reason in rfd4_records.check(residues)


def test_a_chain_of_another_length_than_the_one_asked_for_is_refused_and_not_cut():
    assert rfd4_records.check(chain(6), expected=7) == ["length_not_the_requested"]


def test_a_record_is_the_sequence_and_its_trace_and_nothing_else():
    record = rfd4_records.record_of("d", chain(3, names=["GLY", "SER", "TRP"]))
    assert record == {
        "id": "d",
        "sequence": "GSW",
        "coordinates": [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]],
    }


def test_the_id_is_the_example_name_less_what_is_asked_to_be_dropped():
    from pathlib import Path

    path = Path("x/monomer-short_rep3/monomer-short_rep3_generated.cif")
    assert rfd4_records.design_id(path, ()) == "monomer-short_rep3"
    path0 = Path("s20260929-000-L60_rep0_generated.cif")
    assert rfd4_records.design_id(path0, ("_rep0",)) == "s20260929-000-L60"


def test_the_length_asked_for_comes_from_a_pattern_or_from_the_id():
    assert rfd4_records.requested_length("monomer-long_rep1", {"long": 250}) == 250
    assert rfd4_records.requested_length("s20260929-000-L60", {}) == 60
    assert rfd4_records.requested_length("anything", {}) is None


def write_design(path, residues):
    gemmi = pytest.importorskip("gemmi")
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    ch = gemmi.Chain("A")
    for name, index, xyz in residues:
        residue = gemmi.Residue()
        residue.name = name
        residue.seqid = gemmi.SeqId(index, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(*xyz)
        residue.add_atom(atom)
        ch.add_residue(residue)
    model.add_chain(ch)
    structure.add_model(model)
    structure.setup_entities()
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.make_mmcif_document().write_file(str(path))


def test_designs_are_converted_and_the_refused_are_listed_with_their_reasons(tmp_path):
    pytest.importorskip("gemmi")
    write_design(tmp_path / "a_rep0" / "a_rep0_generated.cif", chain(6))
    write_design(
        tmp_path / "b_rep0" / "b_rep0_generated.cif",
        [*chain()[:3], ("ALA", 4, (20.0, 0.0, 0.0)), *chain()[4:]],
    )
    write_design(tmp_path / "c-L9_rep0" / "c-L9_rep0_generated.cif", chain(6))
    records, report = rfd4_records.convert(tmp_path, drop=("_rep0",))
    assert [r["id"] for r in records] == ["a"]
    assert records[0]["sequence"] == "AAAAAA" and len(records[0]["coordinates"]) == 6
    refused = {e["id"]: e["reasons"] for e in report["excluded"]}
    assert refused == {"b": ["chain_break"], "c-L9": ["length_not_the_requested"]}
    assert report["designs"] == 3 and report["valid"] == 1
    assert report["rules"]["nothing_is_repaired"]
    json.dumps(report)  # the report is plain data


def test_the_files_of_the_fold_tool_are_read_under_an_arms_prefix(tmp_path):
    pytest.importorskip("gemmi")
    write_design(tmp_path / "ubiquitin_s0.cif", chain(6))
    write_design(tmp_path / "ubiquitin_s1.cif", chain(6))
    write_design(tmp_path / "other" / "x_generated.cif", chain(6))
    records, report = rfd4_records.convert(
        tmp_path, pattern="*_s*.cif", strip=".cif", prefix="adapt-rows__"
    )
    assert [r["id"] for r in records] == [
        "adapt-rows__ubiquitin_s0",
        "adapt-rows__ubiquitin_s1",
    ]
    # RFD4's own files are still what the defaults read
    default, _ = rfd4_records.convert(tmp_path)
    assert [r["id"] for r in default] == ["x"]
    assert report["valid"] == 2 and not report["excluded"]
    path = tmp_path / "ubiquitin_s0.cif"
    assert (
        rfd4_records.design_id(path, (), ".cif", "native__") == "native__ubiquitin_s0"
    )


def test_the_readings_that_need_no_judge_are_counted_from_the_records_themselves():
    records = [
        rfd4_records.record_of("a", chain(4, names=["ALA", "GLU", "LYS", "GLY"])),
        rfd4_records.record_of("b", chain(4, names=["LEU", "LEU", "SER", "SER"])),
    ]
    r = rfd4_records.readings(records)
    assert r["a_e_k_l_share"] == pytest.approx(5 / 8)  # A, E, K, L, L of eight
    assert r["per_sample_entropy_median"] == pytest.approx(
        (math.log(4) + math.log(2)) / 2
    )
    assert r["distinct_residues_median"] == 3.0
    assert (
        r["ca_step_median"] == pytest.approx(3.8) and r["ca_steps_in_3.6_to_4.0"] == 1.0
    )
    assert rfd4_records.readings([]) == {}
