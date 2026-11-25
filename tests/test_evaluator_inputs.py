"""The inputs RFD4-Proteina is asked for at the lengths of a panel: the substitution, the probe and the chunks."""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from foldspace.evaluation import rfd4_inputs

pytestmark = pytest.mark.offline

BENCHMARK = Path(
    "/path/to/design/RFD4-Proteina/data/benchmarks/monomer"
)


def read_benchmark(name: str) -> bytes | None:
    """The shipped `monomer-NAME.cif` as bytes: the file if the checkout has it, else its `.cif.zst` decompressed in memory with the `zstd` command (the checkout's main ships it that way). None when neither can be read."""
    plain = BENCHMARK / f"monomer-{name}.cif"
    if plain.exists():
        return plain.read_bytes()
    packed = BENCHMARK / f"monomer-{name}.cif.zst"
    if packed.exists() and shutil.which("zstd"):
        done = subprocess.run(
            ["zstd", "-dc", str(packed)], check=True, capture_output=True
        )
        return done.stdout
    return None


def has_inputs() -> bool:
    return all(
        (BENCHMARK / f"monomer-{n}.cif").exists()
        or ((BENCHMARK / f"monomer-{n}.cif.zst").exists() and shutil.which("zstd"))
        for n in ("short", "medium", "long")
    )


HAS_INPUTS = has_inputs()
NO_INPUTS = "no monomer-*.cif inputs (plain or as .cif.zst with `zstd` on the path) in an RFD4-Proteina checkout"

# the fields of the benchmark's `monomer-short.cif` that carry its name and its length
TEMPLATE = """data_monomer-short.cif
_entry.id       monomer-short.cif
_entity_poly_seq.mon_id      100-100
_chem_comp.id                        100-100
_atom_site.label_comp_id                     100-100
_atom_site.annotation_expsegmax_1_residue    100
_atom_site.annotation_expsegmin_1_residue    100
_rfd_design.sampled_contig   100-100
_atom_site.Cartn_x                           nan
"""


def records(lengths, seed=20260929):
    return [
        {"id": f"s{seed}-{i:03d}-L{n}", "sequence": "A" * n}
        for i, n in enumerate(lengths)
    ]


def test_an_input_asks_for_the_length_in_the_form_l_dash_l_and_holds_its_own_name():
    text = rfd4_inputs.make_input(TEMPLATE, "s1-007-L63", 63)
    assert text.count("63-63") == 4
    assert "monomer-short" not in text and "100" not in text
    assert "data_s1-007-L63.cif" in text
    assert "_atom_site.annotation_expsegmin_1_residue    63" in text


def test_a_name_that_holds_the_digits_of_the_templates_length_is_left_alone():
    text = rfd4_inputs.make_input(TEMPLATE, "s1-100-L250", 250)
    assert "data_s1-100-L250.cif" in text
    assert text.count("250-250") == 4


def test_an_input_of_the_templates_own_length_differs_from_it_by_the_name_alone():
    text = rfd4_inputs.make_input(TEMPLATE, "x", 100)
    assert text == TEMPLATE.replace("monomer-short", "x")


def test_a_template_without_the_length_in_that_form_is_refused():
    with pytest.raises(ValueError):
        rfd4_inputs.make_input("data_monomer-short.cif\n", "x", 63)


@pytest.mark.skipif(
    not HAS_INPUTS,
    reason=NO_INPUTS,
)
@pytest.mark.parametrize("name, length", [("medium", 150), ("long", 250)])
def test_the_substitution_reproduces_the_benchmarks_own_inputs_line_for_line(
    name, length
):
    template = read_benchmark("short").decode()
    shipped = read_benchmark(name).decode()
    made = rfd4_inputs.make_input(template, f"monomer-{name}", length)

    def lines(
        text,
    ):  # the writer's column order and the time stamp are not part of the input
        return collections.Counter(
            line for line in text.splitlines() if not line.startswith("_entry.time")
        )

    assert lines(made) == lines(shipped)


@pytest.mark.skipif(
    not HAS_INPUTS,
    reason=NO_INPUTS,
)
def test_the_benchmark_template_is_the_one_the_substitution_is_known_for():
    data = read_benchmark("short")
    assert hashlib.sha256(data).hexdigest() in rfd4_inputs.KNOWN_TEMPLATES


def test_a_panel_id_must_hold_the_length_of_its_record_and_be_unique(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(records([60, 75])))
    assert rfd4_inputs.load_panel(good) == [
        ("s20260929-000-L60", 60),
        ("s20260929-001-L75", 75),
    ]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "s1-000-L60", "sequence": "A" * 59}]))
    with pytest.raises(ValueError):
        rfd4_inputs.load_panel(bad)
    twice = tmp_path / "twice.json"
    twice.write_text(json.dumps(records([60]) + records([60])))
    with pytest.raises(ValueError):
        rfd4_inputs.load_panel(twice)


def test_a_gzipped_panel_is_read_like_a_plain_one(tmp_path):
    path = tmp_path / "panel.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump(records([60, 61]), handle)
    assert len(rfd4_inputs.load_panel(path)) == 2


def test_the_longest_design_is_the_probe_and_the_rest_are_dealt_in_turn_in_length_order():
    panel = [
        (f"d{i}", n) for i, n in enumerate([90, 60, 300, 120, 75, 299, 61, 150, 200])
    ]
    dealt = rfd4_inputs.deal(panel, 4)
    assert dealt["probe"] == ("d2", 300)
    lengths = [[n for _, n in chunk] for chunk in dealt["chunks"]]
    # the other eight in length order are 60 61 75 90 120 150 200 299, and design i goes to chunk i mod 4
    assert lengths == [[60, 120], [61, 150], [75, 200], [90, 299]]
    everything = [item for chunk in dealt["chunks"] for item in chunk] + [
        dealt["probe"]
    ]
    assert sorted(everything) == sorted(panel)


def test_a_chunk_stopped_early_spans_the_lengths_and_is_not_the_short_end():
    panel = [(f"d{i:03d}", 60 + 2 * i) for i in range(120)]  # 60 to 298
    dealt = rfd4_inputs.deal(panel, 4)
    first = [n for _, n in dealt["chunks"][0]]
    assert min(first) <= 62 and max(first) >= 290
    assert first == sorted(first)


def test_with_equal_longest_lengths_the_first_by_id_is_the_probe():
    dealt = rfd4_inputs.deal([("b", 300), ("a", 300), ("c", 100)], 2)
    assert dealt["probe"] == ("a", 300)


def test_the_cell_is_written_with_a_manifest_of_what_each_file_holds(tmp_path):
    template = tmp_path / "template.cif"
    template.write_text(TEMPLATE)
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps(records([60, 100, 150, 200, 250, 300, 80, 90, 110])))
    out = tmp_path / "inputs"
    manifest = rfd4_inputs.write(
        template, panel, out, seed=20260929, chunks=2, template_sha256=digest
    )
    assert manifest["probe"]["seed"] == 20260929 + rfd4_inputs.PROBE_OFFSET
    assert [c["seed"] for c in manifest["chunks"]] == [20260929, 20260930]
    written = sorted(p.name for p in out.rglob("*.cif"))
    assert len(written) == 9 and len(set(written)) == 9
    listed = [d["id"] for c in manifest["chunks"] for d in c["designs"]] + [
        d["id"] for d in manifest["probe"]["designs"]
    ]
    assert sorted(f"{i}.cif" for i in listed) == written
    entry = manifest["chunks"][0]["designs"][0]
    data = (out / "chunk0" / f"{entry['id']}.cif").read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert manifest["panel"]["designs"] == 9


def test_a_template_that_is_not_the_one_the_substitution_is_known_for_is_refused(
    tmp_path,
):
    template = tmp_path / "template.cif"
    template.write_text(TEMPLATE + "# changed\n")
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps(records([60, 70])))
    with pytest.raises(ValueError):
        rfd4_inputs.write(template, panel, tmp_path / "inputs", seed=1, chunks=2)
