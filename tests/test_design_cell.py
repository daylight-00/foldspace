"""One cell of RFD4-Proteina's designs: what is started, in what order, and when it stops."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import foldspace.rfd4.design_cell as design_cell

pytestmark = pytest.mark.offline

REPOSITORY = Path(__file__).resolve().parents[1]
TEMPLATE = REPOSITORY.parent / "RFD4-Proteina/data/benchmarks/monomer/monomer-short.cif"
SYNTHETIC = """data_monomer-short.cif
_entry.id       monomer-short.cif
_entity_poly_seq.mon_id      100-100
_atom_site.annotation_expsegmax_1_residue    100
_atom_site.annotation_expsegmin_1_residue    100
_rfd_design.sampled_contig   100-100
"""


def test_a_chunk_is_expected_to_take_as_long_as_the_last_with_a_margin_and_the_first_as_the_prior():
    assert design_cell.chunk_expected(None, 700) == 700
    assert design_cell.chunk_expected(600, 700) == pytest.approx(690)


def test_a_chunk_starts_only_if_it_ends_with_the_reserve_inside_the_budget():
    assert design_cell.may_start(1000, 700, 2760, 720)  # 2420 <= 2760
    assert not design_cell.may_start(1500, 700, 2760, 720)  # 2920 > 2760
    assert design_cell.may_start(1340, 700, 2760, 720)  # exactly 2760


def test_the_designs_found_are_the_ids_without_the_replicate_suffix(tmp_path):
    for name in ("a-L60", "b-L61"):
        folder = tmp_path / "x" / "inference_samples" / f"{name}_rep0"
        folder.mkdir(parents=True)
        (folder / f"{name}_rep0_generated.cif").write_text("")
    (tmp_path / "x" / "inference_samples" / "other.cif").write_text("")
    assert design_cell.designs_in(tmp_path) == {"a-L60", "b-L61"}


def fake_sampler(written, fail_above=None, log=None):
    """A sampler that writes the designs of its inputs, and exits with 1 after half of them above `fail_above`."""

    def run(batch_size, inputs, out_dir, cwd, seed):
        names = sorted(p.stem for p in inputs.glob("*.cif"))
        if log is not None:
            log.append((batch_size, inputs.name, seed, names))
        failing = fail_above is not None and batch_size > fail_above
        for name in names[: len(names) // 2] if failing else names:
            folder = out_dir / "inference_samples" / f"{name}_rep0"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{name}_rep0_generated.cif").write_text("")
            written.append(name)
        return (1 if failing else 0), 10.0

    return run


def inputs_of(tmp_path, names):
    folder = tmp_path / "inputs"
    folder.mkdir()
    for name in names:
        (folder / f"{name}.cif").write_text("")
    return folder


def test_a_run_that_gets_through_is_not_repeated(tmp_path):
    names = ["a-L60", "b-L61", "c-L62"]
    log: list = []
    result = design_cell.generate_with_ladder(
        names, inputs_of(tmp_path, names), tmp_path / "w", 1, fake_sampler([], log=log)
    )
    assert result["complete"] and result["designs"] == 3 and result["missing"] == []
    assert [a["batch_size"] for a in result["attempts"]] == [8]
    assert len(log) == 1


def test_a_run_that_fails_is_repeated_on_the_missing_designs_at_the_next_size(tmp_path):
    names = ["a-L60", "b-L61", "c-L62", "d-L63"]
    log: list = []
    result = design_cell.generate_with_ladder(
        names,
        inputs_of(tmp_path, names),
        tmp_path / "w",
        7,
        fake_sampler([], fail_above=4, log=log),
    )
    assert result["complete"]
    assert [a["batch_size"] for a in result["attempts"]] == [8, 4]
    assert [a["missing_after"] for a in result["attempts"]] == [2, 0]
    assert log[0][3] == names  # the first attempt is asked for all of them
    assert log[1][3] == ["c-L62", "d-L63"]  # the second only for what is missing
    assert {entry[2] for entry in log} == {7}  # one seed for the chunk


def test_a_design_that_fails_at_every_size_is_listed_and_the_ladder_is_exhausted(
    tmp_path,
):
    names = ["a-L60", "b-L61"]

    def never(batch_size, inputs, out_dir, cwd, seed):
        return 1, 5.0

    result = design_cell.generate_with_ladder(
        names, inputs_of(tmp_path, names), tmp_path / "w", 1, never
    )
    assert not result["complete"] and result["missing"] == names
    assert [a["batch_size"] for a in result["attempts"]] == [8, 4, 2, 1]
    assert result["seconds"] == pytest.approx(20.0)


def test_the_probe_is_tried_at_batch_size_one_alone(tmp_path):
    log: list = []
    result = design_cell.generate_with_ladder(
        ["p-L300"],
        inputs_of(tmp_path, ["p-L300"]),
        tmp_path / "w",
        1,
        fake_sampler([], log=log),
        ladder=(1,),
    )
    assert result["complete"] and [entry[0] for entry in log] == [1]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def chunks_of(n):
    return [{"index": k} for k in range(n)]


def test_every_chunk_is_made_while_there_is_room():
    clock = Clock()

    def make(chunk):
        clock.t += 600
        return {"complete": True, "designs": 32}

    done, stopped, failed = design_cell.run_chunks(
        chunks_of(4), make, clock, 0.0, 5000, 720, 720
    )
    assert [c["index"] for c in done] == [0, 1, 2, 3]
    assert stopped is None and not failed
    assert done[0]["wall_seconds"] == 600


def test_the_chunks_stop_before_the_one_that_would_not_end_with_the_reserve():
    clock = Clock()
    clock.t = 600  # the probe has taken ten minutes

    def make(chunk):
        clock.t += 600
        return {"complete": True, "designs": 32}

    # budget 2760, reserve 720: chunk 0 expected 720 -> 600 + 720 + 720 = 2040 ok;
    # chunk 1 expected 690 at 1200 -> 2610 ok; chunk 2 at 1800 -> 3210 > 2760: stop
    done, stopped, failed = design_cell.run_chunks(
        chunks_of(4), make, clock, 0.0, 2760, 720, 720
    )
    assert [c["index"] for c in done] == [0, 1]
    assert stopped == 2 and not failed


def test_the_first_chunk_is_not_started_if_the_prior_does_not_fit():
    done, stopped, failed = design_cell.run_chunks(
        chunks_of(4), lambda c: {"complete": True}, Clock(), 0.0, 1000, 720, 720
    )
    assert done == [] and stopped == 0 and not failed


def test_a_chunk_that_cannot_be_completed_ends_the_run_of_chunks():
    clock = Clock()

    def make(chunk):
        clock.t += 100
        return {"complete": chunk["index"] != 1}

    done, stopped, failed = design_cell.run_chunks(
        chunks_of(4), make, clock, 0.0, 10_000, 0, 100
    )
    assert [c["index"] for c in done] == [0, 1]
    assert failed and stopped is None


# the whole flow with a stand-in for the sampler: inputs, probe, chunks, records, evaluation


def stand_in_sampler(path: Path, gemmi_path: str):
    path.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, re, sys
            sys.path.insert(0, {gemmi_path!r})
            import gemmi
            args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
            inputs, out = args["inputs"], args["out_dir"]
            batch = int(args["batch_size"])
            with open(os.environ["STAND_IN_LOG"], "a") as log:
                log.write(f"{{batch}} {{args['seed']}} {{os.path.basename(inputs.rstrip('/'))}}\\n")
            names = sorted(f[:-4] for f in os.listdir(inputs) if f.endswith(".cif"))
            limit = float(os.environ.get("STAND_IN_FAIL_ABOVE", "0"))
            failing = bool(limit) and batch > limit
            for name in names[: len(names) // 2] if failing else names:
                length = int(re.search(r"-L(\\d+)$", name).group(1))
                structure = gemmi.Structure()
                model = gemmi.Model("1")
                chain = gemmi.Chain("A")
                for i in range(length):
                    residue = gemmi.Residue()
                    residue.name = "ALA"
                    residue.seqid = gemmi.SeqId(i + 1, " ")
                    atom = gemmi.Atom()
                    atom.name = "CA"
                    atom.element = gemmi.Element("C")
                    atom.pos = gemmi.Position(3.8 * i, 0.0, 0.0)
                    residue.add_atom(atom)
                    chain.add_residue(residue)
                model.add_chain(chain)
                structure.add_model(model)
                structure.setup_entities()
                folder = os.path.join(out, "inference_samples", name + "_rep0")
                os.makedirs(folder, exist_ok=True)
                structure.make_mmcif_document().write_file(
                    os.path.join(folder, name + "_rep0_generated.cif")
                )
            sys.exit(1 if failing else 0)
            """
        )
    )
    path.chmod(0o755)


def cell(tmp_path, monkeypatch, *extra, fail_above=None):
    gemmi = pytest.importorskip("gemmi")
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    os.symlink(sys.executable, env / "bin" / "python")
    sampler = tmp_path / "sampler.py"
    stand_in_sampler(sampler, str(Path(gemmi.__file__).parent.parent))
    panel = tmp_path / "panel.json"
    lengths = [60, 75, 100, 120, 150, 180, 210, 240, 270, 300, 90, 130, 160]
    panel.write_text(
        json.dumps(
            [
                {"id": f"s1-{i:03d}-L{n}", "sequence": "A" * n}
                for i, n in enumerate(lengths)
            ]
        )
    )
    template = tmp_path / "template.cif"
    template.write_text(SYNTHETIC)
    digest = design_cell_digest(template)
    monkeypatch.setenv("STAND_IN_LOG", str(tmp_path / "sampler.log"))
    if fail_above:
        monkeypatch.setenv("STAND_IN_FAIL_ABOVE", str(fail_above))
    out = tmp_path / "cell"
    command = [
        sys.executable,
        str(REPOSITORY / "src/foldspace/rfd4/design_cell.py"),
        "--stack-root",
        str(REPOSITORY),
        "--env-dir",
        str(env),
        "--template",
        str(template),
        "--template-sha256",
        digest,
        "--panel",
        str(panel),
        "--seed",
        "1000",
        "--revision",
        "abc",
        "--out",
        str(out),
        "--chunks",
        "2",
        "--hf-cache",
        str(tmp_path / "hf"),
        "--python-flags=-B",
        "--generate-cmd",
        str(sampler),
        "--evaluate-shell",
        f"echo '{{}}' > '{out}/evaluation.json'",
        *extra,
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    return done, out, tmp_path / "sampler.log"


def design_cell_digest(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_whole_cell_is_made_in_order_and_converted_and_evaluated(
    tmp_path, monkeypatch
):
    done, out, log = cell(tmp_path, monkeypatch)
    assert done.returncode == 0, done.stdout + done.stderr
    runs = [line.split() for line in log.read_text().splitlines()]
    assert runs[0] == [
        "1",
        "1100",
        "probe",
    ]  # the longest design alone, first, at batch size one
    assert [r[2] for r in runs[1:]] == ["chunk0", "chunk1"]
    assert [r[1] for r in runs[1:]] == ["1000", "1001"]
    record = json.loads((out / "cell.json").read_text())
    names = [s["name"] for s in record["stages"]]
    assert names == ["inputs", "probe", "chunk0", "chunk1", "convert", "evaluate"]
    assert record["designs_generated"] == 13
    assert record["stopped_before_chunk"] is None
    assert len(json.loads((out / "records.json").read_text())) == 13
    validation = json.loads((out / "validation.json").read_text())
    assert validation["valid"] == 13 and validation["excluded"] == []
    assert (out / "evaluation.json").exists()


def test_a_sampler_that_fails_at_a_big_batch_is_repeated_at_a_smaller_one(
    tmp_path, monkeypatch
):
    done, out, log = cell(tmp_path, monkeypatch, fail_above=4)
    assert done.returncode == 0, done.stdout + done.stderr
    sizes = [int(line.split()[0]) for line in log.read_text().splitlines()]
    assert sizes[0] == 1 and 8 in sizes and 4 in sizes
    record = json.loads((out / "cell.json").read_text())
    assert record["designs_generated"] == 13
    chunk = next(s for s in record["stages"] if s["name"] == "chunk0")
    assert [a["batch_size"] for a in chunk["attempts"]] == [8, 4]


def test_a_prior_that_does_not_fit_leaves_only_the_probe_and_nothing_is_evaluated(
    tmp_path, monkeypatch
):
    done, out, _ = cell(tmp_path, monkeypatch, "--prior-s", "5000")
    assert done.returncode == 6, done.stdout + done.stderr
    record = json.loads((out / "cell.json").read_text())
    assert record["stopped_before_chunk"] == 0
    assert [s["name"] for s in record["stages"]] == ["inputs", "probe"]
    assert not (out / "evaluation.json").exists()


def test_a_probe_that_cannot_be_made_ends_the_job_before_anything_else_is_spent(
    tmp_path, monkeypatch
):
    done, _, log = cell(tmp_path, monkeypatch, fail_above=0.5)
    # a failure above batch size 0.5 is a failure at size one, so the probe is not made
    assert done.returncode == 3, done.stdout + done.stderr
    assert len(log.read_text().splitlines()) == 1
