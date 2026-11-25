"""One cell of RFD4-Proteina's designs on the ruler's panel: generated, converted and scored in one allocation.

Generation, conversion and the frozen evaluator are run one after the other inside the one job, so that the
card is not given back between them and the environments are read from disk once. The standard library is all this
file needs: it runs under RFD4's own interpreter and starts the other stages as processes of their own, each in the
environment it belongs to (RFD4's for the sampler and the converter, this repository's stack for the evaluator).

What it does, in order (`src/foldspace/evaluation/rfd4_inputs.py` says why the inputs are dealt as they are):

1. writes the inputs of the cell: the longest design alone, and the rest in interleaved chunks;
2. the probe: generates the longest design alone at batch size 1. It absorbs what a first run costs before its first
   design and shows whether the longest chain fits; a probe that fails ends the job, before anything else is spent;
3. the chunks, in turn, each at the first batch size of the ladder 8, 4, 2, 1 that gets through it (a run that fails
   is repeated on the designs that are missing, at the next size). Before a chunk is started it must be expected to
   end, with the room the conversion and the evaluation need, before the budget does; it is expected to take as long as
   the last one took, with a margin, and the first as long as a prior says. A chunk that is not started is not
   generated, and the cell that results is a sample of the panel's lengths, not its short end;
4. converts every design that exists to the evaluator's records (`src/foldspace/evaluation/rfd4_records.py`, which repairs nothing);
5. scores the records with the frozen evaluator.

It writes `cell.json` after every stage, so that a job that is stopped leaves what it had done, and the memory of the
card every five seconds (`gpu_memory.csv`). A job stopped between stages loses nothing that is on disk: the designs are
in the work directory, and a fold the evaluator had finished is scored later on a CPU (`--predictions`).

    python -I -B src/foldspace/rfd4/design_cell.py --stack-root DIR --env-dir RFD4_VENV --template monomer-short.cif \\
        --panel records.json.gz --seed 20260929 --revision REV --out DIR/runs/cell
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

LADDER = (8, 4, 2, 1)
GROWTH = 1.15
HF_REPOSITORY = "nvidia/RFD4-Proteina"
# what the stages must not inherit from the shell that submitted the job
FOREIGN = (
    "DESIGN_ROOT",
    "EF2AW_ROOT",
    "PYTHONPATH",
    "PYTHONHOME",
    "FS_ROOT",
    "FS_RUNS",
    "FS_CACHE",
    "EF_MODELS",
    "ESMCFOLD_CCD_PATH",
    "VIRTUAL_ENV",
)


def chunk_expected(last: float | None, prior: float, growth: float = GROWTH) -> float:
    """How long the next chunk is expected to take: as long as the last one took, with a margin; the first, the prior."""
    return prior if last is None else growth * last


def may_start(elapsed: float, expected: float, budget: float, reserve: float) -> bool:
    """Whether a chunk expected to take `expected` ends, with `reserve` left for what follows, inside `budget`."""
    return elapsed + expected + reserve <= budget


def designs_in(root: Path) -> set[str]:
    """The ids of the designs RFD4 has written under `root` (`<id>_rep0/<id>_rep0_generated.cif`)."""
    return {
        p.name.removesuffix("_generated.cif").removesuffix("_rep0")
        for p in root.rglob("*_generated.cif")
    }


def generate_with_ladder(
    wanted: list[str],
    inputs: Path,
    work: Path,
    seed: int,
    run: Callable[[int, Path, Path, Path, int], tuple[int, float]],
    ladder: tuple[int, ...] = LADDER,
) -> dict:
    """Generate `wanted` (ids whose inputs are `inputs/<id>.cif`) at the first batch size that gets through them.

    `run(batch_size, inputs, out_dir, cwd, seed)` is one run of the sampler, returning its exit code and seconds. A
    run that fails leaves the designs it had written; the next is asked for the designs that are missing only. Each
    attempt has an output directory of its own two levels down, because the sampler stops at once if a results file
    already stands beside the directory it is given."""
    attempts = []
    current = inputs
    for n, batch_size in enumerate(ladder):
        attempt = work / f"try{n}"
        out = attempt / "out"
        attempt.mkdir(parents=True, exist_ok=True)
        code, seconds = run(batch_size, current, out, attempt, seed)
        have = designs_in(work)
        missing = [name for name in wanted if name not in have]
        attempts.append(
            {
                "batch_size": batch_size,
                "returncode": code,
                "seconds": seconds,
                "missing_after": len(missing),
            }
        )
        if not missing:
            break
        current = attempt / "remaining"
        current.mkdir(parents=True, exist_ok=True)
        for name in missing:
            target = current / f"{name}.cif"
            if not target.exists():
                target.symlink_to((inputs / f"{name}.cif").resolve())
    have = designs_in(work)
    missing = [name for name in wanted if name not in have]
    return {
        "attempts": attempts,
        "designs": len(wanted) - len(missing),
        "missing": missing,
        "complete": not missing,
        "seconds": sum(a["seconds"] for a in attempts),
    }


def run_chunks(
    chunks: list[dict],
    generate_chunk: Callable[[dict], dict],
    now: Callable[[], float],
    start: float,
    budget: float,
    reserve: float,
    prior: float,
) -> tuple[list[dict], int | None, bool]:
    """Generate the chunks in turn while the next is expected to end, with the reserve, inside the budget.

    Returns the chunks done, the index of the chunk it stopped before for want of room (or None), and whether it
    stopped because a chunk could not be completed at any batch size."""
    done: list[dict] = []
    last: float | None = None
    for chunk in chunks:
        elapsed = now() - start
        if not may_start(elapsed, chunk_expected(last, prior), budget, reserve):
            return done, chunk["index"], False
        began = now()
        outcome = generate_chunk(chunk)
        last = now() - began
        done.append({"index": chunk["index"], **outcome, "wall_seconds": last})
        if not outcome["complete"]:
            return done, None, True
    return done, None, False


def environment_of(python: Path, flags: tuple[str, ...] = ("-I", "-B")) -> dict:
    """The distributions installed in an interpreter's environment, as name and version, and a digest of the list."""
    code = (
        "import importlib.metadata as m, json, hashlib;"
        "d=sorted((x.metadata['Name'].lower(), x.version) for x in m.distributions());"
        "print(json.dumps({'count': len(d), 'sha256': hashlib.sha256(json.dumps(d).encode()).hexdigest(),"
        "'versions': {n: v for n, v in d if n in ('torch','rfd4-proteina','tmol','atomworks','gemmi','biotite','triton')}}))"
    )
    try:
        done = subprocess.run(
            [str(python), *flags, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (OSError, IndexError, json.JSONDecodeError) as error:
        return {"error": str(error)[-300:]}


def card_of() -> dict:
    try:
        done = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {"nvidia_smi": None}
    return {"nvidia_smi": done.stdout.strip()}


def peak_memory(path: Path) -> int | None:
    if not path.exists():
        return None
    values = []
    for line in path.read_text().splitlines():
        try:
            values.append(int(line.split(",")[0]))
        except ValueError:
            continue
    return max(values) if values else None


def stage_env(drop: tuple[str, ...] = FOREIGN, **extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(extra)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, required=True)
    parser.add_argument(
        "--env-dir", type=Path, required=True, help="RFD4's virtual environment"
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="the weights' revision on the Hugging Face Hub",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--budget-s", type=float, default=46 * 60)
    parser.add_argument("--reserve-s", type=float, default=12 * 60)
    parser.add_argument("--prior-s", type=float, default=12 * 60)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--label", default="rfd4 cell")
    parser.add_argument(
        "--python-flags",
        default="-I -B",
        help="how the helper stages start the environment's interpreter",
    )
    parser.add_argument(
        "--template-sha256",
        default=None,
        help="for a test; the benchmark's digest otherwise",
    )
    parser.add_argument(
        "--generate-cmd",
        default=None,
        help="the sampler (default: rfd4-generate of the environment)",
    )
    parser.add_argument(
        "--evaluate-shell",
        default=None,
        help="the evaluation stage as a shell command, for a test",
    )
    args = parser.parse_args()

    start = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    python = args.env_dir / "bin" / "python"
    flags = tuple(shlex.split(args.python_flags))
    generate = args.generate_cmd or str(args.env_dir / "bin" / "rfd4-generate")
    root = args.stack_root.resolve()
    inputs = out / "inputs"
    gen = out / "designs"
    record: dict = {
        "label": args.label,
        "cell_seed": args.seed,
        "weights": {"repository": HF_REPOSITORY, "revision": args.revision},
        "budget_s": args.budget_s,
        "reserve_s": args.reserve_s,
        "prior_s": args.prior_s,
        "started": started_at,
        "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "stack": str(root),
        "environment": environment_of(python, flags),
        "card": card_of(),
        "stages": [],
        "stopped_before_chunk": None,
        "stopped_after_failed_chunk": False,
    }

    def save() -> None:
        record["elapsed_s"] = round(time.monotonic() - start, 1)
        (out / "cell.json").write_text(json.dumps(record, indent=1) + "\n")

    def log(message: str) -> None:
        print(f"[{time.monotonic() - start:7.1f}s] {message}", flush=True)

    def stage(name: str, **fields) -> None:
        record["stages"].append({"name": name, **fields})
        save()

    memory = out / "gpu_memory.csv"
    sampler = (
        subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-l",
                "5",
            ],
            stdout=memory.open("w"),
            stderr=subprocess.DEVNULL,
        )
        if shutil.which("nvidia-smi")
        else None
    )

    hf_env = stage_env(
        PATH=f"{args.env_dir / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        VIRTUAL_ENV=str(args.env_dir),
        HF_HUB_CACHE=str(args.hf_cache),
        HF_HOME=str(out / "hf-home"),
        HF_HUB_OFFLINE="1",
    )
    (out / "hf-home").mkdir(exist_ok=True)

    def run_sampler(
        batch_size: int, inputs_dir: Path, out_dir: Path, cwd: Path, seed: int
    ) -> tuple[int, float]:
        command = [
            generate,
            f"hf_ckpt_path={HF_REPOSITORY}",
            f"hf_ckpt_revision={args.revision}",
            "~metrics",
            "compile_lengths=null",
            f"inputs={inputs_dir}",
            "num_replicates=1",
            f"batch_size={batch_size}",
            f"seed={seed}",
            f"out_dir={out_dir}",
        ]
        log(
            f"sampler: batch size {batch_size}, seed {seed}, {inputs_dir.name} -> {out_dir}"
        )
        began = time.monotonic()
        with (logs / f"{cwd.parent.name}-{cwd.name}.log").open("w") as handle:
            code = subprocess.run(
                command,
                cwd=cwd,
                env=hf_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        seconds = time.monotonic() - began
        log(f"sampler: exit {code} after {seconds:.0f} s")
        return code, seconds

    try:
        # 1. the inputs
        done = subprocess.run(
            [
                str(python),
                *flags,
                str(root / "src/foldspace/evaluation/rfd4_inputs.py"),
                "--template",
                str(args.template),
                "--panel",
                str(args.panel),
                "--seed",
                str(args.seed),
                "--chunks",
                str(args.chunks),
                "--out",
                str(inputs),
                *(
                    ["--template-sha256", args.template_sha256]
                    if args.template_sha256
                    else []
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        log(done.stdout.strip() or done.stderr.strip())
        if done.returncode != 0:
            stage("inputs", returncode=done.returncode, error=done.stderr[-500:])
            return 2
        manifest = json.loads((inputs / "manifest.json").read_text())
        stage("inputs", returncode=0, designs=manifest["panel"]["designs"])

        # 2. the probe
        probe = manifest["probe"]
        wanted = [d["id"] for d in probe["designs"]]
        gen.mkdir(exist_ok=True)
        outcome = generate_with_ladder(
            wanted,
            inputs / probe["directory"],
            gen / "probe",
            probe["seed"],
            run_sampler,
            ladder=(1,),
        )
        stage("probe", ids=wanted, length=probe["designs"][0]["length"], **outcome)
        if not outcome["complete"]:
            log(
                "the probe did not make its design: stopping before anything else is spent"
            )
            return 3

        # 3. the chunks
        def generate_chunk(chunk: dict) -> dict:
            ids = [d["id"] for d in chunk["designs"]]
            log(f"chunk {chunk['index']}: {len(ids)} designs, seed {chunk['seed']}")
            return generate_with_ladder(
                ids,
                inputs / chunk["directory"],
                gen / f"chunk{chunk['index']}",
                chunk["seed"],
                run_sampler,
            )

        done_chunks, stopped_before, failed = run_chunks(
            manifest["chunks"],
            generate_chunk,
            time.monotonic,
            start,
            args.budget_s,
            args.reserve_s,
            args.prior_s,
        )
        record["stopped_before_chunk"] = stopped_before
        record["stopped_after_failed_chunk"] = failed
        for chunk in done_chunks:
            stage(
                f"chunk{chunk['index']}",
                **{k: v for k, v in chunk.items() if k != "index"},
            )
        if stopped_before is not None:
            log(
                f"stopped before chunk {stopped_before}: it is not expected to end with room for the evaluation"
            )
        made = designs_in(gen)
        record["designs_generated"] = len(made)
        save()
        if not done_chunks:
            log(
                "no chunk was made: only the probe's design exists, and there is nothing to evaluate"
            )
            return 6

        # 4. the records
        records = out / "records.json"
        done = subprocess.run(
            [
                str(python),
                *flags,
                str(root / "src/foldspace/evaluation/rfd4_records.py"),
                "--designs",
                str(gen),
                "--out-records",
                str(records),
                "--out-report",
                str(out / "validation.json"),
                "--drop-suffix",
                "_rep0",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        log(done.stdout.strip() or done.stderr.strip())
        stage("convert", returncode=done.returncode)
        if done.returncode != 0:
            return 4

        # 5. the evaluator
        if args.evaluate_shell:
            command = ["bash", "-c", args.evaluate_shell]
        else:
            command = [
                "bash",
                "-c",
                (
                    f"cd '{root}' && source ./env.sh >/dev/null 2>&1 && python src/foldspace/evaluation/refold.py "
                    f"--records '{records}' --label '{args.label}' --work '{out / 'fold'}' "
                    f"--out '{out / 'evaluation.json'}'"
                ),
            ]
        began = time.monotonic()
        with (logs / "evaluate.log").open("w") as handle:
            code = subprocess.run(
                command,
                env=stage_env(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        stage("evaluate", returncode=code, seconds=round(time.monotonic() - began, 1))
        log(f"evaluation: exit {code}")
        return 0 if code == 0 else 5
    finally:
        if sampler is not None:
            sampler.terminate()
            sampler.wait()
        record["peak_memory_mib"] = peak_memory(memory)
        save()


if __name__ == "__main__":
    sys.exit(main())
