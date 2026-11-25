"""ESMC's last hidden state for every residue of the corpus, once, as one array: the first GPU step of the pipeline.

The corpus is `data/corpus/chains.npz` (not part of this release): the residues of all chains concatenated, chain `i` the slice
`[starts[i], starts[i] + lengths[i])`, `tokens` in AlphaFold's alphabet. The result is `esmc_last.f16.npy`, `[total, 2560]` in half
precision, with the same rows: the **post-final-norm** state that `src/foldspace/rfd4/esmc_features.py` gives (the tensor ESMC's own masked-LM
head reads), so that a chain's features are the same slice of it as its coordinates are of `xyz`. Nothing is computed twice and the
6B model never runs inside a later training step.

Chains are dealt into batches by length, longest first, each batch at most `--max-tokens` residues of padded width and at most
`--max-chains` chains, so a batch pads little and one forward serves many chains. A batch is written when it is computed.

`progress.json` names the batches done **and what they were done for**: the digest of the plan (the chain indices of every batch, in
order), the digest of the corpus (its tokens and chain boundaries) and the settings that decide a row's value (`max_tokens`,
`max_chains`, the state, the precision of the file). A job that stops at its budget resumes only a record that agrees with all of
them: a count of batches alone does not say whether the next batch is the one the array was written for, since another batching of
the same chains can have as many batches. A record of the first form, which counted batches only, is accepted when the run it counts is
complete (nothing then remains to be computed under the wrong plan) and `meta.json` names the same corpus and settings.

The array is written by `seek` and buffered `write` at each chain's byte range (`RowFile`), not by assignment into a memmap: on GPFS
(abc) a large array built through a memmap is written at about a hundredth of the rate of a buffered write (14 against 1,839 MiB/s).
When every batch is done, `--check` chains chosen by a fixed seed are folded again alone and compared with what was written, as the
**per-vector** relative L2 distance and cosine (a mean or a maximum of absolute differences says nothing about the size of an error
without the size of the vectors); `meta.json` records the comparison, the source and the settings. `--check-only` repeats the
comparison on a finished array (a larger sample, or a second look) without computing anything else and writes `check.json`. A
non-finite state, or one that half precision cannot hold, stops the job: nothing is clipped.

    python src/foldspace/rfd4/esmc_corpus_features.py --chains data/corpus/chains.npz --out-dir cache/rfd4esmc [--budget-s 3000]
    python src/foldspace/rfd4/esmc_corpus_features.py --chains data/corpus/chains.npz --out-dir cache/rfd4esmc --check-only --check 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

import numpy as np

HERE = Path(__file__).resolve().parent
DIM = 2560
#: what half precision holds without overflow; a state beyond it is an error, not something to clip
HALF_MAX = 6.0e4
CHECK_SEED = 20261111
#: what a row of the array is: part of the contract a record of progress is continued under
STATE = "ESMC-6B last_hidden_state, post-final-norm, bf16 forward"


def plan_batches(lengths, max_tokens: int, max_chains: int) -> list[list[int]]:
    """Chains dealt into batches, longest first, ties by index: each batch is at most `max_chains` chains and `width * n <= max_tokens`.

    The width of a batch is its first (longest) chain, so the padded size is exactly `width * n`; a chain longer than `max_tokens`
    is a batch of its own. Every chain is in exactly one batch and the plan is a function of the lengths only.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for i in order:
        if current and (
            len(current) >= max_chains or width * (len(current) + 1) > max_tokens
        ):
            batches.append(current)
            current = []
        if not current:
            width = int(lengths[i])
        current.append(i)
    if current:
        batches.append(current)
    return batches


def batch_arrays(tokens, starts, lengths, idx) -> tuple[np.ndarray, np.ndarray]:
    """`[B, W]` tokens and mask of the chains `idx`, right-padded with 0 to the longest; the mask is 1 on residues."""
    lengths = np.asarray(lengths)
    width = int(max(lengths[i] for i in idx))
    tok = np.zeros((len(idx), width), dtype=np.int64)
    mask = np.zeros((len(idx), width), dtype=np.int64)
    for row, i in enumerate(idx):
        n = int(lengths[i])
        tok[row, :n] = tokens[int(starts[i]) : int(starts[i]) + n]
        mask[row, :n] = 1
    return tok, mask


def to_half(states: np.ndarray) -> np.ndarray:
    """The states in half precision, or an error if one is not finite or beyond what half precision holds."""
    states = np.asarray(states, dtype=np.float32)
    if not np.isfinite(states).all():
        raise ValueError("a state is not finite")
    peak = float(np.abs(states).max()) if states.size else 0.0
    if peak > HALF_MAX:
        raise ValueError(f"a state of {peak:g} does not fit half precision")
    return states.astype(np.float16)


def create_npy(path: Path, total: int) -> None:
    """A `[total, DIM]` half-precision `.npy` of its full size with its header and no row written (the file is sparse until filled)."""
    header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
        "fortran_order": False,
        "shape": (total, DIM),
    }
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, header)
        f.truncate(f.tell() + total * DIM * np.dtype(np.float16).itemsize)


def npy_offset(path: Path, total: int) -> int:
    """The byte offset of the rows of an existing array, after checking that it is the `[total, DIM]` half-precision array of this corpus.

    A file shorter than its header says is refused by the read-only map itself.
    """
    view = np.load(path, mmap_mode="r")
    try:
        if (
            view.shape != (total, DIM)
            or view.dtype != np.float16
            or not view.flags.c_contiguous
        ):
            raise ValueError(
                f"{path} is not the array of this corpus: {view.shape} {view.dtype}"
            )
        return int(view.offset)
    finally:
        del view


class RowFile:
    """The rows of the array as a file, written by `seek` and buffered `write`: `rows[a:b] = block` puts `[b - a, DIM]` half-precision
    values at the byte range of rows `a` to `b`, and `flush` makes them durable (`fsync`) before a record of progress names them.

    Reading is `np.load(path, mmap_mode="r")`: only the writing is slow through a memmap on GPFS.
    """

    ITEM = np.dtype(np.float16).itemsize

    def __init__(self, path: Path, total: int, offset: int):
        self.path, self.total, self.offset = Path(path), int(total), int(offset)
        # held for the life of the object and closed by `close` (a `with` on the object does that)
        self.file = open(self.path, "r+b")  # noqa: SIM115

    @classmethod
    def create(cls, path: Path, total: int) -> RowFile:
        create_npy(path, total)
        return cls.open(path, total)

    @classmethod
    def open(cls, path: Path, total: int) -> RowFile:
        return cls(path, total, npy_offset(path, total))

    def __setitem__(self, key, rows) -> None:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("rows are written as a contiguous slice: rows[a:b] = block")
        start, stop, _ = key.indices(self.total)
        block = np.ascontiguousarray(rows, dtype=np.float16)
        if block.shape != (stop - start, DIM):
            raise ValueError(
                f"rows {start}:{stop} take {(stop - start, DIM)}, not {block.shape}"
            )
        self.file.seek(self.offset + start * DIM * self.ITEM)
        self.file.write(block.tobytes())

    def flush(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self) -> None:
        if not self.file.closed:
            self.flush()
            self.file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_batch(out, starts, lengths, idx, states: np.ndarray) -> None:
    """Rows of each chain of the batch (the first `lengths[i]` rows of its padded state) into `out` at the chain's slice."""
    half = to_half(states)
    for row, i in enumerate(idx):
        n = int(lengths[i])
        out[int(starts[i]) : int(starts[i]) + n] = half[row, :n]


def check_layout(starts, lengths, total: int) -> None:
    """The chains partition `[0, total)` in order: a slice is a chain, and the array has no gap and no overlap."""
    starts = np.asarray(starts)
    lengths = np.asarray(lengths)
    if starts.size and (
        int(starts[0]) != 0
        or not bool((starts[1:] == starts[:-1] + lengths[:-1]).all())
    ):
        raise ValueError("the chains are not contiguous in the order of `starts`")
    if int(lengths.sum()) != total:
        raise ValueError("the lengths do not add up to the number of residues")


def digest_of(tokens, lengths) -> str:
    """A hash of the sequences and their boundaries, so that the array says which corpus it belongs to."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tokens, dtype=np.int8).tobytes())
    h.update(np.ascontiguousarray(lengths, dtype=np.int64).tobytes())
    return h.hexdigest()


def plan_digest(plan: list[list[int]]) -> str:
    """A hash of the plan: the chain indices of every batch in order, each batch led by its size so that where one ends is part of it."""
    h = hashlib.sha256()
    for batch in plan:
        h.update(np.int64(len(batch)).tobytes())
        h.update(np.asarray(batch, dtype=np.int64).tobytes())
    return h.hexdigest()


def make_contract(plan: list[list[int]], corpus_sha256: str, settings: dict) -> dict:
    """What a record of progress has to agree with to be continued: the plan, the corpus and the settings that decide a row's value."""
    return {
        "n_batches": len(plan),
        "plan_sha256": plan_digest(plan),
        "corpus_sha256": corpus_sha256,
        "settings": dict(settings),
    }


def settings_of(max_tokens: int, max_chains: int) -> dict:
    return {
        "max_tokens": int(max_tokens),
        "max_chains": int(max_chains),
        "state": STATE,
        "dtype": "float16",
        "dim": DIM,
    }


def meta_confirms(meta: dict | None, contract: dict) -> bool:
    """Whether `meta.json`, written when a run ended, says that run was complete and made for this corpus and these settings."""
    if not meta or meta.get("complete") is not True:
        return False
    settings = contract["settings"]
    return (
        meta.get("corpus_sha256") == contract["corpus_sha256"]
        and meta.get("max_tokens") == settings["max_tokens"]
        and meta.get("max_chains") == settings["max_chains"]
        and meta.get("state") == settings["state"]
    )


def read_progress(path: Path, contract: dict, meta: dict | None = None) -> int:
    """How many leading batches of the plan are done; 0 if there is no record. A record made for another plan, corpus or settings is an error.

    A record of the first form (batches counted, nothing else) cannot say which plan its batches were of, so it is accepted only when
    it counts every batch of this plan and `meta` confirms the corpus and the settings of that finished run.
    """
    if not path.exists():
        return 0
    record = json.loads(path.read_text())
    done = int(record["done"])
    if "plan_sha256" not in record:
        n = int(record["n_batches"])
        if n != contract["n_batches"] or done != n:
            raise ValueError(
                "progress.json counts batches only and the run is not complete: it cannot say which plan the rows belong to; "
                "remove it and the array"
            )
        if not meta_confirms(meta, contract):
            raise ValueError(
                "progress.json counts batches only and meta.json does not confirm a complete run of this corpus and these settings"
            )
        return done
    for key, what in (
        ("n_batches", "plan"),
        ("plan_sha256", "plan"),
        ("corpus_sha256", "corpus"),
        ("settings", "settings"),
    ):
        if record.get(key) != contract[key]:
            raise ValueError(f"progress.json belongs to another {what} ({key})")
    if not 0 <= done <= contract["n_batches"]:
        raise ValueError(
            f"progress.json names {done} batches of {contract['n_batches']}"
        )
    return done


def write_progress(path: Path, done: int, contract: dict) -> None:
    """The record is written whole and renamed into place, so a job killed in the middle leaves the last complete one."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 2, "done": done, **contract}))
    os.replace(tmp, path)


def extract(
    features: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tokens,
    starts,
    lengths,
    out,
    plan: list[list[int]],
    progress_path: Path,
    contract: dict,
    *,
    meta: dict | None = None,
    budget_s: float | None = None,
    flush_every: int = 20,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> int:
    """Runs the batches of `plan` that are not done, until the budget ends; returns how many batches are done.

    `features(tok, mask)` returns `[B, W, DIM]`; a batch is written as soon as it is computed, and the record of progress follows a
    flush of `out`, so what the record names is on disk. No batch is started once `budget_s` seconds have passed. The record is
    continued only under `contract` (`make_contract`), and `meta` is `meta.json` of an earlier run if there is one.
    """
    done = read_progress(progress_path, contract, meta)
    began = clock()
    since_flush = 0
    for b in range(done, len(plan)):
        if budget_s is not None and clock() - began >= budget_s:
            log(f"budget reached before batch {b} of {len(plan)}")
            break
        tok, mask = batch_arrays(tokens, starts, lengths, plan[b])
        write_batch(out, starts, lengths, plan[b], features(tok, mask))
        done = b + 1
        since_flush += 1
        if since_flush >= flush_every:
            out.flush()
            write_progress(progress_path, done, contract)
            since_flush = 0
            log(f"{done}/{len(plan)} batches, {clock() - began:.0f} s")
    out.flush()
    write_progress(progress_path, done, contract)
    return done


def agreement(again: np.ndarray, stored: np.ndarray) -> dict[str, np.ndarray]:
    """Per residue vector: the relative L2 distance `||a - s|| / ||s||`, the cosine of the angle between `a` and `s`, and `||s||`.

    A stored vector of norm zero (a row that was never written) has no relative distance and no angle: both are NaN there, and
    `self_check` counts such rows by name instead of letting them into a quantile.
    """
    a = np.asarray(again, dtype=np.float64)
    s = np.asarray(stored, dtype=np.float64)
    s_norm = np.linalg.norm(s, axis=-1)
    a_norm = np.linalg.norm(a, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.linalg.norm(a - s, axis=-1) / s_norm
        cos = (a * s).sum(-1) / (a_norm * s_norm)
    return {
        "rel_l2": np.where(s_norm > 0, rel, np.nan),
        "cosine": np.where((s_norm > 0) & (a_norm > 0), cos, np.nan),
        "stored_norm": s_norm,
    }


def summarize(
    values: np.ndarray, quantiles: dict[str, float]
) -> dict[str, float | None]:
    """The quantiles of `values` by name (`None` for each if there is no value)."""
    if values.size == 0:
        return dict.fromkeys(quantiles)
    return {name: float(np.quantile(values, q)) for name, q in quantiles.items()}


def self_check(
    features: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tokens,
    starts,
    lengths,
    out,
    count: int,
) -> dict:
    """`count` chains chosen by a fixed seed, folded again alone and compared with the stored rows.

    The comparison is per residue vector (`agreement`): the relative L2 distance and the cosine, over every residue of the chosen
    chains, with the norm of the stored vectors beside them; the absolute differences are kept for continuity. Two foldings of one
    chain differ in bf16 (another padding and batch composition changes the order of the sums), so what is read is how large that
    is next to the vector, not that it is not zero.
    """
    if count <= 0:
        return {"chains": 0}
    rng = np.random.default_rng(CHECK_SEED)
    chosen = sorted(
        int(i)
        for i in rng.choice(len(lengths), size=min(count, len(lengths)), replace=False)
    )
    worst = 0.0
    mean_abs = []
    rel, cos, norm = [], [], []
    for i in chosen:
        tok, mask = batch_arrays(tokens, starts, lengths, [i])
        again = to_half(features(tok, mask))[0, : int(lengths[i])].astype(np.float32)
        stored = np.asarray(
            out[int(starts[i]) : int(starts[i]) + int(lengths[i])], dtype=np.float32
        )
        diff = np.abs(again - stored)
        worst = max(worst, float(diff.max()))
        mean_abs.append(float(diff.mean()))
        per_vector = agreement(again, stored)
        rel.append(per_vector["rel_l2"])
        cos.append(per_vector["cosine"])
        norm.append(per_vector["stored_norm"])
    rel_all, cos_all, norm_all = (np.concatenate(x) for x in (rel, cos, norm))
    return {
        "chains": len(chosen),
        "residues": int(rel_all.size),
        "max_abs_diff": worst,
        "mean_abs_diff": float(np.mean(mean_abs)),
        "rel_l2": summarize(
            rel_all[np.isfinite(rel_all)],
            {"median": 0.5, "p95": 0.95, "p99": 0.99, "max": 1.0},
        ),
        "cosine": summarize(
            cos_all[np.isfinite(cos_all)], {"min": 0.0, "p01": 0.01, "median": 0.5}
        ),
        "stored_norm": summarize(norm_all, {"min": 0.0, "median": 0.5, "max": 1.0}),
        "zero_rows": int((norm_all == 0).sum()),
        "indices": chosen,
    }


def run_check(
    out_dir: Path,
    tokens,
    starts,
    lengths,
    features: Callable[[np.ndarray, np.ndarray], np.ndarray],
    count: int,
    *,
    stack_commit: str | None = None,
) -> dict:
    """The comparison of `self_check` on a finished array of `out_dir`, written to `check.json`.

    The array must be the complete one `meta.json` describes, for this corpus and this state; otherwise it is an error, because a
    comparison against rows that were never meant for these chains says nothing.
    """
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"{meta_path} is missing: there is no finished array to check")
    meta = json.loads(meta_path.read_text())
    total = int(np.asarray(lengths).sum())
    if meta.get("complete") is not True:
        raise ValueError("meta.json does not describe a complete run")
    if meta.get("corpus_sha256") != digest_of(tokens, lengths):
        raise ValueError("the array is of another corpus than the one given")
    if meta.get("state") != STATE:
        raise ValueError(f"the array holds another state: {meta.get('state')!r}")
    path = out_dir / meta["array"]
    npy_offset(path, total)
    out = np.load(path, mmap_mode="r")
    result = {
        "array": path.name,
        "corpus_sha256": meta["corpus_sha256"],
        "state": meta["state"],
        "stack_commit": stack_commit,
        "check": self_check(features, tokens, starts, lengths, out, count),
    }
    (out_dir / "check.json").write_text(json.dumps(result, indent=2))
    return result


def model_features(device: str):
    """`features(tok, mask)` over ESMC-6B in bf16: the glue of the build (`src/foldspace/rfd4/esmc_features.py`), NumPy in and out."""
    import torch

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    from foldspace.rfd4 import esmc_features

    esmc, directory = esmc_features.load_esmc(device)
    glue = esmc_features.EsmcFeatures(esmc, esmc_features.residue_table())

    def features(tok: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            states = glue(
                torch.as_tensor(tok, device=device),
                torch.as_tensor(mask, device=device),
            )
        return states.float().cpu().numpy()

    return features, directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--chains", required=True, help="the corpus array (data/corpus/chains.npz)"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--max-chains", type=int, default=32)
    parser.add_argument(
        "--budget-s",
        type=float,
        default=None,
        help="start no batch after this many seconds",
    )
    parser.add_argument(
        "--check", type=int, default=8, help="chains folded again alone at the end"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="compare a finished array with chains folded again alone (writes check.json) and compute nothing else",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only the first N chains of the corpus (a smoke run)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    began = time.monotonic()
    z = np.load(args.chains, allow_pickle=False)
    tokens, starts, lengths = z["tokens"], z["starts"], z["lengths"]
    if args.limit:
        keep = args.limit
        total = int(starts[keep - 1] + lengths[keep - 1])
        tokens, starts, lengths = tokens[:total], starts[:keep], lengths[:keep]
    total = int(lengths.sum())
    check_layout(starts, lengths, total)

    out_dir = Path(args.out_dir)
    stack_commit = os.environ.get("FS_STACK_COMMIT")
    if args.check_only:
        features, directory = model_features(args.device)
        result = run_check(
            out_dir,
            tokens,
            starts,
            lengths,
            features,
            args.check,
            stack_commit=stack_commit,
        )
        result["esmc_directory"] = str(directory)
        result["elapsed_seconds"] = round(time.monotonic() - began, 1)
        (out_dir / "check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "esmc_last.f16.npy"
    plan = plan_batches(lengths, args.max_tokens, args.max_chains)
    settings = settings_of(args.max_tokens, args.max_chains)
    corpus_sha256 = digest_of(tokens, lengths)
    contract = make_contract(plan, corpus_sha256, settings)
    progress = out_dir / "progress.json"
    meta_path = out_dir / "meta.json"
    earlier = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if path.exists():
        try:
            rows = RowFile.open(path, total)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        if progress.exists():
            raise SystemExit(
                "progress.json without an array: remove it, or the run is not what it says"
            )
        rows = RowFile.create(path, total)

    features, directory = model_features(args.device)
    loaded = time.monotonic() - began
    print(
        f"model loaded in {loaded:.0f} s from {directory}; {len(plan)} batches for {len(lengths)} chains, {total} residues"
    )
    try:
        done = extract(
            features,
            tokens,
            starts,
            lengths,
            rows,
            plan,
            progress,
            contract,
            meta=earlier,
            budget_s=args.budget_s,
        )
    finally:
        rows.close()
    out = np.load(path, mmap_mode="r")

    meta = {
        "array": path.name,
        "shape": [total, DIM],
        "dtype": "float16",
        "state": STATE,
        "esmc_directory": str(directory),
        "chains": str(args.chains),
        "corpus_sha256": corpus_sha256,
        "plan_sha256": contract["plan_sha256"],
        "limit": int(args.limit),
        "max_tokens": args.max_tokens,
        "max_chains": args.max_chains,
        "batches": len(plan),
        "batches_done": done,
        "complete": done == len(plan),
        "load_seconds": round(loaded, 1),
        "elapsed_seconds": round(time.monotonic() - began, 1),
        "stack_commit": stack_commit,
    }
    if done == len(plan):
        meta["peak_abs"] = float(
            np.abs(np.asarray(out[:: max(1, total // 200_000)], dtype=np.float32)).max()
        )
        meta["check"] = self_check(features, tokens, starts, lengths, out, args.check)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0 if done == len(plan) else 3


if __name__ == "__main__":
    raise SystemExit(main())
