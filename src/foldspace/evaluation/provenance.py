"""What a number depended on besides the data.

Split out because the honest place to record it is the process that *computed*
the number, not the one that summed the files afterwards. An aggregator running
on a login node reports `cuda_available: false` and no GPU, which describes the
summation and misdescribes the measurement.

The numerical stack is in here because this project has twice been caught by
it: a fold that disagreed with itself at a fixed seed until deterministic
kernels were requested, and a cluster whose driver silently sent a
6-billion-parameter forward to the CPU while the job reported healthy. ESMC's
fallback state is recorded for the same reason -- it warns that without a fused
kernel the unnormalized residual stream differs by ~O(100) in bf16.
"""

from __future__ import annotations

import re
from pathlib import Path

#: A Git-LFS blob name: SHA-256, lower-case hex.
_SHA256_NAME = re.compile(r"[0-9a-f]{64}")

__all__ = [
    "DirtyTree",
    "capture",
    "checkpoint_identity",
    "require_clean",
    "verify_checkpoint",
]


def _run(repo: Path, *args: str) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return out.stdout if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 -- provenance must not fail a run
        return None


#: The trees a forward actually runs code from. A clean `foldspace` says
#: nothing about the three upstreams the model is executed by, and a patch in
#: any of them changes the numbers with no trace in the record.
SOURCE_TREES = ("foldspace", "esmfold2-atomworks", "esm", "atomworks")

#: Untracked paths that do **not** make a tree unreproducible, by directory
#: rather than by file type. Everything else untracked blocks.
#:
#: A suffix allowlist was the first attempt and it guesses: `.json`, `.yaml`,
#: a template or a resource file is not importable and is read by runtime code
#: all the same, so a tree carrying an untracked config would have reported
#: itself reproducible. Naming the directories that hold build artifacts says
#: what is known instead of inferring what Python can load.
#:
#: Keyed by tree. `esm/wheels/` holds prebuilt flash-attn wheels that nothing
#: imports through; an entry here is a claim that the directory cannot affect
#: a result, and it has to be argued when added.
ARTIFACT_PATHS: dict[str, tuple[str, ...]] = {
    "esm": ("wheels/",),
}


def _git(repo: Path) -> dict:
    """A commit is not an identity while the tree differs from it.

    A run recorded `foldspace = 22eadf2` while the working tree carried the
    uncommitted changes that produced it, so the field named a tree that had
    never computed anything. The commit alone cannot catch that; the diff can,
    and hashing it keeps the record small while still distinguishing two
    different dirty states of one commit.

    **Untracked files count too.** A new module under `src/` shadows nothing in
    `git diff` and changes what imports resolve to, so a tree with untracked
    Python in it is not reproducible from its commit either. They are named,
    not counted: a count cannot say whether the file was a scratch note or a
    module on the import path.

    Everything untracked blocks except paths under a directory declared in
    `ARTIFACT_PATHS`, and everything untracked is recorded either way. The
    exemption is by directory, not by file type: a suffix allowlist has to
    guess, and would pass a tree carrying an untracked `.json` that runtime
    code reads. `esm/wheels/` is exempt because nothing imports through it,
    which is a claim about that directory and is written down as one.
    """
    head = _run(repo, "rev-parse", "HEAD")
    if head is None:
        return {"head": None, "clean": None}
    diff = _run(repo, "diff", "HEAD") or ""
    untracked = _run(repo, "ls-files", "--others", "--exclude-standard") or ""
    names = [line for line in untracked.strip().splitlines() if line]
    exempt = ARTIFACT_PATHS.get(repo.name, ())
    blocking = [n for n in names if not any(n.startswith(e) for e in exempt)]
    out = {
        "head": head.strip(),
        "dirty": bool(diff.strip()),
        "clean": not diff.strip() and not blocking,
    }
    if diff.strip():
        import hashlib

        out["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()[:16]
        out["diff_lines"] = diff.count("\n")
    if names:
        out["untracked"] = sorted(names)[:50]
        out["untracked_count"] = len(names)
    if blocking:
        out["untracked_blocking"] = sorted(blocking)[:50]
    return out


class DirtyTree(RuntimeError):
    """A run that would not be identifiable from its own record."""


def _trees() -> dict[str, Path]:
    here = Path(__file__).resolve().parents[2]
    root = here.parent
    return {
        "foldspace": here,
        "esmfold2_atomworks": root / "esmfold2-atomworks",
        "esm": root / "esm",
        "atomworks": root / "atomworks",
    }


def require_clean(*, allow: bool = False) -> None:
    """Refuse to produce a kept number from a stack nothing can name.

    Clean means **no tracked diff and no untracked files, in all four source
    trees the forward runs code from** -- not just this one. A patch in `esm`
    or `atomworks` changes the numbers as surely as one here and, checked only
    against `foldspace`, would leave no trace at all.

    `allow=True` is the development escape and writes `reproducible_tree:
    false` into the record rather than passing silently. A run carrying that
    flag is not admissible as a primary result; it is for getting something
    working, and the record says which it was.

    The failure guarded against is not a wrong number. It is a right number
    that cannot be traced to the code that made it, which is cheap to prevent
    here and expensive to discover after a training campaign.
    """
    unclean = {}
    for name, repo in _trees().items():
        state = _git(repo)
        # `is not True`, not `is False`. `_git` returns `clean: None` when it
        # cannot read git metadata at all -- a tree copied without `.git`, a
        # wrong path, a failed invocation -- and `is False` let every one of
        # those through while `capture` went on to record
        # `reproducible_tree: false`. The guard and the record disagreeing is
        # the failure, and a staged tree on a compute node is exactly where it
        # would have happened.
        if state.get("clean") is not True:
            unclean[name] = state
    if not unclean or allow:
        return
    lines = []
    for name, state in unclean.items():
        head = (state.get("head") or "?")[:8]
        bits = []
        if state.get("dirty"):
            bits.append(
                f"diff {state.get('diff_sha256')} ({state.get('diff_lines')} lines)"
            )
        if state.get("untracked_blocking"):
            shown = ", ".join(state["untracked_blocking"][:3])
            bits.append(f"{len(state['untracked_blocking'])} untracked ({shown})")
        if state.get("clean") is None:
            bits.append("git metadata unreadable, so nothing can be checked")
        if not bits:
            bits.append("unclean for a reason this message failed to name")
        lines.append(f"  {name} at {head}: {'; '.join(bits)}")
    raise DirtyTree(
        "these source trees differ from their commits, so a record of this run "
        "could not name the code that produced it:\n"
        + "\n".join(lines)
        + "\nCommit them, or pass the caller's --allow-dirty and accept that "
        "the run is marked not reproducible and is not a primary result."
    )


def capture(extra: dict | None = None) -> dict:
    """Commits, host, numerical stack and kernel fallbacks, as a flat dict."""
    import os
    import platform

    out: dict = {
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
    }
    reproducible = True
    for name, repo in _trees().items():
        state = _git(repo)
        out[name] = state.get("head")
        if state.get("dirty"):
            out[f"{name}_dirty"] = state.get("diff_sha256")
            out[f"{name}_dirty_lines"] = state.get("diff_lines")
        if state.get("untracked_count"):
            out[f"{name}_untracked"] = state["untracked"]
        if state.get("clean") is not True:
            reproducible = False
    # One field a reader can filter on without knowing which trees exist.
    out["reproducible_tree"] = reproducible
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = getattr(torch.version, "cuda", None)
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:  # noqa: BLE001
        out["torch"] = None
    # Which fused kernels were available. ESMC falls back loudly and the
    # fallbacks are not numerically identical to the fused paths.
    for name in ("transformer_engine", "xformers", "flash_attn"):
        try:
            __import__(name)
            out[f"{name}_present"] = True
        except Exception:  # noqa: BLE001
            out[f"{name}_present"] = False
    if extra:
        out.update(extra)
    return out


def checkpoint_identity(weights: Path | str) -> dict:
    """What pins a checkpoint, and how strongly.

    `config.json` and the shard index decide *which tensor lives in which
    file*. They do not check the files: the same config and index over
    different shard bytes read identically, so hashing them supports "the same
    configuration and shard index" and not "the same weights".

    The weights themselves come free where the checkpoint is a Hugging Face
    cache. Its `snapshots/` entries are symlinks into `blobs/`, and a blob is
    named by the SHA-256 of its own content -- verified here against
    `sha256sum` on a 5 GB shard, which agreed exactly. So reading six symlink
    targets pins 24.5 GiB of tensors at the cost of six `readlink` calls.

    `identity_source` says which was available, because it is not always the
    strong one. A copy made with `rsync -aL` dereferences the symlinks, and a
    staged checkpoint on a compute node has real files with no blob names --
    the same weights, identified more weakly, and the record should not
    pretend otherwise.
    """
    import hashlib

    root = Path(weights)
    if not root.is_dir():
        return {"resolved": str(weights), "identity_source": "unresolved"}

    out: dict = {"resolved": str(root)}
    for name in ("config.json", "model.safetensors.index.json"):
        candidate = root / name
        if candidate.is_file():
            out[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()[:16]

    shards: dict[str, str] = {}
    sizes: dict[str, int] = {}
    dereferenced = False
    for shard in sorted(root.glob("*.safetensors")):
        sizes[shard.name] = shard.stat().st_size
        target = shard.resolve()
        if (
            shard.is_symlink()
            and target.parent.name == "blobs"
            # A blob is named by its content hash, but only Git-LFS blobs get
            # the 64-hex SHA-256; an ordinary Git blob gets a 40-hex SHA-1, and
            # a cache laid out some other way could put anything here. Calling
            # the field `hf-blob-sha256` without checking would be the same
            # class of overclaim as pinning the index and calling it the
            # weights.
            and _SHA256_NAME.fullmatch(target.name)
        ):
            shards[shard.name] = target.name
        else:
            dereferenced = True

    if shards and not dereferenced:
        out["identity_source"] = "hf-blob-sha256"
        out["shards"] = shards
        # One short field to compare two runs by, so a reader does not have to
        # diff six hashes to answer "same weights?".
        digest = hashlib.sha256(
            "".join(f"{k}:{v}" for k, v in sorted(shards.items())).encode()
        )
        out["weights_sha256"] = digest.hexdigest()[:16]
    elif sizes:
        # Sizes are weak -- they catch a truncated or swapped-length file and
        # nothing else -- and are labelled as such rather than dressed up.
        out["identity_source"] = "sizes-only"
        out["shard_sizes"] = sizes
    else:
        out["identity_source"] = "config-and-index-only"
    return out


def verify_checkpoint(weights: Path | str) -> dict:
    """Rehash the shards and check them against their blob names.

    The cheap path trusts Hugging Face's content addressing. This is what
    establishes that trust, and what would catch a blob that had been replaced
    in place. It reads 24.5 GiB, so it is a check to run when a checkpoint
    arrives, not on every job.
    """
    import hashlib

    identity = checkpoint_identity(weights)
    if identity.get("identity_source") != "hf-blob-sha256":
        return {
            "verified": False,
            "reason": f"no blob names to check against "
            f"({identity.get('identity_source')})",
        }
    root = Path(weights)
    mismatches = {}
    for name, claimed in identity["shards"].items():
        digest = hashlib.sha256()
        with (root / name).open("rb") as handle:
            while chunk := handle.read(1 << 24):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != claimed:
            mismatches[name] = {"blob_name": claimed, "sha256": actual}
    return {
        "verified": not mismatches,
        "checked": len(identity["shards"]),
        "mismatches": mismatches,
    }
