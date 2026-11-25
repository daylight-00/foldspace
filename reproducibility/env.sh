#!/usr/bin/env bash
# The environment the runs were made in: the substrate's source trees, the checkpoint and the caches on the path.
#
#   source reproducibility/env.sh
#
# foldspace has no interpreter of its own. It consumes the same stack as
# esmfold2-atomworks -- esm, atomworks, torch -- so it shares that stack's
# environment rather than duplicating ~10 GB of torch. FS_VENV overrides.

FS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FS_ROOT

# ---------------------------------------------------------------- source trees
# DESIGN_ROOT is DISCOVERED, never hardcoded: walk up until a directory holds
# the upstream trees. Correct from the repo, from a git worktree (several levels
# deeper), and after the tree is reorganized again.
if [ -z "${DESIGN_ROOT:-}" ]; then
    _fs_dir="${FS_ROOT}"
    while [ "${_fs_dir}" != "/" ]; do
        if [ -d "${_fs_dir}/esm" ] && [ -d "${_fs_dir}/atomworks" ]; then
            DESIGN_ROOT="${_fs_dir}"
            break
        fi
        _fs_dir="$(dirname "${_fs_dir}")"
    done
    DESIGN_ROOT="${DESIGN_ROOT:-$(cd "${FS_ROOT}/.." && pwd)}"
    unset _fs_dir
fi
export DESIGN_ROOT

# The substrate is consumed from its working tree, so that a fix landed in
# esmfold2-atomworks is visible here immediately -- and so that a probe result
# can name the substrate commit that produced it (see reproducibility/).
export EF2AW_ROOT="${EF2AW_ROOT:-${DESIGN_ROOT}/esmfold2-atomworks}"

export PYTHONPATH="${DESIGN_ROOT}/atomworks/src:${PYTHONPATH:-}"
export PYTHONPATH="${DESIGN_ROOT}/esm:${PYTHONPATH:-}"
export PYTHONPATH="${EF2AW_ROOT}/src:${PYTHONPATH:-}"
export PYTHONPATH="${FS_ROOT}/src:${PYTHONPATH:-}"

# ------------------------------------------------------------------- artifacts
export EF_MODELS="${EF_MODELS:-${DESIGN_ROOT}/biohub}"
export FS_RUNS="${FS_RUNS:-${FS_ROOT}/runs}"
# ESMC hidden states are deterministic given a sequence and cost a 6B forward
# pass. They are cached, keyed by sequence + checkpoint.
export FS_CACHE="${FS_CACHE:-${FS_ROOT}/cache}"

# The CCD pickle ESMFold2 builds ligand and modified-residue conformers from.
# esm loads it once per process -- from ESMCFOLD_CCD_PATH when that is set,
# otherwise wherever the first caller says, and its own lazy lookups say
# nothing, which downloads the Hub's latest copy into ~/.cache. Point it at the
# mirror beside the weights so every call agrees.
if [ -z "${ESMCFOLD_CCD_PATH:-}" ] && [ -f "${EF_MODELS}/ESMFold2/ccd.pkl" ]; then
    export ESMCFOLD_CCD_PATH="${EF_MODELS}/ESMFold2/ccd.pkl"
fi

# ---------------------------------------------------------------- thread policy
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# ---------------------------------------------------------------------- venv
# FS_VENV wins; otherwise the shared stack environment, if it exists. If
# neither does, the current interpreter is left alone.
if [ -n "${FS_VENV:-}" ] && [ -f "${FS_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${FS_VENV}/bin/activate"
fi

# ------------------------------------------------------------- site overlay
# A site whose shared venv is missing packages gets them here rather than by
# installing into a venv another project owns. `cache/` is not tracked, so each
# site carries its own and nothing about one cluster leaks into the repository.
#
# **Appended**, never prepended -- but appending is not enough, and the reason
# is worth stating because it has already bitten. Every PYTHONPATH entry
# precedes the venv's site-packages in `sys.path`, so ordering *within*
# PYTHONPATH does not decide anything against the venv: a package present in
# both wins from the overlay whatever order this line uses.
#
# `uv pip install --target` installs dependencies too, so asking the overlay
# for one missing package can quietly drop a second copy of numpy or scipy
# beside it, and that copy then decides results. **After installing into the
# overlay, delete whatever the venv already provides** -- `scripts/check.sh`
# does not catch this, and the symptom is a version that moves under a
# reproducibility check rather than an error.
FS_PYOVERLAY="${FS_PYOVERLAY:-${FS_ROOT}/cache/pyoverlay}"
if [ -d "${FS_PYOVERLAY}" ]; then
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${FS_PYOVERLAY}"
fi

# ------------------------------------------------------------ torch override
# The one case that has to win over the venv, and therefore the one thing here
# that is **prepended**.
#
# The shared venv carries `torch 2.14.0+cu132`, which needs an r580 driver.
# lcdd's nodes are all on 570.x, which tops out at CUDA 12.8, so that build
# cannot initialise CUDA there at all -- it prints "the NVIDIA driver on your
# system is too old", returns no devices, and every job silently runs the 6B
# ESMC pass on CPU. Appending a working torch would not help, because the
# venv's would still be found first.
#
# So a site whose driver cannot run the venv's torch puts a compatible build
# in `cache/torch-site` and it takes precedence. This is deliberately a
# separate directory from the overlay: one is "packages the venv lacks", this
# is "a package the venv has and this machine cannot use", and conflating them
# would let an ordinary missing-package install start shadowing the numerical
# stack. Runs that use it are not numerically identical to runs that do not,
# and a result that turns on the difference has to say so.
FS_TORCH_SITE="${FS_TORCH_SITE:-${FS_ROOT}/cache/torch-site}"
if [ -d "${FS_TORCH_SITE}" ]; then
    export PYTHONPATH="${FS_TORCH_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# Checkout-owned entrypoint; the shared interpreter is not modified.
export PATH="${FS_ROOT}/bin:${PATH}"
