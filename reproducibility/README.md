# Reproducibility

What can be run from this repository, what it needs, and what has and has not been checked.

## Levels

| level | what | needs | checked |
|---|---|---|---|
| 0 | the offline tests: the code, and the reported numbers recomputed from the committed records (`tests/test_results_rfd4.py`, `tests/test_results_esmfold2.py`) | Python ≥ 3.12, `numpy`, `pytest` | passes from a clean copy of this tree with `numpy` alone; modules that need `torch`, `gemmi`, `hydra`, `omegaconf` or an upstream source tree skip themselves |
| 1 | the summaries of the pilot's arms, recomputed from their per-sample records with `python -m foldspace.esmfold2.seqeval summarize` | `numpy` | the recomputed `summary.json` equals the committed one (`tests/test_results_esmfold2.py`) |
| 2 | folding and evaluation: the folding substrate, a checkpoint, an AtlasFold tree, a GPU | see below | not repeated for this release |
| 3 | training: a GPU, the corpora and caches (not in this repository) | see below | not repeated for this release |

The commands of levels 2 and 3, with every setting of every network, are in [`rfd4_training.md`](rfd4_training.md) and [`esmfold2_training.md`](esmfold2_training.md). The records were made on a Slurm cluster; no checkpoint was re-evaluated and no training repeated for this release, and the package was checked by the offline suite and by installing the built wheel in a clean environment.

Levels 0 and 1 run with the command in the [README](../README.md). Of the commands of level 2 and 3, the `--help` of every command-line module (`python -m foldspace.<package>.<module> --help`) was run from a clean checkout in the environment of each architecture.

## Recompute a result on a CPU

The per-sample records of every generated set are committed, so a set's summary can be recomputed without weights or a GPU. From the root of the repository, for the re-encoded route of the pilot's `standardized` configuration:

```bash
B=experiments/esmfold2/latent_generation/results/pilot_seed0
PYTHONPATH=src uv run --no-project --python 3.12 --with numpy -- python -m foldspace.esmfold2.seqeval summarize \
    --fold-dir $B/standardized/reencoded_fold --novelty $B/reencoded_novelty.json --novelty-set standardized \
    --direct $B/standardized/direct_fold --out /tmp/summary.json
```

`/tmp/summary.json` equals `$B/standardized/reencoded_fold/summary.json` (48 sequences, no success, none credited; reference chains 0.917). The same command with the other configurations' directories reproduces their summaries; `tests/test_results_esmfold2.py` does this for every set.

## Reproduction requirements

| element | in this repository |
|---|---|
| model training code and settings | yes |
| upstream revisions | yes (`pins/`) |
| per-sample evaluation records | yes |
| training corpus, chain lists, cluster split | no (not redistributed; a chain and split manifest with input checksums is not part of this release) |
| trained checkpoints | no |
| the novelty search script | no (its parameters are in each `novelty.json`) |
| a GPU end-to-end run | not done |

## External trees and weights

Nothing below is vendored. [`env.sh`](env.sh) shows the layout it expects under `$DESIGN_ROOT` (`source reproducibility/env.sh`).

| needed by | what | revision the runs used |
|---|---|---|
| both | `esmfold2-atomworks`, `esm`, `atomworks` (the substrate) | v0.5.0 (`8deb4c2`), v3.4.3 (`fddac93`), v3.0.4 (`1a9fa25`) |
| both | the ESMFold2 checkpoint (`models--biohub--ESMFold2`; ESMC is inside it) | by digest in the records |
| sequence evaluation, refolding | AtlasFold, the independent judge | v1.0.4 (`6df9f45`) |
| architecture A | RFD4-Proteina (`rfproteina`) with its pretraining checkpoint and the public release | source `6805869`; weights `nvidia/RFD4-Proteina` at `48377e45` |
| architecture A's panels | RF3 (`rc-foundry` 0.2.0), the second judge | in the pins |

`pip install ".[esmfold2]"` installs `esmfold2-atomworks` at the pinned revision (the resolution was checked; the environment itself was not rebuilt from it). The pins are in [`pins/`](pins/2026-11-11.json). A record names the checkpoint it stands on by digest, and a conclusion about a representation is a conclusion about one checkpoint's representation.

## Paths

The author's absolute paths are replaced by placeholders (`/path/to/design`, `/path/to/home`, `/path/to/databases`, `/path/to/sandbox`, `/path/to/job-tmp`) in code, tests and records alike. A script that reads one needs it set (`FS_DESIGN_ROOT`, or the argument the script takes). Several scripts name inputs that are not part of this release, such as the training corpus (`data/corpus/`); the records name the inputs they were made from.
