# RFD4 training settings

For each network the RFD4 experiments trained or reused: what it started from, what was trained and what was frozen. The experiments are in [`experiments/rfd4/`](../experiments/rfd4/); the code is `src/foldspace/rfd4/`. Weights are not redistributed: RFD4-Proteina's release (autoencoder, pretraining flow checkpoint) and ESMC come from their own sources, with the revisions in [`pins/`](pins/).

## Networks

| component | starts from | adapted decoder (all latents below) | flow |
|---|---|---|---|
| ESMC 6B | pretrained | frozen; its last hidden state is computed once for every residue of the corpus and cached (`esmc_corpus_features`), never run during training | frozen; the cached rows are read |
| latent map `z = M(W LN(h))`, `W`: 2,560 to 32 | new, random | trained | frozen, `eval()` under `no_grad` (`flow_model`) |
| decoder | **a deep copy of the release autoencoder's decoder** | **every parameter trained** (`train_autoencoder`: `copy.deepcopy(ae.decoder)`, then `requires_grad_(True)` on all) | frozen |
| release encoder | pretrained | frozen; read only where channels are kept or a residual is used (`--inplace`, `--residual`, `--posterior-native`), not for `esmc_only` or `type_only` | not used |
| flow denoiser | **the release's pretraining checkpoint**, denoiser weights only | | trained; fresh optimizer, a warm-up of its own, EMA 0.99 (`train_flow`, `--init pretrained`, the default) |
| flow denoiser of `esmc_residual8_scratch` | its own initialisation | | trained for 800 updates; `--init scratch`, the no-transfer reference |

Apart from `esmc_residual8_scratch`, no flow and no decoder of the RFD4 experiments was trained from its own initialisation. The decoder and the flow of `esmc_only` both start from the release's weights; what is new is the latent map and the data. An ESMC-derived latent map trained jointly with a decoder initialised from the release's, and the pretrained flow fine-tuned to generate it, describes these experiments; "a new flow trained from scratch" does not.

## Latents

The latents are defined in [methods](../docs/methods.md); what the commands set for them:

* **`esmc_only`** has no residual channel and nothing from the release encoder. The decoder reads `z` with Gaussian noise of 0.2 during training; every score is at zero noise. The decoder of **`type_only`** is trained the same way.
* **`esmc_residual8`, `type_only_residual8`**: the residual channels are `r = M(W_r mu)` of the release encoder's posterior mean on the same chain.
* **The in-place hybrids**: the kept channels come from the frozen release encoder's samples (`--inplace --posterior-native --code-sigma 0.6`; `contextual_hybrid`: `--source esmc`; `type_shuffled_hybrid`: `--source fixed --code-bank`).
* **Reading adapters** (`train_reading_adapter`) train a copy of the release encoder cut to N, Cα, C and O, with new adapters, in front of the release decoder, which is frozen.
* The autoencoder is the release's (`--ae`). The 28 tensors that the public release changes relative to the pretrained base are in the flow's denoiser, so that every flow here starts from the base and the public release is a different flow.

## Settings

| stage | settings |
|---|---|
| reading adapters | 2,048 steps (8 epochs over 2,048 chains), batch 8, learning rates 1e-4 (pretrained) and 1e-3 (new); 256 dev chains |
| adapted decoder | 6,144 steps (24 epochs over 2,048 chains), batch 8, learning rates 1e-4 and 1e-3, latent noise 0.2; 256 dev chains; the final result is `end.complete` of each `summary.json` |
| flow adaptation, five configurations | `--steps 800 --batch 2 --accumulate 4` (8 chains per update, 4 replicates), `--crop 256`, `--warmup 100`, EMA 0.99, `--save-every 200`, the checkpoint's own learning rate (`--lr 0`), `--seed 20261111`; `--init pretrained`, `scratch` for `esmc_residual8_scratch`; three arms continued to 1,600 updates |
| flow, hybrids and `esmc_only` / `type_only` | `--batch 2 --accumulate 4 --checkpointing low --save-every 200 --workers 0 --lr 1e-5`, two legs of 800 updates, seeds 20261111 and 20261112; `esmc_only` extended to 2,400 and 3,200 updates |
| budget of a flow run | 800 updates are 25,600 example slots, 6,400 distinct presentations and 1,172,823 residues; the same chains, windows, rotations and noises at every update in every configuration |
| hybrid layouts | 14 native channels kept from the release encoder's samples, 18 code channels; `contextual_hybrid` code noise 0.6; `identity_hybrid` types at least 4.32 apart; `deterministic_hybrid` read at 1,600 updates |
| validation of the type-shuffled control | before training: held-out context R² below 0.01, type-centroid similarity within 0.02 of the contextual code's, within-type variance shares within 0.01 |
| design panels | two panels of 96 backbones from the release flow, 24 at each of 90, 130, 170 and 210 residues; the second at sampler seed 20261406 |
| type-shuffled bank | the first 1,024 training chains at the posterior mean, at least 500 donors a type |
| unconditional generation | `scale_scan`: seed 20261401, lengths 100, 140, 180, 220, 24 each, latent noise scales 1, 4, 8; `unconditional`: seed 20261402, lengths 90, 130, 170, 210, 48 each, scale 1 (and each configuration's tuned scale, 8 where it applies) |

Each run records the four axes of its budget in `summary.json`: updates, example slots and distinct presentations, residues and the sum of their squares, and GPU-hours; the step-time statistics and the digests of the checkpoints and inputs it read.

## Commands

The modules are scripts that take their arguments on the command line (`python -m foldspace.rfd4.<module> --help`); the head of each says what it reads and writes. They run from the root of this repository with this package and RFD4-Proteina's source on the path (or with this package installed, `pip install -e .`):

```
export PYTHONPATH=$PWD/src:<RFD4-Proteina>/src
# absolute paths: train_flow, sample_flow and flow_eval change into their output directory before they read the inputs
BASE_AE=<ae-ema.ckpt>; BASE_FLOW=<flow_pretrain-ema.ckpt>; CHAINS=<chains.npz>; ESMC_FEATURES=<esmc_last.f16.npy>
SHARDS=$PWD/cache/rfd4esmc/shards; AE_DIR=<adapted-autoencoder-dir>; FLOW_DIR=<flow-run-dir>

python -m foldspace.rfd4.prepare_shards --chains $CHAINS --split-table <split.tsv.gz> --esmc $ESMC_FEATURES \
    --out $SHARDS --train 2048 --dev 256
python -m foldspace.rfd4.train_reading_adapter --shards $SHARDS --ae $BASE_AE --arm esmc --out <dir>
python -m foldspace.rfd4.train_autoencoder --shards $SHARDS --ae $BASE_AE --source esmc --out $AE_DIR
python -m foldspace.rfd4.train_flow --target esmc --adapter $AE_DIR/autoencoder.pt --shards <pool> --chains $CHAINS \
    --esmc $ESMC_FEATURES --flow $BASE_FLOW --ae $BASE_AE --steps 800 --batch 2 --accumulate 4 --out $FLOW_DIR

# the trained flow is the run's checkpoint; without --checkpoint these two read the weights of --flow, the pretrained flow
python -m foldspace.rfd4.sample_flow --target esmc --adapter $AE_DIR/autoencoder.pt --flow $BASE_FLOW --ae $BASE_AE \
    --checkpoint $FLOW_DIR/ckpt/last.ckpt --expect-step 800 --out <samples-dir>
python -m foldspace.rfd4.flow_eval --target esmc --adapter $AE_DIR/autoencoder.pt --flow $BASE_FLOW --ae $BASE_AE \
    --checkpoint $FLOW_DIR/ckpt/last.ckpt --expect-step 800 --shards <pool> --chains $CHAINS --esmc $ESMC_FEATURES \
    --out <eval.json>
```

`--target` is `native` (the release latent, the continuation), `esmc` (the adapted autoencoder's deterministic latent over the chains' ESMC rows; the layout is read from the checkpoint) or `fixed` (the fixed type code in place of ESMC's). Environments and pins: [`README.md`](README.md).
