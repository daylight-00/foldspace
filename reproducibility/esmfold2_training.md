# ESMFold2 side: networks and commands

What each network of the ESMFold2 experiments starts from and what is trained, then the commands that make the records of `experiments/esmfold2/`. The pins of the substrate are in [`pins/`](pins/) and the environment in [`env.sh`](env.sh). A GPU run uses the substrate at the pins of 2026-11-11: under earlier pins a request of `lm_dropout=0` is not honoured and a random mask stays on the channel the experiments perturb (`validate_interface` and `evaluate_robustness` refuse to run in that case).

## Networks

| component | starts from | trained |
|---|---|---|
| ESMC and the folding model | pretrained (ESMC is bundled in the ESMFold2 checkpoint) | no. ESMC makes the training targets (the mixed singles) and nothing else; the folding model folds the generated sequences with the generated single and judges them, and no gradient passes through it |
| sequence head | scratch (`esmfold2/decoder.py`) | `linear` and `contextual` on the cached singles of the training split (`linear`: 5,140 parameters; `contextual`: a Transformer of 4 layers and width 256, 3,246,868 parameters); `lda` is closed form |
| generator of the single | **scratch** (`esmfold2/generator.py`) | by flow matching, in the three configurations: a Transformer denoiser of 8 layers, width 512, 8 heads, feed-forward 2,048; 20,000 updates, learning rate 3e-4, 500 warm-up updates, EMA 0.999 |
| sequence-only model | scratch (`esmfold2/sequence_only.py`) | a decoder-only Transformer of 8 layers and width 640, 39.65 M parameters; two seeds, sampled at temperatures 0.8 and 1.0 |
| ProtGPT2 | the published checkpoint | no; it is run at its model card's settings (top-k 950, repetition penalty 1.2, temperature 1.0, at most 100 tokens) |

The generator of the single is the one network of the ESMFold2 experiments that is a generator trained from scratch.

## Selection and sampling

* **Heads.** A cluster is drawn uniformly, then a chain inside it. The checkpoint is chosen by cluster-weighted dev cross-entropy; the test split is scored once per model, after the head carried forward is chosen on dev; the contextual head is carried only if it beats the linear one by 0.2 points of cluster-weighted dev accuracy on every seed.
* **Interface parity** is read in deterministic mode on an RTX 6000 Ada and on a Blackwell card.
* **Generator samples.** 48 per configuration, 70 to 220 residues, at 48 lengths spread over the quantiles of the dev chains' lengths; 50 Heun steps.

## Commands

The corpus tables `data/corpus/chains.npz` and `data/corpus/train_sequences.tsv` (17,063 training chains in 3,373 clusters at 30 % identity, and the dev and test splits that share no cluster with them) are inputs that are not redistributed; neither are the checkpoints. Everything below is run from the root of this repository with `PYTHONPATH=src`, on a tree whose substrate is at the pins. `USalign` is on the path and AtlasFold's source tree is at `<design root>/atlasfold`.

**1. The singles** (a GPU with the ESMC backbone resident; about 17 sequences per second on an RTX 6000 Ada). A single is reproducible across pins and not across card types (1 to 2 %), so the singles are made on one card type.

```bash
python -m foldspace.esmfold2.sequences --chains data/corpus/chains.npz --split dev  --out cache/esmfold2/sequences/dev.tsv
python -m foldspace.esmfold2.sequences --chains data/corpus/chains.npz --split test --out cache/esmfold2/sequences/test.tsv
for shard in 0 1 2 3; do
  python -m foldspace.esmfold2.extract --sequences data/corpus/train_sequences.tsv --shard ${shard}/4 --require-all \
      --out cache/esmfold2/targets/train/z_shard${shard}.npz
done
for split in dev test; do
  python -m foldspace.esmfold2.extract --sequences cache/esmfold2/sequences/${split}.tsv --require-all \
      --out cache/esmfold2/targets/${split}/z_shard0.npz
done
python -m foldspace.esmfold2.target_audit --targets cache/esmfold2/targets/train --chains data/corpus/chains.npz \
    --out results/target_audit.json --stats-out cache/esmfold2/model_single_train_normalisation.npz
```

**2. Interface validation** (a GPU; every process sets the deterministic mode before its first CUDA call, so each is its own process).

```bash
python -m foldspace.esmfold2.validate_interface parity --deterministic --out OUT
python -m foldspace.esmfold2.validate_interface parity --out OUT
python -m foldspace.esmfold2.validate_interface contribution --deterministic --targets cache/esmfold2/targets/train \
    --stats cache/esmfold2/model_single_train_normalisation.npz --out OUT
python -m foldspace.esmfold2.validate_interface contribution --deterministic --targets cache/esmfold2/targets/train \
    --stats cache/esmfold2/model_single_train_normalisation.npz --pairs 16 --seed 20261120 --tag replicate --out OUT
python -m foldspace.esmfold2.contribution_audit --runs OUT --out OUT/audit.json \
    --targets cache/esmfold2/targets/train --stats cache/esmfold2/model_single_train_normalisation.npz   # the stored folds, read again on a CPU
```

**3. Sequence decoding** (the heads train on one card; the structure check needs ESMC resident). `DEC` is the output directory.

```bash
python -m foldspace.esmfold2.evaluate_decoder --out DEC --final-test        # every head, the choice on dev, the test split once
python -m foldspace.esmfold2.decoder_structure_check routes --panel DEC/panel_decoded.tsv --out DEC/structure
for label in S D R; do
  python -m foldspace.evaluation.refold --records DEC/structure/records_${label}.json --label structure-${label} \
      --usalign USalign --work DEC/atlas_${label}.folds --out DEC/structure/atlas_${label}.json
done
python -m foldspace.esmfold2.decoder_structure_check summarize --routes DEC/structure --atlas DEC/structure \
    --out DEC/structure/panel_summary.json
mkdir -p cache/esmfold2/decoders && cp DEC/contextual-s0/full/best.pt cache/esmfold2/decoders/contextual-s0.best.pt
```

**4. Robustness** (folds from perturbed singles need no ESMC backbone, so a 24 GB card is enough). `--levels` is `standard` or `coarse`.

```bash
COV=cache/esmfold2/model_single_train_covariance.npz
python -m foldspace.esmfold2.evaluate_robustness covariance --out ${COV}
python -m foldspace.esmfold2.evaluate_robustness folds  --panel DEC/panel_decoded.tsv --out ROB --covariance ${COV} --levels standard
python -m foldspace.esmfold2.evaluate_robustness decode --decoders DEC --panel DEC/panel_decoded.tsv --covariance ${COV} \
    --out ROB/robustness_decoding.json --levels standard
python -m foldspace.esmfold2.evaluate_robustness subspace --covariance ${COV} --out ROB/robustness_subspace.json
python -m foldspace.esmfold2.evaluate_robustness summarize --folds ROB --decode ROB/robustness_decoding.json \
    --subspace ROB/robustness_subspace.json --out ROB/robustness_summary.json
```

**5. Latent generation.** Each configuration is its own allocation of one 24 GB card; `--set seed=1` trains the second seed. The samples are folded from their own single (the `direct` route), and AtlasFold refolds each sequence.

```bash
python -m foldspace.esmfold2.latent_generation train --out GEN --arms standardized          # likewise whitened, whitened_with_sequence_loss
python -m foldspace.esmfold2.latent_generation fold --samples GEN/standardized/samples --out GEN/standardized/fold \
    --panel DEC/panel_decoded.tsv
for label in samples reference floor; do
  python -m foldspace.evaluation.refold --records GEN/standardized/fold/records_${label}.json \
      --label pilot-standardized-fold-${label} --usalign USalign --work GEN/standardized/fold/atlas_${label}.folds \
      --out GEN/standardized/fold/atlas_${label}.json
done
python -m foldspace.esmfold2.latent_generation summarize --arm GEN/standardized --arm GEN/whitened \
    --arm GEN/whitened_with_sequence_loss --out GEN/generation_summary.json
```

**6. Evaluation on the `re-encoded` route**, for the generated sets and the controls. `SET` is a directory with `samples.tsv`; `--singles` is given for a set that has generated singles. The novelty record (`novelty.json`) is an MMseqs2 search of each sequence against the training sequences (a copy is a hit of identity 0.9 over 0.8 of the query; sets are clustered at 0.3 over 0.8; search parameters are in each record); the script that wrote it is not part of this release.

```bash
python -m foldspace.esmfold2.seqeval fold --samples SET --out RUN --panel DEC/panel_decoded.tsv [--singles GEN/standardized/samples/samples.npz]
for label in samples floor reference; do
  python -m foldspace.evaluation.refold --records RUN/records_${label}.json --label seqeval-${label} \
      --usalign USalign --work RUN/atlas_${label}.folds --out RUN/atlas_${label}.json
done
python -m foldspace.esmfold2.seqeval summarize --fold-dir RUN --novelty NOVELTY.json --novelty-set NAME \
    [--direct GEN/standardized/fold] --out RUN/summary.json
python -m foldspace.esmfold2.seqeval classes --pilot SUMMARY_standardized --pilot SUMMARY_whitened \
    --pilot SUMMARY_whitened_with_sequence_loss \
    --control ESMC_SUMMARY --set 0:0.8=SUMMARY --set 0:1.0=SUMMARY --set 1:0.8=SUMMARY --set 1:1.0=SUMMARY --out classes.json
```

**7. Controls.**

```bash
python -m foldspace.esmfold2.sequence_only train  --out SEQ/s0 --set seed=0        # then seed=1; 20,000 updates at most, stopped by dev loss
python -m foldspace.esmfold2.sequence_only sample --run-dir SEQ/s0 --which best --temperature 0.8 --out SEQ/s0/samples_t0.8   # and 1.0
python -m foldspace.esmfold2.pretrained   --out PGPT2        # ProtGPT2: 600 sequences generated, 48 chosen at the pilot's lengths
python -m foldspace.esmfold2.esmc_control --out ESMC         # ESMC iterative unmasking at its default policy, 48 sequences
```

The check `tests/test_results_esmfold2.py` recomputes every published summary of step 6 from the per-sample records, so that the numbers can be checked without any of the weights.
