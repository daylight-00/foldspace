# Latent generation

**Can a generator trained from scratch make the mixed single of ESMFold2, so that the sequence decoded from it and the frozen folding model give a protein that folds?**

Design: [methods](../../../docs/methods.md#latent-generation) · results: [results](../../../docs/results.md#latent-generation) · commands: [`reproducibility/esmfold2_training.md`](../../../reproducibility/esmfold2_training.md)

| records | |
|---|---|
| `results/pilot_seed0/<configuration>/` | `training.json`, `samples/`, `direct_fold/` and `reencoded_fold/` (fold, per-sample records, AtlasFold evaluations, `summary.json`); `direct_summary.json`, `reencoded_novelty.json`, `reencoded_decoding.json` beside them |
| `results/pilot_seed1/standardized/` | the second training seed in the same layout; `checkpoints.sha256` names its weights |
| `results/controls/{protgpt2,sequence_only,esmc_unmasking}/` | the controls, evaluated with the same criteria |
| `decode_reencoded_singles.py` | the script that made `reencoded_decoding.json` |

Verify: `python -m pytest tests/test_results_esmfold2.py` recomputes each set's summary from its per-sample records.
