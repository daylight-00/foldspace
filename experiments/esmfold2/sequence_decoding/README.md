# Sequence decoding

**Can the residue types be read back off a mixed single, and how far can a single be from the model's own and still fold and read the same?**

Design: [methods](../../../docs/methods.md#sequence-decoding) · results: [results](../../../docs/results.md#sequence-decoding) · commands: [`reproducibility/esmfold2_training.md`](../../../reproducibility/esmfold2_training.md)

| records | |
|---|---|
| `results/decoders/` | `summary.json` and `lda.json` (choice, dev and test results), `<head>/` (training summary, dev metrics), `panel_decoded.tsv` |
| `results/structure_check/` | routes, AtlasFold evaluations, `panel_summary.json` |
| `results/robustness/{standard_levels,coarse_levels}/` | folds, decoding, subspace, summary |

Verify: `python -m pytest tests/test_results_esmfold2.py`
