# Interface validation

**Can the folding model be driven from a mixed single alone, and what does the single add to the fold?**

Design: [methods](../../../docs/methods.md#interface-validation) · results: [results](../../../docs/results.md#interface-validation) · commands: [`reproducibility/esmfold2_training.md`](../../../reproducibility/esmfold2_training.md)

| records | |
|---|---|
| `results/target_audit.json` | audit of the training singles |
| `results/parity/{rtx6000ada,blackwell}/` | `parity_det.json` (the gate) and `parity_nondet.json` |
| `results/contribution/` | `contribution_det.json` (8 pairs), `contribution_det_replicate.json` (16 other pairs), `audit.json` (the classification at each threshold and under four definitions of TM) |

Verify: `python -m pytest tests/test_results_esmfold2.py`
