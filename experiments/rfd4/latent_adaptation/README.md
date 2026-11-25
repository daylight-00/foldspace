# Latent adaptation

**Can an ESMC-derived representation replace RFD4-Proteina's native latent, with the pretrained decoder and flow adapted to it?**

Design: [methods](../../../docs/methods.md#latent-adaptation) · results: [results](../../../docs/results.md#latent-adaptation) · settings and commands: [`reproducibility/rfd4_training.md`](../../../reproducibility/rfd4_training.md)

| records | |
|---|---|
| `reading_adapters/`, `decoder_adaptation/<latent>/summary.json` | the final result of each adapter and each adapted autoencoder |
| `data_path/`, `channel_audit.json`, `flow_batch_check.json` | checks of the data path, of which latent channels the decoder reads, and of the flow batch against the release pipeline |
| `flow_retarget/<configuration>/` | `summary.json` (budget axes, step times, arguments), `eval_final.json`, `samples/` (sample set and the folding judge's scores); `continued_1600/` for the three configurations continued; `comparison_800.json` and `continued_1600/comparison.json` |

Verify: `python -m pytest tests/test_results_rfd4.py`
