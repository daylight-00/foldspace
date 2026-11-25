# Latent replacement

**Can all 32 channels of the latent come from ESMC, and can the adapted flow generate them?**

Design: [methods](../../../docs/methods.md#latent-replacement) · results: [results](../../../docs/results.md#latent-replacement) · settings and commands: [`reproducibility/rfd4_training.md`](../../../reproducibility/rfd4_training.md)

| records (`.json` with a `.txt` form) | |
|---|---|
| `latent_statistics`, `decoder_geometry` | the latents' statistics and the bond lengths of the decoder's own reconstructions |
| `generation_1600`, `generation_extended` | usable counts at 1,600 updates and at 2,400 and 3,200 |
| `generated_code_diagnostics`, `sampling_trajectory` | what the generated code looks like and where along sampling it goes |
| `calibration`, `calibration/` | generated codes given back the natural moments |

Verify: `python -m pytest tests/test_results_rfd4.py`
