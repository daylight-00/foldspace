# Hybrid latents

**If part of the native latent is kept, does ESMC's residue-specific context help the flow that generates it?**

Design: [methods](../../../docs/methods.md#hybrid-latents) · results: [results](../../../docs/results.md#hybrid-latents) · settings and commands: [`reproducibility/rfd4_training.md`](../../../reproducibility/rfd4_training.md)

| records | |
|---|---|
| `design_panels.{json,txt}`, `base_vs_release.{json,txt}` | the two design panels: every designer, paired comparisons by backbone, the public release and the pretrained base as designers |
| `unconditional_generation.json`, `seed1/` | usable counts of unconditional generation, and the first seed's results for every configuration |
| `type_shuffled_control/` | the validation of the control before it was trained |

Verify: `python -m pytest tests/test_results_rfd4.py`
