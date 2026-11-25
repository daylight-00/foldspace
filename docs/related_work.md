# Related work

## Models and tools used

| model | role here |
|---|---|
| [ESMC](https://evolutionaryscale.ai/blog/esm-cambrian) | protein language model whose last hidden state supplies the latent of RFD4 latent adaptation; bundled in the ESMFold2 checkpoint, where it makes the mixed single |
| [ESMFold2](https://huggingface.co/biohub/ESMFold2) ([code](https://github.com/biohub/esm), [preprint](https://doi.org/10.64898/2026.06.03.729735)) | structure predictor whose mixed single is generated in ESMFold2 latent generation, and which folds the generated pairs |
| [RFD4-Proteina](https://huggingface.co/nvidia/RFD4-Proteina) ([preprint](https://doi.org/10.64898/2026.11.12.766037), [code](https://github.com/RosettaCommons/RFD4-Proteina), [project page](https://rfd4-proteina.org/)) | pretrained autoencoder and flow adapted to ESMC-derived latents in RFD4 latent adaptation |
| [La-Proteina](https://arxiv.org/abs/2507.09466) | joint atomistic sequence–structure flow matching over an explicit coarse backbone and per-residue latents; the partially latent design that RFD4 latent adaptation modifies |
| [AtlasFold](https://sciety.org/articles/activity/10.64898/2026.09.04.749352) | independent MSA-free folder (built on AtlasLM, not ESMC) that scores generated sequences |
| [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) | external sequence-design baseline for the hybrid-latent design panels |
| [RF3](https://github.com/RosettaCommons/foundry) | second structure judge for the RFD4 design panels |
| [US-align](https://github.com/pylelab/USalign) | structural alignment and TM-score evaluation (Zhang, Freddolino and Zhang, *Nat Protoc* 21, 2026; Zhang, Shine, Pyle and Zhang, *Nat Methods* 19, 2022; Zhang and Pyle, *iScience* 25, 2022) |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) ([Steinegger and Söding, 2017](https://doi.org/10.1038/nbt.3988)) | sequence search and clustering for the novelty and cluster-split checks |

## Closest prior work

| work | relevance |
|---|---|
| [PLAID](https://www.biorxiv.org/content/10.1101/2024.12.02.626353) | latent diffusion in ESMFold's latent space, decoded to sequence and all-atom structure by frozen decoders, trained on sequence databases alone: the closest work to ESMFold2 latent generation |
| [CHEAP](https://www.biorxiv.org/content/10.1101/2024.08.06.606920) ([*Patterns*, 2025](https://doi.org/10.1016/j.patter.2025.101289)) | compression of ESMFold's latent (128× along channels, 8× along length, under 2 Å) that keeps sequence and structure information; the representation PLAID generates in |
| [DiMA](https://arxiv.org/abs/2403.03726) | continuous diffusion on protein-language-model encodings, including ESMC, for sequence generation |

## ESMFold2 latent generation and PLAID

PLAID generates in the latent of a folding model and lets frozen decoders produce the sequence and structure without an explicit Cα trace. ESMFold2 latent generation applies the same idea to ESMFold2 and is not claimed as new. The differences:

* **Entry of the latent.** In ESMFold the language-model latent is the input of the folding trunk. In ESMFold2 the mixed single enters only as a pair bias and the sequence enters separately, so a generated single does not fix a protein, and a sequence head is needed.
* **Data.** PLAID is trained on sequence databases, far more data than the 17,063 training chains of the pilot here.
* **Outcome.** PLAID reports generated structures. The pilot here has none that pass its evaluation criteria, and a sequence-only model of the same size fails equally, so it does not settle whether a generator of the mixed single can work.

## Not claimed

* That generating sequence and structure from a folding model's latent without an explicit Cα trace is new.
* That ESMC's latent cannot be generated, or that an explicit Cα trace is necessary.
* That any generator here is a competitive designer.
