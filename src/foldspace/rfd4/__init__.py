"""RFD4 latent adaptation: RFD4-Proteina with an ESMC-derived latent.

The autoencoder and flow trainers, the sampler, the audits and the controls. Most modules are scripts as much as library code:
each has a `main` that runs it from the command line (the commands are in `reproducibility/rfd4_training.md`), and the other
modules import what they share (`flow_data`, `flow_model`, `code_bank`, ...) from here.
"""
