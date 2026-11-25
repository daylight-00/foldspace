"""foldspace: pretrained protein representations as generative latents.

Two architectures are built and measured here: `foldspace.rfd4` replaces or augments the latent of RFD4-Proteina with ESMC-derived
representations; `foldspace.esmfold2` generates ESMFold2's mixed-single representation and folds it with the frozen model.
`foldspace.evaluation` holds the independent refolding evaluator and the geometry both are scored with.
"""

__version__ = "0.1.0"
