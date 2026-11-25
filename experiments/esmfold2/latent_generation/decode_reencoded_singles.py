"""Does the frozen decoder read a generated sequence back off the single the model derives from it?

For each sample of the pilot's arms: the single H(S_hat) that the model derived from the sequence read off the generated single (``derived_singles.npz`` of the
re-encoded run) is standardised with the train split's normalisation and read by the frozen contextual decoder; the share of residues that it reads as S_hat
is reported, beside the same on the 24 reference chains (the decoder's accuracy on natural singles of the dev panela). Run from a stack's tree:

    python experiments/esmfold2/latent_generation/decode_reencoded_singles.py RUN_DIR OUT.json

where RUN_DIR holds ``<arm>/fold.json`` and ``<arm>/derived_singles.npz`` for the arms.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from foldspace.esmfold2 import train_decoder
from foldspace.esmfold2.decode_eval import ALPHABET

run, out = Path(sys.argv[1]), Path(sys.argv[2])
norm = np.load("cache/esmfold2/model_single_train_normalisation.npz", allow_pickle=False)
mean, std = norm["mean"].astype(np.float32), norm["std"].astype(np.float32)
decoder = train_decoder.load_checkpoint(
    Path("cache/esmfold2/decoders/contextual-s0.best.pt"), torch.device("cpu")
)[0].eval()
result = {}
for arm in ("standardized", "whitened", "whitened_with_sequence_loss"):
    fold = json.loads((run / arm / "fold.json").read_text())
    sequences = {c["id"]: c["sequence"] for c in fold["chains"] if c["kind"] == "sample"}
    held = np.load(run / arm / "derived_singles.npz", allow_pickle=False)
    accuracy = []
    for sample_id in sorted(sequences):
        h = held[sample_id]
        z = torch.as_tensor((h - mean) / std)[None]
        with torch.no_grad():
            logits = decoder(z, torch.as_tensor([len(h)]))
        read = "".join(ALPHABET[int(k)] for k in logits[0, : len(h)].argmax(-1))
        accuracy.append(float(np.mean([a == b for a, b in zip(read, sequences[sample_id], strict=True)])))
    result[arm] = {
        "samples": len(accuracy),
        "accuracy_median": float(np.median(accuracy)),
        "accuracy_min": float(np.min(accuracy)),
        "accuracy_max": float(np.max(accuracy)),
    }
    print(arm, result[arm])
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
