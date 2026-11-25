"""ESMC-6B's feature of a corrupted sequence, for a conditioning hook: the glue between the generator's alphabet and ESMC's.

`features(tokens, mask)` is what a conditioning hook calls with the generator's own `aatypes` and `res_mask`: `tokens` are `[B, L]` indices in the **AlphaFold's alphabet** (`evaluation.residues`, AlphaFold's order, 20 the mask) and `mask` is `[B, L]` with 1 at residues and 0 at padding. The
result is `[B, L, 2560]`: ESMC's last hidden state at each residue, zero at padding.

Three things here are where a fold or a feature fails silently, and each is checked:

* *the alphabet.* ESMC's input ids are frequency-ordered and the generator's are AlphaFold's; no integer agrees (`esmfold2.vocabulary`, `evaluation.residues`). The table is built from the letters, through the same encoders the substrate uses, never by casting one integer to the other, and the mask token is the id the upstream signature applies (32).
* *which state.* `last_hidden_state` is the **post-final-norm** state, the tensor ESMC's own masked-LM head reads (`esmfold2.lm_head`), and is the last slice of the 81-layer stack that `compute_lm_hidden_states` returns. Only it is asked for: `output_hidden_states=True` would keep all 81 layers (about 80 times the memory).
* *the wrapping.* Each sequence is `[BOS] residues [EOS]` with padding after it, the chain ids and the padding spelled as `compute_lm_hidden_states` spells them (`sequence_id`: BOS cumulated, padding -1), so a single chain's result is that function's, one layer of it. A sequence's result does not depend on what it is batched with (a check on a small model).

ESMC runs in `eval()` with no gradients (`requires_grad_(False)`, `torch.no_grad()`): nothing is back-propagated through it. The valid positions of each row must be a prefix (right padding), as the plan's micro-batches and a sampler's designs are; anything else is refused.

`load_esmc` loads the backbone alone, from the mirror the substrate would resolve, in bf16: the folding model's own trunk and heads (about a GiB) are not needed and are not resident. The checkpoint is read **one tensor at a time onto the card, already in bf16**: upstream's reader holds every shard in host memory in fp32 first (23.6 GiB for ESMC-6B) and a 6000ada job's host memory is
25000M, so the stock path's peak is at the limit before the process holds anything else (a test compares the result with the stock path's, bit for bit).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

#: ESMC's special tokens, as `compute_lm_hidden_states` writes them
BOS, PAD, EOS = 0, 1, 2


def token_table(letters: str, id_of: Callable[[str], int], mask_id: int) -> list[int]:
    """`[len(letters) + 1]`: the ESMC id of each residue of the generator's alphabet, in its order, then the id the generator's mask token stands for."""
    return [int(id_of(letter)) for letter in letters] + [int(mask_id)]


def residue_table() -> list[int]:
    """The table for AlphaFold's alphabet, from the substrate's own encoders."""
    from foldspace.evaluation import residues
    from foldspace.esmfold2 import vocabulary

    return token_table(
        residues.RESIDUES,
        lambda letter: vocabulary.sequence_to_input_ids(letter)[0],
        vocabulary.mask_token_id(),
    )


class EsmcFeatures:
    """`features(tokens, mask)` for a conditioning hook, over an ESMC backbone (`esm.models.esmc.EsmcModel`)."""

    def __init__(self, esmc, table: Sequence[int]):
        self.esmc = esmc
        self.table = torch.as_tensor(list(table), dtype=torch.long)
        self.forwards = 0

    def hidden(self, lm_ids: torch.Tensor, sequence_id: torch.Tensor):
        """ESMC's last hidden state of wrapped inputs: `[B, T, D]`."""
        return self.esmc(
            input_ids=lm_ids, sequence_id=sequence_id, return_dict=True
        ).last_hidden_state

    def wrap(self, tokens: torch.Tensor, mask: torch.Tensor):
        """`[BOS] residues [EOS]` per row, padded to the longest, and the chain ids `compute_lm_hidden_states` gives them."""
        valid = mask > 0
        if not bool((valid[:, 1:] <= valid[:, :-1]).all()):
            raise ValueError(
                "the residues of a row must be a prefix: right padding only"
            )
        if int(tokens.min()) < 0 or int(tokens.max()) >= len(self.table):
            raise ValueError("a token is outside the generator's alphabet")
        batch = tokens.shape[0]
        lengths = valid.sum(dim=1)
        width = int(lengths.max()) + 2
        ids = self.table.to(tokens.device)[tokens]
        lm = torch.full((batch, width), PAD, dtype=torch.long, device=tokens.device)
        lm[:, 0] = BOS
        body = torch.where(valid, ids, torch.full_like(ids, PAD))[:, : width - 2]
        lm[:, 1 : 1 + body.shape[1]] = body
        lm[torch.arange(batch, device=tokens.device), lengths + 1] = EOS
        sequence_id = (lm == BOS).cumsum(dim=1) - 1
        return lm, sequence_id.masked_fill(lm == PAD, -1), lengths

    @torch.no_grad()
    def __call__(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lm, sequence_id, _ = self.wrap(tokens, mask)
        self.forwards += 1
        states = self.hidden(lm, sequence_id)
        width = tokens.shape[1]
        inner = states[:, 1:-1]
        if inner.shape[1] < width:
            inner = torch.nn.functional.pad(inner, (0, 0, 0, width - inner.shape[1]))
        valid = (mask > 0)[..., None]
        return torch.where(valid, inner[:, :width], torch.zeros_like(inner[:, :width]))


def checkpoint_shards(directory) -> list:
    """The safetensors files of a checkpoint directory: the shards of its index, or the single file."""
    import json
    from pathlib import Path

    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [directory / name for name in names]
    single = directory / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors checkpoint in {directory}")


def read_streamed(device) -> Callable:
    """A reader of a checkpoint directory with `esm.models.hub.read_safetensors_dir`'s signature that holds one tensor on the host at a time.

    Every floating-point tensor comes back in bf16 on `device`; the rest keep their dtype. Upstream's reader returns the whole checkpoint in its stored dtype on the host, and the cast that follows it is the load's own, so a tensor that is already bf16 passes through it unchanged.
    """
    from safetensors import safe_open

    def read(directory) -> dict:
        out = {}
        for shard in checkpoint_shards(directory):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 -- safetensors' handle has keys() and no iteration
                    tensor = handle.get_tensor(key)
                    out[key] = tensor.to(
                        device=device,
                        dtype=torch.bfloat16 if tensor.is_floating_point() else None,
                    )
        return out

    return read


def load_esmc(device, *, source: str | None = None, stream: bool = True):
    """ESMC-6B alone, in bf16 and `eval()`, frozen; and the directory it came from.

    The directory is the substrate's own resolution of the backbone (`esmfold2_atomworks.paths.resolve_esmc`: a local mirror first), so what is loaded is what a fold would use.

    With `stream` (the default) only the reader of upstream's load is replaced (`read_streamed`: one tensor at a time onto `device` in bf16, the host never holding the checkpoint); the layout translation, the key accounting that refuses a silent drop, the cast of the buffers and the move are upstream's own. This is a temporary workaround of a seam upstream does not offer
    (a loader whose host memory does not grow with the checkpoint); it reaches into one private name, `esm.models.hub.read_safetensors_dir`, which a test pins by comparing the result to the stock path's bit for bit.
    """
    from esm.models import hub
    from esm.models.esmc import EsmcModel
    from esm.models.esmc.config import ESMC_6B_HF_REPO
    from esmfold2_atomworks import paths

    directory = str(source or paths.resolve_esmc(ESMC_6B_HF_REPO))
    original = hub.read_safetensors_dir
    if stream:
        hub.read_safetensors_dir = read_streamed(device)
    try:
        model = EsmcModel.from_pretrained(
            directory, device=device, dtype=torch.bfloat16
        )
    finally:
        hub.read_safetensors_dir = original
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if str(device).startswith("cuda"):
        # the load's transients (the checkpoint's tensors before the layout translation fuses them) are not left in the allocator's cache
        torch.cuda.empty_cache()
    return model, directory


def glue_logits_of(features: EsmcFeatures, head):
    """`logits_of(state)` for `esmc_generate.self_check`, through this glue: `state` is a native chain in `STANDARD_AA` order with -1 at the masked positions; the result is `[L, 20]` logits in that order.

    The head is ESMC's own masked-LM head (`perturb.mlm.load_lm_head`), which reads the same post-final-norm state the generator is given, so a recovery of masked residues through it is a check of the whole glue (the alphabet, the mask, the wrapping, the state) against what ESMC is known to do.
    """
    import numpy as np

    from foldspace.evaluation import residues
    from foldspace.esmfold2 import vocabulary

    standard = vocabulary.STANDARD_AA
    to_generator = np.array(
        [residues.RESIDUES.index(letter) for letter in standard], dtype=np.int64
    )
    columns = torch.as_tensor(vocabulary.input_ids_in_aa_order(), dtype=torch.long)

    def logits_of(state):
        state = np.asarray(state)
        tokens = np.where(
            state >= 0, to_generator[np.clip(state, 0, None)], residues.MASK
        )
        tokens = torch.as_tensor(tokens, dtype=torch.long)[None]
        device = next(features.esmc.parameters()).device
        mask = torch.ones(tokens.shape, device=device)
        with torch.no_grad():
            h = features(tokens.to(device), mask)
            logits = head(h)[0].float()
        return logits[:, columns.to(logits.device)].cpu().numpy()

    return logits_of
