"""A pretrained generator's sequences at the pilot's lengths: ProtGPT2.

A masked model such as ESMC's unmasking is no generator, so the ruler is also read on a published autoregressive one. ProtGPT2 is an autoregressive GPT-2 of 738 M parameters trained on
UniRef50 and published as making foldable sequences; its sequences are read by the ruler like any other set, so that it says whether the ruler can be passed by generation and what
"a pretrained sequence generator, then the folding model" scores.

It is made at the settings of its model card (sampling, top-k 950, a repetition penalty of 1.2, temperature 1.0, 100 tokens, stopping at the end-of-text token) and not at the pilot's
lengths, which it cannot be told: a few hundred are generated, those of 70 to 220 residues are kept, and 48 are chosen to meet the pilot's 48 lengths, each target taking the nearest
unused sequence within a tolerance. The weights are a pickled file; ``transformers`` reads it with ``weights_only`` loading.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

from foldspace.esmfold2 import latent_generation as latent_generation, sequence_only

__all__ = ["clean", "keep", "match_lengths"]

REPO = "nferruz/ProtGPT2"
REVISION = "ff981fc96771d6f6d3b6453f94772d3cb6c7b90b"
SEED = 20261127
#: the model card's own settings, fixed in advance
SETTINGS = {
    "top_k": 950,
    "repetition_penalty": 1.2,
    "temperature": 1.0,
    "max_length": 100,
}
LOW, HIGH = 70, 220
TOLERANCE = 0.15
STANDARD = set(latent_generation.LETTERS)
END = "<|endoftext|>"


def clean(text: str) -> str:
    """A decoded generation as a sequence: the end-of-text tokens, the line breaks it was written with and any spaces removed."""
    return re.sub(r"\s+", "", text.replace(END, ""))


def keep(sequence: str, low: int = LOW, high: int = HIGH) -> bool:
    """Whether a generation is a sequence the ruler reads: of the twenty standard residues, and of 70 to 220 of them."""
    return bool(low <= len(sequence) <= high and sequence and set(sequence) <= STANDARD)


def match_lengths(
    lengths: list[int], targets: list[int], tolerance: float = TOLERANCE
) -> list[int | None]:
    """For each target length (in the order given), the index of the nearest unused sequence within ``tolerance`` of it, or ``None``.

    The targets are served shortest first, and a sequence is used once; ties in distance go to the earlier sequence.
    """
    free = set(range(len(lengths)))
    chosen: list[int | None] = [None] * len(targets)
    for position in sorted(range(len(targets)), key=lambda p: (targets[p], p)):
        target = targets[position]
        best = None
        for index in sorted(free):
            gap = abs(lengths[index] - target)
            if gap <= tolerance * target and (
                best is None or gap < abs(lengths[best] - target)
            ):
                best = index
        if best is not None:
            chosen[position] = best
            free.discard(best)
    return chosen


def generate_ids(
    model, *, n: int, seed: int, batch: int = 100, device="cuda", settings=SETTINGS
) -> list[list[int]]:
    """``n`` generations as token ids, from the end-of-text token as the prompt, ``batch`` at a time."""
    import torch

    model.eval()
    prompt = torch.zeros(
        (1, 1), dtype=torch.long, device=device
    )  # the model card's prompt: <|endoftext|>, id 0
    out: list[list[int]] = []
    for k, lo in enumerate(range(0, n, batch)):
        torch.manual_seed(seed + k)
        size = min(batch, n - lo)
        with torch.no_grad():
            ids = model.generate(
                prompt.expand(size, 1),
                do_sample=True,
                top_k=settings["top_k"],
                repetition_penalty=settings["repetition_penalty"],
                temperature=settings["temperature"],
                max_length=settings["max_length"],
                num_return_sequences=1,
                eos_token_id=0,
                pad_token_id=0,
            )
        out += [row.tolist() for row in ids.cpu()]
    return out


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("cache/models/nferruz-ProtGPT2")
    )
    parser.add_argument(
        "--chains", type=Path, default=Path("data/corpus/chains.npz")
    )
    parser.add_argument("--n-generate", type=int, default=600)
    parser.add_argument("--n", type=int, default=latent_generation.N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from foldspace.evaluation.provenance import capture, require_clean

    began = time.perf_counter()
    require_clean(allow=args.allow_dirty)
    weights = args.model_dir / "pytorch_model.bin"
    weights_sha = digest(weights)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), use_safetensors=False, trust_remote_code=False
    ).to(args.device)
    print(
        f"loaded {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M parameters  {time.perf_counter() - began:.0f}s",
        flush=True,
    )

    ids = generate_ids(model, n=args.n_generate, seed=args.seed, device=args.device)
    sequences = [clean(tokenizer.decode(row, skip_special_tokens=False)) for row in ids]
    kept = [i for i, s in enumerate(sequences) if keep(s)]
    targets = latent_generation.sample_lengths(
        sequence_only.dev_lengths(args.chains), args.n, *latent_generation.SAMPLE_LENGTHS
    ).tolist()
    matched = match_lengths([len(sequences[i]) for i in kept], targets)
    chosen = [None if m is None else kept[m] for m in matched]
    missing = [t for t, c in zip(targets, chosen, strict=True) if c is None]
    picked = [c for c in chosen if c is not None]

    args.out.mkdir(parents=True, exist_ok=True)
    kept_set = set(kept)
    lines = ["index\tlength\tkept\tsequence"] + [
        f"{i}\t{len(s)}\t{int(i in kept_set)}\t{s}" for i, s in enumerate(sequences)
    ]
    (args.out / "generated.tsv").write_text("\n".join(lines) + "\n")
    sequence_only.write_samples(
        args.out,
        [sequences[c] for c in picked],
        [float("nan")] * len(picked),
        {
            "n": len(picked),
            "generator": f"{REPO} at {REVISION}",
            "settings": SETTINGS,
            "seed": args.seed,
            "generated": len(sequences),
            "kept_of_70_to_220_residues": len(kept),
            "chosen": len(picked),
            "targets_without_a_sequence": missing,
            "target_lengths": targets,
            "chosen_lengths": [len(sequences[c]) for c in picked],
            "weights_sha256": weights_sha,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "provenance": capture({"allow_dirty": bool(args.allow_dirty)}),
            "seconds": time.perf_counter() - began,
        },
    )
    print(
        f"generated {len(sequences)}, kept {len(kept)}, chosen {len(picked)} of {len(targets)} targets"
        + (f" ({len(missing)} targets without a sequence)" if missing else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
