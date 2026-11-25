"""Latent generation: a pilot generator of the single, trained from scratch, and what it makes.

The arms and how they are read are fixed before a run is read. Three steps:

``train``
    one arm (a coordinate and a loss) through its stages (smoke, overfit, small, full), a denoising reading on dev at fixed
    noise levels, and a fixed number of samples whose singles are written down with the sequences read off them.
``fold``
    the samples folded by the frozen model from their own single (the *direct* route), beside controls and real chains.
``summarize``
    the readings.

A pilot: three arms at one seed, a small model, a budget of an hour or two each. It asks whether the coordinate can be
generated at all, not how well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

__all__ = ["ARMS", "Config", "parse_overrides"]

N_SAMPLES = 48
SAMPLE_LENGTHS = (
    70,
    220,
)  # a sample is no shorter and no longer than the folding model is read on in the decoding and robustness experiments
SAMPLE_STEPS = 50


@dataclass
class Config:
    coordinate: str = (
        "standardized"  # "standardized": the standardised single; "whitened": the same single whitened
    )
    ce_weight: float = 0.0  # weight of the frozen decoder's cross-entropy on the predicted single, times t
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    lr: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    max_tokens: int = 32768
    steps: int = 20000
    ema_decay: float = 0.999
    t_floor: float = 0.1
    floor: float = 1e-3  # added to every eigenvalue before the whitening divides by its square root
    clip: float = 1.0
    eval_every: int = 2000
    n_samples: int = N_SAMPLES
    sample_steps: int = SAMPLE_STEPS
    seed: int = 0


#: the arms of the pilot: the coordinate and the loss are what differ
ARMS: dict[str, Config] = {
    "standardized": Config(coordinate="standardized"),
    "whitened": Config(coordinate="whitened"),
    "whitened_with_sequence_loss": Config(coordinate="whitened", ce_weight=0.1),
}

#: what the stages ask
OVERFIT_CHAINS = 8
OVERFIT_STEPS = 300
OVERFIT_RATIO = (
    0.25  # the error at t = 0.5 must fall to this share of its starting value
)
SMALL_STEPS = 400
SMALL_RATIO = 0.85
EVAL_T = (0.5, 0.8, 0.95)
GROUPS = {
    "leading_64": (0, 0.25),
    "middle_128": (0.25, 0.75),
    "trailing_64": (0.75, 1.0),
}  # shares of the 256


def parse_overrides(text: str) -> dict:
    """``key=value,...`` against ``Config``'s fields, each value in the field's own type."""
    fields = {f: type(getattr(Config(), f)) for f in asdict(Config())}
    out = {}
    for item in [p for p in text.split(",") if p]:
        key, _, value = item.partition("=")
        if key not in fields:
            raise ValueError(f"{key!r} is not a field of Config")
        out[key] = fields[key](value)
    return out


# ----------------------------------------------------------------- readings


def _groups(values: np.ndarray) -> dict[str, float]:
    d = len(values)
    return {
        name: float(values[round(lo * d) : round(hi * d)].sum())
        for name, (lo, hi) in GROUPS.items()
    }


def eigen_error(sq_error: np.ndarray, values: np.ndarray, count: int) -> dict:
    """The error of a prediction, in the units of the robustness experiment: per direction against that direction's own variance.

    ``sq_error`` is the summed squared error along each eigendirection, over ``count`` rows; ``values`` the variances.
    ``relative`` is the root of the mean over directions of (mean squared error / variance): the level of ``anoise`` that has
    the same error. ``by_group`` is the root of the group's total error over its total variance.
    """
    values = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    mean_error = sq_error / count
    gv = _groups(values)
    ge = _groups(mean_error)
    return {
        "relative": float(np.sqrt(np.mean(mean_error / values))),
        "relative_size": float(np.sqrt(mean_error.sum() / values.sum())),
        "by_group": {k: float(np.sqrt(ge[k] / gv[k])) for k in GROUPS},
    }


def sample_lengths(dev_lengths: np.ndarray, n: int, lo: int, hi: int) -> np.ndarray:
    """``n`` lengths spread over the dev lengths between ``lo`` and ``hi``: the quantiles of what is there."""
    eligible = np.sort(dev_lengths[(dev_lengths >= lo) & (dev_lengths <= hi)])
    if len(eligible) < n:
        raise ValueError(
            f"{len(eligible)} dev chains between {lo} and {hi}, fewer than {n}"
        )
    picks = np.linspace(0, len(eligible) - 1, n).round().astype(int)
    return eligible[picks].astype(np.int64)


# --------------------------------------------------------------------- run


def _digest(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


class Harness:
    """What every arm shares: the data, the coordinate maps, the frozen decoder and the dev chains it is read on."""

    def __init__(
        self,
        *,
        cache,
        chains,
        train_table,
        device,
        spectrum,
        decoder_path,
        limit_chains=0,
    ):
        import torch

        from foldspace.esmfold2 import train_decoder
        from foldspace.esmfold2.decoder_data import Normaliser, load_split

        self.torch, self.device = torch, torch.device(device)
        cache = Path(cache)
        self.norm = Normaliser.load(
            cache / "esmfold2" / "model_single_train_normalisation.npz", self.device
        )
        common = {
            "cache": cache,
            "chains": Path(chains),
            "train_table": Path(train_table),
            "device": self.device,
        }
        self.train = load_split("train", limit_chains=limit_chains, **common)
        self.dev = load_split("dev", limit_chains=limit_chains, **common)
        self.spectrum = spectrum
        self.values = np.asarray(spectrum.values, dtype=np.float64)
        self.vectors = torch.as_tensor(
            spectrum.vectors, dtype=torch.float32, device=self.device
        )
        self.decoder = None
        if decoder_path:
            self.decoder = train_decoder.load_checkpoint(
                Path(decoder_path), self.device
            )[0].eval()
            for p in self.decoder.parameters():
                p.requires_grad_(False)
        # one chain of every dev cluster, the first: the chains the denoising is read on
        _, first = np.unique(self.dev.clusters, return_index=True)
        self.eval_chains = np.sort(first)

    def whitening(self, config: Config):
        from foldspace.esmfold2 import generator

        if config.coordinate == "standardized":
            return generator.Whitening.identity()
        if config.coordinate == "whitened":
            return generator.Whitening.from_spectrum(
                self.spectrum.vectors,
                self.spectrum.values,
                floor=config.floor,
                device=self.device,
            )
        raise ValueError(f"coordinate is standardized or whitened, not {config.coordinate!r}")

    def model(self, config: Config):
        from foldspace.esmfold2 import generator

        return generator.Denoiser(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
        ).to(self.device)


def _autocast(device):
    import torch

    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def _step_loss(model, harness, whiten, config, batch, gen):
    """One batch: the flow loss, the cross-entropy term when asked, and the unweighted error of the prediction."""
    import torch
    from torch.nn import functional as F

    from foldspace.esmfold2 import generator
    from foldspace.esmfold2.decoder_data import IGNORE, gather

    z, y, lens = gather(harness.train, batch, harness.norm)
    x1 = whiten.forward(z)
    x_t, t, _ = generator.flow_inputs(x1, lens, gen)
    with _autocast(harness.device):
        x1_hat = model(x_t, t, lens)
    loss, mse = generator.flow_loss(x1_hat, x1, t, lens, t_floor=config.t_floor)
    ce = torch.zeros((), device=harness.device)
    if config.ce_weight > 0 and harness.decoder is not None:
        z_hat = whiten.inverse(x1_hat.float())
        with _autocast(harness.device):
            logits = harness.decoder(z_hat, lens)
        per_position = F.cross_entropy(
            logits.float().transpose(1, 2), y, ignore_index=IGNORE, reduction="none"
        )
        valid = (y != IGNORE).float()
        per_chain = (per_position * valid).sum(1) / valid.sum(1)
        ce = (t * per_chain).mean()
        loss = loss + config.ce_weight * ce
    return loss, mse, ce


def denoise_eval(model, harness, whiten, *, seed: int = 0) -> dict:
    import torch

    with torch.no_grad():
        return _denoise_eval(model, harness, whiten, seed)


def _denoise_eval(model, harness, whiten, seed: int) -> dict:
    """The model's one-step estimate of a clean dev single from a noisy one, at fixed noise levels, read in the units of the robustness experiment.

    For each ``t`` in ``EVAL_T``: the error against the clean single in the standardised coordinate, per direction against
    that direction's variance (``eigen_error``), and the accuracy of the frozen decoder on the estimate.
    """
    import torch

    from foldspace.esmfold2 import batching, generator
    from foldspace.esmfold2.decoder_data import IGNORE, gather

    model.eval()
    gen = torch.Generator(device=harness.device).manual_seed(seed)
    out = {}
    for t_value in EVAL_T:
        sq = torch.zeros(
            harness.values.shape[0], dtype=torch.float64, device=harness.device
        )
        rows = 0
        correct, total = [], []
        for batch in batching.pack(harness.eval_chains, harness.dev.lengths, 65536):
            z, y, lens = gather(harness.dev, batch, harness.norm)
            x1 = whiten.forward(z)
            x_t, t, _ = generator.flow_inputs(
                x1,
                lens,
                gen,
                t=torch.full((len(batch),), t_value, device=harness.device),
            )
            with _autocast(harness.device):
                x1_hat = model(x_t, t, lens)
            z_hat = whiten.inverse(x1_hat.float())
            valid = y != IGNORE
            err = ((z_hat - z)[valid]) @ harness.vectors  # along the eigendirections
            sq += (err.double() ** 2).sum(0)
            rows += int(valid.sum())
            if harness.decoder is not None:
                with _autocast(harness.device):
                    pred = harness.decoder(z_hat, lens).argmax(-1)
                for b in range(len(batch)):
                    v = valid[b]
                    correct.append(float((pred[b][v] == y[b][v]).float().mean()))
                    total.append(int(v.sum()))
        reading = eigen_error(sq.cpu().numpy(), harness.values, rows)
        if correct:
            reading["decode_acc_per_chain"] = float(np.mean(correct))
            reading["decode_acc_micro"] = float(np.average(correct, weights=total))
        out[f"t={t_value}"] = reading
    return out


def _warm_lr(opt, step, total, config):
    from foldspace.esmfold2 import train_decoder

    lr = train_decoder.lr_at(step, total, config)
    for group in opt.param_groups:
        group["lr"] = lr
    return lr


def train_steps(
    harness, config, whiten, model, *, steps, out, log, resume=True, label="full"
) -> dict:
    """``steps`` updates of ``model`` on cluster-uniform draws of whole chains; the average of its weights is what is read."""
    import torch

    from foldspace.esmfold2 import batching, generator

    device = harness.device
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    gen = torch.Generator(device=device).manual_seed(config.seed + 1)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.99),
        weight_decay=config.weight_decay,
    )
    ema = generator.EMA(model, config.ema_decay)
    probs = batching.cluster_probabilities(harness.train.clusters)
    step, history = 0, []
    Path(out).mkdir(parents=True, exist_ok=True)
    last = Path(out) / "last.pt"
    if resume and last.exists():
        state = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        ema.shadow = state["ema"]
        step, history = state["step"], state["history"]
        rng.bit_generator.state = state["rng"]
        log(f"  resumed {label} at step {step}")
    batches = iter(())
    began = time.perf_counter()
    while step < steps:
        try:
            batch = next(batches)
        except StopIteration:
            draws = batching.draw(
                harness.train.n_chains, harness.train.n_chains, rng, probs
            )
            batches = iter(
                batching.pack(draws, harness.train.lengths, config.max_tokens, rng)
            )
            continue
        model.train()
        _warm_lr(opt, step, steps, config)
        loss, mse, ce = _step_loss(model, harness, whiten, config, batch, gen)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step} ({label})")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        opt.step()
        ema.update(model, step)
        step += 1
        if step % config.eval_every == 0 or step == steps:
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            reading = denoise_eval(model, harness, whiten)
            model.load_state_dict(live)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "mse": float(mse),
                "ce": float(ce.detach()),
                "seconds": time.perf_counter() - began,
                "denoise": reading,
            }
            history.append(row)
            r = reading["t=0.8"]
            log(
                f"  {label} step {step:6d}  loss {row['loss']:.4f}  t=0.8 relative {r['relative']:.3f}"
                f" decode {r.get('decode_acc_per_chain', float('nan')):.4f}  {row['seconds']:.0f}s"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "ema": ema.shadow,
                    "step": step,
                    "history": history,
                    "rng": rng.bit_generator.state,
                },
                last,
            )
    ema.copy_to(model)
    return {"steps": step, "history": history, "digest": _digest(model)}


def stage_smoke(harness, config, whiten) -> dict:
    """A batch is finite, a step moves the parameters, and the output does not depend on the padding."""
    import torch

    from foldspace.esmfold2 import batching

    model = harness.model(config)
    gen = torch.Generator(device=harness.device).manual_seed(0)
    batch = next(
        iter(
            batching.pack(
                np.arange(harness.train.n_chains),
                harness.train.lengths,
                config.max_tokens,
            )
        )
    )
    before = [p.detach().clone() for p in model.parameters()]
    loss, mse, _ = _step_loss(model, harness, whiten, config, batch, gen)
    loss.backward()
    finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    moved = any(
        not torch.equal(a, p) for a, p in zip(before, model.parameters(), strict=True)
    )
    return {
        "ok": bool(torch.isfinite(loss)) and finite and moved,
        "loss": float(loss.detach()),
        "initial_mse": float(mse),
        "gradient_finite": finite,
        "parameter_moved": moved,
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch_chains": len(batch),
    }


def _error_at(model, harness, whiten, split, chains, t_value=0.5, seed=0) -> float:
    """The unweighted mean squared error of the one-step estimate at ``t_value`` on the given chains of ``split``."""
    import torch

    from foldspace.esmfold2 import batching, generator
    from foldspace.esmfold2.decoder_data import gather

    model.eval()
    gen = torch.Generator(device=harness.device).manual_seed(seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in batching.pack(chains, split.lengths, 65536):
            z, _, lens = gather(split, batch, harness.norm)
            x1 = whiten.forward(z)
            x_t, t, _ = generator.flow_inputs(
                x1,
                lens,
                gen,
                t=torch.full((len(batch),), t_value, device=harness.device),
            )
            with _autocast(harness.device):
                x1_hat = model(x_t, t, lens)
            total += float(generator.flow_loss(x1_hat, x1, t, lens)[1]) * len(batch)
            count += len(batch)
    return total / count


def stage_overfit(harness, config, whiten) -> dict:
    """Eight chains, trained on alone: the error at ``t = 0.5`` falls to a quarter of where it started."""
    import torch

    from foldspace.esmfold2 import batching

    model = harness.model(config)
    rng = np.random.default_rng(1)
    chains = np.sort(
        rng.choice(
            harness.train.n_chains,
            min(OVERFIT_CHAINS, harness.train.n_chains),
            replace=False,
        )
    )
    before = _error_at(model, harness, whiten, harness.train, chains)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator(device=harness.device).manual_seed(2)
    for _ in range(OVERFIT_STEPS):
        model.train()
        for batch in batching.pack(
            chains, harness.train.lengths, config.max_tokens, rng
        ):
            loss, _, _ = _step_loss(
                model, harness, whiten, replace(config, ce_weight=0.0), batch, gen
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            opt.step()
    after = _error_at(model, harness, whiten, harness.train, chains)
    return {
        "ok": after <= OVERFIT_RATIO * before,
        "before": before,
        "after": after,
        "steps": OVERFIT_STEPS,
        "required_ratio": OVERFIT_RATIO,
    }


def stage_small(harness, config, whiten, out, log) -> dict:
    """A few hundred steps on the whole train split: the error at ``t = 0.5`` on dev falls to 0.7 of its start."""
    model = harness.model(config)
    before = _error_at(model, harness, whiten, harness.dev, harness.eval_chains)
    small = replace(config, warmup_steps=min(config.warmup_steps, 50), eval_every=10**9)
    run = train_steps(
        harness,
        small,
        whiten,
        model,
        steps=SMALL_STEPS,
        out=out,
        log=log,
        resume=False,
        label="small",
    )
    after = _error_at(model, harness, whiten, harness.dev, harness.eval_chains)
    return {
        "ok": after <= SMALL_RATIO * before,
        "before": before,
        "after": after,
        "steps": run["steps"],
        "required_ratio": SMALL_RATIO,
    }


def export_samples(
    model,
    harness,
    whiten,
    out: Path,
    *,
    n: int,
    steps: int,
    seed: int = 7,
    shift: float = 1.0,
) -> dict:
    """``n`` samples at lengths spread over dev's: the singles, and the sequences the frozen decoder reads off them."""
    import torch

    from foldspace.esmfold2 import generator
    from foldspace.esmfold2.decode_eval import ALPHABET

    lengths = sample_lengths(harness.dev.lengths, n, *SAMPLE_LENGTHS)
    t_lengths = torch.as_tensor(lengths, device=harness.device)
    gen = torch.Generator(device=harness.device).manual_seed(seed)
    model.eval()
    singles, rows = {}, ["sample_id\tsequence\tmean_confidence"]
    for lo in range(0, n, 8):  # a few at a time: the longest of them sets the padding
        chunk = t_lengths[lo : lo + 8]
        x = generator.sample(model, chunk, steps=steps, generator=gen, shift=shift)
        z = whiten.inverse(x)
        with torch.no_grad(), _autocast(harness.device):
            logp = torch.log_softmax(harness.decoder(z, chunk).float(), dim=-1)
        for b in range(len(chunk)):
            i, ln = lo + b, int(chunk[b])
            h = (
                (z[b, :ln] * harness.norm.std + harness.norm.mean)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            top = logp[b, :ln].max(-1)
            seq = "".join(ALPHABET[int(k)] for k in logp[b, :ln].argmax(-1))
            singles[f"s{i:03d}"] = h
            rows.append(f"s{i:03d}\t{seq}\t{float(top.values.exp().mean()):.4f}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "samples.npz", **singles)
    (out / "samples.tsv").write_text("\n".join(rows) + "\n")
    return {
        "n": n,
        "lengths": lengths.tolist(),
        "steps": steps,
        "shift": shift,
        "seed": seed,
    }


def run_arm(args, name: str, config: Config, harness: Harness, out: Path, log) -> dict:
    import torch

    whiten = harness.whitening(config)
    arm_dir = out / name
    result: dict = {"config": asdict(config)}
    stages = args.stages.split(",")
    failed = None
    if "smoke" in stages:
        result["smoke"] = stage_smoke(harness, config, whiten)
        _write(arm_dir / "smoke.json", result["smoke"])
        log(
            f"  smoke: {'ok' if result['smoke']['ok'] else 'FAILED'}  {result['smoke']['parameters']} parameters"
        )
        failed = None if result["smoke"]["ok"] else "smoke"
    if failed is None and "overfit" in stages:
        result["overfit"] = stage_overfit(harness, config, whiten)
        _write(arm_dir / "overfit.json", result["overfit"])
        log(
            f"  overfit: {result['overfit']['before']:.3f} -> {result['overfit']['after']:.3f}"
        )
        failed = None if result["overfit"]["ok"] else "overfit"
    if failed is None and "small" in stages:
        result["small"] = stage_small(harness, config, whiten, arm_dir / "small", log)
        _write(arm_dir / "small.json", result["small"])
        log(
            f"  small: {result['small']['before']:.3f} -> {result['small']['after']:.3f}"
        )
        failed = None if result["small"]["ok"] else "small"
    if failed is None and "full" in stages:
        model = harness.model(config)
        run = train_steps(
            harness,
            config,
            whiten,
            model,
            steps=config.steps,
            out=arm_dir / "full",
            log=log,
        )
        result["full"] = run
        _write(arm_dir / "full.json", run)
        torch.save(
            {"model": model.state_dict(), "config": asdict(config)},
            arm_dir / "full" / "ema.pt",
        )
        result["samples"] = export_samples(
            model,
            harness,
            whiten,
            arm_dir / "samples",
            n=config.n_samples,
            steps=config.sample_steps,
        )
        _write(arm_dir / "samples" / "samples.json", result["samples"])
    result["failed"] = failed
    return result


# ------------------------------------------------------------ what a sample is


KMER = 8
#: fixed in advance
CONFIDENT = 0.7  # a direct fold is confident if the model's mean pLDDT is at least this
CONSISTENT = 0.5  # and self-consistent if AtlasFold's refold of the sequence reaches this scTM with it
SUPPORTED_SHARE = (
    0.5  # a supported arm succeeds on at least this share of what the references do
)
OVER_FLOOR = 0.25  # ... and on this much more of its samples than the shuffled-sequence floor does
MEMORISED = 0.5  # a sample shares at least this share of its 8-mers with train: it is a copy of a train chain
LETTERS = "ACDEFGHIKLMNPQRSTVWY"


def kmer_codes(sequence: str, k: int = KMER) -> np.ndarray:
    """Every ``k``-mer of a sequence as an integer (base 20), in order."""
    lookup = {c: i for i, c in enumerate(LETTERS)}
    index = np.array([lookup[c] for c in sequence], dtype=np.int64)
    if len(index) < k:
        return np.empty(0, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(index, k)
    return windows @ (20 ** np.arange(k, dtype=np.int64))


def kmer_set(sequences, k: int = KMER) -> np.ndarray:
    """The sorted distinct ``k``-mers of many sequences."""
    return np.unique(np.concatenate([kmer_codes(s, k) for s in sequences]))


def containment(sequence: str, reference: np.ndarray, k: int = KMER) -> float:
    """The share of a sequence's ``k``-mers that the sorted ``reference`` holds (1 for a copy, near 0 for another protein)."""
    codes = kmer_codes(sequence, k)
    return float(np.isin(codes, reference).mean()) if len(codes) else float("nan")


def composition(sequences) -> np.ndarray:
    counts = np.zeros(20)
    for s in sequences:
        for c in s:
            counts[LETTERS.index(c)] += 1
    return counts / counts.sum()


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """The Jensen-Shannon divergence in bits: 0 for the same composition, 1 for disjoint ones."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def kl(a, b):
        keep = a > 0
        return float((a[keep] * np.log2(a[keep] / b[keep])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def read_samples(directory: Path) -> list[dict]:
    """``samples.tsv`` of one arm: id, sequence and the decoder's mean confidence."""
    lines = (Path(directory) / "samples.tsv").read_text().splitlines()
    if lines[0].split("\t") != ["sample_id", "sequence", "mean_confidence"]:
        raise ValueError(f"{directory}: not a samples table")
    rows = []
    for line in lines[1:]:
        sid, sequence, confidence = line.split("\t")
        rows.append({"id": sid, "sequence": sequence, "confidence": float(confidence)})
    return rows


def run_fold(args) -> dict:
    """The samples of one arm folded from their own singles, beside two controls, and the panel's real chains beside them.

    *direct*: the sample's sequence with the sample's single. *mean*: the same sequence with the mean single at every position
    (nothing the generator made). *shuffled*: the same single with its positions permuted (what it made, in no order).
    *reference*: a real panel chain with its own sequence and single. No ESMC backbone is resident, so the single is the only
    prior; one file per sample, so that an allocation that ends early is resumed.
    """
    import torch

    from foldspace.esmfold2 import validate_interface
    from foldspace.esmfold2 import interface as iface
    from foldspace.esmfold2 import substrate as sub

    settings = validate_interface.configure(True)
    lite, backbone = iface.drop_backbone(
        sub.open_substrate(args.checkpoint, load_esmc=False)
    )
    dropout = validate_interface.lm_dropout_in_force(lite)
    print(f"device {lite.device}  {settings}  {dropout}  lite {backbone}", flush=True)
    if dropout["lm_dropout"] != 0.0:
        raise SystemExit(f"lm_dropout in force is {dropout['lm_dropout']}, not 0")
    samples = read_samples(args.samples)
    held = np.load(Path(args.samples) / "samples.npz", allow_pickle=False)
    singles = {k: held[k] for k in held.files}
    mean = np.load(args.normalisation, allow_pickle=False)["mean"].astype(np.float32)
    references = []
    if args.panel:
        from foldspace.esmfold2 import decoder_structure_check

        rows = decoder_structure_check.read_panel(args.panel)
        cached = validate_interface._cached_rows(Path(args.targets), [r["id"] for r in rows])
        references = [
            {"id": r["id"], "sequence": r["sequence"], "single": cached[r["id"]]}
            for r in rows
        ]
    out = Path(args.out)
    chain_dir = out / "chains"
    chain_dir.mkdir(parents=True, exist_ok=True)
    device = lite.device

    def to_tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None].to(
            device
        )

    began = time.perf_counter()
    jobs = [("sample", r["id"], r["sequence"], singles[r["id"]]) for r in samples]
    jobs += [("reference", r["id"], r["sequence"], r["single"]) for r in references]
    for kind, i, sequence, single in jobs:
        npz, meta = chain_dir / f"{i}.npz", chain_dir / f"{i}.json"
        if npz.exists() and meta.exists():
            print(f"  {i} already folded, not repeated", flush=True)
            continue
        prepared = sub.prepare_sequences(lite, sequence)
        rng = np.random.default_rng(args.seed + zlib_crc(i))
        folds = {
            "direct": iface.fold_single(
                lite, prepared, to_tensor(single), seed=args.seed
            )
        }
        if kind == "sample":
            folds["mean"] = iface.fold_single(
                lite,
                prepared,
                to_tensor(np.tile(mean, (len(sequence), 1))),
                seed=args.seed,
            )
            folds["shuffled"] = iface.fold_single(
                lite,
                prepared,
                to_tensor(single[rng.permutation(len(sequence))]),
                seed=args.seed,
            )
            # the floor: the sample's own sequence shuffled (the composition and the length kept, the order gone), folded with the
            # mean single, which says nothing; what a sequence that is not a protein gets from this model
            floor_sequence = "".join(rng.permutation(list(sequence)))
            folds["floor"] = iface.fold_single(
                lite,
                sub.prepare_sequences(lite, floor_sequence),
                to_tensor(np.tile(mean, (len(sequence), 1))),
                seed=args.seed,
            )
        tmp = chain_dir / f"{i}.tmp.npz"
        np.savez(tmp, **{f"{i}__{name}": f.ca for name, f in folds.items()})
        os.replace(tmp, npz)
        meta.write_text(
            json.dumps(
                {
                    "id": i,
                    "kind": kind,
                    "length": len(sequence),
                    "sequence": sequence,
                    "floor_sequence": floor_sequence if kind == "sample" else None,
                    "ptm": {n: f.metrics.get(validate_interface.PTM) for n, f in folds.items()},
                    "plddt": {n: f.metrics.get(validate_interface.PLDDT) for n, f in folds.items()},
                    "card": torch.cuda.get_device_name(0),
                    "slurm_job": os.environ.get("SLURM_JOB_ID"),
                },
                default=float,
            )
        )
        print(
            f"  {i} L={len(sequence)}  {len(folds)} folds  {time.perf_counter() - began:7.1f}s",
            flush=True,
        )
        del folds, prepared
        torch.cuda.empty_cache()

    records = {}
    chains = [json.loads((chain_dir / f"{i}.json").read_text()) for _, i, _, _ in jobs]
    coords = {}
    for _, i, _, _ in jobs:
        arrays = np.load(chain_dir / f"{i}.npz", allow_pickle=False)
        for key in arrays.files:  # each array once
            coords[key] = arrays[key]
    tmp = out / "fold_coords.tmp.npz"
    np.savez(tmp, **coords)
    os.replace(tmp, out / "fold_coords.npz")
    # AtlasFold refolds each sequence against the direct fold of it: its scTM is the self-consistency of the sample
    for label, kind in (("samples", "sample"), ("reference", "reference")):
        records[label] = [
            {
                "id": c["id"],
                "sequence": c["sequence"],
                "coordinates": coords[f"{c['id']}__direct"].tolist(),
            }
            for c in chains
            if c["kind"] == kind
        ]
        (out / f"records_{label}.json").write_text(json.dumps(records[label]))
    # the floor is refolded by AtlasFold like the samples: the same sequences' shuffles against their own folds
    records["floor"] = [
        {
            "id": c["id"],
            "sequence": c["floor_sequence"],
            "coordinates": coords[f"{c['id']}__floor"].tolist(),
        }
        for c in chains
        if c["kind"] == "sample"
    ]
    (out / "records_floor.json").write_text(json.dumps(records["floor"]))
    result = {
        "experiment": "foldspace.esmfold2.latent_generation.fold",
        "provenance": validate_interface._provenance(args, lite, {"mode": "generation_fold"}),
        "settings": settings,
        "lm_dropout": dropout,
        "chains": chains,
    }
    _write(out / "fold.json", result)
    print(f"wrote {out / 'fold.json'}", flush=True)
    return result


def zlib_crc(text: str) -> int:
    import zlib

    return zlib.crc32(text.encode())


def _median(values) -> float:
    values = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    return float(np.median(values)) if values else float("nan")


def summarize_arm(
    fold: dict,
    samples: list[dict],
    atlas: dict | None,
    train_kmers: np.ndarray,
    train_composition: np.ndarray,
) -> dict:
    """One arm's samples read by the pre-specified criteria.

    ``fold`` is ``fold.json``, ``samples`` the arm's ``samples.tsv``, ``atlas`` maps ``samples`` and ``reference`` to the
    ``per_target`` lists of AtlasFold's evaluations (``scTM`` of the refold against the direct fold), when they were made.
    """
    by_id = {c["id"]: c for c in fold["chains"]}
    own = [by_id[s["id"]] for s in samples]
    refs = [c for c in fold["chains"] if c["kind"] == "reference"]
    sctm = {}
    for label in ("samples", "reference", "floor"):
        if atlas and label in atlas:
            sctm[label] = {t["id"]: t["scTM"] for t in atlas[label]}
    sequences = [s["sequence"] for s in samples]
    contain = [containment(q, train_kmers) for q in sequences]
    sets = [set(kmer_codes(q).tolist()) for q in sequences]
    jaccard = [
        len(a & b) / max(1, len(a | b)) for j, a in enumerate(sets) for b in sets[:j]
    ]

    def confident(chain, route="direct"):
        got = chain["plddt"].get(route)
        return got is not None and got >= CONFIDENT

    def success(chain, table, route="direct"):
        got = table.get(chain["id"]) if table else None
        return bool(confident(chain, route) and got is not None and got >= CONSISTENT)

    own_table, ref_table = sctm.get("samples"), sctm.get("reference")
    floor_table = sctm.get("floor")
    out = {
        "samples": len(samples),
        "sequence": {
            "mean_confidence": _median(s["confidence"] for s in samples),
            "composition_jsd_from_train": jensen_shannon(
                composition(sequences), train_composition
            ),
            "eight_mer_containment_median": _median(contain),
            "share_memorised": float(
                np.mean([c >= MEMORISED for c in contain if not np.isnan(c)])
            ),
            "mean_pairwise_eight_mer_jaccard": float(np.mean(jaccard))
            if jaccard
            else float("nan"),
        },
        "direct_fold": {
            "plddt_median": _median(c["plddt"]["direct"] for c in own),
            "ptm_median": _median(c["ptm"]["direct"] for c in own),
            "share_confident": float(np.mean([confident(c) for c in own])),
        },
        "controls": {
            route: {
                "plddt_median": _median(c["plddt"][route] for c in own),
                "share_confident": float(np.mean([confident(c, route) for c in own])),
            }
            for route in ("mean", "shuffled")
        },
        "floor": {
            "plddt_median": _median(c["plddt"].get("floor") for c in own),
            "share_confident": float(np.mean([confident(c, "floor") for c in own])),
        },
        "reference": {
            "chains": len(refs),
            "plddt_median": _median(c["plddt"]["direct"] for c in refs),
            "share_confident": float(np.mean([confident(c) for c in refs]))
            if refs
            else float("nan"),
        },
    }
    if own_table is not None:
        out["self_consistency"] = {
            "samples_sctm_median": _median(own_table.get(c["id"]) for c in own),
            "samples_share_consistent": float(
                np.mean([(own_table.get(c["id"]) or 0.0) >= CONSISTENT for c in own])
            ),
        }
        out["success_share"] = float(np.mean([success(c, own_table) for c in own]))
        if ref_table is not None and refs:
            out["self_consistency"]["reference_sctm_median"] = _median(
                ref_table.get(c["id"]) for c in refs
            )
            out["reference"]["success_share"] = float(
                np.mean([success(c, ref_table) for c in refs])
            )
        if floor_table is not None:
            out["floor"]["success_share"] = float(
                np.mean([success(c, floor_table, "floor") for c in own])
            )
        ref_share = out["reference"].get("success_share", float("nan"))
        floor_share = out["floor"].get("success_share", float("nan"))
        known = not (np.isnan(ref_share) or np.isnan(floor_share))
        out["verdict"] = {
            "success_share": out["success_share"],
            "reference_success_share": ref_share,
            "floor_success_share": floor_share,
            "supported": bool(
                out["success_share"] >= SUPPORTED_SHARE * ref_share
                and out["success_share"] >= floor_share + OVER_FLOOR
            )
            if known
            else None,
        }
    return out


def run_summarize(args) -> dict:
    from foldspace.esmfold2 import sequences as seqs

    train_sequences = [s for _, s in seqs.read_table(args.train_table)]
    train_kmers, train_composition = (
        kmer_set(train_sequences),
        composition(train_sequences),
    )
    summary = {"arms": {}}
    for directory in args.arm:
        directory = Path(directory)
        fold = json.loads((directory / args.fold_dir / "fold.json").read_text())
        atlas = {}
        for label in ("samples", "reference", "floor"):
            path = directory / args.fold_dir / f"atlas_{label}.json"
            if path.exists():
                atlas[label] = json.loads(path.read_text())["per_target"]
        samples = read_samples(directory / args.samples_dir)
        summary["arms"][directory.name] = summarize_arm(
            fold, samples, atlas or None, train_kmers, train_composition
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    for name, arm in summary["arms"].items():
        print(
            f"{name:9s} success {arm.get('success_share', float('nan')):.2f}  confident {arm['direct_fold']['share_confident']:.2f}"
            f"  pLDDT {arm['direct_fold']['plddt_median']:.2f}  memorised {arm['sequence']['share_memorised']:.2f}"
            f"  supported {arm.get('verdict', {}).get('supported')}"
        )
    return summary


# --------------------------------------------- a diagnostic: complete a real single

PARTIAL_T0 = (0.5, 0.7, 0.9)


def run_partial(args) -> dict:
    """What a trained arm does with a real single that is only partly noised: the generator completes it.

    For each chain of a panel and each ``t0``, the chain's real single in the arm's coordinate is noised to ``t0`` (the same noise
    a training example at ``t0`` has) and the Heun steps of the sampler run from there to ``t = 1``; the frozen decoder reads the
    result. The identity of the read sequence with the real one, and (with AtlasFold, against the deposited trace) whether it folds,
    say how local the generator's competence is: a generator that completes a single from ``t0 = 0.7`` and cannot make one from
    nothing is missing global coherence, not the local structure of the coordinate. Not one of the pre-specified readings.
    """
    import torch

    from foldspace.esmfold2 import batching, validate_interface, generator
    from foldspace.esmfold2 import decoder_structure_check
    from foldspace.esmfold2 import perturb as P
    from foldspace.esmfold2.decode_eval import ALPHABET
    from foldspace.esmfold2.decoder_data import gather
    from foldspace.evaluation.provenance import capture

    spectrum = P.spectrum_from_covariance(
        np.load(args.covariance, allow_pickle=False)["covariance"]
    )
    harness = Harness(
        cache=args.cache,
        chains=args.chains,
        train_table=args.train_table,
        device=args.device,
        spectrum=spectrum,
        decoder_path=args.decoder,
    )
    arm_dir = Path(args.arm_dir)
    state = torch.load(
        arm_dir / "full" / "ema.pt", map_location=harness.device, weights_only=True
    )
    config = Config(**state["config"])
    model = harness.model(config)
    model.load_state_dict(state["model"])
    model.eval()
    whiten = harness.whitening(config)
    rows = decoder_structure_check.read_panel(args.panel)
    index_of = {c: k for k, c in enumerate(harness.dev.ids)}
    chains = np.array([index_of[r["id"]] for r in rows])
    deposited = {c["id"]: c["ca"] for c in validate_interface.corpus_chains(args.chains, "dev")}
    gen = torch.Generator(device=harness.device).manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {"arm": arm_dir.name, "steps": args.steps, "t0": {}}
    for t0 in [float(v) for v in str(args.t0).split(",")]:
        records, table = [], []
        with torch.no_grad():
            for batch in batching.pack(chains, harness.dev.lengths, 65536):
                z, _, lens = gather(harness.dev, batch, harness.norm)
                x_t, _, _ = generator.flow_inputs(
                    whiten.forward(z),
                    lens,
                    gen,
                    t=torch.full((len(batch),), t0, device=harness.device),
                )
                x = generator.sample(
                    model, lens, steps=args.steps, start=x_t, t_start=t0
                )
                with _autocast(harness.device):
                    logp = torch.log_softmax(
                        harness.decoder(whiten.inverse(x), lens).float(), dim=-1
                    )
                for b, chain in enumerate(batch):
                    n = int(lens[b])
                    true = harness.dev.sequence(int(chain))
                    top = logp[b, :n].max(-1)
                    decoded = "".join(ALPHABET[int(k)] for k in top.indices)
                    ident = float(
                        np.mean([a == c for a, c in zip(true, decoded, strict=True)])
                    )
                    cid = harness.dev.ids[int(chain)]
                    table.append(
                        {
                            "id": cid,
                            "length": n,
                            "identity": ident,
                            "mean_confidence": float(top.values.exp().mean()),
                            "decoded": decoded,
                        }
                    )
                    records.append(
                        {
                            "id": cid,
                            "sequence": decoded,
                            "coordinates": deposited[cid].tolist(),
                        }
                    )
        result["t0"][f"{t0:g}"] = table
        (out / f"records_t{t0:g}.json").write_text(json.dumps(records))
        print(
            f"  t0 {t0:g}: median identity {np.median([r['identity'] for r in table]):.3f}",
            flush=True,
        )
    result["provenance"] = capture({"allow_dirty": bool(args.allow_dirty)})
    _write(out / "partial.json", result)
    return result


def summarize_partial(
    partial: dict, atlas: dict | None = None, reference: dict | None = None
) -> dict:
    """Per ``t0``: the identity of the completed sequence with the real one and, with AtlasFold's evaluations (``atlas`` maps
    ``t0`` to id -> scTM of the refold against the deposited trace), whether it folds, beside the real sequences' (``reference``)."""
    out = {}
    for t0, table in partial["t0"].items():
        ident = [r["identity"] for r in table]
        row = {
            "chains": len(table),
            "identity_median": float(np.median(ident)),
            "identity_share_over_0.9": float(np.mean([v >= 0.9 for v in ident])),
            "mean_confidence_median": float(
                np.median([r["mean_confidence"] for r in table])
            ),
        }
        got = (atlas or {}).get(t0)
        if got:
            sctm = [got[r["id"]] for r in table if r["id"] in got]
            row["sctm_median"] = float(np.median(sctm))
            row["sctm_share_over_0.5"] = float(np.mean([v >= 0.5 for v in sctm]))
        out[t0] = row
    if reference:
        out["real_sequences"] = {
            "sctm_median": float(np.median(list(reference.values())))
        }
    return out


def run_sample(args) -> dict:
    """Samples from a trained arm at a chosen number of steps and time shift, into a directory of their own.

    The pilot's samples were made with 50 Heun steps on an even grid. The partial diagnostic found foldability to turn on between
    ``t0`` of 0.1 and 0.2, and an even grid spends about five steps there; this makes the same samples (the same lengths and the
    same seed) with more steps and a grid denser near ``t = 0``, with no retraining, for the fold step to read as before.
    """
    import torch

    from foldspace.esmfold2 import perturb as P

    spectrum = P.spectrum_from_covariance(
        np.load(args.covariance, allow_pickle=False)["covariance"]
    )
    harness = Harness(
        cache=args.cache,
        chains=args.chains,
        train_table=args.train_table,
        device=args.device,
        spectrum=spectrum,
        decoder_path=args.decoder,
    )
    arm_dir = Path(args.arm_dir)
    state = torch.load(
        arm_dir / "full" / "ema.pt", map_location=harness.device, weights_only=True
    )
    config = Config(**state["config"])
    model = harness.model(config)
    model.load_state_dict(state["model"])
    model.eval()
    info = export_samples(
        model,
        harness,
        harness.whitening(config),
        Path(args.out),
        n=args.n,
        steps=args.steps,
        seed=args.seed,
        shift=args.shift,
    )
    _write(Path(args.out) / "samples.json", info)
    print(
        f"  {arm_dir.name}: {args.n} samples, {args.steps} steps, shift {args.shift:g}",
        flush=True,
    )
    return info


def run_partial_summary(args) -> dict:
    """``summarize_partial`` for each arm directory that has a ``partial/`` with AtlasFold's evaluations, written as one file."""
    reference = None
    if args.reference_atlas:
        reference = {
            t["id"]: t["scTM"]
            for t in json.loads(Path(args.reference_atlas).read_text())["per_target"]
        }
    out = {}
    for directory in args.arm_dir:
        directory = Path(directory)
        partial = json.loads((directory / args.subdir / "partial.json").read_text())
        atlas = {}
        for t0 in partial["t0"]:
            path = directory / args.subdir / f"atlas_t{t0}.json"
            if path.exists():
                atlas[t0] = {
                    t["id"]: t["scTM"]
                    for t in json.loads(path.read_text())["per_target"]
                }
        out[directory.name] = summarize_partial(partial, atlas or None, reference)
    _write(Path(args.out), out)
    for arm, rows in out.items():
        for t0, row in rows.items():
            if t0 == "real_sequences":
                print(f"{arm:9s} real sequences  scTM median {row['sctm_median']:.3f}")
            else:
                print(
                    f"{arm:9s} t0 {t0:>4s}  identity median {row['identity_median']:.3f}"
                    f"  scTM median {row.get('sctm_median', float('nan')):.3f}"
                )
    return out


def run_train(args) -> int:
    from foldspace.esmfold2 import perturb as P
    from foldspace.evaluation.provenance import capture

    out = Path(args.out)
    began = time.perf_counter()

    def log(*a):
        print(*a, flush=True)

    spectrum = P.spectrum_from_covariance(
        np.load(args.covariance, allow_pickle=False)["covariance"]
    )
    harness = Harness(
        cache=args.cache,
        chains=args.chains,
        train_table=args.train_table,
        device=args.device,
        spectrum=spectrum,
        decoder_path=args.decoder,
        limit_chains=args.limit_chains,
    )
    log(
        f"device {harness.device}  train {harness.train.n_chains} chains  dev {harness.dev.n_chains} chains "
        f"({len(harness.eval_chains)} read)  loaded in {time.perf_counter() - began:.0f}s"
    )
    overrides = parse_overrides(args.set)
    summary: dict = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "provenance": capture(
            {"allow_dirty": bool(args.allow_dirty), "device": str(args.device)}
        ),
        "normalisation_digest": harness.norm.digest,
        "arms": {},
    }
    for name in [a for a in args.arms.split(",") if a]:
        config = replace(ARMS[name], **overrides)
        log(f"\n=== {name} ({config.coordinate}, ce {config.ce_weight})")
        summary["arms"][name] = run_arm(args, name, config, harness, out, log)
        _write(out / "generation_summary.json", summary)
    failed = {k: v["failed"] for k, v in summary["arms"].items() if v["failed"]}
    log(f"\ndone in {time.perf_counter() - began:.0f}s; failed stages: {failed}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub_ = parser.add_subparsers(dest="command", required=True)
    t = sub_.add_parser("train")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--cache", type=Path, default=Path("cache"))
    t.add_argument("--chains", type=Path, default=Path("data/corpus/chains.npz"))
    t.add_argument(
        "--train-table",
        type=Path,
        default=Path("data/corpus/train_sequences.tsv"),
    )
    t.add_argument(
        "--covariance",
        type=Path,
        default=Path("cache/esmfold2/model_single_train_covariance.npz"),
    )
    t.add_argument(
        "--decoder",
        type=Path,
        default=Path("cache/esmfold2/decoders/contextual-s0.best.pt"),
    )
    t.add_argument("--arms", default="standardized,whitened,whitened_with_sequence_loss")
    t.add_argument("--stages", default="smoke,overfit,small,full")
    t.add_argument("--set", default="")
    t.add_argument("--limit-chains", type=int, default=0)
    t.add_argument("--allow-dirty", action="store_true")
    f = sub_.add_parser("fold")
    f.add_argument("--samples", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--panel", type=Path, default=None)
    f.add_argument("--checkpoint", default="ESMFold2")
    f.add_argument("--targets", type=Path, default=Path("cache/esmfold2/targets/dev"))
    f.add_argument(
        "--normalisation",
        type=Path,
        default=Path("cache/esmfold2/model_single_train_normalisation.npz"),
    )
    f.add_argument("--seed", type=int, default=20261122)
    f.add_argument("--allow-dirty", action="store_true")
    p = sub_.add_parser("partial")
    p.add_argument("--arm-dir", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--chains", type=Path, default=Path("data/corpus/chains.npz"))
    p.add_argument(
        "--train-table",
        type=Path,
        default=Path("data/corpus/train_sequences.tsv"),
    )
    p.add_argument(
        "--covariance",
        type=Path,
        default=Path("cache/esmfold2/model_single_train_covariance.npz"),
    )
    p.add_argument(
        "--decoder",
        type=Path,
        default=Path("cache/esmfold2/decoders/contextual-s0.best.pt"),
    )
    p.add_argument("--t0", default=",".join(f"{v:g}" for v in PARTIAL_T0))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--allow-dirty", action="store_true")
    sm = sub_.add_parser("sample")
    sm.add_argument("--arm-dir", type=Path, required=True)
    sm.add_argument("--out", type=Path, required=True)
    sm.add_argument("--device", default="cuda")
    sm.add_argument("--cache", type=Path, default=Path("cache"))
    sm.add_argument("--chains", type=Path, default=Path("data/corpus/chains.npz"))
    sm.add_argument(
        "--train-table",
        type=Path,
        default=Path("data/corpus/train_sequences.tsv"),
    )
    sm.add_argument(
        "--covariance",
        type=Path,
        default=Path("cache/esmfold2/model_single_train_covariance.npz"),
    )
    sm.add_argument(
        "--decoder",
        type=Path,
        default=Path("cache/esmfold2/decoders/contextual-s0.best.pt"),
    )
    sm.add_argument("--n", type=int, default=N_SAMPLES)
    sm.add_argument("--steps", type=int, default=SAMPLE_STEPS)
    sm.add_argument("--shift", type=float, default=1.0)
    sm.add_argument("--seed", type=int, default=7)
    sm.add_argument("--allow-dirty", action="store_true")
    ps = sub_.add_parser("partial-summary")
    ps.add_argument("--arm-dir", type=Path, action="append", required=True)
    ps.add_argument("--reference-atlas", type=Path, default=None)
    ps.add_argument("--subdir", default="partial")
    ps.add_argument("--out", type=Path, required=True)
    m = sub_.add_parser("summarize")
    m.add_argument("--arm", type=Path, action="append", required=True)
    m.add_argument("--samples-dir", default="samples")
    m.add_argument("--fold-dir", default="fold")
    m.add_argument(
        "--train-table",
        type=Path,
        default=Path("data/corpus/train_sequences.tsv"),
    )
    m.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    if args.command not in ("summarize", "partial-summary") and not args.allow_dirty:
        from foldspace.evaluation.provenance import require_clean

        require_clean()
    result = {
        "train": run_train,
        "fold": run_fold,
        "partial": run_partial,
        "sample": run_sample,
        "partial-summary": run_partial_summary,
        "summarize": run_summarize,
    }[args.command](args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
