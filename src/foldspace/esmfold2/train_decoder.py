"""Training a sequence head on the standardised singles, and reading it back.

One loop for both decoders. An epoch is as many draws as there are train chains, drawn *a cluster uniformly, then
a chain of it* (or chain-uniform, the one ablation), packed into batches of whole chains up to a padded-token budget.
The model is chosen by the **cluster-weighted cross-entropy on the dev split**, evaluated after every epoch: the
dev split is 400 clusters and one holds 30.5 % of its chains, so a micro figure is that family's. The best
checkpoint is written as it is found, and reading it back must reproduce what was recorded (``restore_check``).

Nothing here looks at the test split.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from foldspace.esmfold2 import batching, decode_eval
from foldspace.esmfold2.decoder import build_decoder, parameter_count
from foldspace.esmfold2.decoder_data import IGNORE, Normaliser, Split, gather

__all__ = [
    "Config",
    "Predictions",
    "fit",
    "load_checkpoint",
    "predict",
    "restore_check",
    "save_checkpoint",
    "selection",
]


@dataclass
class Config:
    arch: str = "contextual"
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    max_tokens: int = 32768
    epochs: int = 40
    patience: int = 8
    clip: float = 1.0
    sampling: str = "cluster"
    seed: int = 0


@dataclass
class Predictions:
    """What a decoder says about a set of whole chains, rows back to back in the order asked."""

    logp: np.ndarray
    labels: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    clusters: np.ndarray
    chains: np.ndarray

    def summary(self) -> dict:
        return decode_eval.summarize(
            self.logp, self.labels, self.starts, self.lengths, self.clusters
        )

    def selection(self) -> dict:
        return decode_eval.selection_stats(
            self.logp, self.labels, self.starts, self.lengths, self.clusters
        )


def _autocast(device):
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(
    model,
    split: Split,
    norm: Normaliser,
    chain_indices: np.ndarray | None = None,
    *,
    max_tokens: int = 65536,
    transform=None,
) -> Predictions:
    """Log-probabilities for the chains asked for (all of the split by default), in that order.

    ``transform(x, batch, lengths)``, when given, is applied to each standardised batch before the model reads it
    (robustness perturbs the singles this way); it returns an array of the same shape whose padding is still zero.
    """
    model.eval()
    asked = (
        np.arange(split.n_chains)
        if chain_indices is None
        else np.asarray(chain_indices, dtype=np.int64)
    )
    lengths = split.lengths[asked]
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    logp = np.empty((int(lengths.sum()), 20), dtype=np.float32)
    labels = np.empty(int(lengths.sum()), dtype=np.int64)
    where = np.empty(split.n_chains, dtype=np.int64)
    where[asked] = np.arange(len(asked))
    device = split.z.device
    for batch in batching.pack(asked, split.lengths, max_tokens):
        x, _, lens = gather(split, batch, norm)
        if transform is not None:
            x = transform(x, batch, lens)
        with _autocast(device):
            logits = model(x, lens)
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        for b, chain in enumerate(batch):
            j = int(where[chain])
            n = int(split.lengths[chain])
            logp[starts[j] : starts[j] + n] = lp[b, :n]
            s = int(split.starts[chain])
            labels[starts[j] : starts[j] + n] = split.labels_np[s : s + n]
    return Predictions(logp, labels, starts, lengths, split.clusters[asked], asked)


def selection(model, split: Split, norm: Normaliser, **kwargs) -> dict:
    return predict(model, split, norm, **kwargs).selection()


def lr_at(step: int, total: int, config: Config) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, total - config.warmup_steps)
    floor = config.min_lr_ratio
    return config.lr * (
        floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    )


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def save_checkpoint(
    path: Path, model, config: Config, *, epoch: int, dev: dict, norm_digest: str
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "dev": dev,
            "norm_digest": norm_digest,
            "parameters": parameter_count(model),
        },
        path,
    )
    return file_digest(path)


def load_checkpoint(path: Path, device):
    """``(model, config, meta)``: the model rebuilt from its own config, in eval mode."""
    held = torch.load(path, map_location=device, weights_only=True)
    config = Config(**held["config"])
    model = build_decoder(asdict(config)).to(device)
    model.load_state_dict(held["model"])
    model.eval()
    return (
        model,
        config,
        {k: held[k] for k in ("epoch", "dev", "norm_digest", "parameters")},
    )


def restore_check(
    path: Path,
    split: Split,
    norm: Normaliser,
    recorded: dict,
    *,
    tolerance: float = 1e-3,
) -> dict:
    """Read a checkpoint back and score it again: what it gives must be what was recorded."""
    model, _, meta = load_checkpoint(path, split.z.device)
    again = selection(model, split, norm)
    return {
        "recorded": recorded,
        "again": again,
        "cw_ce_difference": abs(again["cw_ce"] - recorded["cw_ce"]),
        "cw_acc_difference": abs(again["cw_acc"] - recorded["cw_acc"]),
        "restored": abs(again["cw_ce"] - recorded["cw_ce"]) <= tolerance
        and abs(again["cw_acc"] - recorded["cw_acc"]) <= tolerance,
        "norm_digest_matches": meta["norm_digest"] == norm.digest,
    }


def fit(
    model,
    train: Split,
    dev: Split,
    norm: Normaliser,
    config: Config,
    out_dir: Path,
    *,
    epochs: int | None = None,
    log=print,
) -> dict:
    """Train; keep the epoch with the lowest cluster-weighted dev cross-entropy as ``best.pt``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = train.z.device
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed + 1000)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay,
    )
    probabilities = (
        batching.cluster_probabilities(train.clusters)
        if config.sampling == "cluster"
        else None
    )
    epochs = epochs or config.epochs
    n = train.n_chains

    def epoch_batches():
        drawn = batching.draw(n, n, rng, probabilities)
        return batching.pack(drawn, train.lengths, config.max_tokens, rng)

    total_steps = epochs * len(epoch_batches())
    history: list[dict] = []
    best: dict | None = None
    best_digest = None
    stale = 0
    step = 0
    began = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        tokens = 0
        for batch in epoch_batches():
            for group in optimiser.param_groups:
                group["lr"] = lr_at(step, total_steps, config)
            x, y, lengths = gather(train, batch, norm)
            with _autocast(device):
                logits = model(x, lengths)
            loss = F.cross_entropy(
                logits.float().reshape(-1, 20), y.reshape(-1), ignore_index=IGNORE
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            if not torch.isfinite(grad):
                raise FloatingPointError(f"step {step}: gradient norm {float(grad)}")
            optimiser.step()
            n_tokens = int(lengths.sum())
            loss_sum += loss.detach() * n_tokens
            tokens += n_tokens
            step += 1
        stats = selection(model, dev, norm)
        row = {
            "epoch": epoch,
            "steps": step,
            "train_loss": float(loss_sum) / tokens,
            **stats,
            "seconds": time.perf_counter() - began,
        }
        history.append(row)
        if not np.isfinite(row["train_loss"]):
            raise FloatingPointError(f"epoch {epoch}: train loss {row['train_loss']}")
        if best is None or stats["cw_ce"] < best["cw_ce"] - 1e-6:
            best = {"epoch": epoch, **stats}
            best_digest = save_checkpoint(
                out_dir / "best.pt",
                model,
                config,
                epoch=epoch,
                dev=stats,
                norm_digest=norm.digest,
            )
            stale = 0
        else:
            stale += 1
        log(
            f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  dev cw ce {stats['cw_ce']:.4f} "
            f"acc {stats['cw_acc']:.4f} (micro {stats['micro_acc']:.4f})  {row['seconds']:.0f}s"
            f"{'  *' if stale == 0 else ''}",
        )
        if stale >= config.patience:
            log(f"  no improvement in {config.patience} epochs; stopping")
            break
    save_checkpoint(
        out_dir / "last.pt",
        model,
        config,
        epoch=epoch,
        dev=stats,
        norm_digest=norm.digest,
    )
    return {
        "config": asdict(config),
        "parameters": parameter_count(model),
        "steps": step,
        "history": history,
        "best": best,
        "best_digest": best_digest,
        "seconds": time.perf_counter() - began,
    }
