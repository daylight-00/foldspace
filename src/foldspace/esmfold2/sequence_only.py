"""A sequence-only generator of the pilot's size: an autoregressive Transformer on residues.

What it is for: the pilot generator made sequences that fold no better than shuffles, and could not say whether that is the
coordinate or the scale, because nothing of its size on its data made sequences without the single. This is that: the same corpus,
the same cluster-uniform draws of whole chains in the same padded-token batches, a model of the same size, and a sequence read by
the same ruler (``foldspace.esmfold2.seqeval``). It is a control chosen for being the simplest, not a candidate.

It is given a length and a seed. The first token of a chain says its length in bins of ten, the residues follow, and generating a
chain of ``L`` residues is ``L`` draws, so that a set of samples can be made at the lengths of the pilot's own.

What differs from the pilot is the objective (the likelihood of the residues), the regularisation (dropout and weight decay: a
likelihood model of this size overfits 3 million residues long before 20,000 steps) and the stopping: the cluster-weighted dev
cross-entropy is read every ``eval_every`` steps, the best checkpoint is kept, and the run stops after ``patience`` readings
without a better one. The four budget axes (updates; example slots and de-duplicated presentations; valid residues and the sum of
``L^2``; GPU-hours) are recorded for the checkpoint kept and for the run, and none is claimed to be matched to the pilot's. The test
split is not read.

Commands: ``train`` (a corpus, a seed, its checkpoints and history) and ``sample`` (a set of sequences at the pilot's lengths, at a
temperature, from a checkpoint).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from foldspace.esmfold2 import batching, latent_generation

__all__ = [
    "Budget",
    "Corpus",
    "LMConfig",
    "cluster_weighted",
    "lr_at",
    "make_batch",
    "ngram_nll",
    "parse_overrides",
]

LETTERS = latent_generation.LETTERS  # "ACDEFGHIKLMNPQRSTVWY": a residue's token is its index here
PAD = 20
LEN_BIN = 10  # residues per length token
LEN_BINS = 41  # 0 to 400
LEN_BASE = 21  # the first length token
VOCAB = LEN_BASE + LEN_BINS
IGNORE = -100


@dataclass
class LMConfig:
    d_model: int = 640
    n_layers: int = 8
    n_heads: int = 10
    d_ff: int = 2560
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    clip: float = 1.0
    max_tokens: int = 32768
    steps: int = 20000
    eval_every: int = 100
    patience: int = 12
    max_len: int = 320
    seed: int = 0
    select: str = "dev"  # what chooses the checkpoint: "dev" (the PDB dev chains) or "heldout" (a slice of the corpus itself)
    heldout_clusters: int = (
        0  # clusters of the training corpus held out, for select = heldout
    )


def parse_overrides(text: str) -> dict:
    """``key=value,...`` against ``LMConfig``'s fields, each value in the field's own type."""
    fields = {f: type(getattr(LMConfig(), f)) for f in asdict(LMConfig())}
    out = {}
    for item in [p for p in text.split(",") if p]:
        key, _, value = item.partition("=")
        if key not in fields:
            raise ValueError(f"{key!r} is not a field of LMConfig")
        out[key] = fields[key](value)
    return out


def lr_at(step: int, total: int, cfg: LMConfig) -> float:
    """Linear warm-up, then a cosine to ``min_lr_ratio`` of the peak at ``total`` steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


# --------------------------------------------------------------------- data


@dataclass
class Corpus:
    """Whole chains as residue codes, with their clusters, back to back."""

    ids: list[str]
    lengths: np.ndarray
    clusters: np.ndarray
    flat: np.ndarray
    starts: np.ndarray

    @property
    def n_chains(self) -> int:
        return len(self.ids)

    def sequence(self, i: int) -> str:
        s, n = int(self.starts[i]), int(self.lengths[i])
        return "".join(LETTERS[int(c)] for c in self.flat[s : s + n])


def encode(sequence: str) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(LETTERS)}
    return np.array([lookup[c] for c in sequence], dtype=np.int64)


def corpus_from(chains: list[dict]) -> Corpus:
    """A corpus from rows with ``id``, ``sequence`` and ``cluster``."""
    codes = [encode(c["sequence"]) for c in chains]
    lengths = np.array([len(x) for x in codes], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    return Corpus(
        ids=[str(c["id"]) for c in chains],
        lengths=lengths,
        clusters=np.array([int(c["cluster"]) for c in chains], dtype=np.int64),
        flat=np.concatenate(codes) if codes else np.empty(0, dtype=np.int64),
        starts=starts,
    )


def load_split(chains_npz: Path, split: str) -> Corpus:
    """One split of the corpus (``data/corpus/chains.npz``): its chains, sequences and 30 % clusters."""
    from foldspace.esmfold2 import validate_interface

    return corpus_from(validate_interface.corpus_chains(Path(chains_npz), split))


def load_table(path: Path) -> Corpus:
    """A corpus table: ``seq_id <TAB> sequence <TAB> cluster``, as ``foldspace.esmfold2.corpus`` writes it."""
    lines = Path(path).read_text().splitlines()
    if lines[0].split("\t") != ["seq_id", "sequence", "cluster"]:
        raise ValueError(f"{path}: not a corpus table")
    rows = []
    for line in lines[1:]:
        if line:
            i, s, c = line.split("\t")
            rows.append({"id": i, "sequence": s, "cluster": int(c)})
    return corpus_from(rows)


HOLDOUT_SEED = 20261125  # the slice held out of a corpus is the same for every seed


def subset(corpus: Corpus, keep: np.ndarray) -> Corpus:
    """The chains of a corpus at the indices ``keep``, in that order."""
    keep = np.asarray(keep, dtype=np.int64)
    lengths = corpus.lengths[keep]
    pieces = [
        corpus.flat[corpus.starts[i] : corpus.starts[i] + corpus.lengths[i]]
        for i in keep
    ]
    return Corpus(
        ids=[corpus.ids[i] for i in keep],
        lengths=lengths,
        clusters=corpus.clusters[keep],
        flat=np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64),
        starts=np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64),
    )


def split_heldout(
    corpus: Corpus, n_clusters: int, seed: int = HOLDOUT_SEED
) -> tuple[Corpus, Corpus]:
    """``(train, heldout)``: ``n_clusters`` whole clusters taken out, so that the held-out chains share no cluster with what is left."""
    unique = np.unique(corpus.clusters)
    if not 0 < n_clusters < len(unique):
        raise ValueError(f"{n_clusters} clusters of {len(unique)} cannot be held out")
    chosen = np.random.default_rng(seed).choice(unique, size=n_clusters, replace=False)
    inside = np.isin(corpus.clusters, chosen)
    return subset(corpus, np.flatnonzero(~inside)), subset(
        corpus, np.flatnonzero(inside)
    )


def dev_lengths(chains_npz: Path) -> np.ndarray:
    """The lengths of the dev split's chains, each array read once."""
    held = np.load(chains_npz, allow_pickle=False)
    splits, lengths = held["splits"], held["lengths"]
    return lengths[splits == "dev"]


def make_batch(corpus: Corpus, indices: np.ndarray):
    """Whole chains right-padded: ``(inputs, targets)``, ``[n, T]``.

    The input of a chain of length ``L`` is its length token and its first ``L - 1`` residues; the target is its ``L``
    residues. Padding is on the right, so with causal attention no real position sees it, and the loss ignores it.
    """
    indices = np.asarray(indices, dtype=np.int64)
    lengths = corpus.lengths[indices]
    t = int(lengths.max())
    position = np.arange(t)[None, :]
    mask = position < lengths[:, None]
    gather = np.minimum(
        corpus.starts[indices][:, None] + position, len(corpus.flat) - 1
    )
    codes = np.where(mask, corpus.flat[gather], PAD)
    inputs = np.empty_like(codes)
    inputs[:, 0] = LEN_BASE + np.minimum(lengths // LEN_BIN, LEN_BINS - 1)
    inputs[:, 1:] = codes[:, :-1]
    targets = np.where(mask, codes, IGNORE)
    return inputs, targets


def cluster_weighted(chain_mean: np.ndarray, clusters: np.ndarray) -> float:
    """The mean over clusters of the mean over each cluster's chains: the weighting of the split's clusters, not of its chains."""
    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=chain_mean)
    counts = np.bincount(inverse)
    return float((sums / counts).mean())


def ngram_nll(train: Corpus, dev: Corpus, order: int, alpha: float = 0.5) -> dict:
    """What a model with no more than the composition (order 0) or the previous residue (order 1) reads of held-out clusters.

    A Markov model of ``order`` 0 or 1 with add-``alpha`` smoothing, its counts drawn from the train chains with the weight the
    sequence-only model's draws give them (a cluster uniformly, then a chain of it), and scored as the model is: nats per residue,
    cluster-weighted over the dev chains. It is the floor a sequence model of any size is read against: a likelihood close to it has
    learnt the composition and almost nothing of the context.
    """
    if order not in (0, 1):
        raise ValueError(f"order is 0 or 1, not {order}")
    start = 20

    def positions(corpus: Corpus):
        previous = np.empty_like(corpus.flat)
        previous[1:] = corpus.flat[:-1]
        previous[corpus.starts] = start
        chain = np.repeat(np.arange(corpus.n_chains), corpus.lengths)
        return corpus.flat, (previous if order else np.zeros_like(previous)), chain

    following, previous, chain = positions(train)
    weight = batching.cluster_probabilities(train.clusters)
    counts = np.bincount(
        previous * 20 + following,
        weights=weight[chain] * len(weight),
        minlength=21 * 20,
    ).reshape(21, 20)
    table = counts + alpha
    logp = np.log(table / table.sum(axis=1, keepdims=True))
    following, previous, chain = positions(dev)
    chain_sum = np.bincount(
        chain, weights=-logp[previous, following], minlength=dev.n_chains
    )
    cw = cluster_weighted(chain_sum / dev.lengths, dev.clusters)
    return {
        "order": order,
        "alpha": alpha,
        "cw_nll": cw,
        "cw_perplexity": math.exp(cw),
        "micro_nll": float(chain_sum.sum() / dev.lengths.sum()),
    }


# ------------------------------------------------------------------- budget


class Budget:
    """The four axes of a run (updates; slots and de-duplicated presentations; residues and ``L^2``; GPU-hours)."""

    def __init__(self, n_chains: int):
        self.seen = np.zeros(n_chains, dtype=bool)
        self.updates = 0
        self.slots = 0
        self.residues = 0
        self.padded = 0
        self.sum_l2 = 0
        self.seconds = 0.0

    def add(self, indices: np.ndarray, lengths: np.ndarray) -> None:
        n = lengths[indices].astype(np.int64)
        self.updates += 1
        self.slots += len(indices)
        self.residues += int(n.sum())
        self.padded += int(n.max()) * len(indices)
        self.sum_l2 += int((n**2).sum())
        self.seen[indices] = True

    def snapshot(self) -> dict:
        return {
            "updates": self.updates,
            "example_slots": self.slots,
            "presentations_unique": int(self.seen.sum()),
            "valid_residues": self.residues,
            "padded_tokens": self.padded,
            "sum_L2": self.sum_l2,
            "gpu_hours": self.seconds / 3600.0,
            "gpus": 1,
        }


# ------------------------------------------------------------------- model


def build_model(cfg: LMConfig):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d_model
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.ff1, self.ff2 = nn.Linear(d, cfg.d_ff), nn.Linear(cfg.d_ff, d)

        def forward(self, x):
            b, t, d = x.shape
            h = cfg.n_heads
            q, k, v = (
                self.qkv(self.n1(x)).view(b, t, 3, h, d // h).permute(2, 0, 3, 1, 4)
            )
            a = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=cfg.dropout if self.training else 0.0
            )
            x = x + F.dropout(
                self.proj(a.transpose(1, 2).reshape(b, t, d)),
                cfg.dropout,
                self.training,
            )
            return x + F.dropout(
                self.ff2(F.gelu(self.ff1(self.n2(x)))), cfg.dropout, self.training
            )

    class SequenceLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, cfg.d_model)
            self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layers)])
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, 20, bias=False)
            self.apply(self._init)
            scale = 1.0 / math.sqrt(2 * cfg.n_layers)
            for block in self.blocks:  # the residual branches start small
                block.proj.weight.data.mul_(scale)
                block.ff2.weight.data.mul_(scale)

        @staticmethod
        def _init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        def forward(self, ids):
            x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
            x = F.dropout(x, cfg.dropout, self.training)
            for block in self.blocks:
                x = block(x)
            return self.head(self.norm(x))

    return SequenceLM()


def parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _autocast(device):
    import torch

    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def evaluate(model, corpus: Corpus, max_tokens: int, device) -> dict:
    """The cross-entropy of every chain of a corpus, in nats per residue: cluster-weighted first, and the residue-weighted mean."""
    import torch
    from torch.nn import functional as F

    model.eval()
    chain_sum = np.zeros(corpus.n_chains, dtype=np.float64)
    order = batching.pack(np.arange(corpus.n_chains), corpus.lengths, max_tokens)
    with torch.no_grad():
        for idx in order:
            inputs, targets = make_batch(corpus, idx)
            x = torch.as_tensor(inputs, device=device)
            y = torch.as_tensor(targets, device=device)
            with _autocast(device):
                logits = model(x)
            nll = F.cross_entropy(
                logits.float().reshape(-1, 20),
                y.reshape(-1),
                ignore_index=IGNORE,
                reduction="none",
            ).reshape(y.shape)
            chain_sum[idx] = nll.sum(1).double().cpu().numpy()
    chain_mean = chain_sum / corpus.lengths
    cw = cluster_weighted(chain_mean, corpus.clusters)
    micro = float(chain_sum.sum() / corpus.lengths.sum())
    return {
        "cw_nll": cw,
        "micro_nll": micro,
        "cw_perplexity": math.exp(cw),
        "chains": corpus.n_chains,
    }


# ------------------------------------------------------------------ training


def train(
    train_corpus: Corpus,
    dev_corpus: Corpus,
    cfg: LMConfig,
    out: Path,
    *,
    device,
    log=print,
    resume: bool = True,
    heldout: Corpus | None = None,
) -> dict:
    """The run: cluster-uniform draws of whole chains, the best checkpoint kept (by the PDB dev chains, or by a held-out slice of the corpus), a stop after ``patience`` readings."""
    import torch
    from torch.nn import functional as F

    if cfg.select not in ("dev", "heldout"):
        raise ValueError(f"select is dev or heldout, not {cfg.select!r}")
    if cfg.select == "heldout" and heldout is None:
        raise ValueError("select = heldout needs a held-out corpus")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay
    )
    probs = batching.cluster_probabilities(train_corpus.clusters)
    budget = Budget(train_corpus.n_chains)
    history: list[dict] = []
    best = {"cw_nll": float("inf"), "step": 0}
    since_best, step = 0, 0
    last = out / "last.pt"
    if resume and last.exists():
        state = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, history, best = state["step"], state["history"], state["best"]
        since_best = state["since_best"]
        budget.seen = state["seen"].cpu().numpy().astype(bool)
        for k in ("updates", "slots", "residues", "padded", "sum_l2", "seconds"):
            setattr(budget, k, state["budget"][k])
        rng.bit_generator.state = state["rng"]
        log(f"  resumed at step {step}")
    log(
        f"  {parameters(model) / 1e6:.2f} M parameters, {train_corpus.n_chains} train chains"
    )
    batches = iter(())
    began = time.perf_counter() - budget.seconds
    stopped = None
    while step < cfg.steps:
        try:
            idx = next(batches)
        except StopIteration:
            draws = batching.draw(
                train_corpus.n_chains, train_corpus.n_chains, rng, probs
            )
            batches = iter(
                batching.pack(draws, train_corpus.lengths, cfg.max_tokens, rng)
            )
            continue
        model.train()
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg.steps, cfg)
        inputs, targets = make_batch(train_corpus, idx)
        x = torch.as_tensor(inputs, device=device)
        y = torch.as_tensor(targets, device=device)
        with _autocast(device):
            logits = model(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, 20), y.reshape(-1), ignore_index=IGNORE
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        step += 1
        budget.add(idx, train_corpus.lengths)
        budget.seconds = time.perf_counter() - began
        if step % cfg.eval_every == 0 or step == cfg.steps:
            dev = evaluate(model, dev_corpus, cfg.max_tokens, device)
            held = (
                evaluate(model, heldout, cfg.max_tokens, device)
                if heldout is not None
                else None
            )
            chosen = held if cfg.select == "heldout" else dev
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "dev": dev,
                "heldout": held,
                "budget": budget.snapshot(),
            }
            history.append(row)
            if chosen["cw_nll"] < best["cw_nll"] - 1e-9:
                best = {
                    "cw_nll": chosen["cw_nll"],
                    "step": step,
                    "selected_by": cfg.select,
                    "dev_nll": dev["cw_nll"],
                    "heldout_nll": held["cw_nll"] if held else None,
                    "budget": budget.snapshot(),
                }
                since_best = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "step": step,
                        "dev": dev,
                        "heldout": held,
                        "budget": budget.snapshot(),
                    },
                    out / "best.pt",
                )
            else:
                since_best += 1
            held_text = f"  heldout {held['cw_nll']:.4f}" if held else ""
            log(
                f"  step {step:6d}  train {row['train_loss']:.3f}  dev cw {dev['cw_nll']:.4f}{held_text}"
                f" (best {cfg.select} {best['cw_nll']:.4f} at {best['step']})"
                f"  {budget.seconds:.0f}s"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "opt": opt.state_dict(),
                    "step": step,
                    "history": history,
                    "best": best,
                    "since_best": since_best,
                    "seen": torch.as_tensor(budget.seen),
                    "budget": {
                        "updates": budget.updates,
                        "slots": budget.slots,
                        "residues": budget.residues,
                        "padded": budget.padded,
                        "sum_l2": budget.sum_l2,
                        "seconds": budget.seconds,
                    },
                    "rng": rng.bit_generator.state,
                },
                last,
            )
            if since_best >= cfg.patience:
                stopped = f"no better {cfg.select} loss in {cfg.patience} readings"
                break
    torch.save(
        {"model": model.state_dict(), "config": asdict(cfg), "step": step},
        out / "final.pt",
    )
    result = {
        "config": asdict(cfg),
        "parameters": parameters(model),
        "steps": step,
        "stopped": stopped or "the step ceiling",
        "best": best,
        "run_budget": budget.snapshot(),
        "heldout": None
        if heldout is None
        else {
            "chains": heldout.n_chains,
            "clusters": len(np.unique(heldout.clusters)),
        },
        "history": history,
    }
    latent_generation._write(out / "train.json", result)
    return result


# ------------------------------------------------------------------ sampling


def load_checkpoint(path: Path, device):
    import torch

    state = torch.load(path, map_location=device, weights_only=True)
    cfg = LMConfig(**state["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), cfg, state


def sample(
    model, lengths, *, temperature: float, generator, device, chunk: int = 64
) -> tuple[list[str], list[float]]:
    """Sequences of the given lengths by ancestral sampling, and the mean probability the model gave each residue it drew.

    The temperature divides the logits before the draw; the probability reported is the model's own (temperature 1).
    Whole prefixes are recomputed at each step: the model is small and the chains short.
    """
    import torch

    sequences: list[str] = []
    confidences: list[float] = []
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(lengths), chunk):
            these = np.asarray(lengths[lo : lo + chunk], dtype=np.int64)
            n, longest = len(these), int(these.max())
            ids = torch.as_tensor(
                (LEN_BASE + np.minimum(these // LEN_BIN, LEN_BINS - 1))[:, None],
                device=device,
            )
            taken = torch.zeros(n, longest, device=device)
            drawn = torch.zeros(n, longest, dtype=torch.long, device=device)
            for t in range(longest):
                logits = model(ids)[:, -1, :].float()
                p_model = torch.softmax(logits, dim=-1)
                p_draw = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(p_draw, 1, generator=generator)
                taken[:, t] = p_model.gather(1, nxt)[:, 0]
                drawn[:, t] = nxt[:, 0]
                ids = torch.cat([ids, nxt], dim=1)
            for b in range(n):
                length = int(these[b])
                sequences.append("".join(LETTERS[int(c)] for c in drawn[b, :length]))
                confidences.append(float(taken[b, :length].mean()))
    return sequences, confidences


def write_samples(out: Path, sequences, confidences, info: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = ["sample_id\tsequence\tmean_confidence"]
    rows += [
        f"s{i:03d}\t{s}\t{c:.4f}"
        for i, (s, c) in enumerate(zip(sequences, confidences, strict=True))
    ]
    (out / "samples.tsv").write_text("\n".join(rows) + "\n")
    latent_generation._write(out / "samples.json", info)


# ----------------------------------------------------------------------- CLI


def run_train(args) -> int:
    import torch

    from foldspace.evaluation.provenance import capture

    cfg = replace(LMConfig(), **parse_overrides(args.set))
    began = time.perf_counter()
    train_corpus = (
        load_table(args.corpus_table)
        if args.corpus_table
        else load_split(args.chains, "train")
    )
    dev_corpus = load_split(args.chains, "dev")
    heldout = None
    if cfg.heldout_clusters:
        train_corpus, heldout = split_heldout(train_corpus, cfg.heldout_clusters)
    print(
        f"device {args.device}  train {train_corpus.n_chains} chains in "
        f"{len(np.unique(train_corpus.clusters))} clusters, dev {dev_corpus.n_chains} chains "
        f"in {len(np.unique(dev_corpus.clusters))}"
        + (
            f", held out of the corpus {heldout.n_chains} chains in {len(np.unique(heldout.clusters))}"
            if heldout is not None
            else ""
        )
        + f"  loaded in {time.perf_counter() - began:.0f}s",
        flush=True,
    )
    out = Path(args.out)
    result = train(
        train_corpus,
        dev_corpus,
        cfg,
        out,
        device=torch.device(args.device),
        log=lambda *a: print(*a, flush=True),
        heldout=heldout,
    )
    result["provenance"] = capture(
        {"allow_dirty": bool(args.allow_dirty), "device": str(args.device)}
    )
    result["train_corpus"] = {
        "table": str(args.corpus_table)
        if args.corpus_table
        else "the train split of the corpus",
        "chains": train_corpus.n_chains,
        "clusters": len(np.unique(train_corpus.clusters)),
    }
    latent_generation._write(out / "train.json", result)
    print(
        f"best {cfg.select} {result['best']['cw_nll']:.4f} at step {result['best']['step']}; "
        f"stopped: {result['stopped']}; run {result['run_budget']['gpu_hours']:.2f} GPU-hours",
        flush=True,
    )
    return 0


def run_sample(args) -> int:
    import torch

    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    model, cfg, state = load_checkpoint(run_dir / f"{args.which}.pt", device)
    lengths = latent_generation.sample_lengths(dev_lengths(args.chains), args.n, *latent_generation.SAMPLE_LENGTHS)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sequences, confidences = sample(
        model, lengths, temperature=args.temperature, generator=generator, device=device
    )
    info = {
        "n": args.n,
        "lengths": lengths.tolist(),
        "temperature": args.temperature,
        "seed": args.seed,
        "checkpoint": args.which,
        "step": state["step"],
        "model_digest": latent_generation._digest(model),
        "config": asdict(cfg),
    }
    write_samples(Path(args.out), sequences, confidences, info)
    print(
        f"  {args.n} samples at temperature {args.temperature:g} from {args.which} (step {state['step']})",
        flush=True,
    )
    return 0


def run_ngram(args) -> int:
    """The floors on the dev chains of the pilot's corpus, or, with a corpus table, on the slice of it that ``select = heldout`` holds out."""
    if args.corpus_table:
        corpus = load_table(args.corpus_table)
        train_corpus, dev_corpus = split_heldout(corpus, args.heldout_clusters)
    else:
        train_corpus = load_split(args.chains, "train")
        dev_corpus = load_split(args.chains, "dev")
    result = {
        "train_chains": train_corpus.n_chains,
        "dev_chains": dev_corpus.n_chains,
        "read_on": "the held-out slice of the corpus"
        if args.corpus_table
        else "the PDB dev chains",
        "uniform_nll": math.log(20),
        "models": {
            "unigram": ngram_nll(train_corpus, dev_corpus, 0),
            "bigram": ngram_nll(train_corpus, dev_corpus, 1),
        },
    }
    latent_generation._write(Path(args.out), result)
    for name, m in result["models"].items():
        print(
            f"  {name}: dev cluster-weighted {m['cw_nll']:.4f} nats (perplexity {m['cw_perplexity']:.2f})"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--chains", type=Path, default=Path("data/corpus/chains.npz"))
    t.add_argument(
        "--corpus-table",
        type=Path,
        default=None,
        help="train on this corpus instead of the train split of the corpus",
    )
    t.add_argument("--set", default="", help="overrides of LMConfig, key=value,...")
    t.add_argument("--allow-dirty", action="store_true")
    s = sub.add_parser("sample")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--which", choices=("best", "last", "final"), default="best")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--chains", type=Path, default=Path("data/corpus/chains.npz"))
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--n", type=int, default=latent_generation.N_SAMPLES)
    s.add_argument("--seed", type=int, default=7)
    g = sub.add_parser(
        "ngram", help="the composition and the previous residue as sequence models"
    )
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--chains", type=Path, default=Path("data/corpus/chains.npz"))
    g.add_argument(
        "--corpus-table",
        type=Path,
        default=None,
        help="read the floors on the slice held out of this corpus",
    )
    g.add_argument("--heldout-clusters", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.command == "ngram":
        return run_ngram(args)
    return run_train(args) if args.command == "train" else run_sample(args)


if __name__ == "__main__":
    raise SystemExit(main())
