"""Sequence decoding: can the sequence be read back off the single, and is the best reading worth putting behind a generator?

One process loads the train and dev singles onto the card once and runs every arm in turn.

Arms: ``lda`` (closed form, fitted two ways), ``linear`` (a residue-wise linear classifier), ``contextual`` (a small
Transformer, two seeds) and ``contextual_chain`` (the same with chain-uniform draws, the one ablation). Each trained arm
goes through **stages, each of which must pass before the next is paid for**:

    smoke     the largest batch the packer makes: finite loss and gradient, a parameter that moves, the loader's
              throughput and the card's memory; a decoder's log-probabilities that sum to one.
    overfit   (contextual only) eight chains, dropout off: it must reach 99.9 % of their residues in 400 steps, or the
              architecture or the code cannot learn what it is shown.
    small     two epochs on the whole train split: dev accuracy of at least 0.50 (chance is 0.05: it shows that the code
              learns, not how well), and the checkpoint it writes must be read back to the number that was recorded.
    full      up to ``epochs`` epochs, the best by cluster-weighted dev cross-entropy kept.

A failed stage stops that arm and the run goes on to the next. **The test split is read once, last, after the arm that
will be carried forward has been chosen on dev, and each model is scored on it at most once** (a lock per model digest).
The decoder carried forward is the contextual one only if it beats the linear one by at least 0.2 points of
cluster-weighted dev accuracy on every seed; otherwise the linear one, which is then the finding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from foldspace.esmfold2 import batching, decode_eval, decoder_structure_panel, lda, train_decoder
from foldspace.esmfold2.decoder import build_decoder, parameter_count
from foldspace.esmfold2.decoder_data import IGNORE, Normaliser, gather, load_split
from foldspace.esmfold2.train_decoder import Config

ARMS = {
    "linear": Config(
        arch="linear",
        lr=3e-3,
        warmup_steps=100,
        epochs=30,
        patience=6,
        weight_decay=0.0,
        dropout=0.0,
    ),
    "contextual": Config(arch="contextual"),
    "contextual_chain": Config(arch="contextual", sampling="chain"),
}

#: fixed in advance
ADOPTION_MARGIN = 0.002  # cluster-weighted dev accuracy by which contextual must beat linear, on every seed
OVERFIT_CHAINS = 8
OVERFIT_STEPS = 400
OVERFIT_ACCURACY = 0.999
SMALL_EPOCHS = 2
SMALL_WARMUP = 25
SMALL_DEV_ACCURACY = 0.50

__all__ = ["ARMS", "main", "run"]


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")


def parse_overrides(text: str) -> dict:
    """``key=value,...`` for fields of ``Config``, typed by the field's own default."""
    defaults = asdict(Config())
    out = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        key, _, value = item.partition("=")
        if key not in defaults:
            raise ValueError(f"{key!r} is not a field of Config")
        kind = type(defaults[key])
        out[key] = (
            kind(value) if kind is not bool else value.lower() in ("1", "true", "yes")
        )
    return out


# ------------------------------------------------------------------- stages


def stage_smoke(config: Config, train, dev, norm) -> dict:
    """The largest batch, two optimiser steps, and what the loader costs."""
    device = train.z.device
    torch.manual_seed(config.seed)
    model = build_decoder(asdict(config)).to(device)
    longest = int(train.lengths.max())
    biggest = np.argsort(-train.lengths)[: max(1, config.max_tokens // longest)]
    x, y, lengths = gather(train, biggest, norm)
    checks = {
        "features_finite": bool(torch.isfinite(x).all()),
        "feature_width": int(x.shape[-1]),
        "padding_rows_are_zero": bool((x.abs().sum(-1)[y == IGNORE] == 0).all()),
        "labels_in_range": bool(((y == IGNORE) | ((y >= 0) & (y < 20))).all()),
    }
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr)
    before = [p.detach().clone() for p in model.parameters()]
    model.train()
    losses = []
    began = time.perf_counter()
    for _ in range(2):
        logits = model(x, lengths)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, 20), y.reshape(-1), ignore_index=IGNORE
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimiser.step()
        losses.append(float(loss.detach()))
        checks["gradient_finite"] = bool(torch.isfinite(grad))
    if device.type == "cuda":
        torch.cuda.synchronize()
    step_seconds = (time.perf_counter() - began) / 2
    moved = any(
        not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
    )

    model.eval()
    probe = np.arange(min(20, dev.n_chains))
    p = train_decoder.predict(model, dev, norm, probe)
    sums = np.exp(p.logp).sum(axis=1)
    checks["log_probabilities_sum_to_one"] = bool(np.allclose(sums, 1.0, atol=1e-3))

    # what an epoch's batching costs: packing plus cutting 50 batches out of the resident tensors
    rng = np.random.default_rng(0)
    began = time.perf_counter()
    batches = batching.pack(
        np.arange(train.n_chains), train.lengths, config.max_tokens, rng
    )
    pack_seconds = time.perf_counter() - began
    began = time.perf_counter()
    for b in batches[:50]:
        gather(train, b, norm)
    if device.type == "cuda":
        torch.cuda.synchronize()
    cut_seconds = (time.perf_counter() - began) / min(50, len(batches))

    ok = (
        all(v for k, v in checks.items() if k != "feature_width")
        and checks["feature_width"] == 256
        and moved
        and all(np.isfinite(losses))
    )
    return {
        "ok": bool(ok),
        "checks": checks,
        "parameter_moved": bool(moved),
        "losses": losses,
        "initial_loss_over_uniform": losses[0] / float(np.log(20)),
        "step_seconds_at_the_largest_batch": step_seconds,
        "batches_per_epoch": len(batches),
        "tokens_in_the_largest_batch": int(lengths.sum()),
        "pack_seconds_per_epoch": pack_seconds,
        "cut_seconds_per_batch": cut_seconds,
        "peak_memory_gib": (torch.cuda.max_memory_allocated() / 2**30)
        if device.type == "cuda"
        else None,
        "parameters": parameter_count(model),
    }


def stage_overfit(config: Config, train, norm) -> dict:
    """Eight chains of different lengths and clusters, dropout off: can it learn what it is shown?"""
    device = train.z.device
    order = np.argsort(train.lengths, kind="stable")
    taken, seen = [], set()
    for position in (
        np.linspace(0, len(order) - 1, 4 * OVERFIT_CHAINS).round().astype(int)
    ):
        chain = int(order[position])
        if int(train.clusters[chain]) not in seen:
            taken.append(chain)
            seen.add(int(train.clusters[chain]))
        if len(taken) == OVERFIT_CHAINS:
            break
    chains = np.array(taken)
    cfg = replace(config, dropout=0.0)
    torch.manual_seed(cfg.seed)
    model = build_decoder(asdict(cfg)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    x, y, lengths = gather(train, chains, norm)
    valid = y != IGNORE
    accuracy, used = 0.0, OVERFIT_STEPS
    for step in range(1, OVERFIT_STEPS + 1):
        model.train()
        logits = model(x, lengths)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, 20), y.reshape(-1), ignore_index=IGNORE
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                accuracy = float(
                    (model(x, lengths).argmax(-1)[valid] == y[valid]).float().mean()
                )
            if accuracy >= OVERFIT_ACCURACY:
                used = step
                break
    return {
        "ok": accuracy >= OVERFIT_ACCURACY,
        "chains": len(chains),
        "residues": int(valid.sum()),
        "accuracy": accuracy,
        "steps": used,
        "required": OVERFIT_ACCURACY,
        "final_loss": float(loss.detach()),
    }


def stage_small(config: Config, train, dev, norm, out: Path, log) -> dict:
    # Two epochs are about 200 steps, and the full run's warm-up is 200: left alone the whole stage would be warm-up,
    # the rate would never reach its peak, and the gate would be failing on the schedule and not on the code.
    config = replace(config, warmup_steps=min(config.warmup_steps, SMALL_WARMUP))
    model = build_decoder(asdict(config)).to(train.z.device)
    run = train_decoder.fit(
        model, train, dev, norm, config, out, epochs=SMALL_EPOCHS, log=log
    )
    restored = train_decoder.restore_check(out / "best.pt", dev, norm, run["best"])
    final = run["history"][-1]
    ok = bool(
        np.isfinite(final["train_loss"])
        and final["cw_acc"] >= SMALL_DEV_ACCURACY
        and restored["restored"]
    )
    return {
        "ok": ok,
        "dev_cw_acc": final["cw_acc"],
        "required": SMALL_DEV_ACCURACY,
        "restore": restored,
        "seconds": run["seconds"],
        "steps": run["steps"],
    }


def stage_full(config: Config, train, dev, norm, out: Path, log) -> dict:
    model = build_decoder(asdict(config)).to(train.z.device)
    run = train_decoder.fit(model, train, dev, norm, config, out, log=log)
    restored = train_decoder.restore_check(out / "best.pt", dev, norm, run["best"])
    best, _, _ = train_decoder.load_checkpoint(out / "best.pt", train.z.device)
    summary = train_decoder.predict(best, dev, norm).summary()
    _write(out / "dev_metrics.json", summary)
    return {
        "best_epoch": run["best"]["epoch"],
        "best": run["best"],
        "best_digest": run["best_digest"],
        "epochs_run": len(run["history"]),
        "seconds": run["seconds"],
        "restore": restored,
        "history": run["history"],
        "dev_summary": summary,
        "parameters": run["parameters"],
    }


# --------------------------------------------------------------------- lda


def lda_analysis(train, dev, norm) -> tuple[dict, lda.LDA]:
    """The discriminant fitted two ways; returns the report and the model fitted on every train chain."""
    everything = np.arange(train.n_chains)
    final = lda.fit_lda(train, norm, everything)
    report = {
        "fit_on_all_train": {"dev": train_decoder.predict(final, dev, norm).summary()}
    }
    if train.shard.max() >= 3:
        fit_on = np.flatnonzero(train.shard != 3)
        held_out = np.flatnonzero(train.shard == 3)
        partial = lda.fit_lda(train, norm, fit_on)
        shards = train_decoder.predict(partial, train, norm, held_out).summary()
        report["fit_on_shards_0_1_2"] = {
            "held_out_shard_3": shards,
            "dev": train_decoder.predict(partial, dev, norm).summary(),
        }
        report["shard_3_clusters_shared_with_fit"] = float(
            np.isin(train.clusters[held_out], np.unique(train.clusters[fit_on])).mean()
        )
    return report, final


# ----------------------------------------------------------------- the run


def digest_of(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def score_test_once(
    name: str, model, test, norm, lock_dir: Path, *, force: bool = False
) -> dict:
    """The test split, for one model, at most once: a lock per digest says it has been done."""
    digest = digest_of(model)
    lock = lock_dir / f"{digest}.json"
    if lock.exists() and not force:
        raise RuntimeError(
            f"the test split has already been scored for {name} ({digest}): {lock}"
        )
    summary = train_decoder.predict(model, test, norm).summary()
    record = {"name": name, "digest": digest, "summary": summary}
    _write(lock, record)
    return record


def _provenance(args) -> dict:
    """What a number from this run is quoted with: the four trees' commits, the host and job, torch, CUDA and the card.

    The decoders read singles that were made by the folding model, so no checkpoint is loaded here and none is
    named; the targets' own record names it.
    """
    from foldspace.evaluation.provenance import capture

    return capture(
        {
            "allow_dirty": bool(getattr(args, "allow_dirty", False)),
            "device": str(args.device),
        }
    )


def run(args) -> int:
    device = torch.device(args.device)
    out = Path(args.out)
    cache = Path(args.cache)
    overrides = parse_overrides(args.set)

    def log(*a):
        print(*a, flush=True)

    norm = Normaliser.load(
        cache / "esmfold2" / "model_single_train_normalisation.npz", device
    )
    began = time.perf_counter()
    common = {
        "cache": cache,
        "chains": Path(args.chains),
        "train_table": Path(args.train_table),
        "device": device,
        "limit_chains": args.limit_chains,
    }
    train = load_split("train", **common)
    dev = load_split("dev", **common)
    log(
        f"device {device}  train {train.n_chains} chains {train.n_rows} rows ({len(np.unique(train.clusters))} clusters)  "
        f"dev {dev.n_chains} chains {dev.n_rows} rows  loaded in {time.perf_counter() - began:.0f}s"
    )

    summary: dict = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "provenance": _provenance(args),
        "normalisation_digest": norm.digest,
        "arms": {},
        "failed": [],
    }
    lda_model = None
    if "lda" in args.arms.split(","):
        log("lda")
        report, lda_model = lda_analysis(train, dev, norm)
        summary["lda"] = report
        _write(out / "lda.json", report)
        full = report["fit_on_all_train"]["dev"]
        log(
            f"  fit on all train -> dev: cw acc {full['cluster_weighted']['acc']:.4f}  micro {full['micro']['acc']:.4f}"
        )
        if "fit_on_shards_0_1_2" in report:
            leaky = report["fit_on_shards_0_1_2"]["held_out_shard_3"]["micro"]["acc"]
            log(f"  fit on shards 0-2 -> shard 3 (the old protocol): {leaky:.4f}")

    stages = args.stages.split(",")
    for spec in [a for a in args.arms.split(",") if a and a != "lda"]:
        name, _, seed = spec.partition(":")
        config = replace(ARMS[name], seed=int(seed or 0), **overrides)
        arm = f"{name}-s{config.seed}"
        directory = out / arm
        log(f"\n=== {arm} ({config.arch}, {config.sampling} draws)")
        result: dict = {"config": asdict(config)}
        summary["arms"][arm] = result
        failed = None
        if "smoke" in stages:
            result["smoke"] = stage_smoke(config, train, dev, norm)
            _write(directory / "smoke.json", result["smoke"])
            log(f"  smoke: {'ok' if result['smoke']['ok'] else 'FAILED'}")
            failed = None if result["smoke"]["ok"] else "smoke"
        if failed is None and "overfit" in stages and config.arch == "contextual":
            result["overfit"] = stage_overfit(config, train, norm)
            _write(directory / "overfit.json", result["overfit"])
            log(
                f"  overfit: {result['overfit']['accuracy']:.4f} in {result['overfit']['steps']} steps"
            )
            failed = None if result["overfit"]["ok"] else "overfit"
        if failed is None and "small" in stages:
            result["small"] = stage_small(
                config, train, dev, norm, directory / "small", log
            )
            _write(directory / "small.json", result["small"])
            log(
                f"  small: dev cw acc {result['small']['dev_cw_acc']:.4f} restore {result['small']['restore']['restored']}"
            )
            failed = None if result["small"]["ok"] else "small"
        if failed is None and "full" in stages:
            result["full"] = stage_full(
                config, train, dev, norm, directory / "full", log
            )
            _write(
                directory / "full.json",
                {k: v for k, v in result["full"].items() if k != "dev_summary"},
            )
            b = result["full"]["best"]
            log(
                f"  full: best epoch {b['epoch']}  dev cw ce {b['cw_ce']:.4f} acc {b['cw_acc']:.4f} (micro {b['micro_acc']:.4f})"
            )
        if failed:
            summary["failed"].append({"arm": arm, "stage": failed})
            log(f"  stopped at {failed}")

    # what is carried forward: chosen on dev, before the test split is read
    chosen = choose(summary)
    summary["chosen"] = chosen
    log(f"\nchosen: {chosen}")
    _write(out / "decoder_summary.json", {k: v for k, v in summary.items()})

    if args.final_test:
        test = load_split("test", **common)
        lock_dir = cache / "esmfold2" / "decoders" / "test_once"
        tested = {}
        models = {}
        if lda_model is not None:
            models["lda"] = lda_model
        for key in ("linear", "contextual"):
            arm = chosen.get(key)
            if arm:
                models[arm] = train_decoder.load_checkpoint(
                    out / arm / "full" / "best.pt", device
                )[0]
        for name, model in models.items():
            record = score_test_once(
                name, model, test, norm, lock_dir, force=args.force_test
            )
            tested[name] = record
            s = record["summary"]
            log(
                f"test {name}: cw acc {s['cluster_weighted']['acc']:.4f} micro {s['micro']['acc']:.4f} ({record['digest']})"
            )
        summary["test"] = tested
        _write(out / "decoder_summary.json", summary)

    # the decoder that goes on to the structure check, and the panel it is read on
    carried = chosen.get("carried")
    if carried and (out / carried / "full" / "best.pt").exists():
        model = train_decoder.load_checkpoint(
            out / carried / "full" / "best.pt", device
        )[0]
        panel = decoder_structure_panel.select_panel(dev.ids, dev.lengths, dev.clusters)
        predictions = train_decoder.predict(model, dev, norm, np.array(panel))
        rows = ["seq_id\tsequence\tdecoded\tsubstitutions"]
        for j, chain in enumerate(panel):
            s0, n = int(predictions.starts[j]), int(predictions.lengths[j])
            decoded = "".join(
                decode_eval.ALPHABET[i] for i in predictions.logp[s0 : s0 + n].argmax(1)
            )
            true = dev.sequence(chain)
            rows.append(
                f"{dev.ids[chain]}\t{true}\t{decoded}\t{len(decoder_structure_panel.substitutions(true, decoded))}"
            )
        (out / "panel_decoded.tsv").write_text("\n".join(rows) + "\n")
        log(f"panel of {len(panel)} dev chains decoded by {carried}")
    else:
        log("no decoder was carried forward")

    log(
        f"\ndone in {time.perf_counter() - began:.0f}s; failed stages: {summary['failed']}"
    )
    return 1 if summary["failed"] else 0


def choose(summary: dict) -> dict:
    """The decoders carried forward, from dev alone.

    ``contextual`` is the seed with the lower cluster-weighted dev cross-entropy; it is *adopted* over ``linear`` only
    if it beats it by ``ADOPTION_MARGIN`` of cluster-weighted dev accuracy on every contextual seed that finished.
    """
    arms = summary["arms"]

    def finished(prefix):
        return {
            k: v for k, v in arms.items() if k.startswith(prefix + "-") and "full" in v
        }

    chosen: dict = {}
    linear = finished("linear")
    if linear:
        chosen["linear"] = min(linear, key=lambda k: linear[k]["full"]["best"]["cw_ce"])
    contextual = finished("contextual")
    if contextual:
        chosen["contextual"] = min(
            contextual, key=lambda k: contextual[k]["full"]["best"]["cw_ce"]
        )
    if "linear" in chosen and contextual:
        base = linear[chosen["linear"]]["full"]["best"]["cw_acc"]
        gains = {k: v["full"]["best"]["cw_acc"] - base for k, v in contextual.items()}
        chosen["gains_over_linear"] = gains
        chosen["contextual_adopted"] = bool(
            all(g >= ADOPTION_MARGIN for g in gains.values())
        )
        chosen["carried"] = (
            chosen["contextual"] if chosen["contextual_adopted"] else chosen["linear"]
        )
    elif "contextual" in chosen:
        chosen["carried"] = chosen["contextual"]
    elif "linear" in chosen:
        chosen["carried"] = chosen["linear"]
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument(
        "--chains", type=Path, default=Path("data/corpus/chains.npz")
    )
    parser.add_argument(
        "--train-table",
        type=Path,
        default=Path("data/corpus/train_sequences.tsv"),
    )
    parser.add_argument(
        "--arms", default="lda,linear:0,contextual:0,contextual:1,contextual_chain:0"
    )
    parser.add_argument("--stages", default="smoke,overfit,small,full")
    parser.add_argument(
        "--set",
        default="",
        help="key=value,... overrides of Config (a CPU smoke uses small ones)",
    )
    parser.add_argument("--limit-chains", type=int, default=0)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--force-test", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    if not args.allow_dirty:
        from foldspace.evaluation.provenance import require_clean

        require_clean()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
