"""Train RFD4-Proteina's flow on the corpus, over the release autoencoder's latent or over an ESM-derived one.

It is the release's own training step (`Proteina.training_step`: exploration over `exploration_k` noises, recycling, Muon and Adam stepped
by hand) run by Lightning over `src/foldspace/rfd4/flow_data.py`'s batches, from the denoiser weights of the release's pretraining checkpoint
(`pretrain_ckpt_path`'s way: weights only, a fresh optimizer, a warm-up of its own because the optimizer state is new) with the same
EMA callback the release uses. What differs between the two targets is only the autoencoder object (`src/foldspace/rfd4/flow_model.py`):

    --target native   the release encoder's sample, as in the release's training (a continuation, the control)
    --target esmc     the adapted encoder's deterministic latent over the chains' ESMC rows, with `--adapter` its checkpoint (an in-place layout checkpoint
                      too: the layout is read from it)
    --target fixed    the same with the identity code of adapted autoencoder's `fixed` source in place of ESMC's, no ESMC rows

and the batches are the same for both: the seed fixes the chains, the crop windows and the rotations of every step.

`--align-weight` adds, to any target, one term to that step: the denoiser's final token state, through a small projector, against the ESMC rows of the
same residues (`src/foldspace/rfd4/flow_align.py`; with `--target native` the flow stays the release's and the term is the only change). It uses no new trainer: the
term joins the sum that the release's `training_step` already backpropagates, and the projector is an ordinary weight of the module.

A run writes, in `--out`: `ckpt/last.ckpt` (resumed from when present: the step, the optimizer, the EMA, the stream's position), `steps.csv`
(time of every step, the tokens of its batch, peak memory), `losses.npz` (the loss terms and the times of every kept replicate, for the
analysis by time and length that a mean over a run hides), `summary.json` (the four axes of the budget: updates, example slots and
distinct presentations, residues and the sum of their squares, GPU-hours; the step-time statistics; the provenance) and the Lightning
logs. Nothing here is compared with anything: `src/foldspace/rfd4/flow_eval.py` reads the checkpoints.

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/train_flow.py --target esmc --adapter <autoencoder.pt> \\
        --shards cache/rfd4esmc/pool --chains data/corpus/chains.npz --esmc cache/rfd4esmc/esmc_last.f16.npy \\
        --flow <flow_pretrain-ema.ckpt> --ae <ae-ema.ckpt> --steps 2000 --batch 8 --out runs/rfd4esmc/flow/esmc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foldspace.rfd4.flow_data as fd
import foldspace.rfd4.flow_donors as flow_donors
import foldspace.rfd4.flow_model as fm


def sha256(path: Path, limit: int = 0) -> str:
    """SHA-256 of a file (of its first `limit` bytes when given: a digest that is cheap and still names the file)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        left = limit or None
        while chunk := f.read(1 << 24):
            if left is not None:
                chunk, left = chunk[:left], left - len(chunk[:left])
            h.update(chunk)
            if left == 0:
                break
    return h.hexdigest()


def blob_name(path: Path) -> str:
    """A Hub snapshot file's blob name is the SHA-256 of its bytes (CLAUDE.md): the cheap identity of a checkpoint."""
    return os.path.basename(os.path.realpath(path))


def recorder_class():
    import lightning as L
    import torch

    class FlowRecorder(L.Callback):
        """Time, memory, tokens and the per-replicate losses of every step; written out at the end and at every checkpoint."""

        def __init__(self, out: Path, batch: int, replicates: int) -> None:
            self.out = out
            self.batch = batch
            self.replicates = replicates
            self.rows: list[dict] = []
            self.losses: dict[str, list] = {}
            self.times: dict[str, list] = {}
            self.lengths: list[int] = []
            self.sample_step: list[int] = []
            self._t0 = 0.0
            self.began = time.time()

        def _sync(self) -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
            self._sync()
            self._t0 = time.time()

        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx
        ) -> None:
            self._sync()
            mask = batch["padding_mask"][:: self.replicates]
            lengths = mask.sum(1).cpu().numpy()
            step = int(trainer.global_step)
            self.rows.append(
                {
                    "step": step,
                    "seconds": time.time() - self._t0,
                    "chains": len(lengths),
                    "residues": int(lengths.sum()),
                    "residues_squared": int((lengths.astype(np.int64) ** 2).sum()),
                    "padded_length": int(mask.shape[1]),
                    "peak_allocated_gb": (
                        torch.cuda.max_memory_allocated() / 2**30
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    "loss": float(outputs["loss"]),
                }
            )
            for name, value in outputs["losses_per_sample"].items():
                self.losses.setdefault(name, []).extend(value.float().cpu().tolist())
            for name, value in outputs["t"].items():
                self.times.setdefault(name, []).extend(value.float().cpu().tolist())
            self.lengths.extend(lengths.tolist())
            self.sample_step.extend([step] * len(lengths))

        def flush(self) -> None:
            import csv

            if self.rows:
                with open(self.out / "steps.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
                    w.writeheader()
                    w.writerows(self.rows)
            np.savez(
                self.out / "losses.npz",
                step=np.array(self.sample_step),
                length=np.array(self.lengths),
                **{f"loss_{k}": np.array(v) for k, v in self.losses.items()},
                **{f"t_{k}": np.array(v) for k, v in self.times.items()},
            )

        def on_train_end(self, trainer, pl_module) -> None:
            self.flush()

        def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
            self.flush()

    return FlowRecorder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", required=True, choices=("native", "esmc", "fixed"))
    parser.add_argument(
        "--adapter", default="", help="the adapted autoencoder checkpoint (--target esmc)"
    )
    parser.add_argument("--shards", required=True)
    parser.add_argument(
        "--chains", required=True, help="the corpus's chains.npz (ESMC row positions)"
    )
    parser.add_argument("--esmc", default="", help="the ESMC array (--target esmc)")
    parser.add_argument(
        "--esmc-donors",
        default="",
        help="a donor map (`src/foldspace/rfd4/flow_donors.py`): every chain's ESMC rows are those of its donor, a chain of the exact length (the SHAM control "
        "of --align-weight: the same head and stream without the information); needs --esmc and --align-weight",
    )
    parser.add_argument(
        "--flow", required=True, help="the flow checkpoint to start from"
    )
    parser.add_argument(
        "--ae", required=True, help="the release autoencoder checkpoint"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="updates in all, counting those of a run resumed",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="chains per update (each is made `exploration_k` times)",
    )
    parser.add_argument("--limit-chains", type=int, default=0)
    parser.add_argument(
        "--crop",
        type=int,
        default=fd.CROP,
        help="residues kept of a longer chain (the release's crop is 256)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0, help="override the checkpoint's learning rate"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="updates of linear warm-up of both optimizers",
    )
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--buckets",
        default="",
        help="comma-separated padded lengths (static shapes for a compiled run)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="keep the checkpoint's torch.compile setting (off by default)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="a smoke run on the CPU: eager, torch kernels, single precision",
    )
    parser.add_argument(
        "--init",
        default="pretrained",
        choices=("pretrained", "scratch"),
        help="the denoiser starts from the checkpoint's weights, or from its own initialisation (the no-transfer reference)",
    )
    parser.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="micro-batches of `--batch` chains per optimizer update (memory for time: an update then has `--batch` times this many chains)",
    )
    parser.add_argument(
        "--checkpointing",
        default="",
        choices=("", "none", "low", "medium"),
        help="the network's activation checkpointing (memory for time: the pair updates, then the recycle cycles too); the checkpoint's own setting by default",
    )
    parser.add_argument(
        "--align-weight",
        type=float,
        default=0.0,
        help="add the ESMC alignment term (`src/foldspace/rfd4/flow_align.py`) at this weight: the denoiser's final token state, through a small projector, against the "
        "ESMC rows that the batches carry (needs --esmc); 0 leaves the release's training step as it is",
    )
    parser.add_argument(
        "--align-hidden", type=int, default=1024, help="the projector's hidden width"
    )
    parser.add_argument(
        "--align-probe",
        action="store_true",
        help="record, for every micro-batch, the gradient norm of the alignment term and of the release's loss on a few shared trunk weights and on the projector "
        "(`align_probe.jsonl`; two extra backward passes: for a smoke run, not for a training run)",
    )
    parser.add_argument(
        "--condition-fraction",
        type=float,
        default=0.0,
        help="the share of chains that are given their whole sequence as the release's own condition (its `folding` task: the 28 sequence-probability conditions and `C_SEQ` "
        "exact); a coin of its own stream, so that crops, rotations and noise are the arms' without it. The control of --esmc-adapter",
    )
    parser.add_argument(
        "--esmc-adapter",
        action="store_true",
        help="add the ESMC context adapter (`src/foldspace/rfd4/flow_adapt.py`): ESMC's rows of the sequence the denoiser is given, added to its initial token representation "
        "(needs --esmc and --condition-fraction above 0; ESMC is then an input at inference too, not only a teacher)",
    )
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=0.0,
        help="the share of the chains given their sequence that are given it in part (a design: a few spans of the chain are masked, and the denoiser generates them; the release's "
        "conditions name the known residues only, and the adapter reads ESMC's rows of the chain read with the spans masked, the script that writes the masked rows); a coin of its own "
        "stream. Needs --esmc-adapter, --condition-fraction and --esmc-masked",
    )
    parser.add_argument(
        "--structure-fraction",
        type=float,
        default=0.0,
        help="the share of the masked chains that are also given the backbone atoms (N, CA, C, O) of their known residues, in the release's absolute coordinate conditions "
        "(`condition_coordinate_1_atom`, `condition_CoordinateGroup_1_atom`, `coord_cond_mask`; `src/foldspace/rfd4/structure_condition.py`), the frame centred on those atoms for the "
        "target too, as the release centres its own coordinate-conditioned examples; the residues to generate are never given. A coin of its own stream, among the masked chains. "
        "Needs --mask-fraction",
    )
    parser.add_argument(
        "--esmc-masked",
        default="",
        help="the directory of the script that writes the masked rows (`masked_rows.f16.npy`, `index.json`): the rows of the masked slots of the steps this run reads",
    )
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args(argv)

    import lightning as L
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from rfproteina.utils.simple_ema import SimpleEMA
    from torch.utils.data import DataLoader

    began = time.time()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # the autoencoder makes ./tmp_ae/val_stuff in the working directory
    shards, chains, flow, ae = (
        str(Path(p).resolve()) for p in (args.shards, args.chains, args.flow, args.ae)
    )
    esmc = str(Path(args.esmc).resolve()) if args.esmc else ""
    adapter = str(Path(args.adapter).resolve()) if args.adapter else None
    if args.target == "esmc" and not (adapter and esmc):
        parser.error("--target esmc needs --adapter and --esmc")
    if args.target == "fixed" and not adapter:
        parser.error(
            "--target fixed needs --adapter (an adapted autoencoder checkpoint of the fixed identity code)"
        )
    if args.target == "native" and adapter:
        parser.error("--target native takes no adapter")

    nn_overrides = {} if args.compile and not args.cpu else {"use_torch_compile": False}
    if args.cpu:
        nn_overrides.update(tri_mult_backend="torch", norm_type="rmsnorm_torch")
    if args.checkpointing:
        nn_overrides["checkpointing"] = args.checkpointing
    opt_overrides = {"muon_warmup_steps": args.warmup, "adam_warmup_steps": args.warmup}
    if args.lr:
        opt_overrides["lr"] = args.lr
    if (
        args.accumulate > 1
    ):  # the release's own accumulation: the loss is averaged over the micro-batches and the optimizers step once
        opt_overrides["accumulate_grad_batches"] = args.accumulate
    L.seed_everything(args.seed, workers=True)
    model, ck = fm.load_flow(
        flow, ae, out / "store", adapter, nn_overrides, opt_overrides, args.init
    )
    model.train()
    del ck
    align = None
    if args.align_weight > 0:
        if not esmc:
            parser.error("--align-weight needs --esmc (the ESMC rows are the target)")
        import foldspace.rfd4.flow_align as fa

        align = fa.attach(model, args.align_weight, args.align_hidden)
        if args.align_probe:
            align.probe_params = {
                "trunk_first": next(model.nn.transformer_layers[0].parameters()),
                "trunk_last": next(model.nn.transformer_layers[-1].parameters()),
                "projector": next(getattr(model.nn, fa.HEAD_NAME).parameters()),
            }
        print(
            f"alignment term on: weight {args.align_weight}, projector {sum(p.numel() for p in getattr(model.nn, fa.HEAD_NAME).parameters()) / 1e6:.2f} M weights",
            flush=True,
        )

    context = None
    if args.esmc_adapter:
        if not (esmc and args.condition_fraction > 0):
            parser.error("--esmc-adapter needs --esmc and --condition-fraction above 0")
        import foldspace.rfd4.flow_adapt as fad

        context = fad.attach(model)
        print(
            f"ESMC context adapter on: {sum(p.numel() for p in getattr(model.nn, fad.HEAD_NAME).parameters()) / 1e6:.2f} M weights, "
            f"{args.condition_fraction:.2f} of the chains given their sequence",
            flush=True,
        )
    elif args.condition_fraction > 0:
        print(
            f"sequence conditioning without the adapter (the control): {args.condition_fraction:.2f} of the chains given their sequence",
            flush=True,
        )

    ids = fd.split_ids(Path(shards) / "index.json", "train", args.limit_chains)
    rows = fd.corpus_rows(chains) if esmc else None
    if args.esmc_donors:
        if not (esmc and args.align_weight > 0):
            parser.error("--esmc-donors needs --esmc and --align-weight")
        donor_record = json.loads(Path(args.esmc_donors).read_text())
        # a stream that meets a chain without a donor stops here, before a step is spent
        rows = flow_donors.remap_rows(rows, donor_record["donors"], ids)
        print(
            f"SHAM: the alignment term's ESMC rows are the donors' ({len(ids)} chains of the stream)",
            flush=True,
        )
    if args.structure_fraction > 0 and args.mask_fraction <= 0:
        parser.error(
            "--structure-fraction is for the masked chains: it needs --mask-fraction"
        )
    if not 0.0 <= args.structure_fraction <= 1.0:
        parser.error("--structure-fraction is a share: it is between 0 and 1")
    masked_rows = None
    if args.mask_fraction > 0:
        if not (context is not None and args.esmc_masked):
            parser.error("--mask-fraction needs --esmc-adapter and --esmc-masked")
        if args.align_weight > 0:
            parser.error(
                "--mask-fraction is refused with --align-weight: a masked example's alignment target would be the whole chain's rows"
            )
        masked_dir = Path(args.esmc_masked).resolve()
        masked_rows = fd.MaskedRows(
            masked_dir / "masked_rows.f16.npy", masked_dir / "index.json"
        )
        meta = masked_rows.index["meta"]
        # the rows were made for one stream: the seed, the batch and the two fractions decide which slots are masked and the rows of which chains are in the file
        for key, want in (
            ("seed", args.seed),
            ("batch", args.batch),
            ("condition_fraction", args.condition_fraction),
            ("mask_fraction", args.mask_fraction),
            ("limit_chains", args.limit_chains),
        ):
            if meta[key] != want:
                parser.error(
                    f"the masked rows were made with {key} {meta[key]} and this run has {want}"
                )
    last = out / "ckpt" / "last.ckpt"
    resume = last if last.is_file() else None
    start = 0
    if resume is not None:
        start = int(
            torch.load(resume, map_location="cpu", weights_only=False)["global_step"]
        )
        print(f"resuming {resume} at update {start}", flush=True)
    stream = fd.FlowStream(
        shards,
        ids,
        batch=args.batch,
        replicates=int(model.cfg_exp.training.exploration_k),
        seed=args.seed,
        # the stream is indexed by micro-batch: an update is `accumulate` of them
        steps=args.steps * args.accumulate,
        offset=start * args.accumulate,
        crop=args.crop,
        esmc_array=esmc or None,
        esmc_rows=rows,
        buckets=[int(b) for b in args.buckets.split(",") if b] or None,
        condition_fraction=args.condition_fraction,
        context_rows=context is not None,
        mask_fraction=args.mask_fraction,
        masked_rows=masked_rows,
        structure_fraction=args.structure_fraction,
    )
    if masked_rows is not None:
        # the slots this run will read, against the slots the rows were made for: a run that would meet an unmasked-rows slot stops here, before a step is spent
        steps_read = range(start * args.accumulate, args.steps * args.accumulate)
        missing = [
            f"{s}:{j}"
            for s, j, _ in stream.masked_slots(steps_read)
            if f"{s}:{j}" not in masked_rows.index["slots"]
        ]
        if missing:
            raise SystemExit(
                f"{len(missing)} masked slots of steps {steps_read.start} to {steps_read.stop} have no rows (the first: {missing[0]}): "
                f"{args.esmc_masked} was made for steps {masked_rows.index['meta']['steps']}"
            )
    loader = DataLoader(
        stream,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        prefetch_factor=2 if args.workers else None,
        persistent_workers=bool(args.workers),
        # a fresh interpreter for each worker: a forked one would carry the parent's whole heap (the model was built there), and
        # touching shared pages copies them
        multiprocessing_context="spawn" if args.workers else None,
    )
    recorder = recorder_class()(out, args.batch, stream.replicates)
    checkpoint = ModelCheckpoint(
        dirpath=out / "ckpt",
        every_n_train_steps=args.save_every,
        save_last=True,
        save_top_k=1,
        filename="step{step}",
        auto_insert_metric_name=False,
        # a run continued from a copy of an earlier checkpoint finds `last.ckpt` there already, and with Lightning's version counter its own
        # last checkpoint is written as `last-v1.ckpt` while `last.ckpt` stays the copy it began from, which every later reading then opens
        enable_version_counter=False,
    )
    callbacks = [
        SimpleEMA(decay=args.ema_decay, update_every_n=1),
        recorder,
        checkpoint,
    ]
    if align is not None and args.align_probe:

        class AlignProbe(L.Callback):
            """The gradient norms of the alignment term and of the release's loss on the probe weights, one line for every micro-batch."""

            def on_train_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx
            ) -> None:
                if align.probe:
                    with open(out / "align_probe.jsonl", "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "step": int(trainer.global_step),
                                    "batch": batch_idx,
                                    **align.probe,
                                }
                            )
                            + "\n"
                        )

        callbacks.append(AlignProbe())
    trainer = L.Trainer(
        accelerator="cpu" if args.cpu else "gpu",
        devices=1,
        precision="32-true" if args.cpu else str(model.cfg_exp.precision),
        max_steps=args.steps,
        logger=CSVLogger(str(out), name="logs"),
        callbacks=callbacks,
        enable_progress_bar=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_model_summary=False,
        default_root_dir=str(out),
    )
    print(
        f"{len(ids)} chains, {stream.replicates} replicates, target {args.target}, "
        f"update {start} of {args.steps}, {time.time() - began:.0f} s to set up",
        flush=True,
    )
    fit = time.time()
    # a checkpoint of this run holds the module's configuration (an OmegaConf object), which Lightning's default of a restricted
    # unpickler refuses; the file is this run's own
    trainer.fit(
        model,
        loader,
        ckpt_path=str(resume) if resume else None,
        weights_only=False,
    )
    fit = time.time() - fit
    # the checkpoint callback saves at multiples of --save-every: a run whose update count is not one has not saved its last update (300 updates at --save-every 200 left the weights of
    # update 200 and lost the ones that the run ended on), so the last is saved here, before the check below
    saved = (
        int(
            torch.load(last, map_location="cpu", weights_only=False, mmap=True)[
                "global_step"
            ]
        )
        if last.is_file()
        else -1
    )
    if saved != int(trainer.global_step):
        trainer.save_checkpoint(str(last))
        print(
            f"the last update, {int(trainer.global_step)}, is not a multiple of --save-every {args.save_every}: saved to {last}",
            flush=True,
        )
    # the checkpoint that the readings open (`last.ckpt`) must be the one this run ended on
    if last.is_file():
        final = int(
            torch.load(last, map_location="cpu", weights_only=False, mmap=True)[
                "global_step"
            ]
        )
        if final != int(trainer.global_step):
            raise SystemExit(
                f"{last} is at update {final} and the run ended at update {int(trainer.global_step)}: a reading of it would be of the wrong weights"
            )

    done = [r for r in recorder.rows]
    seconds = np.array([r["seconds"] for r in done]) if done else np.zeros(1)
    steady = seconds[min(5, len(seconds) - 1) :]
    residues = int(sum(r["residues"] for r in done))
    summary = {
        "target": args.target,
        "updates_this_run": len(done) // args.accumulate,
        "micro_batches_this_run": len(done),
        "accumulate": args.accumulate,
        "updates_total": int(trainer.global_step),
        "resumed_from": start,
        "chains_per_update": args.batch * args.accumulate,
        "replicates": stream.replicates,
        "example_slots": len(done) * args.batch * stream.replicates,
        "distinct_presentations": len(done) * args.batch,
        "chains_in_pool": len(ids),
        "residues": residues,
        "residues_squared": int(sum(r["residues_squared"] for r in done)),
        "gpu_hours": fit / 3600 if not args.cpu else 0.0,
        "wall_seconds_fit": round(fit, 1),
        "step_seconds_median": float(np.median(steady)),
        "step_seconds_p90": float(np.percentile(steady, 90)),
        "step_seconds_first": float(seconds[0]),
        "peak_allocated_gb": max((r["peak_allocated_gb"] for r in done), default=0.0),
        "adapter": None
        if context is None
        else {
            "weights": int(sum(p.numel() for p in model.nn.esmc_adapter.parameters())),
            "condition_fraction": args.condition_fraction,
            "term": "s_init += LayerNorm(2560) -> Linear(2560, 768) of the ESMC rows of the given sequence, gated by the row, src/foldspace/rfd4/flow_adapt.py",
            "calls_with_rows": context.with_rows,
            "calls": context.calls,
        },
        "condition_fraction": args.condition_fraction,
        "mask_fraction": args.mask_fraction,
        "structure_fraction": args.structure_fraction,
        "masked_rows": None
        if masked_rows is None
        else {
            "directory": str(Path(args.esmc_masked).resolve()),
            "slots": len(masked_rows.index["slots"]),
            "meta": masked_rows.index["meta"],
        },
        "align": None
        if align is None
        else {
            "weight": args.align_weight,
            "hidden": args.align_hidden,
            "projector_weights": int(
                sum(p.numel() for p in model.nn.esmc_align.parameters())
            ),
            "term": "1 - cos(P(final token state), LayerNorm(ESMC row)), src/foldspace/rfd4/flow_align.py",
        },
        "args": vars(args),
        "provenance": {
            "stack_commit": os.environ.get("FS_STACK_COMMIT"),
            "flow_checkpoint_blob": blob_name(Path(flow)),
            "ae_checkpoint_blob": blob_name(Path(ae)),
            "adapter_sha256": sha256(Path(adapter)) if adapter else None,
            "shards_index_sha256": sha256(Path(shards) / "index.json"),
            "esmc_array_head_sha256": sha256(Path(esmc), 1 << 26) if esmc else None,
            "esmc_donors_sha256": sha256(Path(args.esmc_donors))
            if args.esmc_donors
            else None,
            "masked_rows_index_sha256": sha256(Path(args.esmc_masked) / "index.json")
            if masked_rows is not None
            else None,
            "torch": torch.__version__,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("args", "provenance")},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
