"""What a flow has learned of a latent, against what the latent's own marginals and the noise level already give.

The flow's loss is the squared error of its prediction of the clean sample, per token and summed over the channels, times
`1 / (1 - t)^2` (`rdn_flow_matcher.compute_fm_loss`). Its value depends on the target (the release latent and an ESM-derived one have
different spreads, and a run's mean over times is a mean over its own t-distribution), so no two such numbers are compared. This reads
a checkpoint on the dev chains with the noise and the times held fixed and reports, per time `t`, per mode and per group of channels:

* the model's squared error `E_model(t)` (per token, summed over channels, with no time weight);
* `E_mean`: the error of predicting the dev set's mean latent, whatever `x_t` is;
* `E_xt`: the error of the best predictor that sees `x_t` and `t` and nothing else of the chain: the posterior mean under the empirical
  marginal of each channel, independently (`E[x_1 | x_t] = sum_m a_m w_m / sum_m w_m`, `w_m = N(x_t; t a_m, (1 - t)^2)`, the atoms `a_m` the
  latents of training chains the model was not shown here). A flow that only learned the marginals and the interpolation reaches it,
  and the context (the Cα trace, the other residues) is what lies below it;
* `skill = 1 - E_model / E_xt` (and against `E_mean`): 0 when the flow adds nothing to the marginals, 1 when it is exact;
* `E_joint` and `skill_vs_joint` (secondary; the primary reading is `skill_vs_xt`): the same against the best predictor that sees the
  token's *whole* noisy latent (all channels at once) and `t`, under the empirical joint distribution of 65,536 training tokens'
  latents. The per-channel marginals are a weak baseline for channels that depend on one another (the 24 ESMC channels are a map of a
  state that is near one of twenty kinds), and this one isolates what the context adds beyond the token's own latent. A joint density of
  32 channels is only as good as its atoms: its skill is an upper bound on what a stronger baseline would leave, not a claim. It is
  also far weaker than the marginals at a high `t` (the 16 channels at the native latent's prior make every training token far from
  every other, so that the estimate snaps to one of them): the pretraining checkpoint on its own latent has a skill over it of 0.98
  at t = 0.9 and over the marginals of 0.19. A skill over it at a high `t` says that the baseline is poor, and it is not read;
* `E_tok` and `skill_vs_tok` (with `--token-chains`): the same against a predictor that was fitted for the purpose, a small network on
  the token's whole noisy latent and `t` (`flow_token.py`), checked on held-out training tokens against the marginals'. What is left
  below it is what the chain's context gives and the token's own latent does not.

For the Cα mode the mean predictor is the origin (the data are centred), and `E_xt` is the isotropic-Gaussian posterior mean of the
standardised coordinates. The flow's sequence readout is scored too (accuracy over the supervised residues). Every figure is a mean over
the same batches, the same crop windows and rotations, and for each (t, repeat) the same noise, whichever checkpoint is read, so two
checkpoints are comparable batch by batch. With a native target the latent is a sample of the release encoder's posterior and its draw
is fixed by the batch's seed.

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/flow_eval.py --target esmc --adapter <autoencoder.pt> \\
        --flow <flow_pretrain-ema.ckpt> --ae <ae-ema.ckpt> --checkpoint <run>/ckpt/last.ckpt --shards cache/rfd4esmc/pool \\
        --chains data/corpus/chains.npz --esmc cache/rfd4esmc/esmc_last.f16.npy --out <run>/eval.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foldspace.rfd4.flow_data as fd
import foldspace.rfd4.flow_model as fm
import foldspace.rfd4.sample_flow as fs

#: the 16 channels of the release latent that carry information; the other 16 sit at the prior
NATIVE_ACTIVE = [0, 1, 4, 5, 8, 10, 13, 14, 15, 16, 17, 22, 25, 27, 28, 31]


def marginal_posterior_mean(xt, t: float, atoms, chunk: int = 1024):
    """`E[x_1 | x_t]` per channel under the empirical marginal `atoms` `[M, d]`, for `x_t = t x_1 + (1 - t) x_0` with `x_0` standard normal."""
    import torch

    out = torch.empty_like(xt)
    scale = 2.0 * max(1.0 - t, 1e-3) ** 2
    for start in range(0, xt.shape[0], chunk):
        x = xt[start : start + chunk]  # [T, d]
        logw = -((x[:, None, :] - t * atoms[None]) ** 2) / scale  # [T, M, d]
        w = torch.softmax(logw, dim=1)
        out[start : start + chunk] = (w * atoms[None]).sum(1)
    return out


def joint_posterior_mean(xt, t: float, atoms, chunk: int = 2048):
    """`E[x_1 | x_t]` under the empirical *joint* distribution of a token's whole latent (`atoms` `[M, d]`): the best predictor that sees
    the token's own noisy latent and `t` and nothing of its context, which the per-channel marginals cannot be when the channels depend
    on one another (the 24 ESMC channels are a map of a state that is near one of twenty kinds)."""
    import torch

    out = torch.empty_like(xt)
    scale = 2.0 * max(1.0 - t, 1e-3) ** 2
    a2 = (atoms**2).sum(1)  # [M]
    for start in range(0, xt.shape[0], chunk):
        x = xt[start : start + chunk]  # [T, d]
        d2 = (x**2).sum(1, keepdim=True) - 2.0 * t * (x @ atoms.T) + (t * t) * a2[None]
        out[start : start + chunk] = torch.softmax(-d2 / scale, dim=1) @ atoms
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", required=True, choices=("native", "esmc", "fixed"))
    parser.add_argument("--adapter", default="")
    parser.add_argument("--flow", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="a run's checkpoint; without it the weights of --flow (the zero-shot reading)",
    )
    parser.add_argument("--state", default="ema", choices=("ema", "raw"))
    parser.add_argument(
        "--expect-step",
        type=int,
        default=None,
        help="refuse a --checkpoint that is not at this update (the number the run was asked for)",
    )
    parser.add_argument("--shards", required=True)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--esmc", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--chains-per-batch", type=int, default=16)
    parser.add_argument(
        "--crop",
        type=int,
        default=fd.CROP,
        help="residues kept of a longer chain (the release's crop is 256)",
    )
    parser.add_argument("--limit-chains", type=int, default=0)
    parser.add_argument("--t-grid", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument(
        "--atom-chains",
        type=int,
        default=400,
        help="training chains whose latents are the atoms of the baselines (about 180 tokens each)",
    )
    parser.add_argument("--atoms", type=int, default=1024)
    parser.add_argument(
        "--joint-atoms",
        type=int,
        default=65536,
        help="whole latents of training tokens that make the joint predictor (the more, the stronger the baseline: a joint density of "
        "32 channels is only as good as its atoms)",
    )
    parser.add_argument(
        "--token-chains",
        type=int,
        default=0,
        help="training chains whose latents fit the learned token-only baseline (`flow_token.py`; 0: not read). It adds `E_tok` and "
        "`skill_vs_tok`; every other figure is unchanged, the noise and the atoms of the other baselines included",
    )
    parser.add_argument("--token-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    import foldspace.rfd4.flow_token as ft
    import torch
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.model_masks import (
        compute_flow_emit_masks,
        compute_flow_supervision_masks,
    )

    began = time.time()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(out.parent)
    flow, ae, shards, chains = (
        str(Path(p).resolve()) for p in (args.flow, args.ae, args.shards, args.chains)
    )
    esmc = str(Path(args.esmc).resolve()) if args.esmc else None
    adapter = str(Path(args.adapter).resolve()) if args.adapter else None
    if args.target == "esmc" and not (adapter and esmc):
        parser.error("--target esmc needs --adapter and --esmc")
    if args.target == "fixed" and not adapter:
        parser.error("--target fixed needs --adapter")
    if args.target == "native" and (adapter or esmc):
        parser.error("--target native takes neither --adapter nor --esmc")
    device = torch.device("cpu" if args.cpu else "cuda")
    overrides = {"use_torch_compile": False}
    if args.cpu:
        overrides.update(tri_mult_backend="torch", norm_type="rmsnorm_torch")
    model, _ = fm.load_flow(flow, ae, out.parent / "store", adapter, overrides)
    restored = (
        fs.restore(model, args.checkpoint, args.state, args.expect_step)
        if args.checkpoint
        else {"weights": flow}
    )
    model.eval().to(device)
    rows = fd.corpus_rows(chains) if esmc else None
    modes = list(model.fm.data_modes)
    dim = int(model.latent_dim)
    if args.target != "native" and getattr(
        model.autoencoder.encoder.latent, "inplace", False
    ):
        # in-place layout: the channels that carry the code and the informative native channels kept where the pretrained flow expects them
        latent_layout = model.autoencoder.encoder.latent
        groups = {
            "code_channels": list(latent_layout.code_idx),
            "native_kept_channels": list(latent_layout.keep),
        }
    elif args.target != "native":
        residual = int(model.autoencoder.encoder.residual)
        groups = {
            "sequence_channels": list(range(dim - residual)),
            "residual_channels": list(range(dim - residual, dim)),
        }
    else:
        groups = {
            "active_channels": NATIVE_ACTIVE,
            "prior_channels": [c for c in range(dim) if c not in NATIVE_ACTIVE],
        }
    # a latent without residual channels (the full-width ESMC latent and its type-only control) has no group of them: a skill over nothing is not 0 / 0
    groups = {name: idx for name, idx in groups.items() if idx}
    groups["all_channels"] = list(range(dim))

    def prepare(stream, step: int, seed: int) -> dict:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in stream[step].items()
        }
        batch["coords_nm"] = batch["coords"] / 10.0
        batch["coord_mask"] = batch["atom37_mask"]
        torch.manual_seed(seed)
        with (
            torch.no_grad(),
            torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.cpu),
        ):
            return model.add_clean_samples(batch)

    # the marginals' atoms: latents of training chains, none of them a dev chain
    train_ids = fd.split_ids(Path(shards) / "index.json", "train", args.atom_chains)
    atom_stream = fd.FlowStream(
        shards,
        train_ids,
        batch=args.chains_per_batch,
        replicates=1,
        seed=args.seed,
        steps=1 << 20,
        crop=args.crop,
        esmc_array=esmc,
        esmc_rows=rows,
    )
    pool = []
    for step in range(
        (len(train_ids) + args.chains_per_batch - 1) // args.chains_per_batch
    ):
        b = prepare(atom_stream, step, args.seed + step)
        sup = compute_flow_supervision_masks(b)["local_latents"]
        pool.append(b["x_1"]["local_latents"][sup].float())
    pool = torch.cat(pool)
    pick = torch.randperm(
        len(pool), generator=torch.Generator().manual_seed(args.seed)
    )[: args.atoms]
    atoms = pool[pick.to(pool.device)]
    joint_pick = torch.randperm(
        len(pool), generator=torch.Generator().manual_seed(args.seed + 1)
    )[: args.joint_atoms]
    atoms_joint = pool[joint_pick.to(pool.device)]
    del pool

    ids = fd.split_ids(Path(shards) / "index.json", args.split, args.limit_chains)
    per = args.chains_per_batch
    if len(ids) % per:
        ids = ids[: len(ids) - len(ids) % per]
    stream = fd.FlowStream(
        shards,
        ids,
        batch=per,
        replicates=1,
        seed=args.seed,
        steps=len(ids) // per,
        crop=args.crop,
        esmc_array=esmc,
        esmc_rows=rows,
    )
    batches = [prepare(stream, s, args.seed + 1000 + s) for s in range(len(stream))]
    with torch.no_grad():
        lat = torch.cat(
            [
                b["x_1"]["local_latents"][
                    compute_flow_supervision_masks(b)["local_latents"]
                ].float()
                for b in batches
            ]
        )
    mu = lat.mean(0)
    stats = {
        "dev_latent_mean_abs": float(mu.abs().mean()),
        "dev_latent_std_mean": float(lat.std(0).mean()),
        "dev_latent_std_per_channel": [round(float(x), 3) for x in lat.std(0)],
        "tokens": len(lat),
    }
    del lat
    t_grid = [float(x) for x in args.t_grid.split(",")]
    # the learned token-only baseline (`flow_token.py`): fitted on the latents of training chains drawn from a stream of its own, so
    # that the atoms of the other baselines, the dev batches and their noise are what they are without it; the network's held-out error
    # against the marginals' is kept with the reading, and a time at which it is worse is not read with it
    token_net, token_info = None, None
    if args.token_chains:
        tok_ids = fd.split_ids(Path(shards) / "index.json", "train", args.token_chains)
        tok_stream = fd.FlowStream(
            shards,
            tok_ids,
            batch=args.chains_per_batch,
            replicates=1,
            seed=args.seed + 2,
            steps=1 << 20,
            crop=args.crop,
            esmc_array=esmc,
            esmc_rows=rows,
        )
        tok_pool = []
        for step in range((len(tok_ids) + per - 1) // per):
            b = prepare(tok_stream, step, args.seed + 5000 + step)
            sup = compute_flow_supervision_masks(b)["local_latents"]
            tok_pool.append(b["x_1"]["local_latents"][sup].float())
        tok_pool = torch.cat(tok_pool)
        token_net, held, token_info = ft.fit_token_denoiser(
            tok_pool, steps=args.token_steps, seed=args.seed
        )
        token_info["chains"] = len(tok_ids)
        token_info["validation"] = ft.validate(
            token_net,
            held,
            t_grid,
            lambda x, t: marginal_posterior_mean(x, t, atoms),
        )
        print(
            "token-only baseline, held-out error against the marginals': "
            + ", ".join(
                f"t={t}: {v['network']:.3f} vs {v['reference']:.3f}"
                for t, v in token_info["validation"].items()
            ),
            flush=True,
        )
        del tok_pool, held
    result = {"t": {}}
    per_chain: dict[str, dict] = {}
    base_fm = model.fm.base_flow_matchers["local_latents"]
    with torch.no_grad():
        for t in t_grid:
            acc = {
                "latent_model": 0.0,
                "latent_mean": 0.0,
                "latent_xt": 0.0,
                "bb_model": 0.0,
                "bb_mean": 0.0,
                "bb_xt": 0.0,
                "loss_" + modes[0]: 0.0,
                "loss_" + modes[1]: 0.0,
            }
            # per chain, summed over the repeats: what a bootstrap over the dev chains resamples
            n_chains = len(batches) * per
            kinds = ("model", "mean", "xt", "joint") + (
                ("tok",) if token_net is not None else ()
            )
            chain = {k: np.zeros((n_chains, dim)) for k in kinds}
            chain_tokens = np.zeros(n_chains)
            chain_bb = {k: np.zeros(n_chains) for k in ("model", "mean", "xt")}
            chain_seq = {k: np.zeros(n_chains) for k in ("correct", "tokens")}
            for s, batch in enumerate(batches):
                rows_of_batch = slice(s * per, (s + 1) * per)
                for r in range(args.repeats):
                    b = dict(batch)
                    torch.manual_seed(args.seed + 7919 * s + 104729 * r + int(t * 1000))
                    bs = b["padding_mask"].shape[0]
                    b = model.fm.corrupt_batch(
                        b, t={dm: torch.full((bs,), t, device=device) for dm in modes}
                    )
                    with torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=not args.cpu
                    ):
                        nn_out = model.call_nn(b, n_recycle=args.recycles)
                    losses = model.fm.compute_loss(batch=b, nn_out=nn_out)
                    for dm in modes:
                        acc["loss_" + dm] += float(losses[dm].float().mean()) / (
                            len(batches) * args.repeats
                        )
                    emit = compute_flow_emit_masks(b)
                    sup = compute_flow_supervision_masks(b)
                    chain_tokens[rows_of_batch] += (
                        sup["local_latents"].sum(1).cpu().numpy()
                    )
                    x1, xt, x0 = (
                        b["x_1"]["local_latents"].float(),
                        b["x_t"]["local_latents"].float(),
                        b["x_0"]["local_latents"].float(),
                    )
                    # the interpolation the marginal baseline assumes
                    gap = float(
                        (
                            (xt - (t * x1 + (1 - t) * x0))
                            * sup["local_latents"][..., None]
                        )
                        .abs()
                        .max()
                    )
                    if gap > 1e-3:
                        raise RuntimeError(
                            f"x_t is not t x_1 + (1 - t) x_0 (largest gap {gap:.3g}): the marginal baseline does not apply"
                        )
                    pred = base_fm.nn_out_add_clean_sample_prediction(
                        x_t=b["x_t"]["local_latents"],
                        t=b["t"]["local_latents"],
                        emit_mask=emit["local_latents"],
                        nn_out=dict(nn_out["local_latents"]),
                    )["x_1"].float()
                    m = sup["local_latents"][..., None].float()
                    marg = marginal_posterior_mean(xt[sup["local_latents"]], t, atoms)
                    marg_full = torch.zeros_like(xt)
                    marg_full[sup["local_latents"]] = marg
                    joint_full = torch.zeros_like(xt)
                    joint_full[sup["local_latents"]] = joint_posterior_mean(
                        xt[sup["local_latents"]], t, atoms_joint
                    )
                    predictors = [
                        ("model", pred),
                        ("mean", mu.expand_as(x1)),
                        ("xt", marg_full),
                        ("joint", joint_full),
                    ]
                    if token_net is not None:
                        tok_full = torch.zeros_like(xt)
                        tok_full[sup["local_latents"]] = token_net(
                            xt[sup["local_latents"]], t
                        )
                        predictors.append(("tok", tok_full))
                    for name, p in predictors:
                        chain[name][rows_of_batch] += (
                            ((x1 - p) ** 2 * m).sum(1).cpu().numpy()
                        )
                    # the Cα mode: the model against the origin and against the isotropic-Gaussian posterior mean of the standardised coordinates
                    c1, ct = b["x_1"]["bb_ca"].float(), b["x_t"]["bb_ca"].float()
                    cm = sup["bb_ca"][..., None].float()
                    cpred = (
                        model.fm.base_flow_matchers["bb_ca"]
                        .nn_out_add_clean_sample_prediction(
                            x_t=b["x_t"]["bb_ca"],
                            t=b["t"]["bb_ca"],
                            emit_mask=emit["bb_ca"],
                            nn_out=dict(nn_out["bb_ca"]),
                        )["x_1"]
                        .float()
                    )
                    chain_bb["model"][rows_of_batch] += (
                        ((c1 - cpred) ** 2 * cm).sum((1, 2)).cpu().numpy()
                    )
                    chain_bb["mean"][rows_of_batch] += (
                        (c1**2 * cm).sum((1, 2)).cpu().numpy()
                    )
                    sd = (
                        float((c1**2 * cm).sum() / (cm.sum() + 1e-9)) ** 0.5
                    )  # the coordinates' own spread (nm), which the baseline is given
                    gain = t * sd**2 / (t**2 * sd**2 + (1 - t) ** 2)
                    chain_bb["xt"][rows_of_batch] += (
                        ((c1 - gain * ct) ** 2 * cm).sum((1, 2)).cpu().numpy()
                    )
                    logits = nn_out.get("sequence_logits")
                    if logits is not None:
                        guess = logits[..., : ENC.n_tokens].argmax(-1)
                        ok = (guess == b["residue_type"]) & sup["local_latents"]
                        chain_seq["correct"][rows_of_batch] += ok.sum(1).cpu().numpy()
                        chain_seq["tokens"][rows_of_batch] += (
                            sup["local_latents"].sum(1).cpu().numpy()
                        )
            tokens = float(chain_tokens.sum())
            channel = {k: v.sum(0) for k, v in chain.items()}
            correct, seq_tokens = (
                int(chain_seq["correct"].sum()),
                int(chain_seq["tokens"].sum()),
            )
            acc["bb_model"], acc["bb_mean"], acc["bb_xt"] = (
                float(chain_bb[k].sum()) for k in ("model", "mean", "xt")
            )
            per_chain[f"{t:g}"] = {
                "tokens": chain_tokens,
                **{f"E_{k}": v for k, v in chain.items()},
                **{f"bb_{k}": v for k, v in chain_bb.items()},
                **{f"seq_{k}": v for k, v in chain_seq.items()},
            }
            denom = max(tokens, 1.0)
            row = {
                "tokens": int(tokens),
                "weighted_loss_mean_over_batches": {
                    dm: acc["loss_" + dm] for dm in modes
                },
            }
            for g, idx in groups.items():
                em = float(channel["model"][idx].sum() / denom)
                ex = float(channel["xt"][idx].sum() / denom)
                e0 = float(channel["mean"][idx].sum() / denom)
                ej = float(channel["joint"][idx].sum() / denom)
                row[g] = {
                    "E_model": em,
                    "E_xt": ex,
                    "E_mean": validate_interface,
                    "E_joint": ej,
                    "skill_vs_xt": 1 - em / ex,
                    "skill_vs_mean": 1 - em / validate_interface,
                    # beyond the token's own noisy latent as a whole (a secondary reading: the primary reading is skill_vs_xt)
                    "skill_vs_joint": 1 - em / ej,
                }
                if "tok" in channel:
                    # beyond the best predictor of the token's own noisy latent that was fitted (`flow_token.py`)
                    et = float(channel["tok"][idx].sum() / denom)
                    row[g]["E_tok"] = et
                    row[g]["skill_vs_tok"] = 1 - em / et
            bb_n = denom
            row["bb_ca"] = {
                "E_model": acc["bb_model"] / bb_n,
                "E_mean": acc["bb_mean"] / bb_n,
                "E_xt_gaussian": acc["bb_xt"] / bb_n,
                "skill_vs_xt": 1 - acc["bb_model"] / max(acc["bb_xt"], 1e-12),
                "skill_vs_mean": 1 - acc["bb_model"] / max(acc["bb_mean"], 1e-12),
            }
            if seq_tokens:
                row["sequence_readout_accuracy"] = correct / seq_tokens
            row["per_channel_skill_vs_xt"] = [
                round(
                    1
                    - float(channel["model"][c]) / max(float(channel["xt"][c]), 1e-12),
                    3,
                )
                for c in range(dim)
            ]
            result["t"][f"{t:g}"] = row
            print(
                f"t={t:g}: "
                + ", ".join(
                    f"{g} skill_vs_xt {row[g]['skill_vs_xt']:.3f}"
                    + (
                        f" (vs tok {row[g]['skill_vs_tok']:.3f})"
                        if "skill_vs_tok" in row[g]
                        else ""
                    )
                    for g in groups
                )
                + f", bb_ca {row['bb_ca']['skill_vs_xt']:.3f}",
                flush=True,
            )
    report = {
        "target": args.target,
        "restored": restored,
        "args": vars(args),
        "chains": len(ids),
        "batches": len(batches),
        "channel_groups": groups,
        "dev_latent": stats,
        "atoms": {"chains": len(train_ids), "atoms": len(atoms)},
        "token_baseline": token_info,
        **result,
        "seconds": round(time.time() - began, 1),
        "stack_commit": os.environ.get("FS_STACK_COMMIT"),
    }
    out.write_text(json.dumps(report, indent=1))
    # the per-chain sums, in the order the stream deals the dev chains (the same for every checkpoint read with the same seed)
    order = list(
        itertools.chain.from_iterable(
            stream.chains_of_step(s) for s in range(len(stream))
        )
    )
    np.savez(
        out.with_suffix(".npz"),
        chain_ids=np.array(order),
        channel_groups=json.dumps(groups),
        **{f"{k}@t{t}": v for t, d in per_chain.items() for k, v in d.items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
