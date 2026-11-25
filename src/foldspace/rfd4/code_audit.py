"""The representation audit (five readings), on stored artifacts: what the learned code of the ESM-derived in-place latents holds, against the release encoder's own identity channels. CPU, nothing trained.

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/code_audit.py --shards cache/rfd4esmc/shards --ae <ckpt> \\
        --adapter learned_posterior_hybrid-b1e-2=<autoencoder.pt> --adapter learned_posterior_hybrid-b1e-3=<autoencoder.pt> --adapter contextual_hybrid-s0.6=<autoencoder.pt> --adapter identity_hybrid-s0.6=<autoencoder.pt> --out audit.json

The release encoder reads the dev chains once (its posterior means), and each adapter's code is computed from the chains' ESMC rows: `mu` (the code), its posterior mean (`mu` for a fixed-σ or deterministic code,
`sqrt(1 - σ²) mu` for a learned width) and σ. The two *identity slots* of a code are the channels at the release's channels 0 and 10. `src/foldspace/rfd4/code_audit_stats.py` has the arithmetic.

* **alignment**: the per-type centroids of the identity slots (posterior means) against the release encoder's channels 0 and 10: the R² of the best similarity transform, its permutation null and the ceiling of two halves of the chains.
* **context**: the within-type residual of the identity slots regressed on the neighbours' types, on the Cα geometry and on both (ridge, five folds by chain), for the code and for the release encoder's own channels, with a shuffled-feature null.
* **information_dimension**: the share of each channel's variance within types and the participation ratio of the within-type residual covariance.
* **regularizers**: the exact KL and the rate term of a learned posterior, by channel group.
* **decoder_reading**: a learned posterior's autoencoder read on the dev chains at its code, at its posterior mean and sampled (`train_autoencoder.read_dev`), the code reading checked against the recorded `end` of its summary.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foldspace.rfd4.code_audit_stats as cs
import foldspace.rfd4.code_bank as bk
import foldspace.rfd4.train_autoencoder as autoencoder
from foldspace.rfd4.train_reading_adapter import KEYS, Shards, collate

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
IDENTITY = (0, 10)  # the release latent's identity channels


def standard_map(enc) -> np.ndarray:
    """Token id -> index in `ALPHABET` for the twenty standard residues, -1 for every other token."""
    from foldspace.rfd4.ae_datapath import THREE_TO_ONE

    table = np.full(enc.n_tokens, -1, dtype=np.int64)
    for i in range(enc.n_tokens):
        name = enc.idx_to_token[i]
        if name in THREE_TO_ONE:
            table[i] = ALPHABET.index(THREE_TO_ONE[name])
    return table


def build_adapter(
    path: str, ae, in_dim: int, standard_rows: tuple[int, ...], enc, torch
) -> dict:
    """A saved adapted autoencoder checkpoint as its latent and its decoder, in evaluation mode."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    source = "fixed" if "code" in state["latent"] else "esmc"
    learned = "sigma_head.weight" in state["latent"]
    latent = autoencoder.anchored_class()(
        source,
        enc.n_tokens,
        residual=0,
        inplace=True,
        in_dim=in_dim,
        standard_rows=standard_rows,
        stochastic=learned,
        calibrated="code_chol" in state["latent"],
        banked=int(state["latent"]["code_bank"].shape[0])
        if "code_bank" in state["latent"]
        else 0,
    )
    latent.load_state_dict(state["latent"])
    posterior = state.get("posterior") or {}
    latent.code_sigma = float(posterior.get("code_sigma", 0.0))
    latent.eval()
    decoder = copy.deepcopy(ae.decoder)
    decoder.load_state_dict(state["decoder"])
    decoder.eval()
    slots = [latent.code_idx.index(c) for c in IDENTITY]
    return {
        "source": source,
        "learned": learned,
        "latent": latent,
        "decoder": decoder,
        "slots": slots,
        "posterior": posterior,
    }


def collect(
    ae,
    dev,
    adapters: dict,
    defaults: dict,
    mask_token: int,
    device,
    batch_size: int,
    std_of_token: np.ndarray,
    torch,
) -> dict:
    """One pass over the dev chains: the release encoder's posterior means, and each adapter's code, at the standard residues that carry a latent; and the chains' Cα and types for the context features."""
    from rfproteina.model_masks import compute_ae_masks

    native, types, chain, position = [], [], [], []
    per_adapter = {k: {"mu": [], "mean": [], "sigma": []} for k in adapters}
    chain_inputs = []
    cache: dict = {}
    for start in range(0, len(dev), batch_size):
        items = dev.items[start : start + batch_size]
        batch = collate(items, defaults, mask_token, device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            out = ae.encoder(batch)
        mean = out["mean"].float()
        cache[(tuple(batch["coords"].shape), float(batch["coords"].double().sum()))] = (
            mean
        )
        emit_ae = compute_ae_masks(batch)[1]
        emit = emit_ae & batch["is_resolved_mask"].bool()
        std = torch.from_numpy(std_of_token)[batch["residue_type"].cpu()].to(
            emit.device
        )
        select = emit & (std >= 0)
        native.append(mean[select].cpu().numpy())
        types.append(std[select].cpu().numpy())
        for i, item in enumerate(items):
            n = int(item["residue_type"].shape[0])
            ids = np.full(int(select[i].sum()), start + i)
            chain.append(ids)
            position.append(np.flatnonzero(select[i].cpu().numpy()))
            valid = (emit[i, :n] & batch["atom37_mask"][i, :n, 1].bool()).cpu().numpy()
            t_std = np.where(valid, std[i, :n].cpu().numpy(), -1)
            chain_inputs.append(
                (t_std, batch["coords"][i, :n, 1, :].float().cpu().numpy(), valid)
            )
        for key, ad in adapters.items():
            latent = ad["latent"]
            with torch.no_grad():
                unshrunk = latent(batch, native=mean, unshrunk=True)[
                    ..., latent.code_idx
                ]
                posterior_mean = latent(batch, native=mean)[..., latent.code_idx]
                per_adapter[key]["mu"].append(unshrunk[select].cpu().numpy())
                per_adapter[key]["mean"].append(posterior_mean[select].cpu().numpy())
                if ad["learned"]:
                    full = torch.zeros(*emit_ae.shape, len(latent.code_idx))
                    full[emit_ae.cpu()] = latent.posterior_scale(batch).cpu()
                    per_adapter[key]["sigma"].append(full[select.cpu()].numpy())
    result = {
        "native": np.concatenate(native).astype(np.float64),
        "types": np.concatenate(types),
        "chain": np.concatenate(chain),
        "position": np.concatenate(position),
        "chain_inputs": chain_inputs,
        "cache": cache,
        "adapters": {},
    }
    for key, parts in per_adapter.items():
        result["adapters"][key] = {
            "mu": np.concatenate(parts["mu"]).astype(np.float64),
            "mean": np.concatenate(parts["mean"]).astype(np.float64),
            "sigma": np.concatenate(parts["sigma"]).astype(np.float64)
            if parts["sigma"]
            else None,
        }
    return result


def context_matrix(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The context features of the collected residues (sequence, structure) and which of them are usable (interior, with a Cα at the window), in the order of `data`'s residue arrays."""
    seq_rows, str_rows, usable_rows = [], [], []
    for c, (types, ca, valid) in enumerate(data["chain_inputs"]):
        seq, structure, usable = cs.chain_features(types, ca, valid)
        at = data["position"][data["chain"] == c]
        seq_rows.append(seq[at])
        str_rows.append(structure[at])
        usable_rows.append(usable[at])
    return (
        np.concatenate(seq_rows),
        np.concatenate(str_rows),
        np.concatenate(usable_rows),
    )


def context_r2(
    y: np.ndarray,
    t: np.ndarray,
    seq: np.ndarray,
    structure: np.ndarray,
    usable: np.ndarray,
    chain: np.ndarray,
    seed: int,
) -> dict:
    """Held-out R² of the within-type residual of each column of `y` on the neighbours' types, the Cα geometry and both, and the same with the features' rows shuffled."""
    resid = cs.within_type_residual(y, t)[usable]
    groups = chain[usable]
    sets = {
        "sequence": seq[usable],
        "structure": structure[usable],
        "both": np.hstack([seq[usable], structure[usable]]),
    }
    rng = np.random.default_rng(seed)
    out = {}
    for name, x in sets.items():
        out[name] = cs.grouped_ridge_r2(x, resid, groups, seed=seed).tolist()
        out[name + "_shuffled"] = cs.grouped_ridge_r2(
            x[rng.permutation(len(x))], resid, groups, seed=seed
        ).tolist()
    out["residues"] = int(usable.sum())
    return out


def analyse(data: dict, adapters: dict, perm: int, seed: int) -> dict:
    native = data["native"]
    t, chain = data["types"], data["chain"]
    seq, structure, usable = context_matrix(data)
    native_id = native[:, list(IDENTITY)]
    native_centroids, counts = cs.type_centroids(native_id, t)
    keep = ~np.isnan(native_centroids).any(1)
    report: dict = {
        "residues": len(t),
        "chains": len(np.unique(chain)),
        "usable_for_context": int(usable.sum()),
        "type_counts": counts.tolist(),
        "native": {
            "identity_channels": list(IDENTITY),
            "centroids": native_centroids.tolist(),
            "within_type_share": cs.within_type_share(native, t).tolist(),
            "split_half_ceiling": cs.split_half_ceiling(
                native_id, t, chain, seed=seed
            ).tolist(),
            "context": context_r2(native_id, t, seq, structure, usable, chain, seed),
            "participation_ratio_32": cs.participation_ratio(
                np.cov(cs.within_type_residual(native, t).T)
            ),
            "participation_ratio_identity": cs.participation_ratio(
                np.cov(cs.within_type_residual(native_id, t).T)
            ),
        },
        "adapters": {},
    }
    for key, ad in adapters.items():
        arrays = data["adapters"][key]
        mean, mu, sigma = arrays["mean"], arrays["mu"], arrays["sigma"]
        slots = ad["slots"]
        code_id = mean[:, slots]
        centroids, _ = cs.type_centroids(code_id, t)
        both = keep & ~np.isnan(centroids).any(1)
        null = cs.permutation_null(
            native_centroids[both], centroids[both], n=perm, seed=seed
        )
        r1 = {
            "similarity_r2": cs.similarity_r2(native_centroids[both], centroids[both]),
            "null_p95": float(np.quantile(null, 0.95)),
            "null_p99": float(np.quantile(null, 0.99)),
            "null_median": float(np.median(null)),
            "affine_r2": cs.affine_r2(native_centroids[both], centroids[both]),
            "types": int(both.sum()),
        }
        # the identity slots of a deterministic code have no residual within types (a fixed code): the context reading and the participation ratio are not defined for it
        residual = cs.within_type_residual(code_id, t)
        spread = float(residual.var())
        entry: dict = {
            "kind": ad.get("kind")
            or (
                "learned"
                if ad["learned"]
                else (
                    "fixed-code"
                    if ad["source"] == "fixed"
                    else "fixed-sigma-or-deterministic"
                )
            ),
            "slots": slots,
            "centroids": centroids.tolist(),
            "alignment": r1,
            "within_type_share": cs.within_type_share(mean, t).tolist(),
            "identity_within_type_variance": spread,
        }
        if spread > 1e-12:
            entry["context"] = context_r2(code_id, t, seq, structure, usable, chain, seed)
            entry["participation_ratio_identity"] = cs.participation_ratio(
                np.cov(residual.T)
            )
            entry["participation_ratio_18"] = cs.participation_ratio(
                np.cov(cs.within_type_residual(mean, t).T)
            )
        if sigma is not None:
            by_channel = sigma.mean(0)
            quiet = np.flatnonzero(by_channel < 0.9)
            entry["sigma_by_channel"] = by_channel.tolist()
            entry["channels_below_0.9"] = quiet.tolist()
            if len(quiet) > 1:
                entry["participation_ratio_below_0.9"] = cs.participation_ratio(
                    np.cov(cs.within_type_residual(mean[:, quiet], t).T)
                )
            others = [c for c in range(sigma.shape[1]) if c not in slots]
            entry["regularizers"] = {
                "all": {
                    "rate": cs.kl_rate(sigma),
                    "exact": cs.kl_exact(mu, sigma),
                    "omitted": cs.kl_omitted(mu, sigma),
                },
                "identity_slots": {
                    "rate": cs.kl_rate(sigma[:, slots]),
                    "exact": cs.kl_exact(mu[:, slots], sigma[:, slots]),
                    "omitted": cs.kl_omitted(mu[:, slots], sigma[:, slots]),
                },
                "other_16": {
                    "rate": cs.kl_rate(sigma[:, others]),
                    "exact": cs.kl_exact(mu[:, others], sigma[:, others]),
                    "omitted": cs.kl_omitted(mu[:, others], sigma[:, others]),
                },
                "mean_mu_squared_by_group": {
                    "identity_slots": float((mu[:, slots] ** 2).mean()),
                    "other_16": float((mu[:, others] ** 2).mean()),
                },
            }
        report["adapters"][key] = entry
    return report


def text(report: dict) -> str:
    lines = [
        f"{report['chains']} chains, {report['residues']} standard residues ({report['usable_for_context']} usable for the context features)"
    ]
    nat = report["native"]
    lines.append(
        f"release encoder, channels 0 and 10: split-half ceiling of the centroids {np.median(nat['split_half_ceiling']):.3f} (median of {len(nat['split_half_ceiling'])}); "
        f"context R² sequence {nat['context']['sequence'][0]:.3f}/{nat['context']['sequence'][1]:.3f}, structure {nat['context']['structure'][0]:.3f}/{nat['context']['structure'][1]:.3f}, "
        f"both {nat['context']['both'][0]:.3f}/{nat['context']['both'][1]:.3f} (shuffled both {nat['context']['both_shuffled'][0]:.3f}/{nat['context']['both_shuffled'][1]:.3f}); "
        f"variance within types of channels 0, 10: {nat['within_type_share'][0]:.3f}, {nat['within_type_share'][10]:.3f}; participation ratio of the within-type residual {nat['participation_ratio_identity']:.2f} (identity), {nat['participation_ratio_32']:.2f} (32)"
    )
    for key, e in report["adapters"].items():
        r1 = e["alignment"]
        line = f"{key} [{e['kind']}] alignment: similarity R² {r1['similarity_r2']:.3f} (null median {r1['null_median']:.3f}, p95 {r1['null_p95']:.3f}, p99 {r1['null_p99']:.3f}; affine {r1['affine_r2']:.3f}, {r1['types']} types)"
        if "context" in e:
            c = e["context"]
            line += (
                f"; context both {c['both'][0]:.3f}/{c['both'][1]:.3f} (sequence {c['sequence'][0]:.3f}/{c['sequence'][1]:.3f}, structure {c['structure'][0]:.3f}/{c['structure'][1]:.3f}, "
                f"shuffled {c['both_shuffled'][0]:.3f}/{c['both_shuffled'][1]:.3f})"
            )
            line += f"; information_dimension identity within-type share {e['within_type_share'][e['slots'][0]]:.3f}/{e['within_type_share'][e['slots'][1]]:.3f}, participation ratio {e['participation_ratio_identity']:.2f} (identity), {e['participation_ratio_18']:.2f} (18)"
        lines.append(line)
        if "regularizers" in e:
            for group, v in e["regularizers"].items():
                if group != "mean_mu_squared_by_group":
                    lines.append(
                        f"   regularizers {group}: rate {v['rate']:.4f}, exact KL {v['exact']:.4f}, omitted {v['omitted']:.4f}"
                    )
            lines.append(
                f"   mean mu^2: {e['regularizers']['mean_mu_squared_by_group']}; channels with mean sigma below 0.9: {e['channels_below_0.9']}"
            )
        if "decoder_reading" in e:
            for mode, r in e["decoder_reading"]["readings"].items():
                c = r["complete"]
                lines.append(
                    f"   decoder_reading {mode}: types {c['sequence_accuracy']:.4f}, per-residue RMSD {c['per_residue_rmsd']:.3f} A, side chain {c['side_chain_rms']:.3f} A"
                )
            if "regression" in e["decoder_reading"]:
                lines.append(
                    f"   decoder_reading regression of the code reading against the recorded `end`: {e['decoder_reading']['regression']}"
                )
    return "\n".join(lines) + "\n"


def decode_readings(
    label: str,
    ad: dict,
    ae,
    dev,
    defaults: dict,
    mask_token: int,
    legal,
    device,
    batch_size: int,
    seed: int,
    modes: list[str],
    cache: dict,
    summary: dict | None,
    torch,
) -> dict:
    """The autoencoder of a learned posterior read on the dev chains at each of `modes`, the release encoder's means taken from the pass of `collect` where the mode needs no samples."""

    def native_fn(batch, sample=False):
        key = (tuple(batch["coords"].shape), float(batch["coords"].double().sum()))
        if not sample and key in cache:
            return cache[key].to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            out = ae.encoder(batch)
            return out["z_latent"] if sample else out["mean"]

    readings = {}
    for mode in modes:
        began = time.time()
        readings[mode] = autoencoder.read_dev(
            latent=ad["latent"],
            decoder=ad["decoder"],
            dev=dev,
            native_fn=native_fn,
            defaults=defaults,
            mask_token=mask_token,
            legal=legal,
            device=device,
            batch_size=batch_size,
            seed=seed,
            mode=mode,
            posterior_native=bool(ad["posterior"].get("native_sample", False)),
            label=f"{label} {mode}",
        )
        print(f"   {label} {mode} read in {time.time() - began:.0f} s", flush=True)
    out: dict = {"readings": readings}
    if summary is not None and "code" in readings and "end" in summary:
        # the recorded `end` of the summary is the reading of the unshrunk code (the trainer before the posterior mean was shrunk): the two must agree to what the hardware and the bf16 give
        a, b = readings["code"]["complete"], summary["end"]["complete"]
        out["regression"] = {
            "types_diff": a["sequence_accuracy"] - b["sequence_accuracy"],
            "rmsd_diff": a["per_residue_rmsd"] - b["per_residue_rmsd"],
            "side_chain_diff": a["side_chain_rms"] - b["side_chain_rms"],
            "recorded_types": b["sequence_accuracy"],
            "recorded_rmsd": b["per_residue_rmsd"],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="LABEL=path of an adapted autoencoder checkpoint (autoencoder.pt; its summary.json beside it is used for the regression of the decoder reading)",
    )
    parser.add_argument(
        "--bank",
        action="append",
        default=[],
        help="LABEL=path of a code bank (`src/foldspace/rfd4/code_bank.py`): the code of each dev residue is a donor of its type drawn at random from the bank, read by alignment, context and information-dimension readings as an adapter's code is "
        "(its context R² must be at the shuffled null, which is the check no_context of the control)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--dev-chains", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument(
        "--decode",
        default="",
        help="comma list of the labels whose autoencoder is read (the decoder reading)",
    )
    parser.add_argument("--decode-modes", default="code,mean,sampled")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.adapter:
        parser.error("give at least one --adapter LABEL=path")
    labels = [a.split("=", 1)[0] for a in args.adapter]
    banks_given = {b.split("=", 1)[0]: b.split("=", 1)[1] for b in args.bank}
    if len(set(labels)) != len(labels) or set(labels) & set(banks_given):
        parser.error("two adapters or banks have the same label")
    decode = [d for d in args.decode.split(",") if d]
    for d in decode:
        if d not in labels:
            parser.error(f"--decode names {d}, which is not an --adapter")

    import torch
    from rfproteina.datapipes.schema.proteina import SPECS
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.partial_autoencoder.autoencoder import AutoEncoder

    began = time.time()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    args.shards, args.ae = (str(Path(p).resolve()) for p in (args.shards, args.ae))
    adapters_given = {
        a.split("=", 1)[0]: str(Path(a.split("=", 1)[1]).resolve())
        for a in args.adapter
    }
    os.chdir(
        out.parent
    )  # the autoencoder makes ./tmp_ae/val_stuff in the working directory
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    index = json.loads((Path(args.shards) / "index.json").read_text())
    dev_ids = [r["id"] for r in index["dev"]][: args.dev_chains or None]
    dev = Shards(Path(args.shards), dev_ids)
    ae = AutoEncoder.load_from_checkpoint(
        args.ae, strict=True, weights_only=False, map_location="cpu"
    ).to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    defaults = {k: SPECS[k][1] for k in KEYS if k in SPECS}
    mask_token = ENC.token_to_idx["<M>"]
    legal = ae.decoder.residue_atom_mask
    std_of_token = standard_map(ENC)
    standard_rows = tuple(int(i) for i in np.flatnonzero(std_of_token >= 0))
    in_dim = int(dev.items[0]["esmc"].shape[-1])
    adapters = {
        k: build_adapter(p, ae, in_dim, standard_rows, ENC, torch)
        for k, p in adapters_given.items()
    }
    print(
        f"{len(dev_ids)} dev chains and {len(adapters)} adapters read in {time.time() - began:.0f} s",
        flush=True,
    )

    data = collect(
        ae, dev, adapters, defaults, mask_token, device, args.batch, std_of_token, torch
    )
    print(
        f"collected {len(data['types'])} residues in {time.time() - began:.0f} s",
        flush=True,
    )
    if banks_given:
        # a bank is read as an adapter whose code at each dev residue is a donor of the residue's type: the same alignment, context and information-dimension readings on a code that has nothing of the residue's own context
        token_of_std = np.full(len(ALPHABET), -1, dtype=np.int64)
        for token in np.flatnonzero(std_of_token >= 0):
            token_of_std[std_of_token[token]] = token
        code_idx = [c for c in range(32) if c not in autoencoder.INPLACE_KEEP]
        slots = [code_idx.index(c) for c in IDENTITY]
        rng = np.random.default_rng(args.seed)
        for key, path in banks_given.items():
            layout = dict(np.load(str(Path(path).resolve())))
            if [int(c) for c in layout["channels"]] != code_idx:
                parser.error(
                    f"{path} is a bank of channels {layout['channels'].tolist()}, not of the code channels {code_idx}"
                )
            draws = bk.draw_donors(layout, token_of_std[data["types"]], rng).astype(
                np.float64
            )
            data["adapters"][key] = {"mu": draws, "mean": draws, "sigma": None}
            adapters[key] = {
                "source": "fixed",
                "learned": False,
                "slots": slots,
                "kind": "swapped-code",
            }
    report = analyse(data, adapters, args.permutations, args.seed)
    for key in decode:
        summary_path = Path(adapters_given[key]).with_name("summary.json")
        summary = (
            json.loads(summary_path.read_text()) if summary_path.is_file() else None
        )
        report["adapters"][key]["decoder_reading"] = decode_readings(
            key,
            adapters[key],
            ae,
            dev,
            defaults,
            mask_token,
            legal,
            device,
            args.batch,
            args.seed,
            args.decode_modes.split(","),
            data["cache"],
            summary,
            torch,
        )
    report["args"] = {k: v for k, v in vars(args).items()}
    report["commit"] = os.environ.get("FS_STACK_COMMIT")
    report["seconds"] = round(time.time() - began, 1)
    out.write_text(json.dumps(report, indent=1) + "\n")
    out.with_suffix(".txt").write_text(text(report))
    sys.stdout.write(text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
