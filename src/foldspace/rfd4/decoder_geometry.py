"""The backbone geometry of an autoencoder's own reconstructions of real chains: is a broken backbone in a *generated* sample the decoder's rendering or the generator's?

A sample of the type-only control `type_only` has decoded `Cα` steps of 3.9 ± 0.17 Å, 57 to 61 % of them in 3.6 to 4.0, and a peptide C–N⁺ distance of 3.1 Å, against 3.8 Å, all of them and 1.34 Å for the native continuation. Its generated codes lie on its
table and its decoder reads them, so the break is in the joint of the generated `Cα` trace and code or in what the decoder renders from a code that holds no geometry. This reads the second alone: each autoencoder (the adapted decoder of an adapter behind its own
latent, the release's own decoder behind its encoder) decodes the dev chains from their *true* `Cα` and the code of their *true* sequence, and the geometry of what it returns is measured with the true chain's beside it, over consecutive residues that both
carry N, Cα and C: the `Cα` step and its share in 3.6 to 4.0 Å, the bonds N–Cα, Cα–C and C–N⁺, and how far the decoder's `Cα` lies from the `Cα` it was given (the generated samples' `Cα` is the decoder's own output). Two readings per autoencoder: the latent as it is
(the means) and with independent noise of 0.2 added (the posterior the adapted autoencoders were trained under).

    PYTHONPATH=<stack>/RFD4-Proteina/src python src/foldspace/rfd4/decoder_geometry.py --shards cache/rfd4esmc/shards --ae <release ae> \\
        --adapter esmc_only=<autoencoder.pt> --adapter type_only=<autoencoder.pt> --out geometry.json [--dev-chains 64]
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

N_SLOT, CA_SLOT, C_SLOT = 0, 1, 2
STEP_WINDOW = (3.6, 4.0)
TRAINING_NOISE = 0.2
SHOWN = ("ca_step_mean", "ca_step_std", "ca_step_in_window", "n_ca", "ca_c", "c_n_next")


def geometry_of(
    coords: np.ndarray, present: np.ndarray, valid: np.ndarray
) -> dict[str, list[float]]:
    """The backbone figures of one chain, as lists of per-pair or per-residue values: `coords` `[n, 37, 3]` in Å, `present` `[n, 37]` (the atoms the chain has), `valid` `[n]` (the residues scored).

    A residue counts when it is valid and has N, Cα and C; the `Cα` step and the peptide bond C–N⁺ are measured between a counted residue and the counted residue after it, and N–Cα and Cα–C within a counted residue.
    """
    ok = (
        valid.astype(bool)
        & present[:, N_SLOT].astype(bool)
        & present[:, CA_SLOT].astype(bool)
        & present[:, C_SLOT].astype(bool)
    )
    n, ca, c = coords[:, N_SLOT], coords[:, CA_SLOT], coords[:, C_SLOT]
    both = ok[:-1] & ok[1:]
    step = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)[both]
    return {
        "ca_step": step.tolist(),
        "n_ca": np.linalg.norm(n - ca, axis=-1)[ok].tolist(),
        "ca_c": np.linalg.norm(c - ca, axis=-1)[ok].tolist(),
        "c_n_next": np.linalg.norm(c[:-1] - n[1:], axis=-1)[both].tolist(),
    }


def summarize(pooled: dict[str, list[float]]) -> dict[str, float]:
    """The means of the pooled figures, the spread of the `Cα` step and the share of the steps in `STEP_WINDOW`."""
    step = np.asarray(pooled["ca_step"])
    return {
        "pairs": len(step),
        "ca_step_mean": float(step.mean()),
        "ca_step_std": float(step.std()),
        "ca_step_in_window": float(
            ((step > STEP_WINDOW[0]) & (step < STEP_WINDOW[1])).mean()
        ),
        "n_ca": float(np.mean(pooled["n_ca"])),
        "ca_c": float(np.mean(pooled["ca_c"])),
        "c_n_next": float(np.mean(pooled["c_n_next"])),
    }


def deviation_of(
    given: np.ndarray, returned: np.ndarray, valid: np.ndarray
) -> list[float]:
    """How far each valid residue's returned `Cα` lies from the `Cα` it was given, in Å: `given` and `returned` `[n, 3]`."""
    return np.linalg.norm(returned - given, axis=-1)[valid.astype(bool)].tolist()


def pool(chains: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for chain in chains:
        for key, values in chain.items():
            out.setdefault(key, []).extend(values)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument(
        "--adapter", action="append", default=[], help="LABEL=path of an adapted autoencoder checkpoint"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--dev-chains", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    labels = [a.split("=", 1)[0] for a in args.adapter]
    if not labels or len(set(labels)) != len(labels) or "release" in labels:
        parser.error(
            "give --adapter LABEL=path, each label once and none called release"
        )

    import torch
    from foldspace.rfd4.esmc_only_audit import build_latent
    from foldspace.rfd4.train_reading_adapter import KEYS as BATCH_KEYS
    from foldspace.rfd4.train_reading_adapter import Shards, collate
    from rfproteina.datapipes.schema.proteina import SPECS
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.partial_autoencoder.autoencoder import AutoEncoder

    began = time.time()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    adapters = {
        a.split("=", 1)[0]: str(Path(a.split("=", 1)[1]).resolve())
        for a in args.adapter
    }
    shards, ae_path = str(Path(args.shards).resolve()), str(Path(args.ae).resolve())
    os.chdir(
        out.parent
    )  # the autoencoder makes ./tmp_ae/val_stuff in the working directory
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    index = json.loads((Path(shards) / "index.json").read_text())
    dev_ids = [r["id"] for r in index["dev"]][: args.dev_chains or None]
    dev = Shards(Path(shards), dev_ids)
    ae = AutoEncoder.load_from_checkpoint(
        ae_path, strict=True, weights_only=False, map_location="cpu"
    ).to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    defaults = {k: SPECS[k][1] for k in BATCH_KEYS if k in SPECS}
    mask_token = ENC.token_to_idx["<M>"]
    in_dim = int(dev.items[0]["esmc"].shape[-1])

    readers = {"release": (ae.decoder, None)}
    for label, path in adapters.items():
        state = torch.load(path, map_location="cpu", weights_only=False)
        decoder = copy.deepcopy(ae.decoder)
        decoder.load_state_dict(state["decoder"])
        decoder.eval()
        readers[label] = (decoder, build_latent(path, in_dim, ENC, torch))
    gens = {
        k: torch.Generator().manual_seed(args.seed + i) for i, k in enumerate(readers)
    }
    truth: list[dict] = []
    # the release's own latent has its own posterior: only the adapted autoencoders are read under the noise they were trained with
    modes = {k: (("mean",) if k == "release" else ("mean", "noisy")) for k in readers}
    given_vs_returned: dict[str, list[float]] = {
        f"{k}/{m}": [] for k in readers for m in modes[k]
    }
    pooled: dict[str, list[dict]] = {f"{k}/{m}": [] for k in readers for m in modes[k]}
    for start in range(0, len(dev), args.batch):
        batch = collate(
            dev.items[start : start + args.batch], defaults, mask_token, device
        )
        with torch.no_grad():
            for label, (decoder, latent) in readers.items():
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    if latent is None:
                        z = ae.encoder(batch)["mean"].float()
                    else:
                        z = latent(batch).float()
                for mode in modes[label]:
                    zz = (
                        z
                        if mode == "mean"
                        else z
                        + TRAINING_NOISE
                        * torch.randn(z.shape, generator=gens[label]).to(z.device)
                    )
                    with torch.autocast(device.type, dtype=torch.bfloat16):
                        feed = dict(batch)
                        feed["z_latent"] = zz
                        dec = decoder(feed)
                    emit = (
                        (dec["emit_mask"] & batch["is_resolved_mask"].bool())
                        .cpu()
                        .numpy()
                    )
                    pred = dec["coors_nm"].float().cpu().numpy() * 10.0
                    true = batch["coords"].float().cpu().numpy()
                    present = batch["atom37_mask"].bool().cpu().numpy()
                    for i in range(len(emit)):
                        pooled[f"{label}/{mode}"].append(
                            geometry_of(pred[i], present[i], emit[i])
                        )
                        given_vs_returned[f"{label}/{mode}"].extend(
                            deviation_of(
                                true[i][:, CA_SLOT],
                                pred[i][:, CA_SLOT],
                                emit[i] & present[i][:, CA_SLOT],
                            )
                        )
                        if label == "release" and mode == "mean":
                            truth.append(geometry_of(true[i], present[i], emit[i]))
        print(
            f"{min(start + args.batch, len(dev))} of {len(dev)} chains, {time.time() - began:.0f} s",
            flush=True,
        )
    result = {
        "chains": len(dev),
        "true_chains": summarize(pool(truth)),
        "readings": {},
        "seconds": round(time.time() - began, 1),
    }
    for key, chains in pooled.items():
        entry = summarize(pool(chains))
        dev_ca = np.asarray(given_vs_returned[key])
        entry["ca_deviation_mean"] = float(dev_ca.mean())
        entry["ca_deviation_p90"] = float(np.quantile(dev_ca, 0.9))
        result["readings"][key] = entry
    out.write_text(json.dumps(result, indent=1))
    print(text(result), end="")
    return 0


def text(result: dict) -> str:
    lines = [f"{result['chains']} dev chains"]
    t = result["true_chains"]
    lines.append("true chains: " + ", ".join(f"{k} {t[k]:.3f}" for k in SHOWN))
    for key, e in result["readings"].items():
        lines.append(
            f"{key:14s}: "
            + ", ".join(f"{k} {e[k]:.3f}" for k in SHOWN)
            + f"; Ca given vs returned: mean {e['ca_deviation_mean']:.3f}, 90th percentile {e['ca_deviation_p90']:.3f} A"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
