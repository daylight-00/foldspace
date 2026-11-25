"""Unconditional samples from RFD4-Proteina's flow over the release latent or an ESM-derived one, and what they are.

A sample is a Cα trace and a latent per residue, integrated by the flow from noise (`fm.full_simulation`, the release's sampler with its
own schedule and recycle depths, from the configuration the checkpoint carries), and atoms and residue types read from them by the
autoencoder's decoder (`Proteina.sample_formatting`). Two sequences exist for it and they are kept apart:

* **decoder**: the decoder's own residue types, `condition_decoder_on_sequence_prediction` off, which is what an ESM-derived decoder was
  trained for (it reads the sequence from the latent and was given no sequence condition);
* **readout**: the flow's own sequence head (`sequence_readout`, detached from the trunk in training), the argmax of its logits over
  the amino acids at the end of the trajectory; for the release autoencoder this is what the release decodes with by default
  (`--decoder-sequence readout` gives the decoder the readout as its sequence condition, the release's setting).

For an ESM-derived latent the agreement of the two says whether the flow's sequence head and the latent it generated tell the same
sequence. The records of the decoder sequence and the Cα trace go to the frozen evaluator (`[{id, sequence, coordinates}]`); the
geometry here (consecutive Cα distances, Cα contacts, the radius of gyration against the corpus's fit, backbone bond lengths) is the cheap
check that a sample is a chain at all, and says nothing about whether it folds.

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/sample_flow.py --target esmc --adapter <autoencoder.pt> \\
        --flow <flow_pretrain-ema.ckpt> --ae <ae-ema.ckpt> --checkpoint <run>/ckpt/last.ckpt --lengths 80,120,160,200 --per-length 8 \\
        --out runs/rfd4esmc/flow/esmc/samples
"""

from __future__ import annotations

import argparse
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
import foldspace.rfd4.flow_trace as ft
from foldspace.rfd4.flow_align import drop_training_only

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
#: the corpus's own fit of the radius of gyration, `Rg = c * L ** k` (`natural_geometry.json`, as `src/foldspace/evaluation/rfd4_records.py`)
RG_COEFFICIENT, RG_EXPONENT = 2.843, 0.331


def generation_batch(
    lengths: list[int], extra: list[dict[str, np.ndarray]] | None = None
) -> dict:
    """A batch of unconditional monomers of the given lengths, as the release's collate makes it: every conditioning key at its default.

    `extra` gives chain `k` features of its own beyond those (the sequence conditioning of `src/foldspace/rfd4/flow_data.py`, an adapter's rows): the fold script asks for a
    structure given a sequence; with none the batch is the unconditional one that every arm's samples were drawn from."""
    import torch
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.encoding import TOKEN_TYPE_IDS

    samples = []
    for k, n in enumerate(lengths):
        path, ordering = fd.pair_features(n)
        feats = {
            "token_type": np.full(n, TOKEN_TYPE_IDS["aa"], dtype=np.int32),
            "chain_iid": np.zeros(n, dtype=np.int16),
            **fd.monomer_index_features(n),
            "supports_local_latent": np.ones(n, dtype=bool),
            "is_in_context_conditioning": np.zeros(n, dtype=bool),
            "is_resolved_mask": np.ones(n, dtype=bool),
            "token_bond_path_length": path,
            "canonical_polymer_ordering": ordering,
            **(extra[k] if extra is not None else {}),
        }
        samples.append(
            {
                "example_id": f"sample{k}",
                "feats": {k_: torch.from_numpy(v) for k_, v in feats.items()},
            }
        )
    padded = collate_batch(samples, fd.SCHEMA, keep_atom_array=False)
    batch = dict(padded.data_samples.tensors)
    batch["padding_mask"] = padded["padding_mask"]
    batch["example_id_batch"] = padded["example_id_batch"]
    return batch


def restore(
    model, checkpoint: str | Path, state: str, expect_step: int | None = None
) -> dict:
    """Load a run's weights into `model`: the EMA's (the callback's state) or the raw ones; returns what the checkpoint says of itself. With `expect_step` a
    checkpoint that is at another update is refused: a run continued from a copy of an earlier checkpoint can leave that copy under the name a later
    reading opens, and a reading of the wrong weights is then a result that looks like a finding."""
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if expect_step is not None and int(ck["global_step"]) != int(expect_step):
        raise SystemExit(
            f"{checkpoint} is at update {int(ck['global_step'])}, and update {int(expect_step)} was asked for: the weights read are not the ones meant"
        )
    if state == "ema":
        weights = ck["callbacks"]["SimpleEMA"]["ema_state_dict"]
    else:
        weights = ck["state_dict"]
    # the alignment projector of a checkpoint trained with `src/foldspace/rfd4/flow_align.py` exists for training only: generation does not call ESMC and does not need it
    weights, dropped = drop_training_only(weights)
    from foldspace.rfd4.conditional_lora import attach, checkpoint_config

    conditional = checkpoint_config(ck)
    if conditional is not None:
        attach(model, conditional)
    model.load_state_dict(weights, strict=True)
    return {
        "global_step": int(ck["global_step"]),
        "state": state,
        "nsamples_processed": ck.get("nsamples_processed"),
        "training_only_weights_dropped": len(dropped),
    }


#: the modalities of the generation configuration a flow checkpoint carries (`generation.model.<modality>.simulation_step_params`): the Cα trace and the latent
MODALITIES = ("bb_ca", "local_latents")


def apply_noise_scales(model_cfg, scales: str) -> None:
    """Set the injected-noise scale of the Cα and of the latent (`"1.0,1.0"`, in that order); an empty string keeps the checkpoint's own, which for both flow
    checkpoints of the release is 0.3 and 0.1 while the release's inference defaults (`configs/generation/inference_base.yaml`) are 1.0 and 1.0: the two
    numbers are what differs between the arms' sampler and the release's, and this is how a draw is made at the other's."""
    if not scales:
        return
    values = [float(x) for x in scales.split(",")]
    if len(values) != 2 or min(values) < 0:
        raise ValueError(
            f"noise scales are two numbers that are not negative, the Cα's and the latent's, not {scales!r}"
        )
    for modality, value in zip(MODALITIES, values, strict=True):
        model_cfg[modality]["simulation_step_params"]["sc_scale_noise"] = value


def sampler_record(model_cfg) -> dict:
    """What the sampler integrates with, read back from the configuration in force and not from what was asked for (the sampling mode and the injected-noise
    scale of each modality): a setting that was requested and did not take is the kind that invalidates a comparison without showing."""
    out = {}
    for modality in MODALITIES:
        step = model_cfg[modality]["simulation_step_params"]
        out[modality] = {
            "sampling_mode": str(step["sampling_mode"]),
            "sc_scale_noise": float(step["sc_scale_noise"]),
        }
    return out


def geometry(coors: np.ndarray, atom_mask: np.ndarray) -> dict:
    """The cheap chain checks of one sample: `coors` `[n, 37, 3]` in Å, `atom_mask` `[n, 37]`; Cα is slot 1, N slot 0, C slot 2."""
    n = len(coors)
    ca = coors[:, 1]
    step = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    far = np.abs(np.arange(n)[:, None] - np.arange(n)[None]) > 2
    out = {
        "length": n,
        "ca_step_mean": float(step.mean()),
        "ca_step_std": float(step.std()),
        "ca_step_in_3p6_4p0": float(((step > 3.6) & (step < 4.0)).mean()),
        "ca_step_broken": int((step > 4.5).sum()),
        "ca_clashes_under_3p0": int(((d < 3.0) & far).sum() // 2),
        "rg": float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean())),
        "rg_expected": RG_COEFFICIENT * n**RG_EXPONENT,
        "contacts_per_residue": float(((d < 8.0) & far).sum() / n),
    }
    ok = atom_mask[:, 0] & atom_mask[:, 1] & atom_mask[:, 2]
    if ok.sum() > 1:
        out["n_ca_mean"] = float(
            np.linalg.norm(coors[ok, 0] - coors[ok, 1], axis=-1).mean()
        )
        out["ca_c_mean"] = float(
            np.linalg.norm(coors[ok, 2] - coors[ok, 1], axis=-1).mean()
        )
        both = ok[:-1] & ok[1:]
        if both.any():
            out["c_n_next_mean"] = float(
                np.linalg.norm(coors[:-1][both, 2] - coors[1:][both, 0], axis=-1).mean()
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", required=True, choices=("native", "esmc", "fixed"))
    parser.add_argument("--adapter", default="")
    parser.add_argument(
        "--flow",
        required=True,
        help="the flow checkpoint whose configuration (and, without --checkpoint, weights) is used",
    )
    parser.add_argument("--ae", required=True)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="a run's Lightning checkpoint; without it the weights of --flow",
    )
    parser.add_argument(
        "--bundle",
        default="",
        help="a release `model.safetensors` (the adapter merged into the base): the denoiser's weights, with --flow giving the configuration only",
    )
    parser.add_argument("--state", default="ema", choices=("ema", "raw"))
    parser.add_argument(
        "--expect-step",
        type=int,
        default=None,
        help="refuse a --checkpoint that is not at this update (the number the run was asked for)",
    )
    parser.add_argument("--lengths", default="80,120,160,200")
    parser.add_argument("--per-length", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--nsteps",
        type=int,
        default=0,
        help="integration steps (the checkpoint's setting by default)",
    )
    parser.add_argument(
        "--decoder-sequence", default="decoder", choices=("decoder", "readout")
    )
    parser.add_argument(
        "--noise-scales",
        default="",
        help="the injected-noise scales of the Cα and of the latent, `1.0,1.0` (the release's inference defaults); the checkpoint's own (0.3 and 0.1) by default",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=0,
        help="keep the moments of the generated latent along the trajectory at every K-th step (`fm.full_simulation`'s `save_trajectory_every`): `trace.npz` beside the samples (`src/foldspace/rfd4/flow_trace.py`)",
    )
    parser.add_argument(
        "--calibrate",
        default="",
        help="a calibration (`src/foldspace/rfd4/code_calibration.py`): the generated latents are mapped by its affine map before they are decoded; the samples are then those of a distribution changed after training and are not the result of an arm",
    )
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    began = time.time()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    flow, ae = (str(Path(p).resolve()) for p in (args.flow, args.ae))
    adapter = str(Path(args.adapter).resolve()) if args.adapter else None
    if (args.target != "native") != bool(adapter):
        parser.error(
            "--target esmc and fixed need --adapter, and --target native takes none"
        )
    device = torch.device("cpu" if args.cpu else "cuda")
    overrides = {"use_torch_compile": False}
    if args.cpu:
        overrides.update(tri_mult_backend="torch", norm_type="rmsnorm_torch")
    bundle = str(Path(args.bundle).resolve()) if args.bundle else None
    if bundle and args.checkpoint:
        parser.error("--bundle and --checkpoint both name the denoiser's weights")
    model, _ = fm.load_flow(flow, ae, out / "store", adapter, overrides, bundle=bundle)
    restored = (
        restore(model, args.checkpoint, args.state, args.expect_step)
        if args.checkpoint
        else {"weights": bundle or flow, "configuration": flow}
    )
    model.eval().to(device)
    cfg = model.inf_cfg
    cfg.args.condition_decoder_on_sequence_prediction = (
        args.decoder_sequence == "readout"
    )
    apply_noise_scales(cfg.model, args.noise_scales)
    sampler = sampler_record(cfg.model)
    print(f"sampler in force: {json.dumps(sampler)}", flush=True)
    nsteps = args.nsteps or int(cfg.args.nsteps)
    letters = {}
    for i in range(ENC.n_tokens):
        name = ENC.idx_to_token[i]
        if name in THREE_TO_ONE:
            letters[i] = THREE_TO_ONE[name]
    amino = torch.tensor(sorted(letters), device=device)
    calibration = None
    if args.calibrate:
        cal = json.loads(Path(args.calibrate).read_text())
        calibration = (
            torch.tensor(cal["map"], dtype=torch.float32, device=device),
            torch.tensor(cal["shift"], dtype=torch.float32, device=device),
        )
        print(
            f"calibration {args.calibrate}: {cal['label']}, ridge {cal['ridge']}",
            flush=True,
        )
    tracer = ft.Trace() if args.trace_every else None

    lengths = [int(x) for x in args.lengths.split(",") if x]
    records, rows, arrays = [], [], {}
    for length in lengths:
        for first in range(0, args.per_length, args.batch):
            b = min(args.batch, args.per_length - first)
            torch.manual_seed(args.seed + 100003 * length + first)
            batch = generation_batch([length] * b)
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            batch = model.preprocess_batch_for_generation(batch)
            with (
                torch.no_grad(),
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.cpu),
            ):
                gen, extra = model.fm.full_simulation(
                    batch=batch,
                    predict_for_sampling=model.predict_for_sampling,
                    recycle_depths=model.recycle_depths(nsteps),
                    nsteps=nsteps,
                    nsamples=b,
                    n=length,
                    sampling_model_args=cfg.model,
                    device=device,
                    guidance_w=cfg.args.get("guidance_w", 1.0),
                    guidance_model=None,
                    save_trajectory_every=args.trace_every,
                    dump_noisy_trajectories=bool(args.trace_every),
                )
                if tracer is not None:
                    tracer.add(extra["trajectory_steps"])
                if calibration is not None:
                    gen = dict(gen)
                    z = gen["local_latents"]
                    with torch.autocast(device.type, enabled=False):
                        gen["local_latents"] = (
                            z.float() @ calibration[0] + calibration[1]
                        ).to(z.dtype)
                batch["sequence_logits"] = extra.get("sequence_logits")
                prots = model.sample_formatting(
                    x=gen, ret_mode="coors37_n_aatype", batch=batch
                )
            logits = extra.get("sequence_logits")
            for i in range(b):
                name = f"flow-{args.target}-L{length}-{first + i:03d}"
                types = prots["residue_type"][i].cpu().numpy()
                coors = prots["coors"][i].float().cpu().numpy()
                mask = prots["atom_mask"][i].cpu().numpy()
                decoder_seq = "".join(letters.get(int(t), "X") for t in types)
                row = {"id": name, "length": length, "sequence_decoder": decoder_seq}
                if logits is not None:
                    pick = logits[i][:, : ENC.n_tokens].float()[:, amino].argmax(-1)
                    readout = "".join(
                        letters[int(amino[j])] for j in pick.cpu().numpy()
                    )
                    row["sequence_readout"] = readout
                    row["readout_agrees_with_decoder"] = float(
                        np.mean(
                            [a == c for a, c in zip(decoder_seq, readout, strict=True)]
                        )
                    )
                row.update(geometry(coors, mask))
                row["decoder_unknown_residues"] = int(decoder_seq.count("X"))
                rows.append(row)
                # the evaluator reads twenty letters: a design with a residue the decoder left unknown is not given to it (and counted)
                if "X" not in decoder_seq:
                    records.append(
                        {
                            "id": name,
                            "sequence": decoder_seq,
                            "coordinates": coors[:, 1].round(3).tolist(),
                        }
                    )
                arrays[name] = {
                    "coors": coors.astype(np.float16),
                    "atom_mask": mask,
                    "z": gen["local_latents"][i]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                }
        print(
            f"length {length}: {args.per_length} samples, {time.time() - began:.0f} s",
            flush=True,
        )
    torch.save(arrays, out / "samples.pt")
    if tracer is not None:
        ts, _ = model.fm.sample_schedule(nsteps=nsteps, sampling_model_args=cfg.model)
        times = {
            m: torch.as_tensor(ts[m]).float().cpu().numpy()
            for m in ("local_latents", "bb_ca")
        }
        np.savez(
            out / "trace.npz", **tracer.arrays(times["local_latents"], times["bb_ca"])
        )
    (out / "records.json").write_text(json.dumps(records))
    keys = (
        "ca_step_mean", "ca_step_in_3p6_4p0", "ca_step_broken", "ca_clashes_under_3p0",
        "contacts_per_residue", "n_ca_mean", "ca_c_mean", "c_n_next_mean", "readout_agrees_with_decoder", "decoder_unknown_residues",
    )  # fmt: skip
    summary = {
        k: float(np.mean([r[k] for r in rows if k in r]))
        for k in keys
        if any(k in r for r in rows)
    }
    summary["rg_over_expected"] = float(
        np.mean([r["rg"] / r["rg_expected"] for r in rows])
    )
    composition = {
        c: float(np.mean([r["sequence_decoder"].count(c) / r["length"] for r in rows]))
        for c in sorted(set(letters.values()))
    }
    report = {
        "args": vars(args),
        "restored": restored,
        "nsteps": nsteps,
        "sampler": sampler,
        "samples": len(rows),
        "records_for_the_evaluator": len(records),
        "summary": summary,
        "composition_decoder_sequence": composition,
        "rows": rows,
        "seconds": round(time.time() - began, 1),
        "stack_commit": os.environ.get("FS_STACK_COMMIT"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"summary": summary, "seconds": report["seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
