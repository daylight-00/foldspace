"""The data path: corpus chains through RFD4-Proteina's own pipeline and its native autoencoder.

The corpus (`data/corpus/chains.npz`) holds N, Cα and C of each chain; the autoencoder reads all atoms. So a chain is read again from
the mmCIF mirror with RFD4's parse preset and validation pipeline (`build_design_validation_pipeline`, what the autoencoder's own
validation runs), cut to the residues the corpus admitted, collated as RFD4 collates, and passed through the release autoencoder:
encoder -> posterior (mean, log scale) -> decoder. Nothing of the pipeline is re-implemented; what this file adds is the comparison
with the corpus and the scores. It runs in RFD4's own environment (`.venv-rfd4`) on a CPU:

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/ae_datapath.py \\
        --chains data/corpus/chains.npz --split-table data/corpus/split.tsv.gz \\
        --ae <snapshot>/ae_checkpoints/rfd4_proteina_ae-ema.ckpt --out runs/rfd4esmc/datapath/<label>

Three checks, each a number in `summary.json`, and a record per chain in `chains.jsonl`:

1. **The mapping.** The residues the pipeline returns for a chain are the corpus's residues in the corpus's order (the longest run of
   consecutive residues with N, Cα and C: `scripts/pdb_reference_set.py:backbone_trace`): same count, same residue types, Cα and the
   backbone within a tolerance after one rigid superposition (the pipeline recentres), bond lengths in angstroms (Cα to Cα 3.8, not
   0.38), the masks the encoder derives all true on the chain's residues.
2. **The reconstruction, without the answer.** The decoder reads `condition_sequence_1_residue` (`C_SEQ`) as an input, and `decode()`
   lets it override the predicted residue type. Every score here is with `C_SEQ` set to the mask token on every residue (asserted);
   the same decoder with the true sequence given is reported apart, as what a leak looks like and not as a recovery. The latent is
   decoded as the posterior mean and as a sample, both reported. Scored: residue-type accuracy, atom-presence F1, the distance of the
   predicted atoms from the structure's (per residue, backbone, side chain).
3. **The latent.** The posterior's mean and scale per channel over all residues of the chains, its KL to N(0, I), the covariance of the
   mean, and how much of the mean the residue type alone fixes (`src/foldspace/rfd4/datapath_stats.py`).

A chain that fails any step is a record with its reason, not a stop.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foldspace.rfd4.datapath_stats as ds

#: a Cα (or N, C) of the pipeline within this many angstroms of the corpus's, after one rigid fit, is the same atom
BACKBONE_TOLERANCE = 0.05
#: Cα to Cα of consecutive residues in a chain, in angstroms: what a unit mistake would move by a factor of ten
CA_CA = (3.6, 4.0)

# AlphaFold's one-letter ordering, which the corpus's `tokens` use (`foldspace.evaluation.residues`)
CORPUS_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip


def longest_complete_run(res_ids: np.ndarray, complete: np.ndarray) -> list[int]:
    """The residue ids of the longest run of consecutive ids that are all complete (N, Cα and C present); the first of equals.

    `res_ids` are the distinct residue ids in ascending order and `complete[i]` says whether residue `res_ids[i]` has all three
    atoms. This is the corpus's rule (`backbone_trace`): ids that differ by one are consecutive, a gap or an incomplete residue
    ends a run, and of two runs of one length the earlier is kept.
    """
    ordered = [int(r) for r, ok in zip(res_ids, complete, strict=True) if ok]
    if not ordered:
        return []
    best: list[int] = []
    run = [ordered[0]]
    for value in ordered[1:]:
        if value - run[-1] == 1:
            run.append(value)
        else:
            best = max(best, run, key=len)
            run = [value]
    return max(best, run, key=len)


def split_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def corpus_chain(z, index: int) -> dict:
    """The corpus's own record of chain `index`: letters, and the N, Cα, C coordinates."""
    start, length = int(z["starts"][index]), int(z["lengths"][index])
    letters = "".join(
        CORPUS_ALPHABET[int(t)] for t in z["tokens"][start : start + length]
    )
    return {
        "letters": letters,
        "xyz": np.asarray(z["xyz"][start : start + length], dtype=np.float64),
    }


def parsed_chain(data: dict, asym: str):
    """`(array, source)`: the polymer atoms of label chain `asym` as a biotite array, from the first model of assembly 1 or, when
    the corpus read a copy that assembly 1 does not contain, from the asymmetric unit.

    The pipeline needs the annotations of an assembly (`pn_unit_iid` and the like), which the asymmetric unit does not carry; a chain
    taken from it gets the instance annotations of an assembly's first transform (`<id>_1`). When assembly 1 holds several copies of the
    chain, the first by instance id is taken (the identity transform).
    """
    array = data["assemblies"]["1"][0]
    keep = (array.chain_id == asym) & array.is_polymer
    instances = sorted({str(i) for i in array.chain_iid[keep]})
    if instances:
        return array[keep & (array.chain_iid == instances[0])], "assembly 1"
    unit = data["asym_unit"]
    model = unit[0] if getattr(unit, "stack_depth", lambda: 0)() > 0 else unit
    keep = (model.chain_id == asym) & model.is_polymer
    if not keep.any():
        raise LookupError(
            f"chain {asym} is in neither assembly 1 nor the asymmetric unit"
        )
    chain = model[keep]
    categories = set(chain.get_annotation_categories())
    for name in ("chain", "pn_unit", "molecule"):
        if f"{name}_id" in categories:
            ids = np.asarray(chain.get_annotation(f"{name}_id")).astype(str)
            chain.set_annotation(f"{name}_iid", np.char.add(ids, "_1"))
    chain.set_annotation("transformation_id", np.full(chain.array_length(), "1"))
    return chain, "asymmetric unit"


def residue_table(chain) -> dict:
    """Per residue of a parsed chain, in `res_id` order: its name and whether N, Cα and C are all there with finite coordinates."""
    names = chain.atom_name
    coords = chain.coord
    finite = np.isfinite(coords).all(-1)
    out_ids, out_names, complete = [], [], []
    for rid in np.unique(chain.res_id):
        rows = np.flatnonzero(chain.res_id == rid)
        present = {str(names[i]) for i in rows if finite[i]}
        out_ids.append(int(rid))
        out_names.append(str(chain.res_name[rows[0]]))
        complete.append({"N", "CA", "C"} <= present)
    return {
        "ids": np.array(out_ids),
        "names": out_names,
        "complete": np.array(complete),
    }


def read_chain(
    name: str,
    z,
    index_of,
    row_of,
    mirror,
    parse,
    preset,
    pipeline,
    collate_batch,
    schema: str,
    record: dict,
) -> tuple[dict, dict, str]:
    """The corpus chain `name` as a collated batch of one, with the corpus's own record of it and the letters of the residues read.

    The chain is read from the mmCIF mirror, cut to the corpus's run of residues, and passed through RFD4's validation pipeline and
    collate. What is learned on the way (lengths, sequence agreement, atoms without coordinates, the pipeline's time) is written
    into `record`.
    """
    row = row_of[name]
    entry, asym = row["entry"].lower(), row["chain"]
    reference = corpus_chain(z, index_of[name])
    data = parse(mirror / f"{entry}.cif", config=preset)
    chain, source = parsed_chain(data, asym)
    record["chain_from"] = source
    table = residue_table(chain)
    run = longest_complete_run(table["ids"], table["complete"])
    at = {int(r): i for i, r in enumerate(table["ids"])}
    letters = "".join(THREE_TO_ONE.get(table["names"][at[r]], "X") for r in run)
    record.update(
        {
            "corpus_length": len(reference["letters"]),
            "run_length": len(run),
            "chain_residues": len(table["ids"]),
            "same_length": len(run) == len(reference["letters"]),
            "same_sequence": letters == reference["letters"],
            "nonstandard_residues": letters.count("X"),
            "nonstandard_names": sorted(
                {
                    table["names"][at[r]]
                    for r in run
                    if table["names"][at[r]] not in THREE_TO_ONE
                }
            ),
            "sequence_mismatches_standard": int(
                sum(
                    a != b
                    for a, b in zip(letters, reference["letters"], strict=False)
                    if a != "X"
                )
            ),
        }
    )
    window = chain[np.isin(chain.res_id, run)]
    # An unresolved atom is absent, not a placeholder: the autoencoder's own held-out structures omit non-finite
    # coordinates, and one NaN coordinate makes every latent of the chain NaN.
    finite = np.isfinite(window.coord).all(-1)
    heavy = window.element != "H"
    record["atoms_without_coordinates"] = int(
        ((~finite) & heavy & (window.atom_name != "OXT")).sum()
    )
    record["terminal_oxygens_without_coordinates"] = int(
        ((~finite) & (window.atom_name == "OXT")).sum()
    )
    window = window[finite]
    data["atom_array"] = window
    data["example_id"] = name
    t_pipeline = time.time()
    sample = pipeline(data)
    record["pipeline_seconds"] = round(time.time() - t_pipeline, 2)
    feats = sample["feats"]
    n_tokens = int(feats["residue_type"].shape[0])
    record["tokens"] = n_tokens
    batch = collate_batch([sample], schema=schema, keep_atom_array=False)
    batch = {**batch}
    return batch, reference, letters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chains", required=True)
    parser.add_argument("--split-table", required=True)
    parser.add_argument(
        "--mirror", default="~/workspaces/databases/alphafold3/mmcif_files"
    )
    parser.add_argument(
        "--ae",
        required=True,
        help="the autoencoder checkpoint (rfd4_proteina_ae-ema.ckpt)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-bin", type=int, default=32)
    parser.add_argument("--stress", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated ids: read these chains of the corpus, not the chosen ones",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only the first N chosen chains (a smoke run)",
    )
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    import rfproteina
    from atomworks.io import parse
    from atomworks.ml.conditions import C_SEQ
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
    from rfproteina.utils.parse_presets import RCSB_WITH_METALS

    began = time.time()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.chains, args.split_table, args.ae = (
        str(Path(p).resolve()) for p in (args.chains, args.split_table, args.ae)
    )
    os.chdir(out)  # the autoencoder makes ./tmp_ae/val_stuff in the working directory
    z = np.load(args.chains, allow_pickle=False)
    ids = [str(i) for i in z["ids"]]
    index_of = {name: i for i, name in enumerate(ids)}
    lengths = {name: int(z["lengths"][i]) for name, i in index_of.items()}
    rows = split_rows(Path(args.split_table))
    row_of = {f"{r['entry']}_{r['entity']}": r for r in rows}
    chosen = ds.choose_chains(
        rows, lengths, per_bin=args.per_bin, stress=args.stress, seed=args.seed
    )
    names = chosen["ids"][: args.limit] if args.limit else chosen["ids"]
    if args.only:
        names = [n for n in args.only.split(",") if n]
        chosen["why"].update({n: chosen["why"].get(n, "named") for n in names})
    (out / "selection.json").write_text(json.dumps(chosen, indent=1))

    mirror = Path(args.mirror).expanduser()
    ae = AutoEncoder.load_from_checkpoint(
        args.ae, strict=True, weights_only=False, map_location="cpu"
    )
    ae.eval()
    pipeline = build_design_validation_pipeline()
    mask_token = ENC.token_to_idx["<M>"]
    schema = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
    print(
        f"{len(names)} chains; rfproteina {rfproteina.__file__}; "
        f"encoder {ae.nparams_enc / 1e6:.1f} M, decoder {ae.nparams_dec / 1e6:.1f} M parameters",
        flush=True,
    )

    records: list[dict] = []
    pool: dict[str, list[np.ndarray]] = {
        "mean": [],
        "log_scale": [],
        "sample": [],
        "residue_type": [],
    }
    gen = torch.Generator().manual_seed(args.seed)
    for n, name in enumerate(names):
        record: dict = {"id": name, "why": chosen["why"][name]}
        t0 = time.time()
        try:
            batch, reference, letters = read_chain(
                name,
                z,
                index_of,
                row_of,
                mirror,
                parse,
                RCSB_WITH_METALS,
                pipeline,
                collate_batch,
                schema,
                record,
            )
            n_tokens = int(batch["residue_type"].shape[1])
            coords = batch["coords"]  # [1, n, 37, 3], angstroms
            mask37 = batch["atom37_mask"].bool()
            # -- 1. the mapping --------------------------------------------------------------------------------
            ca = coords[0, :, 1].numpy().astype(np.float64)
            record["ca_ca_median"] = float(
                np.median(np.linalg.norm(np.diff(ca, axis=0), axis=-1))
            )
            record["units_ok"] = bool(CA_CA[0] < record["ca_ca_median"] < CA_CA[1])
            one_to_one = n_tokens == len(reference["letters"])
            record["one_to_one"] = one_to_one
            if one_to_one:
                record["ca_rmsd_to_corpus"] = ds.kabsch_rmsd(ca, reference["xyz"][:, 1])
                record["backbone_rmsd_to_corpus"] = ds.kabsch_rmsd(
                    coords[0][:, [0, 1, 2]].reshape(-1, 3).numpy().astype(np.float64),
                    reference["xyz"].reshape(-1, 3),
                )
                record["mapping_ok"] = bool(
                    record["backbone_rmsd_to_corpus"] < BACKBONE_TOLERANCE
                    and record["sequence_mismatches_standard"] == 0
                    and record["units_ok"]
                )
            else:
                record["mapping_ok"] = False
            rtype = batch["residue_type"][0].numpy()
            record["residue_types_in_corpus_alphabet"] = bool(
                one_to_one
                and all(
                    ENC.tokens[int(t)] == a
                    for t, a, letter in zip(
                        rtype, _three(reference["letters"]), letters, strict=True
                    )
                    if letter != "X"
                )
            )
            masks = {
                "padding_mask": bool(batch["padding_mask"][0].all()),
                "supports_local_latent": bool(batch["supports_local_latent"][0].all()),
                "is_resolved_mask": bool(batch["is_resolved_mask"][0].all()),
                "none_in_context": bool(~batch["is_in_context_conditioning"][0].any()),
            }
            record["masks"] = masks
            # The pipeline gives the identity of a modified residue as a sequence condition; a standard residue's is the mask token.
            given = batch[C_SEQ.full_name] != mask_token
            record["c_seq_given_by_the_pipeline"] = int(given.sum())
            record["atoms_per_residue_mean"] = float(mask37[0].sum(-1).float().mean())
            legal = ae.decoder.residue_atom_mask[batch["residue_type"]]
            # OXT (slot 36) is legal for every residue and removed by the pipeline: a residue is complete without it
            required = legal.clone()
            required[..., 36] = False
            complete = ~(required & ~mask37).any(-1)  # [1, n]
            record["residues_complete"] = float(complete.float().mean())
            record["missing_legal_by_slot"] = (required & ~mask37)[0].sum(0).tolist()
            # -- 2. the autoencoder ----------------------------------------------------------------------------
            with torch.no_grad():
                feed = dict(batch)
                feed["coord_mask"] = batch["atom37_mask"]
                feed["coords_nm"] = coords / 10.0
                feed["ca_coors_nm"] = feed["coords_nm"][..., 1, :]
                # Every score below is with the sequence condition removed on every residue; the decoder reads it as an input.
                feed[C_SEQ.full_name] = torch.full_like(
                    batch[C_SEQ.full_name], mask_token
                )
                record["c_seq_all_masked"] = bool(
                    (feed[C_SEQ.full_name] == mask_token).all()
                )
                enc = ae.encoder(feed)
                mean, log_scale = enc["mean"], enc["log_scale"]
                eps = torch.randn(mean.shape, generator=gen)
                sample_z = (mean + eps * torch.exp(log_scale)) * enc["emit_mask"][
                    ..., None
                ]
                emit = (enc["emit_mask"] & batch["is_resolved_mask"].bool())[0].numpy()
                emitting = enc["emit_mask"][0].numpy()
                results = {}
                for label, latent in (("mean", mean), ("sample", sample_z)):
                    feed["z_latent"] = latent
                    dec = ae.decoder(feed)
                    results[label] = dec
                # the leak control: the same decoder, given the true sequence on every residue
                leak = dict(feed)
                leak["z_latent"] = mean
                leak[C_SEQ.full_name] = batch["residue_type"].clone()
                leak_dec = ae.decoder(leak)
                leak_full = ae.decode(leak)
                clean_full = ae.decode({**feed, "z_latent": mean})
            true_types = batch["residue_type"][0].numpy()
            target = coords[0].numpy().astype(np.float64)
            tmask = mask37[0].numpy()
            legal_np = legal[0].numpy()
            complete_np = complete[0].numpy()
            parts = {
                "all": emit,
                "complete": emit & complete_np,
                "truncated": emit & ~complete_np,
            }
            record["residues_scored"] = {k: int(v.sum()) for k, v in parts.items()}
            for label, dec in results.items():
                pred = dec["coors_nm"][0].numpy() * 10.0
                predicted_types = dec["aatype_max"][0].numpy()
                presence = (dec["atom_presence_logits"][0] > 0).numpy()
                for part, scored in parts.items():
                    key = label if part == "all" else f"{label}_{part}"
                    record[f"{key}_structure"] = ds.residue_scores(
                        target, pred, tmask, scored
                    )
                    record[f"{key}_sequence"] = ds.sequence_scores(
                        true_types, predicted_types, scored, ENC.n_tokens
                    )
                    record[f"{key}_presence"] = ds.presence_scores(
                        tmask, presence, legal_np, scored
                    )
            record["leak_sequence"] = ds.sequence_scores(
                true_types, leak_dec["aatype_max"][0].numpy(), emit, ENC.n_tokens
            )
            record["leak_decode_sequence"] = ds.sequence_scores(
                true_types, leak_full["residue_type"][0].numpy(), emit, ENC.n_tokens
            )
            record["clean_decode_sequence"] = ds.sequence_scores(
                true_types, clean_full["residue_type"][0].numpy(), emit, ENC.n_tokens
            )
            keep = np.flatnonzero(emitting)
            pool["mean"].append(mean[0].numpy()[keep])
            pool["log_scale"].append(log_scale[0].numpy()[keep])
            pool["sample"].append(sample_z[0].numpy()[keep])
            pool["residue_type"].append(true_types[keep])
            record["ok"] = True
        except Exception as error:  # noqa: BLE001 -- a chain that fails is a record
            record["ok"] = False
            record["failure"] = f"{type(error).__name__}: {error}"
        record["seconds"] = round(time.time() - t0, 2)
        records.append(record)
        if (n + 1) % 8 == 0 or n + 1 == len(names):
            print(
                f"{n + 1}/{len(names)} chains, {sum(r['ok'] for r in records)} ok, {time.time() - began:.0f} s",
                flush=True,
            )

    with (out / "chains.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    good = [r for r in records if r["ok"]]
    summary = {
        "chains": len(records),
        "ok": len(good),
        "failed": {r["id"]: r["failure"] for r in records if not r["ok"]},
        "rfproteina": str(rfproteina.__file__),
        "ae": str(args.ae),
        "seed": args.seed,
        "stack_commit": os.environ.get("FS_STACK_COMMIT"),
        "seconds": round(time.time() - began, 1),
    }
    summary["mapping"] = mapping_summary(records)
    if good:
        summary["reconstruction"] = reconstruction_summary(good)
    if pool["mean"]:
        mean_all = np.concatenate(pool["mean"])
        summary["latent"] = ds.latent_summary(
            mean_all,
            np.concatenate(pool["log_scale"]),
            np.concatenate(pool["residue_type"]),
            sample=np.concatenate(pool["sample"]),
        )
        np.savez_compressed(
            out / "latent_pool.npz",
            mean=mean_all.astype(np.float32),
            log_scale=np.concatenate(pool["log_scale"]).astype(np.float32),
            residue_type=np.concatenate(pool["residue_type"]).astype(np.int16),
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("latent",)}, indent=1)
    )
    return 0


def _three(letters: str) -> list[str]:
    """RFD4's token names for the corpus's one-letter residues."""
    back = {v: k for k, v in THREE_TO_ONE.items()}
    return [back[a] for a in letters]


def is_random_stratum(record: dict) -> bool:
    """The length strata are draws from the corpus; the stress pools oversample what a draw rarely shows, so a rate of the corpus
    is read from the strata alone."""
    return str(record["why"]).startswith("length")


def _share(group: list[dict], key: str) -> float | None:
    return float(np.mean([bool(r.get(key, False)) for r in group])) if group else None


def _quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(v)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(v.max()),
        "n": int(v.size),
    }


def mapping_block(group: list[dict]) -> dict:
    """The mapping of a set of chains: counted before the autoencoder runs, so a chain that fails later is still in it."""
    return {
        "chains": len(group),
        "one_to_one": _share(group, "one_to_one"),
        "mapping_ok": _share(group, "mapping_ok"),
        "same_length": _share(group, "same_length"),
        "same_sequence": _share(group, "same_sequence"),
        "units_ok": _share(group, "units_ok"),
        "residue_types_in_corpus_alphabet_of_one_to_one": _share(
            [r for r in group if r.get("one_to_one")],
            "residue_types_in_corpus_alphabet",
        ),
        "backbone_rmsd_to_corpus": _quantiles(
            [
                r["backbone_rmsd_to_corpus"]
                for r in group
                if "backbone_rmsd_to_corpus" in r
            ]
        ),
        "chains_with_a_condition_given_by_the_pipeline": float(
            np.mean([r.get("c_seq_given_by_the_pipeline", 0) > 0 for r in group])
        ),
        "chains_with_atoms_without_coordinates": float(
            np.mean([r.get("atoms_without_coordinates", 0) > 0 for r in group])
        ),
        "masks_all_true": float(
            np.mean([all(r["masks"].values()) for r in group if "masks" in r])
        )
        if any("masks" in r for r in group)
        else None,
        "residues_complete_mean": float(
            np.mean([r["residues_complete"] for r in group if "residues_complete" in r])
        )
        if any("residues_complete" in r for r in group)
        else None,
        "failed_before_the_autoencoder": int(
            sum("residues_complete" not in r for r in group)
        ),
    }


def mapping_summary(records: list[dict]) -> dict:
    by_why = {
        why: mapping_block([r for r in records if r["why"] == why])
        for why in sorted({r["why"] for r in records})
    }
    missing = np.zeros(37, dtype=int)
    for r in records:
        if "missing_legal_by_slot" in r:
            missing += np.asarray(r["missing_legal_by_slot"], dtype=int)
    return {
        "random_strata": mapping_block([r for r in records if is_random_stratum(r)]),
        "all": mapping_block(records),
        "by_stratum": by_why,
        "legal_slots_without_an_atom_by_slot": missing.tolist(),
    }


def _pooled(group: list[dict], prefix: str) -> dict | None:
    """Residue-weighted reconstruction scores of one decoding (`mean` or `sample`) over one part (all, complete, truncated)."""
    structure = [
        r[f"{prefix}_structure"]
        for r in group
        if r.get(f"{prefix}_structure", {}).get("residues")
    ]
    if not structure:
        return None
    weights = np.array([s["residues"] for s in structure], dtype=float)

    def avg(key: str) -> float:
        return float(np.average([s[key] for s in structure], weights=weights))

    sequence = [
        r[f"{prefix}_sequence"]
        for r in group
        if r.get(f"{prefix}_sequence", {}).get("residues")
    ]
    total = sum(s["residues"] for s in sequence)
    presence = [
        r[f"{prefix}_presence"]
        for r in group
        if r.get(f"{prefix}_presence", {}).get("target_atoms")
    ]
    return {
        "chains": len(structure),
        "residues": int(weights.sum()),
        "per_residue_rmsd_mean": avg("per_residue_mean"),
        "per_residue_rmsd_median_of_chains": float(
            np.median([s["per_residue_median"] for s in structure])
        ),
        "backbone_rms": avg("backbone_rms"),
        "side_chain_rms": avg("side_chain_rms"),
        "all_atom_rms": avg("all_atom_rms"),
        "sequence_accuracy": float(
            sum(s["accuracy"] * s["residues"] for s in sequence) / total
        ),
        "presence_f1": float(np.mean([p["f1"] for p in presence]))
        if presence
        else None,
    }


def reconstruction_summary(records: list[dict]) -> dict:
    out: dict = {}
    for name, group in (
        ("random_strata", [r for r in records if is_random_stratum(r)]),
        ("all", records),
    ):
        block: dict = {}
        for label in ("mean", "sample"):
            for part in ("all", "complete", "truncated"):
                prefix = label if part == "all" else f"{label}_{part}"
                block[f"{label}/{part}"] = _pooled(group, prefix)
        leak = [r["leak_sequence"] for r in group if r["leak_sequence"].get("residues")]
        leak_decode = [
            r["leak_decode_sequence"]
            for r in group
            if r["leak_decode_sequence"].get("residues")
        ]
        clean_decode = [
            r["clean_decode_sequence"]
            for r in group
            if r["clean_decode_sequence"].get("residues")
        ]
        block["leak_control"] = {
            "decoder_given_the_true_sequence": float(
                np.mean([s["accuracy"] for s in leak])
            )
            if leak
            else None,
            "decode_given_the_true_sequence": float(
                np.mean([s["accuracy"] for s in leak_decode])
            )
            if leak_decode
            else None,
            "decode_given_nothing": float(
                np.mean([s["accuracy"] for s in clean_decode])
            )
            if clean_decode
            else None,
        }
        out[name] = block
    return out


if __name__ == "__main__":
    raise SystemExit(main())
