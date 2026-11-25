"""Robustness: how far a single can be from the model's own and still fold to the same protein, and still read as the same sequence.

The perturbations and how they are read are fixed before a run is read. Four steps:

``covariance``
    the training split's second moment of the standardised single, which the ``rank`` family projects on.
``folds``
    the panel's chains folded from perturbed singles, deterministically, with no ESMC backbone resident (the single is
    then the only prior the model has). One file per chain, so that an allocation that ends early is resumed and not
    repeated.
``decode``
    the decoders chosen on dev read the dev split under the same perturbations, and the panel chains one by one.
``subspace``
    a discriminant refitted on the leading and on the trailing directions of the training covariance, read on dev: where
    in the single the residue type is.
``summarize``
    both readings, and the chains where they part.

Nothing here is a gate. The experiment measures a scale: the error a generator's single may carry, in the corpus's own units.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from foldspace.esmfold2 import decoder_structure_check
from foldspace.esmfold2 import perturb as P

#: fixed in advance; the first three are the structure check's, so that "keeps the fold" means one thing
STABLE_TM = decoder_structure_check.STABLE_TM
KEEP_TM = decoder_structure_check.KEEP_TM
KEEP_SHARE = decoder_structure_check.KEEP_SHARE
#: a chain is readable under a perturbation if the decoder recovers at least this share of its residues
READABLE = 0.95
#: the cluster-weighted accuracy below which the decoder is taken to no longer read the sequence
DECODE_FLOOR = 0.95

SPECIAL = ("teacher", "teacher_other_seed", "no_prior")

__all__ = ["summarize", "summarize_decode", "summarize_folds"]


# ------------------------------------------------------------------ reading


def _median(values) -> float:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else float("nan")


def _tolerance(family: str, share_at, which: str = "standard") -> dict:
    """The most severe level reached before the first level at which the share falls under the required one.

    ``share_at(level)`` is the share of chains (folded: that keep the fold; read: that the decoder reads) at a level.
    ``censored`` says that no level failed, so the tolerance is *at least* the most severe level that was tried.
    """
    levels = P.severity_order(family, which)
    shares = [share_at(level) for level in levels]
    passed = []
    for level, share in zip(levels, shares, strict=True):
        if not share >= KEEP_SHARE:  # a share that is not a number fails
            break
        passed.append(level)
    return {
        "levels_mild_to_severe": levels,
        "shares": shares,
        "tolerance": passed[-1] if passed else None,
        "censored": len(passed) == len(levels),
    }


def summarize_folds(result: dict, coords: dict) -> dict:
    """The folds read by the pre-specified criteria.

    ``result`` is ``robustness_folds.json`` and ``coords`` maps ``<id>__<route>`` to Cα. A chain is *stable* if its teacher at
    another seed has a TM of at least ``STABLE_TM`` with it; only stable chains can show what a perturbation does,
    because on the others the fold already moves with nothing changed. A condition *keeps* the fold of a chain if its TM
    with the teacher is at least ``KEEP_TM``; the tolerance of a family is the most severe level before the share of stable
    chains that keep it falls under ``KEEP_SHARE``.
    """
    from foldspace.evaluation import geometry

    which = result.get("level_set", "standard")
    names = {(f, level): P.name_of(f, level) for f, level in P.conditions(which)}
    per_chain = []
    for c in result["chains"]:
        i = c["id"]
        teacher, deposited = coords[f"{i}__teacher"], coords[f"{i}__deposited"]
        seed_tm = decoder_structure_check._tm(geometry, coords[f"{i}__teacher_other_seed"], teacher)
        row = {
            "id": i,
            "length": c["length"],
            "stable": seed_tm >= STABLE_TM,
            "teacher_seed_tm": seed_tm,
            "teacher_tm_deposited": decoder_structure_check._tm(geometry, teacher, deposited),
            "no_prior_tm_teacher": decoder_structure_check._tm(geometry, coords[f"{i}__no_prior"], teacher),
            "conditions": {},
        }
        for name in names.values():
            fold = coords[f"{i}__{name}"]
            tm = decoder_structure_check._tm(geometry, fold, teacher)
            row["conditions"][name] = {
                "tm_teacher": tm,
                "tm_deposited": decoder_structure_check._tm(geometry, fold, deposited),
                "keeps": bool(tm >= KEEP_TM),
                "plddt": c["plddt"][name],
                "ptm": c["ptm"][name],
                "relative_size": c["relative_size"][name],
            }
        per_chain.append(row)

    stable = [r for r in per_chain if r["stable"]]
    unstable = [r for r in per_chain if not r["stable"]]
    per_condition = {}
    for (family, level), name in names.items():
        held = [r["conditions"][name] for r in stable]
        per_condition[name] = {
            "family": family,
            "level": level,
            "stable_chains": len(stable),
            "keeps": (
                float(np.mean([h["keeps"] for h in held])) if held else float("nan")
            ),
            "median_tm_teacher": _median(h["tm_teacher"] for h in held),
            "median_tm_deposited": _median(h["tm_deposited"] for h in held),
            "median_plddt": _median(h["plddt"] for h in held),
            "median_relative_size": _median(h["relative_size"] for h in held),
            "unstable_median_tm_teacher": _median(
                r["conditions"][name]["tm_teacher"] for r in unstable
            ),
        }
    tolerance = {
        family: _tolerance(
            family,
            lambda level, family=family: per_condition[P.name_of(family, level)][
                "keeps"
            ],
            which,
        )
        for family in P.FAMILIES
    }
    return {
        "chains": len(per_chain),
        "stable_chains": len(stable),
        "unstable_chains": len(unstable),
        "floor": {
            "reading": "median TM of the fold with no language-model prior at all against the teacher, over the stable chains",
            "median_tm_teacher": _median(r["no_prior_tm_teacher"] for r in stable),
        },
        "teacher_tm_deposited_median": _median(
            r["teacher_tm_deposited"] for r in stable
        ),
        "conditions": per_condition,
        "tolerance": tolerance,
        "per_chain": per_chain,
    }


def summarize_decode(decode: dict, folds: dict | None = None) -> dict:
    """The decoders under the same perturbations, and (with the folds) where reading and folding part.

    ``decode`` is ``robustness_decoding.json``. The tolerance of a decoder under a family is the most severe level before its
    cluster-weighted accuracy falls under ``DECODE_FLOOR``. With ``folds`` (the output of ``summarize_folds``) the panel
    chains are tabulated: for each condition how many stable chains keep the fold and are readable, only one of the two,
    or neither, on the very same perturbed single.
    """
    arms = decode["arms"]
    conditions = decode["conditions"]
    which = decode.get("level_set", "standard")
    out: dict = {
        "arms": arms,
        "baseline": decode["conditions"]["none"],
        "conditions": {},
        "tolerance": {},
    }
    for family, level in P.conditions(which):
        name = P.name_of(family, level)
        out["conditions"][name] = {
            "family": family,
            "level": level,
            "median_relative_size": conditions[name].get("median_relative_size"),
            **{arm: conditions[name][arm]["cw_acc"] for arm in arms},
        }
    for arm in arms:
        out["tolerance"][arm] = {}
        for family in P.FAMILIES:
            floor_met = {
                level: float(
                    conditions[P.name_of(family, level)][arm]["cw_acc"] >= DECODE_FLOOR
                )
                for level in P.levels_of(family, which)
            }
            out["tolerance"][arm][family] = _tolerance(
                family, lambda level, floor_met=floor_met: floor_met[level], which
            )
    if folds is not None and decode.get("panel"):
        carried = decode.get("carried") or arms[-1]
        stable = {r["id"] for r in folds["per_chain"] if r["stable"]}
        table = {}
        for family, level in P.conditions(which):
            name = P.name_of(family, level)
            cells = dict.fromkeys(
                (
                    "keeps_and_readable",
                    "keeps_not_readable",
                    "readable_not_kept",
                    "neither",
                ),
                0,
            )
            for r in folds["per_chain"]:
                if r["id"] not in stable:
                    continue
                keeps = r["conditions"][name]["keeps"]
                readable = decode["panel"][r["id"]][name][carried] >= READABLE
                key = (
                    "keeps_and_readable"
                    if keeps and readable
                    else "keeps_not_readable"
                    if keeps
                    else "readable_not_kept"
                    if readable
                    else "neither"
                )
                cells[key] += 1
            table[name] = cells
        out["panel_cross_tab"] = {"decoder": carried, "readable_at": READABLE, **table}
    return out


def summarize(folds_result: dict, coords: dict, decode: dict | None = None) -> dict:
    folds = summarize_folds(folds_result, coords)
    out = {"folds": folds}
    if decode is not None:
        out["decode"] = summarize_decode(decode, folds)
    return out


# ------------------------------------------------------------------- steps


def run_covariance(args) -> dict:
    import torch

    from foldspace.esmfold2 import batching
    from foldspace.esmfold2.decoder_data import IGNORE, Normaliser, gather, load_split
    from foldspace.evaluation.provenance import capture

    device = torch.device(args.device)
    cache = Path(args.cache)
    norm = Normaliser.load(
        cache / "esmfold2" / "model_single_train_normalisation.npz", device
    )
    train = load_split(
        "train",
        cache=cache,
        chains=Path(args.chains),
        train_table=Path(args.train_table),
        device=device,
    )
    d = train.z.shape[1]
    scatter = torch.zeros(d, d, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for batch in batching.pack(np.arange(train.n_chains), train.lengths, 65536):
            x, y, _ = gather(train, batch, norm)
            rows = x[y != IGNORE].double()
            scatter += rows.T @ rows
            count += int(rows.shape[0])
    covariance = (scatter / count).cpu().numpy()
    values = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # a file another allocation may be reading is replaced whole, never written in place
    tmp = out.parent / f"{out.stem}.tmp.npz"
    np.savez(
        tmp,
        covariance=covariance,
        count=np.int64(count),
        normalisation_digest=np.array(norm.digest),
    )
    os.replace(tmp, out)
    report = {
        "rows": count,
        "normalisation_digest": norm.digest,
        "variance_in_leading_directions": {
            str(k): float(values[:k].sum() / values.sum()) for k in (16, 32, 64, 128)
        },
        "provenance": capture({"allow_dirty": bool(args.allow_dirty)}),
    }
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(json.dumps(report["variance_in_leading_directions"]), flush=True)
    return report


def run_folds(args) -> dict:
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

    rows = decoder_structure_check.read_panel(args.panel)
    deposited = {c["id"]: c["ca"] for c in validate_interface.corpus_chains(args.chains, "dev")}
    cached = validate_interface._cached_rows(Path(args.targets), [r["id"] for r in rows])
    held = np.load(args.normalisation, allow_pickle=False)
    mean, std = held["mean"].astype(np.float64), held["std"].astype(np.float64)
    spectrum = P.spectrum_from_covariance(
        np.load(args.covariance, allow_pickle=False)["covariance"]
    )
    out = Path(args.out)
    chain_dir = out / "chains"
    chain_dir.mkdir(parents=True, exist_ok=True)
    device = lite.device

    def to_tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None].to(
            device
        )

    s0, s1 = args.seed, args.seed + 1
    began = time.perf_counter()
    shard, shards = (int(v) for v in str(args.shard).split("/"))
    if not 0 <= shard < shards:
        raise SystemExit(f"--shard is K/N with 0 <= K < N, not {args.shard}")
    for j, r in enumerate(rows):
        if j % shards != shard:
            continue  # another allocation's chain
        i = r["id"]
        npz, meta = chain_dir / f"{i}.npz", chain_dir / f"{i}.json"
        if npz.exists() and meta.exists():
            print(f"  {i} already folded, not repeated", flush=True)
            continue
        prepared = sub.prepare_sequences(lite, r["sequence"])
        h = cached[i]  # float32 [L, 256]: the model's own single
        z = (h.astype(np.float64) - mean) / std
        single = to_tensor(h)
        folds = {
            "teacher": iface.fold_single(lite, prepared, single, seed=s0),
            "teacher_other_seed": iface.fold_single(lite, prepared, single, seed=s1),
            "no_prior": iface.fold_without_prior(lite, prepared, seed=s0),
        }
        sizes = {}
        for family, level in P.conditions(args.levels):
            name = P.name_of(family, level)
            zhat = P.perturb(
                z,
                family,
                level,
                P.rng_for(args.perturb_seed, i, family, level),
                spectrum=spectrum,
            )
            sizes[name] = P.relative_size(z, zhat)
            hhat = (zhat.astype(np.float64) * std + mean).astype(np.float32)
            folds[name] = iface.fold_single(lite, prepared, to_tensor(hhat), seed=s0)
        arrays = {f"{i}__{name}": f.ca for name, f in folds.items()}
        arrays[f"{i}__deposited"] = deposited[i]
        # the coordinates first and the record last: a chain is done when its record is there
        tmp = chain_dir / f"{i}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, npz)
        meta.write_text(
            json.dumps(
                {
                    "id": i,
                    "length": len(r["sequence"]),
                    "ptm": {n: f.metrics.get(validate_interface.PTM) for n, f in folds.items()},
                    "plddt": {n: f.metrics.get(validate_interface.PLDDT) for n, f in folds.items()},
                    "relative_size": sizes,
                    "card": torch.cuda.get_device_name(0),
                    "slurm_job": os.environ.get("SLURM_JOB_ID"),
                },
                default=float,
            )
        )
        print(
            f"  {i} L={len(r['sequence'])}  {len(folds)} folds  {time.perf_counter() - began:7.1f}s",
            flush=True,
        )
        del folds, prepared, single
        torch.cuda.empty_cache()

    missing = [
        r["id"]
        for r in rows
        if not (
            (chain_dir / f"{r['id']}.json").exists()
            and (chain_dir / f"{r['id']}.npz").exists()
        )
    ]
    if missing:
        # the last allocation to finish aggregates; this one has done its share
        print(
            f"shard {shard}/{shards} done; {len(missing)} chains of the others are not there yet",
            flush=True,
        )
        return {"missing": missing}
    chains, coords = [], {}
    for r in rows:
        chains.append(json.loads((chain_dir / f"{r['id']}.json").read_text()))
        arrays = np.load(chain_dir / f"{r['id']}.npz", allow_pickle=False)
        for key in arrays.files:  # each array once
            coords[key] = arrays[key]
    tmp = out / "robustness_coordinates.tmp.npz"
    np.savez(tmp, **coords)
    os.replace(tmp, out / "robustness_coordinates.npz")
    result = {
        "experiment": "foldspace.esmfold2.evaluate_robustness.folds",
        "provenance": validate_interface._provenance(args, lite, {"mode": "robustness_folds"}),
        "settings": settings,
        "lm_dropout": dropout,
        "lite_backbone": backbone,
        "seeds": {
            "fold": s0,
            "fold_other": s1,
            "perturbation": args.perturb_seed,
        },
        "level_set": args.levels,
        "levels": {f: list(P.levels_of(f, args.levels)) for f in P.FAMILIES},
        "chains": chains,
    }
    tmp_json = out / "robustness_folds.tmp.json"
    tmp_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float) + "\n"
    )
    os.replace(
        tmp_json, out / "robustness_folds.json"
    )  # the file the summary waits for, whole or not at all
    print(f"wrote {out / 'robustness_folds.json'}", flush=True)
    return result


class Perturbed:
    """A ``transform`` for ``train_decoder.predict``: each chain perturbed once under one condition, and remembered.

    Every decoder that reads the split under a condition therefore reads the same perturbed singles, and the sizes of
    the perturbations are kept.
    """

    def __init__(self, split, family: str, level: float, seed: int, spectrum):
        self.split, self.family, self.level = split, family, level
        self.seed, self.spectrum = seed, spectrum
        self.seen: dict[int, object] = {}
        self.sizes: dict[int, float] = {}

    def __call__(self, x, batch, lengths):
        import torch

        out = torch.zeros_like(x)
        for b, chain in enumerate(batch):
            chain, n = int(chain), int(lengths[b])
            if chain not in self.seen:
                z = x[b, :n].double().cpu().numpy()
                rng = P.rng_for(
                    self.seed, self.split.ids[chain], self.family, self.level
                )
                zhat = P.perturb(
                    z, self.family, self.level, rng, spectrum=self.spectrum
                )
                self.sizes[chain] = P.relative_size(z, zhat)
                self.seen[chain] = torch.from_numpy(zhat).to(x.device)
            out[b, :n] = self.seen[chain]
        return out


def run_decode(args) -> dict:
    import torch

    from foldspace.esmfold2 import train_decoder
    from foldspace.esmfold2.decoder_data import Normaliser, load_split
    from foldspace.evaluation.provenance import capture

    device = torch.device(args.device)
    cache = Path(args.cache)
    norm = Normaliser.load(
        cache / "esmfold2" / "model_single_train_normalisation.npz", device
    )
    dev = load_split(
        "dev",
        cache=cache,
        chains=Path(args.chains),
        train_table=Path(args.train_table),
        device=device,
    )
    spectrum = P.spectrum_from_covariance(
        np.load(args.covariance, allow_pickle=False)["covariance"]
    )
    decoder_dir = Path(args.decoders)
    chosen = json.loads((decoder_dir / "decoder_summary.json").read_text())["chosen"]
    models = {}
    for key in ("linear", "contextual"):
        arm = chosen.get(key)
        if arm:
            models[arm] = train_decoder.load_checkpoint(
                decoder_dir / arm / "full" / "best.pt", device
            )[0]
    if not models:
        raise SystemExit(f"no decoder was chosen in {decoder_dir}")
    panel = {}
    if args.panel:
        index_of = {c: k for k, c in enumerate(dev.ids)}
        panel = {r["id"]: index_of[r["id"]] for r in decoder_structure_check.read_panel(args.panel)}

    conditions = {}
    panel_acc: dict = {i: {} for i in panel}
    began = time.perf_counter()
    for family, level in [(None, None), *P.conditions(args.levels)]:
        name = "none" if family is None else P.name_of(family, level)
        transform = (
            None
            if family is None
            else Perturbed(dev, family, level, args.perturb_seed, spectrum)
        )
        conditions[name] = {}
        for arm, model in models.items():
            pred = train_decoder.predict(model, dev, norm, transform=transform)
            conditions[name][arm] = pred.selection()
            for i, chain in panel.items():
                s, n = int(pred.starts[chain]), int(pred.lengths[chain])
                hit = pred.logp[s : s + n].argmax(1) == pred.labels[s : s + n]
                panel_acc[i].setdefault(name, {})[arm] = float(hit.mean())
        if transform is not None:
            conditions[name]["median_relative_size"] = float(
                np.median(list(transform.sizes.values()))
            )
        print(
            f"  {name:12s} "
            + "  ".join(
                f"{arm} cw {conditions[name][arm]['cw_acc']:.4f}" for arm in models
            )
            + f"  {time.perf_counter() - began:6.1f}s",
            flush=True,
        )
        del transform
    result = {
        "experiment": "foldspace.esmfold2.evaluate_robustness.decode",
        "level_set": args.levels,
        "provenance": capture({"allow_dirty": bool(args.allow_dirty)}),
        "arms": list(models),
        "carried": chosen.get("carried"),
        "chosen": chosen,
        "normalisation_digest": norm.digest,
        "seeds": {"perturbation": args.perturb_seed},
        "conditions": conditions,
        "panel": panel_acc,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {out}", flush=True)
    return result


#: the directions a discriminant is refitted on: the leading ``k`` of the training covariance and the trailing ``m``
LEADING = (16, 32, 64, 128)
TRAILING = (128, 64)


def run_subspace(args) -> dict:
    """Where in the single the residue type is: a discriminant refitted on a subset of the directions, read on dev.

    The fixed decoders of ``decode`` cannot say whether a truncated single *holds* the sequence, only whether *they* can
    read it. A discriminant is closed-form, so it is refitted on the projection onto the leading directions of the
    training covariance and onto the trailing ones, and read on dev (never test).
    """
    import torch

    from foldspace.esmfold2 import lda, train_decoder
    from foldspace.esmfold2.decoder_data import Normaliser, load_split
    from foldspace.evaluation.provenance import capture

    device = torch.device(args.device)
    cache = Path(args.cache)
    norm = Normaliser.load(
        cache / "esmfold2" / "model_single_train_normalisation.npz", device
    )
    common = {
        "cache": cache,
        "chains": Path(args.chains),
        "train_table": Path(args.train_table),
        "device": device,
    }
    train, dev = load_split("train", **common), load_split("dev", **common)
    spectrum = P.spectrum_from_covariance(
        np.load(args.covariance, allow_pickle=False)["covariance"]
    )
    stats = lda.statistics(train, norm, np.arange(train.n_chains))
    vectors = torch.tensor(spectrum.vectors, dtype=torch.float64, device=device)
    d = vectors.shape[1]
    total = float(spectrum.values.sum())
    subsets = {"full": slice(0, d)}
    subsets.update({f"leading_{k}": slice(0, k) for k in LEADING})
    subsets.update({f"trailing_{m}": slice(d - m, d) for m in TRAILING})
    results = {}
    for name, cols in subsets.items():
        model = lda.solve(stats, basis=vectors[:, cols]).to(device)
        selection = train_decoder.predict(model, dev, norm).selection()
        results[name] = {
            "directions": len(range(*cols.indices(d))),
            "variance_share": float(spectrum.values[cols].sum() / total),
            **selection,
        }
        print(
            f"  {name:13s} {results[name]['directions']:3d} directions  "
            f"{results[name]['variance_share']:.4f} of the variance  cw acc {selection['cw_acc']:.4f}",
            flush=True,
        )
    report = {
        "experiment": "foldspace.esmfold2.evaluate_robustness.subspace",
        "provenance": capture({"allow_dirty": bool(args.allow_dirty)}),
        "normalisation_digest": norm.digest,
        "reading": "closed-form discriminant refitted on the directions named, fitted on train, read on dev",
        "subspaces": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {out}", flush=True)
    return report


def run_summarize(args) -> dict:
    folds_dir = Path(args.folds)
    result = json.loads((folds_dir / "robustness_folds.json").read_text())
    held = np.load(folds_dir / "robustness_coordinates.npz", allow_pickle=False)
    coords = {k: held[k] for k in held.files}
    decode = (
        json.loads(Path(args.decode).read_text())
        if args.decode and Path(args.decode).exists()
        else None
    )
    summary = summarize(result, coords, decode)
    if args.subspace and Path(args.subspace).exists():
        summary["subspace"] = json.loads(Path(args.subspace).read_text())["subspaces"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    f = summary["folds"]
    print(
        f"{f['chains']} chains, {f['stable_chains']} stable; floor TM {f['floor']['median_tm_teacher']:.3f}"
    )
    for family in P.FAMILIES:
        t = f["tolerance"][family]
        print(
            f"  {family:8s} fold tolerance {t['tolerance']}"
            f"{' (at least; none failed)' if t['censored'] else ''}  keeps {[round(s, 2) for s in t['shares']]}"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub_ = parser.add_subparsers(dest="command", required=True)

    def common(p, *, device=False):
        p.add_argument("--cache", type=Path, default=Path("cache"))
        p.add_argument("--chains", type=Path, default=Path("data/corpus/chains.npz"))
        p.add_argument("--allow-dirty", action="store_true")
        if device:
            p.add_argument("--device", default="cuda")
            p.add_argument(
                "--train-table",
                type=Path,
                default=Path("data/corpus/train_sequences.tsv"),
            )

    default_cov = Path("cache/esmfold2/model_single_train_covariance.npz")
    c = sub_.add_parser("covariance")
    common(c, device=True)
    c.add_argument("--out", type=Path, default=default_cov)

    f = sub_.add_parser("folds")
    common(f)
    f.add_argument("--panel", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--checkpoint", default="ESMFold2")
    f.add_argument("--targets", type=Path, default=Path("cache/esmfold2/targets/dev"))
    f.add_argument(
        "--normalisation",
        type=Path,
        default=Path("cache/esmfold2/model_single_train_normalisation.npz"),
    )
    f.add_argument("--covariance", type=Path, default=default_cov)
    f.add_argument("--seed", type=int, default=20261120)
    f.add_argument("--perturb-seed", type=int, default=20261121)
    f.add_argument("--levels", default="standard", choices=("standard", "coarse"))
    f.add_argument(
        "--shard",
        default="0/1",
        help="K/N: this allocation folds the chains whose position is K modulo N; the last to finish aggregates",
    )

    d = sub_.add_parser("decode")
    common(d, device=True)
    d.add_argument("--decoders", type=Path, required=True)
    d.add_argument("--panel", type=Path, default=None)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--covariance", type=Path, default=default_cov)
    d.add_argument("--perturb-seed", type=int, default=20261121)
    d.add_argument("--levels", default="standard", choices=("standard", "coarse"))

    u = sub_.add_parser("subspace")
    common(u, device=True)
    u.add_argument("--covariance", type=Path, default=default_cov)
    u.add_argument("--out", type=Path, required=True)

    s = sub_.add_parser("summarize")
    s.add_argument("--folds", type=Path, required=True)
    s.add_argument("--decode", type=Path, default=None)
    s.add_argument("--subspace", type=Path, default=None)
    s.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command != "summarize":
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
        if args.command == "folds" and not args.allow_dirty:
            from foldspace.evaluation.provenance import require_clean

            require_clean()
    {
        "covariance": run_covariance,
        "folds": run_folds,
        "decode": run_decode,
        "subspace": run_subspace,
        "summarize": run_summarize,
    }[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
