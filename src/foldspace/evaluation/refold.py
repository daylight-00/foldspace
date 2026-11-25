"""Refold a sequence with an independent folder and score it against a structure.

A generator conditioned on ESMFold2 is not judged by ESMFold2 alone: the structural judge is **AtlasFold**,
which is MSA-free and built on AtlasLM rather than ESMC. This is the harness that runs it.

    (sequence, coordinates)  ->  AtlasFold(sequence)  ->  scTM, scRMSD

`coordinates` is whatever the sequence is supposed to describe: the natural Cα
trace when calibrating the evaluator's ceiling, a generator's own output when
scoring it. The two cases run the same code, which is the point -- a ceiling
measured by a different path than the samples is not a ceiling for them.

**The reported scTM is US-align's.** This file also implements TM-score under
a known correspondence, and the two agree to 1e-3 wherever scTM >= 0.5. But
scores are read below 0.5 as well, where the agreement is 3e-2 rather
than 1e-3 -- among pairs that do not superimpose, where a superposition search
has many nearly equivalent answers. US-align's is the published
implementation of exactly this problem, folding costs seconds per target and
scoring costs milliseconds, so there is no reason for the reported number to
be the local one. The local score is computed anyway and recorded beside it:
every report carries its own cross-check, and the local path stays exercised.

**Two RMSDs, because there are two questions.** `scRMSD` is the conventional
one -- the Kabsch fit over every matched residue, which minimises RMSD.
`tm_fit_rmsd` is the RMSD under the superposition that maximised TM, which is
a different objective and therefore a different number. Reporting only the
second under the first's name is what the earlier version did, and the two
were already confused once.

**Folding and scoring are separable.** Folding wants a card; scoring wants
none. `--predictions` reuses a directory that has already been folded, so a
scoring change never costs a refold.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ATLASFOLD = Path("/path/to/design/atlasfold")
THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def load_records(path: Path) -> list[dict]:
    """`[{id, sequence, coordinates}]` from a JSON list or JSONL."""
    text = path.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def predicted_ca(cif: Path):
    """Cα coordinates of the first chain, in file order."""
    import numpy as np
    from biotite.structure.io import pdbx

    block = pdbx.CIFFile.read(cif)
    atoms = pdbx.get_structure(block, model=1)
    chain = atoms[np.isin(atoms.chain_id, [atoms.chain_id[0]])]
    return np.asarray(chain[chain.atom_name == "CA"].coord, dtype=np.float64)


def _kabsch(fixed, mobile):
    """Rotation and translation putting `mobile` onto `fixed`."""
    import numpy as np

    a, b = fixed - fixed.mean(0), mobile - mobile.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return rotation, fixed.mean(0) - mobile.mean(0) @ rotation


def tm_score(reference, subject, reference_length=None) -> tuple[float, float]:
    """``(TM-score, RMSD)`` for two traces with a known 1:1 correspondence.

    `biotite`'s `tm_score` does not superimpose, and its
    `superimpose_structural_homologs` is a remote-homolog *search* -- it looks
    for a correspondence, needs a backbone richer than Cα to build its 3Di
    alphabet, and on this input raises "structures are too dissimilar". None
    of that applies here: a refold of a sequence has residue *i* matching
    residue *i* by construction, so what is needed is the superposition that
    maximises TM under a correspondence already known.

    That is the TM-score program's own procedure: seed on fragments of several
    lengths, superimpose, keep the residues inside a distance cutoff,
    re-superimpose until the set stops changing, and take the best TM over all
    seeds. A single Kabsch fit over everything is *not* the same thing -- it
    minimises RMSD, a different objective, and reports a lower TM.

    The denominator is the length of the **whole** reference, read before the
    pair is cut to their common prefix. Reading it after would renormalise a
    truncated prediction onto the part it did produce, which is the one place
    a generator could score well by emitting less.
    """
    import numpy as np

    reference = np.asarray(reference, dtype=np.float64)
    subject = np.asarray(subject, dtype=np.float64)
    length = reference_length or len(reference)
    n = min(len(reference), len(subject))
    reference, subject = reference[:n], subject[:n]
    d0 = max(1.24 * (length - 15) ** (1 / 3) - 1.8, 0.5) if length > 15 else 0.5

    def tm_of(rotation, shift):
        moved = subject @ rotation + shift
        d = np.linalg.norm(reference - moved, axis=1)
        return float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / length), d

    best, best_fit = 0.0, None
    seeds = [size for size in (n, n // 2, n // 4, n // 8, 4) if size >= 4]
    for span in dict.fromkeys(seeds):
        for start in range(0, n - span + 1, max(span // 2, 1)):
            subset = np.arange(start, start + span)
            for _ in range(20):
                if len(subset) < 3:
                    break
                rotation, shift = _kabsch(reference[subset], subject[subset])
                score_now, d = tm_of(rotation, shift)
                if score_now > best:
                    best, best_fit = score_now, (rotation, shift)
                cutoff = max(d0, 3.5)
                grown = np.flatnonzero(d < cutoff)
                if len(grown) == len(subset) and np.array_equal(grown, subset):
                    break
                if len(grown) < 3:
                    break
                subset = grown
    rotation, shift = best_fit
    moved = subject @ rotation + shift
    return best, float(np.sqrt(((reference - moved) ** 2).sum(axis=1).mean()))


def kabsch_rmsd(reference, subject) -> float:
    """RMSD over every matched residue, under the fit that minimises it.

    The conventional `scRMSD`. `tm_score` also returns an RMSD, but under the
    superposition that maximised TM, which is a different objective: on a pair
    that superimposes the two agree to a hundredth of an angstrom, and on a
    pair that does not they can differ by tens.
    """
    import numpy as np

    reference = np.asarray(reference, dtype=np.float64)
    subject = np.asarray(subject, dtype=np.float64)
    n = min(len(reference), len(subject))
    reference, subject = reference[:n], subject[:n]
    rotation, shift = _kabsch(reference, subject)
    moved = subject @ rotation + shift
    return float(np.sqrt(((reference - moved) ** 2).sum(axis=1).mean()))


def write_ca_pdb(path: Path, coordinates, sequence: str | None = None) -> None:
    """Cα-only PDB, chain A, residues numbered from 1."""
    lines = []
    for i, (x, y, z) in enumerate(coordinates, start=1):
        residue = THREE.get((sequence or "")[i - 1 : i], "GLY")
        lines.append(
            f"ATOM  {i:>5d}  CA  {residue} A{i:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C"
        )
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n")


def usalign(binary: str, prediction: Path, reference: Path) -> dict:
    """US-align's own TM-score under the residue-index correspondence.

    `-TMscore 1` is documented as "superpose two structures by assuming that a
    pair of residues with the same residue index are equivalent", which is the
    self-consistency problem exactly. Chain 2 is the reference, so `TM2` is
    the one normalised by the length the score is a statement about.
    """
    out = subprocess.run(
        [binary, str(prediction), str(reference), "-TMscore", "1", "-outfmt", "2"],
        capture_output=True,
        text=True,
        check=True,
    )
    row = None
    for line in out.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        row = line.split("\t")
    if row is None or len(row) < 11:
        raise SystemExit(f"could not parse US-align output:\n{out.stdout}")
    # #PDBchain1 PDBchain2 TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali
    return {
        "tm_by_prediction": float(row[2]),
        "tm_by_reference": float(row[3]),
        "rmsd": float(row[4]),
        "prediction_length": int(row[8]),
        "reference_length": int(row[9]),
        "aligned": int(row[10]),
    }


def usalign_version(binary: str) -> str:
    banner = subprocess.run(
        [binary, "-v"], capture_output=True, text=True, check=False
    ).stdout
    return next(
        (line.strip(" *") for line in banner.splitlines() if "Version" in line),
        "unknown",
    )


def score(reference, subject, sequence=None, binary=None, work=None, name="pair"):
    """Both scTMs, both RMSDs. `scTM` is the authoritative one.

    Without a US-align binary the authoritative field is the local score and
    the report says so; that path is for CI and for a machine that has no
    binary, not for a number anyone reports.
    """
    local, fit_rmsd = tm_score(reference, subject)
    out = {
        "length": len(reference),
        "predicted_length": len(subject),
        "scTM": local,
        "scTM_local": local,
        "scTM_source": "local",
        "scRMSD": kabsch_rmsd(reference, subject),
        "tm_fit_rmsd": fit_rmsd,
    }
    if binary:
        at = Path(work or tempfile.mkdtemp(prefix="usalign-"))
        at.mkdir(parents=True, exist_ok=True)
        write_ca_pdb(at / f"{name}.ref.pdb", reference, sequence)
        write_ca_pdb(at / f"{name}.pred.pdb", subject, sequence)
        theirs = usalign(binary, at / f"{name}.pred.pdb", at / f"{name}.ref.pdb")
        out["scTM"] = theirs["tm_by_reference"]
        out["scTM_source"] = "usalign"
        out["scTM_difference"] = out["scTM_local"] - out["scTM"]
        out["usalign_aligned"] = theirs["aligned"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="an already-folded directory; without it, AtlasFold is run",
    )
    parser.add_argument("--work", type=Path, default=None, help="where to fold")
    parser.add_argument("--label", default="", help="what this set is")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--usalign",
        default="USalign",
        help="the authoritative scorer; --usalign '' falls back to the local one",
    )
    # AtlasFold's own defaults for these are `auto` and `null`, which resolve
    # against whatever is installed and however long the sequence is. Both
    # resolved the same way for every run so far -- triton, and 20 steps
    # because the corpus is capped at 512 -- so pinning them changes no
    # number. It changes what a *future* run means: an evaluator that picks
    # its kernel by what it can import is one an environment can alter
    # silently, and a step schedule keyed on length would make a longer
    # target a differently-sampled one.
    parser.add_argument("--kernel", default="triton")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--recycles", type=int, default=4)
    parser.add_argument("--mlm-prob", type=float, default=0.15)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    import numpy as np

    from foldspace.evaluation.provenance import capture, require_clean

    require_clean(allow=args.allow_dirty)
    provenance = capture({"allow_dirty": bool(args.allow_dirty)})

    binary = shutil.which(args.usalign) if args.usalign else None
    if args.usalign and not binary:
        raise SystemExit(
            f"no US-align at {args.usalign!r}; it is the authoritative scorer. "
            "Pass --usalign '' to report the local implementation instead."
        )

    records = load_records(args.records)
    print(f"{len(records)} records", flush=True)

    predictions = args.predictions
    if predictions is None:
        work = args.work or args.out.parent / f"{args.out.stem}.folds"
        work.mkdir(parents=True, exist_ok=True)
        fasta = work / "input.fasta"
        fasta.write_text("".join(f">{r['id']}\n{r['sequence']}\n" for r in records))
        predictions = work / "predictions"
        # AtlasFold is a source tree, not an installed package, so its own
        # `src` has to be on the path of the interpreter that runs it. Putting
        # it here rather than in the caller's shell is what makes a refold
        # reproducible from the command line alone: the harness failed once
        # because an invocation set PYTHONPATH and clobbered what `reproducibility/env.sh`
        # had put there, and a missing folder should not look like a missing
        # dependency.
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ATLASFOLD / "src"), *filter(None, [environment.get("PYTHONPATH")])]
        )
        subprocess.run(
            [
                sys.executable,
                str(ATLASFOLD / "run_atlasfold.py"),
                "monomer",
                "--input-fasta",
                str(fasta),
                "--out-dir",
                str(predictions),
                "--seeds",
                str(args.seed),
                "--num-samples",
                "1",
                "--kernel",
                args.kernel,
                "--num-steps",
                str(args.steps),
                "--num-recycles",
                str(args.recycles),
                "--mlm-prob",
                str(args.mlm_prob),
            ],
            cwd=ATLASFOLD,
            env=environment,
            check=True,
        )

    scoring = Path(tempfile.mkdtemp(prefix="usalign-"))
    scored, missing = [], []
    for record in records:
        cif = predictions / record["id"] / f"{record['id']}_ranked_model.cif"
        if not cif.exists():
            missing.append(record["id"])
            continue
        confidence = json.loads(
            (
                predictions / record["id"] / f"{record['id']}_ranked_confidence.json"
            ).read_text()
        )
        got = score(
            np.asarray(record["coordinates"], dtype=np.float64),
            predicted_ca(cif),
            record["sequence"],
            binary=binary,
            work=scoring,
            name=record["id"],
        )
        scored.append(
            {
                "id": record["id"],
                **got,
                "plddt": confidence.get("avg_plddt", confidence.get("plddt")),
                "ptm": confidence.get("ptm"),
            }
        )

    def spread(field):
        values = sorted(r[field] for r in scored if r.get(field) is not None)
        if not values:
            return None
        return {
            "n": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "min": values[0],
            "max": values[-1],
        }

    report = {
        "label": args.label,
        "records": str(args.records),
        "evaluator": {
            "name": "atlasfold-monomer",
            "seed": args.seed,
            "samples": 1,
            "repo": str(ATLASFOLD),
            "kernel": args.kernel,
            "num_steps": args.steps,
            "num_recycles": args.recycles,
            "mlm_prob": args.mlm_prob,
            "pinned": (
                "passed explicitly rather than left to AtlasFold's `auto` "
                "kernel and length-keyed step schedule"
            ),
        },
        "scorer": {
            "authoritative": "usalign" if binary else "local",
            "usalign": usalign_version(binary) if binary else None,
            "usalign_binary": binary,
            "mode": "-TMscore 1",
            "scRMSD": "Kabsch over every matched residue",
            "tm_fit_rmsd": "RMSD under the superposition that maximised TM",
        },
        "scored": len(scored),
        "missing": missing,
        "scTM": spread("scTM"),
        "scTM_local": spread("scTM_local"),
        "scRMSD": spread("scRMSD"),
        "tm_fit_rmsd": spread("tm_fit_rmsd"),
        "plddt": spread("plddt"),
        "ptm": spread("ptm"),
        "scTM_local_minus_authoritative": (
            spread("scTM_difference") if binary else None
        ),
        "per_target": scored,
        "provenance": provenance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("label", "scored", "scTM", "scRMSD", "plddt")},
            indent=2,
        )
    )
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
