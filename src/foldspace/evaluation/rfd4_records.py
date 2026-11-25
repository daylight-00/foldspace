"""RFD4-Proteina's designs as the frozen evaluator's records, with every refusal on the record.

The evaluator reads `[{id, sequence, coordinates}]`: a sequence and its Cα trace (`src/foldspace/evaluation/refold.py`).
RFD4 writes one all-atom CIF a design, `{example}_generated.cif`, so this reads the designed chain of each file and
writes its sequence and its Cα trace. It repairs nothing. A design the evaluator could score wrongly is left out
and listed with its reason: a residue that is not one of the twenty, an index twice or not in sequence, a residue
without a Cα, a Cα that is not a number, a step between two Cα that no peptide has (a break or two on top of each
other), a coordinate list that is not as long as the sequence, a file that is not one protein chain, a length that is
not the one asked for.

CPU only. It reads files and builds no model. The composition and the geometry of the records are read afterwards by
the frozen `scripts/natural_geometry.py --against`, which every cell's composition record is made by.

    python src/foldspace/evaluation/rfd4_records.py --designs DIR --out-records records.json --out-report report.json \\
        [--requested short=100 medium=150 long=250] [--drop-suffix _rep0]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
from pathlib import Path

THREE = {
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
# A peptide's Cα-Cα distance is 3.8 Å (3.0 for a cis peptide); past 4.5 the chain is broken, under 2.5 two Cα overlap.
LONGEST_STEP = 4.5
SHORTEST_STEP = 2.5
LENGTH_IN_ID = re.compile(r"-L(\d+)")
# the corpus's own fit of the radius of gyration, `Rg = c * L ** k` (`natural_geometry.json`)
RG_COEFFICIENT, RG_EXPONENT = 2.843, 0.331


def check(residues: list[tuple], expected: int | None = None) -> list[str]:
    """Why a chain cannot be a record, empty if it can.

    `residues` is `[(three-letter name, sequence index, (x, y, z) or None)]` in file order."""
    if not residues:
        return ["no_protein_residues"]
    reasons = []
    if any(name not in THREE for name, _, _ in residues):
        reasons.append("non_standard_residue")
    index = [i for _, i, _ in residues]
    if len(set(index)) != len(index):
        reasons.append("duplicate_residue_index")
    elif index != list(range(index[0], index[0] + len(index))):
        reasons.append("residue_indices_not_consecutive")
    traces = [xyz for _, _, xyz in residues]
    if any(xyz is None for xyz in traces):
        reasons.append("missing_CA")
    elif any(not all(math.isfinite(v) for v in xyz) for xyz in traces):
        reasons.append("non_finite_coordinate")
    else:
        steps = [math.dist(a, b) for a, b in itertools.pairwise(traces)]
        if steps and max(steps) > LONGEST_STEP:
            reasons.append("chain_break")
        if steps and min(steps) < SHORTEST_STEP:
            reasons.append("CA_overlap")
    if expected is not None and len(residues) != expected:
        reasons.append("length_not_the_requested")
    return reasons


def record_of(name: str, residues: list[tuple]) -> dict:
    """The evaluator's record of a chain that passed `check`."""
    sequence = "".join(THREE[n] for n, _, _ in residues)
    coordinates = [list(xyz) for _, _, xyz in residues]
    assert len(sequence) == len(
        coordinates
    )  # a record's sequence and trace are one length by construction
    return {"id": name, "sequence": sequence, "coordinates": coordinates}


def read_chain(path: Path) -> tuple[list[tuple], list[str]]:
    """The designed chain of one CIF as `check` takes it, and the reasons the file is not one protein chain."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    chains = [c for c in structure[0] if len(c.get_polymer()) > 0]
    if len(chains) != 1:
        return [], [f"not_one_protein_chain ({len(chains)} polymer chains)"]
    residues = []
    for r in chains[0].get_polymer():
        atom = r.find_atom("CA", "*")
        number = r.label_seq if r.label_seq is not None else r.seqid.num
        residues.append(
            (
                r.name,
                int(number),
                (atom.pos.x, atom.pos.y, atom.pos.z) if atom else None,
            )
        )
    return residues, []


def design_id(
    path: Path, drop: tuple[str, ...], strip: str = "_generated.cif", prefix: str = ""
) -> str:
    name = path.name.removesuffix(strip)
    for suffix in drop:
        name = name.removesuffix(suffix)
    return prefix + name


def requested_length(name: str, asked: dict[str, int]) -> int | None:
    for pattern, length in asked.items():
        if pattern in name:
            return length
    found = LENGTH_IN_ID.search(name)
    return int(found.group(1)) if found else None


def readings(records: list[dict]) -> dict:
    """The readings of a set of records that need no judge, as the ruler defines them: the pooled share of A, E, K
    and L, the per-sample entropy (nats) and distinct residues at the median, the Cα spacing and the radius of
    gyration against the corpus's fit. The composition's distance to the corpus is the frozen script's."""
    if not records:
        return {}
    pooled = collections.Counter("".join(r["sequence"] for r in records))
    total = sum(pooled.values())
    entropies, distinct, ratios, steps = [], [], [], []
    for r in records:
        n = len(r["sequence"])
        counts = collections.Counter(r["sequence"])
        entropies.append(-sum(c / n * math.log(c / n) for c in counts.values()))
        distinct.append(len(counts))
        xyz = r["coordinates"]
        centre = [sum(p[i] for p in xyz) / n for i in range(3)]
        rg = math.sqrt(
            sum(sum((p[i] - centre[i]) ** 2 for i in range(3)) for p in xyz) / n
        )
        ratios.append(rg / (RG_COEFFICIENT * n**RG_EXPONENT))
        steps += [math.dist(a, b) for a, b in itertools.pairwise(xyz)]
    return {
        "a_e_k_l_share": sum(pooled[a] for a in "AEKL") / total,
        "per_sample_entropy_median": statistics.median(entropies),
        "distinct_residues_median": statistics.median(distinct),
        "ca_step_median": statistics.median(steps),
        "ca_steps_in_3.6_to_4.0": sum(3.6 <= d <= 4.0 for d in steps) / len(steps),
        "radius_of_gyration_over_the_corpus_fit_median": statistics.median(ratios),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(
    designs: Path,
    *,
    asked: dict[str, int] | None = None,
    drop: tuple[str, ...] = (),
    pattern: str = "*_generated.cif",
    strip: str = "_generated.cif",
    prefix: str = "",
) -> tuple[list[dict], dict]:
    """The records of every design under `designs` that can be one, and a report of all of them.

    RFD4's own files are `{example}_generated.cif`; the files of the fold script are `{input}_s{sample}.cif` (`pattern="*_s*.cif"`, `strip=".cif"`), and `prefix` is put before
    every id (an arm's name, so that several arms' records can be one file for one evaluator run)."""
    asked = asked or {}
    files = sorted(designs.rglob(pattern))
    records, excluded, seen = [], [], set()
    for path in files:
        name = design_id(path, drop, strip, prefix)
        residues, why = read_chain(path)
        if name in seen:
            why = [*why, "duplicate_design_id"]
        seen.add(name)
        if not why:
            why = check(residues, requested_length(name, asked))
        if why:
            excluded.append({"id": name, "file": path.name, "reasons": why})
        else:
            records.append(record_of(name, residues))
    lengths: dict[str, int] = {}
    for record in records:
        key = str(len(record["sequence"]))
        lengths[key] = lengths.get(key, 0) + 1
    manifest = "".join(f"{p.name} {sha256(p)}\n" for p in files)
    report = {
        "source": {
            "directory": str(designs),
            "files": len(files),
            "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        },
        "designs": len(files),
        "valid": len(records),
        "excluded": excluded,
        "readings_without_a_judge": readings(records),
        "lengths_of_the_valid": dict(
            sorted(lengths.items(), key=lambda kv: int(kv[0]))
        ),
        "rules": {
            "longest_ca_step": LONGEST_STEP,
            "shortest_ca_step": SHORTEST_STEP,
            "nothing_is_repaired": True,
        },
    }
    return records, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", type=Path, required=True)
    parser.add_argument("--out-records", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument(
        "--requested",
        nargs="*",
        default=[],
        metavar="PATTERN=LENGTH",
        help="a design whose id holds PATTERN must have LENGTH residues (an id with -L<length> is held to it)",
    )
    parser.add_argument(
        "--drop-suffix",
        nargs="*",
        default=[],
        help="suffixes to cut from an id, e.g. _rep0",
    )
    parser.add_argument(
        "--pattern",
        default="*_generated.cif",
        help="the files read (`*_s*.cif` for the fold script's)",
    )
    parser.add_argument(
        "--strip-suffix",
        default="_generated.cif",
        help="what is cut from a file name to make its id (`.cif` for the fold script's)",
    )
    parser.add_argument(
        "--id-prefix", default="", help="put before every id (an arm's name)"
    )
    args = parser.parse_args()
    asked = {k: int(v) for k, v in (item.split("=") for item in args.requested)}
    records, report = convert(
        args.designs,
        asked=asked,
        drop=tuple(args.drop_suffix),
        pattern=args.pattern,
        strip=args.strip_suffix,
        prefix=args.id_prefix,
    )
    args.out_records.parent.mkdir(parents=True, exist_ok=True)
    args.out_records.write_text(json.dumps(records))
    report["records"] = {
        "path": str(args.out_records),
        "sha256": sha256(args.out_records),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"{report['designs']} designs, {report['valid']} records, {len(report['excluded'])} excluded; "
        f"lengths {report['lengths_of_the_valid']}"
    )
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
