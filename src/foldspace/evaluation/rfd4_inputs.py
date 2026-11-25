"""The inputs RFD4-Proteina is asked for, one annotated CIF a design, at the lengths of a panel.

A residue named `L-L` in an annotated CIF asks for a chain of L residues; the benchmark's `monomer-short.cif` asks
for 100. An input of another length is that file with `100` replaced by the length and its name replaced by the
design's id, which reproduces the benchmark's own `monomer-medium.cif` and `monomer-long.cif` line for line (only the
writer's column order and the time stamp differ). A template that is not one of the two files it is known for (the
benchmark's as it was before upstream canonicalized and compressed its inputs on 2026-10-20, and as it ships now, read
from its `.cif.zst`) is refused, because the substitution is only known to be right for them.

The designs of one cell are not generated in one run. The longest is asked for first, alone, so that what a run costs
before its first design (the environment read from disk, the weights, the kernels built) and whether the longest chain
fits in memory are known before anything else is spent; the rest are dealt into interleaved chunks, so that a cell
stopped after some of them is a sample of the panel's lengths and not its short end. Each chunk is in length order, so
that the designs of one batch are near each other in length.

    python src/foldspace/evaluation/rfd4_inputs.py --template monomer-short.cif --panel records.json.gz --seed 20260929 \\
        --chunks 4 --out DIR

writes `DIR/probe/`, `DIR/chunk0/` ... and `DIR/manifest.json`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

TEMPLATE_NAME = "monomer-short"
TEMPLATE_LENGTH = 100
TEMPLATE_SHA256 = "2211795f856d1df247b828ed2362d5508726fd19d8b6c661417630857f35b8e1"
#: `monomer-short.cif` as RFD4-Proteina has shipped it since 2026-10-20 (commit 51eb53f, canonicalized and compressed; this is the digest of
#: the decompressed `monomer-short.cif.zst`); the line-for-line reproduction of the medium and long inputs is tested from it
CANONICAL_TEMPLATE_SHA256 = (
    "bc5b9b81ce4ab936295d0acaf6abb5a88d154b8bfa290569c2983d18c3f940d6"
)
KNOWN_TEMPLATES = (TEMPLATE_SHA256, CANONICAL_TEMPLATE_SHA256)
LENGTH_IN_ID = re.compile(r"-L(\d+)$")
# a chunk's seed is the cell's plus its index; the probe's is the cell's plus this, so that no two runs of a
# cell start from the same stream (two designs of different lengths drawn from one stream share their first residues)
PROBE_OFFSET = 100


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_input(template: str, name: str, length: int) -> str:
    """The template asking for `length` residues, under the name `name`.

    The length is replaced before the name goes in, because a name may hold the digits of the template's own length
    (design 100 of a panel)."""
    asked = template.replace(str(TEMPLATE_LENGTH), str(length))
    if length != TEMPLATE_LENGTH and str(TEMPLATE_LENGTH) in asked:
        raise ValueError("the template still holds its own length")
    if f"{length}-{length}" not in asked:
        raise ValueError("the length is not asked for in the form L-L")
    text = asked.replace(TEMPLATE_NAME, name)
    if TEMPLATE_NAME in text:
        raise ValueError("the template still holds its own name")
    return text


def load_panel(path: Path) -> list[tuple[str, int]]:
    """`[(id, length)]` of a set of records, each id holding its own length as `-L<n>`."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        records = json.load(handle)
    panel = []
    for record in records:
        found = LENGTH_IN_ID.search(record["id"])
        length = len(record["sequence"])
        if not found or int(found.group(1)) != length:
            raise ValueError(f"{record['id']} does not hold its length {length}")
        panel.append((record["id"], length))
    if len({name for name, _ in panel}) != len(panel):
        raise ValueError("an id is used twice")
    return panel


def deal(panel: list[tuple[str, int]], chunks: int) -> dict:
    """The probe (the longest design, the first of them if several) and `chunks` interleaved chunks of the rest.

    The panel is taken in length order, then id; design number i of that order goes to chunk i mod `chunks`."""
    ordered = sorted(panel, key=lambda item: (item[1], item[0]))
    probe = max(ordered, key=lambda item: item[1])
    rest = [item for item in ordered if item != probe]
    dealt: list[list[tuple[str, int]]] = [[] for _ in range(chunks)]
    for i, item in enumerate(rest):
        dealt[i % chunks].append(item)
    return {"probe": probe, "chunks": dealt}


def write(
    template_path: Path,
    panel_path: Path,
    out: Path,
    seed: int,
    chunks: int,
    template_sha256: str | None = None,
) -> dict:
    """Write the cell's inputs. The template must have `template_sha256` if it is given and be one of `KNOWN_TEMPLATES` if not."""
    template_bytes = template_path.read_bytes()
    digest = sha256_of(template_bytes)
    if digest not in ((template_sha256,) if template_sha256 else KNOWN_TEMPLATES):
        raise ValueError(
            f"{template_path} is not the benchmark's {TEMPLATE_NAME}.cif that the substitution is known for"
        )
    template = template_bytes.decode()
    panel = load_panel(panel_path)
    dealt = deal(panel, chunks)
    manifest = {
        "template": {"path": str(template_path), "sha256": digest},
        "panel": {
            "path": str(panel_path),
            "sha256": sha256_of(panel_path.read_bytes()),
            "designs": len(panel),
        },
        "cell_seed": seed,
        "probe": None,
        "chunks": [],
    }

    def put(directory: Path, items: list[tuple[str, int]]) -> list[dict]:
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for name, length in items:
            data = make_input(template, name, length).encode()
            (directory / f"{name}.cif").write_bytes(data)
            files.append({"id": name, "length": length, "sha256": sha256_of(data)})
        return files

    manifest["probe"] = {
        "seed": seed + PROBE_OFFSET,
        "directory": "probe",
        "designs": put(out / "probe", [dealt["probe"]]),
    }
    for k, items in enumerate(dealt["chunks"]):
        manifest["chunks"].append(
            {
                "index": k,
                "seed": seed + k,
                "directory": f"chunk{k}",
                "designs": put(out / f"chunk{k}", items),
            }
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument(
        "--template-sha256",
        default=None,
        help="the digest the template must have (one of the two known benchmark files if not given)",
    )
    args = parser.parse_args()
    manifest = write(
        args.template,
        args.panel,
        args.out,
        args.seed,
        args.chunks,
        template_sha256=args.template_sha256,
    )
    sizes = [len(c["designs"]) for c in manifest["chunks"]]
    print(
        f"probe {manifest['probe']['designs'][0]['id']}, chunks {sizes}, "
        f"{manifest['panel']['designs']} designs in all"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
