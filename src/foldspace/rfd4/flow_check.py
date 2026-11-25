"""Does the flow batch of `src/foldspace/rfd4/flow_data.py` equal what the release pipeline gives for the same chain?

The flow batch is built from shards and rules (a pair feature from the chain's length, defaults for what a monomer does not have); the
release pipeline builds it from the structure file. The two must agree on every tensor the module can read, and this reads a few chains
of the corpus through the pipeline again and compares them, key by key, without crop or rotation:

* **equal**: every key the module builds is compared with the pipeline's, as the schema's dtype (floats to 1e-4);
* **missing**: every key the pipeline carries that the module does not build, and whether the pipeline's value is the schema's default
  everywhere (a default is what the collate would fill in); a key that differs from its default is a feature the module would lack.

    PYTHONPATH=<stack>/RFD4-Proteina/src .venv-rfd4/bin/python -P -B src/foldspace/rfd4/flow_check.py \\
        --shards cache/rfd4esmc/shards --chains data/corpus/chains.npz --split-table data/corpus/split.tsv.gz \\
        --out runs/rfd4esmc/flow/check.json --count 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foldspace.rfd4.flow_data as fd
import foldspace.rfd4.prepare_shards as prep

#: what the pipeline carries that is not built here, and why the network reads it as the schema's default (each reason is a fact of the
#: code at the revision of the stack, and a reason that stops holding is a failure here)
EQUIVALENT_TO_DEFAULT = {
    "centroid_of_atoms_without_coordinate_conditioning": "no feature of the network reads it (nn/feature_factory.py lists none), and after centring it is 0 to 1e-5",
    "condition_ChainCentroid_1_atom": "NaN means 'not given' and ChainCentroidXyzSeqFeat maps a non-finite centroid to zeros, which is what the default 0.0 gives",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--split-table", required=True)
    parser.add_argument(
        "--esmc",
        default="",
        help="the ESMC array; the crop of its rows is checked when given",
    )
    parser.add_argument(
        "--mirror", default="~/workspaces/databases/alphafold3/mmcif_files"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--split", default="dev")
    args = parser.parse_args(argv)

    import torch
    from rfproteina.datapipes.schema.proteina import SPECS

    began = time.time()
    chains, table, shards = (
        str(Path(p).resolve()) for p in (args.chains, args.split_table, args.shards)
    )
    prep._init(chains, table, args.esmc or chains, args.mirror)
    s = prep._STATE
    ids = fd.split_ids(Path(shards) / "index.json", args.split)
    # the shortest, the longest and the rest at even spacing, so that padding-free and long chains are both read
    lengths = {
        r["id"]: r["tokens"]
        for r in json.loads((Path(shards) / "index.json").read_text())[args.split]
    }
    ordered = sorted(ids, key=lambda i: lengths[i])
    picks = [ordered[int(k)] for k in np.linspace(0, len(ordered) - 1, args.count)]
    results = []
    for name in dict.fromkeys(picks):
        record: dict = {}
        batch, _reference, _letters = s["dp"].read_chain(
            name,
            s["z"],
            s["index_of"],
            s["row_of"],
            s["mirror"],
            s["parse"],
            s["preset"],
            s["pipeline"],
            s["collate"],
            fd.SCHEMA,
            record,
        )
        theirs = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        shard = torch.load(Path(shards) / f"{name}.pt", weights_only=False)
        item = {k: shard[k].numpy() for k in fd.SHARD_KEYS}
        n = int(item["residue_type"].shape[0])
        mine = fd.chain_features(
            item, np.random.default_rng(0), crop=10**6, rotate=False
        )
        rows, differing = {}, []
        for key, value in mine.items():
            if key == "esmc":
                continue
            if key not in theirs:
                rows[key] = "absent in the pipeline's batch"
                differing.append(key)
                continue
            dtype = SPECS[key][0]
            a = torch.from_numpy(value).to(dtype)
            b = theirs[key][0].to(dtype)
            if key == "token_bond_path_length":
                # the validation pipeline keeps every distance; training's cutoff gives a connected pair beyond it cutoff + 1
                b = b.clamp(max=fd.PATH_CUTOFF + 1)
            if a.shape != b.shape:
                rows[key] = f"shape {tuple(a.shape)} against {tuple(b.shape)}"
                differing.append(key)
            elif dtype.is_floating_point:
                err = float((a - b).abs().max())
                rows[key] = f"max abs difference {err:.2g}"
                if err > 1e-4:
                    differing.append(key)
            else:
                bad = int((a != b).sum())
                rows[key] = f"{bad} entries differ"
                if bad:
                    differing.append(key)
                    rows[key] += (
                        f" (built here: {sorted(set(a.flatten().tolist()))[:6]},"
                        f" the pipeline's: {sorted(set(b.flatten().tolist()))[:6]})"
                    )
        missing, explained = {}, {}
        for key, value in theirs.items():
            if key in mine or key not in SPECS:
                continue
            default = SPECS[key][1]
            v = value[0]
            if isinstance(default, float) and math.isnan(default):
                same = torch.isnan(v)  # a NaN default is "not given"
            else:
                same = v == default
            if not bool(same.all()) and key in EQUIVALENT_TO_DEFAULT:
                explained[key] = EQUIVALENT_TO_DEFAULT[key]
            elif not bool(same.all()):
                missing[key] = {
                    "entries_not_default": int((~same).sum()),
                    "entries": int(same.numel()),
                    "example_values": sorted(
                        {str(x) for x in v[~same].flatten()[:6].tolist()}
                    ),
                }
        results.append(
            {
                "id": name,
                "tokens": n,
                "compared": rows,
                "differing": differing,
                "pipeline_keys_not_built_here_and_not_default": missing,
                "pipeline_keys_the_network_reads_as_the_default": explained,
            }
        )
        print(
            name,
            n,
            "differing:",
            differing or "none",
            "| not built, not default:",
            sorted(missing) or "none",
            flush=True,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"seconds": round(time.time() - began, 1), "chains": results}, indent=1
        )
    )
    return int(
        any(
            r["differing"] or r["pipeline_keys_not_built_here_and_not_default"]
            for r in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
