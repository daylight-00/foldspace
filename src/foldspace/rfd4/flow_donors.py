"""A donor for every chain of the corpus: the SHAM control of the ESMC alignment term (`src/foldspace/rfd4/flow_align.py`).

SHAM trains the same projector on the same stream with the same weight against the ESMC rows of *another protein* in place of the chain's own. It is what an extra regression head
does without the information: if ALIGN's weights or samples differ from A's by something SHAM shares, the difference is the head's and not ESMC's. The donor of a chain is the
fused generator's (`src/foldspace/rfd4/donor_sampling.py`, whose guard is used as it stands): a chain of another cluster, of another sequence, of **the same exact length** and with no more than
20 % of its residues equal to the chain's at the same positions (the best over shifts of up to 8). The exact length lets the donor's rows be read through the chain's own crop window,
so the arm's stream, crops and corruption are ALIGN's to the integer, and only the rows differ.

The donor is drawn once for a chain, from `(seed, chain)`, so that every presentation of a chain in a stream has one donor: a chain the stream repeats is not given two different
control targets. A chain with no eligible donor at its length is listed in `missing`, not given a donor of another length; the trainer refuses a stream that meets one.

    python src/foldspace/rfd4/flow_donors.py --chains data/corpus/chains.npz --seed 20261112 --out cache/rfd4esmc/donors_s20261112.json

CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def donor_rules():
    """The donor guard (`src/foldspace/rfd4/donor_sampling.py`), imported where it is used: it reads the project's own package (`foldspace.evaluation.streams`), which the offline
    build has on its path and a training job, whose path is RFD4's source and nothing else, does not. The trainer needs `remap_rows` and nothing of this."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    from foldspace.rfd4 import donor_sampling as donors

    return donors


def build(
    ids,
    lengths,
    clusters,
    tokens,
    starts,
    seed: int,
    *,
    max_identity: float | None = None,
):
    """`({id: donor id}, [ids without a donor], the pool)` for chains given as the corpus stores them (`tokens` the residues of every chain end to end)."""
    rules = donor_rules()
    max_identity = rules.MAX_IDENTITY if max_identity is None else max_identity
    per_chain = [
        np.asarray(tokens[s : s + n]) for s, n in zip(starts, lengths, strict=True)
    ]
    classes: dict[bytes, int] = {}
    sequences = np.array(
        [classes.setdefault(t.tobytes(), len(classes)) for t in per_chain]
    )
    pool = rules.Pool(
        lengths, clusters, sequences, per_chain, max_identity=max_identity
    )
    donors: dict[str, str] = {}
    missing: list[str] = []
    for i, name in enumerate(ids):
        try:
            donors[str(name)] = str(ids[pool.draw(i, seed=seed, step=0)])
        except rules.NoDonor:
            missing.append(str(name))
    return donors, missing, pool


def remap_rows(
    rows: dict[str, tuple[int, int]],
    donors: dict[str, str],
    names=None,
) -> dict[str, tuple[int, int]]:
    """The corpus's `{id: (start, length)}` in the ESMC array, with the rows of each chain in `names` (all of them by default) the donor's.

    A chain of `names` without a donor, or with a donor of another length, is an error: it is not given its own rows, which would be ALIGN's, in an arm that is the control."""
    out = dict(rows)
    for name in rows if names is None else names:
        length = rows[name][1]
        if name not in donors:
            raise KeyError(f"{name} has no donor")
        start, donor_length = rows[donors[name]]
        if donor_length != length:
            raise ValueError(
                f"the donor of {name} has {donor_length} rows and the chain {length}: a donor is of the exact length"
            )
        out[name] = (start, length)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    z = np.load(args.chains, allow_pickle=False)
    donors, missing, pool = build(
        z["ids"], z["lengths"], z["clusters"], z["tokens"], z["starts"], args.seed
    )
    # every donor is read back against the guard it was drawn under
    index = {str(n): i for i, n in enumerate(z["ids"])}
    for name, donor in donors.items():
        pool.check([index[name]], [index[donor]])
    identity = [
        float(np.mean(pool.tokens[index[n]] == pool.tokens[index[d]]))
        for n, d in list(donors.items())[:: max(1, len(donors) // 2000)]
    ]
    record = {
        "chains": str(args.chains.resolve()),
        "chains_sha256": hashlib.sha256(args.chains.read_bytes()).hexdigest(),
        "seed": args.seed,
        "max_identity": donor_rules().MAX_IDENTITY,
        "max_shift": donor_rules().MAX_SHIFT,
        "n": len(z["ids"]),
        "with_donor": len(donors),
        "missing": missing,
        "identity_same_positions_mean_over_a_sample": float(np.mean(identity)),
        "identity_same_positions_max_over_a_sample": float(np.max(identity)),
        "donors": donors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record) + "\n")
    print(
        json.dumps({k: v for k, v in record.items() if k != "donors"}, indent=1)[:2000]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
