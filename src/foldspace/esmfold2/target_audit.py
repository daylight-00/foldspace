"""The distribution of the target a generator would have to learn.

The mixed single of a protein, ``[L, 256]``, is what the generator emits. Before
any denoiser is chosen the cached training targets are read for the properties
that decide whether a diffusion or flow model can learn them in the form they are
stored in: the scale of each channel, how heavy the tails are, how many
directions carry the variance, and how much of it is a property of the protein
rather than of the position.

Nothing here needs torch, weights or a GPU: it reads the four
``z_shard*.npz`` files of the corpus and the corpus table, and a reading is a
function of those files and of nothing else. Its outputs are

``mean`` / ``std``
    the channel-wise normalisation of the *training* split, which is what
    every cached chain is. The generator is trained on ``(z - mean) / std`` and
    the inverse is applied before the frozen folding model sees a feature, so
    the transform has to be invertible and has to be fitted on the training
    split only.

a JSON reading
    each number a decision rests on, with the thresholds that gave it.

**What a reading does not say.** A feature can be badly scaled and still
carry the structure (the outlier channels of ESMFold's latent are the example in
the literature), so a heavy tail is a reason to look at what normalisation
costs the folding model, not a reason to drop the channel. The check that does
that is ``foldspace.esmfold2.validate_interface``'s perturbation arm, not this file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "Reading",
    "cluster_overlap",
    "load_shard",
    "read",
    "spectrum_summary",
    "summarize_provenance",
]

#: Lags, in residues, at which the autocorrelation along a chain is read.
LAGS = (1, 2, 3, 4, 8, 16, 32)
#: Residues from either end at which an edge effect is looked for.
EDGE = 16
#: Multiples of a channel's standard deviation that a value is counted beyond.
TAILS = (4.0, 6.0, 8.0)
#: Chain-length bins (lower edge, residues); the last is open above.
LENGTH_BINS = (0, 100, 140, 180, 220, 260)
#: Rows per chunk when a pass is over rows and not over chains.
CHUNK = 1 << 18


@dataclass
class Shard:
    """One ``z_shard*.npz``: the rows of its chains, back to back."""

    path: Path
    ids: np.ndarray
    lengths: np.ndarray
    starts: np.ndarray
    z: np.ndarray
    provenance: dict


def load_shard(path: Path) -> Shard:
    held = np.load(path, allow_pickle=False)
    provenance = (
        json.loads(str(held["provenance"])) if "provenance" in held.files else {}
    )
    return Shard(
        path=Path(path),
        ids=held["ids"],
        lengths=held["lengths"],
        starts=held["starts"],
        z=held["z"],
        provenance=provenance,
    )


def shard_paths(directory: Path) -> list[Path]:
    paths = sorted(Path(directory).glob("z_shard*.npz"))
    if not paths:
        raise FileNotFoundError(f"no z_shard*.npz under {directory}")
    return paths


# ------------------------------------------------------------------ pass 1


@dataclass
class ChannelScale:
    """Mean and standard deviation of every channel, over every row."""

    n: int
    mean: np.ndarray
    std: np.ndarray


def channel_scale(blocks, *, shift: np.ndarray | None = None) -> ChannelScale:
    """Mean and standard deviation from sums taken about a shift.

    ``blocks`` yields ``[n, D]`` arrays. The sums are of ``x - shift`` in float64
    so that a channel whose mean is large against its spread does not lose its
    variance to cancellation; ``shift`` defaults to the first block's mean.
    """
    n = 0
    s1 = s2 = None
    for block in blocks:
        x = np.asarray(block, dtype=np.float64)
        if shift is None:
            shift = x.mean(axis=0)
        d = x - shift
        s1 = d.sum(axis=0) if s1 is None else s1 + d.sum(axis=0)
        s2 = (d * d).sum(axis=0) if s2 is None else s2 + (d * d).sum(axis=0)
        n += len(x)
    if n == 0:
        raise ValueError("no rows")
    mean_d = s1 / n
    var = np.maximum(s2 / n - mean_d**2, 0.0)
    return ChannelScale(n=n, mean=shift + mean_d, std=np.sqrt(var))


def iter_rows(shards: list[Shard], chunk: int = CHUNK):
    for shard in shards:
        for lo in range(0, len(shard.z), chunk):
            yield shard.z[lo : lo + chunk]


# ------------------------------------------------------------------ pass 2


@dataclass
class ChainPass:
    """What one pass over the chains, in standardised units, accumulates."""

    dim: int
    n_rows: int = 0
    n_chains: int = 0
    m3: np.ndarray = field(init=False)
    m4: np.ndarray = field(init=False)
    corr: np.ndarray = field(init=False)
    tails: dict = field(init=False)
    within_ss: float = 0.0
    between_ss: float = 0.0
    lag_num: dict = field(init=False)
    lag_den: dict = field(init=False)
    lag_num_centred: dict = field(init=False)
    lag_den_centred: dict = field(init=False)
    edge_n: np.ndarray = field(init=False)
    edge_sq: dict = field(init=False)
    edge_vec: dict = field(init=False)
    interior_sq: float = 0.0
    interior_n: int = 0
    interior_vec: np.ndarray = field(init=False)
    bin_sq: np.ndarray = field(init=False)
    bin_n: np.ndarray = field(init=False)
    chain_means: list = field(default_factory=list)
    chain_lengths: list = field(default_factory=list)

    def __post_init__(self) -> None:
        d = self.dim
        self.m3 = np.zeros(d)
        self.m4 = np.zeros(d)
        self.corr = np.zeros((d, d))
        self.tails = {t: np.zeros(d, dtype=np.int64) for t in TAILS}
        self.lag_num = {k: np.zeros(d) for k in LAGS}
        self.lag_den = {k: np.zeros(d) for k in LAGS}
        self.lag_num_centred = {k: np.zeros(d) for k in LAGS}
        self.lag_den_centred = {k: np.zeros(d) for k in LAGS}
        self.edge_n = np.zeros(2, dtype=np.int64)
        self.edge_sq = {end: np.zeros(EDGE) for end in ("N", "C")}
        self.edge_vec = {end: np.zeros((EDGE, d)) for end in ("N", "C")}
        self.interior_vec = np.zeros(d)
        self.bin_sq = np.zeros(len(LENGTH_BINS))
        self.bin_n = np.zeros(len(LENGTH_BINS))

    def add_chain(self, y: np.ndarray) -> None:
        """One chain's rows, already standardised, ``[L, D]`` float64."""
        length = len(y)
        mean = y.mean(axis=0)
        centred = y - mean
        self.n_rows += length
        self.n_chains += 1
        self.chain_means.append(mean)
        self.chain_lengths.append(length)
        self.within_ss += float((centred**2).sum())
        self.between_ss += float(length * (mean**2).sum())
        y2 = y * y
        self.m3 += (y2 * y).sum(axis=0)
        self.m4 += (y2 * y2).sum(axis=0)
        self.corr += y.T @ y
        a = np.abs(y)
        for t in TAILS:
            self.tails[t] += (a > t).sum(axis=0)
        for k in LAGS:
            if length <= k:
                continue
            self.lag_num[k] += (y[:-k] * y[k:]).sum(axis=0)
            self.lag_den[k] += 0.5 * (y2[:-k].sum(axis=0) + y2[k:].sum(axis=0))
            c2 = centred * centred
            self.lag_num_centred[k] += (centred[:-k] * centred[k:]).sum(axis=0)
            self.lag_den_centred[k] += 0.5 * (c2[:-k].sum(axis=0) + c2[k:].sum(axis=0))
        sq = y2.sum(axis=1)
        if length > 2 * EDGE:
            self.edge_n += 1
            for j in range(EDGE):
                self.edge_sq["N"][j] += sq[j]
                self.edge_sq["C"][j] += sq[length - 1 - j]
                self.edge_vec["N"][j] += y[j]
                self.edge_vec["C"][j] += y[length - 1 - j]
            inner = sq[EDGE : length - EDGE]
            self.interior_sq += float(inner.sum())
            self.interior_n += len(inner)
            self.interior_vec += y[EDGE : length - EDGE].sum(axis=0)
        b = int(np.searchsorted(LENGTH_BINS, length, side="right") - 1)
        self.bin_sq[b] += float(sq.sum())
        self.bin_n[b] += length * self.dim


def run_chain_pass(shards: list[Shard], scale: ChannelScale) -> ChainPass:
    result = ChainPass(dim=len(scale.mean))
    safe = np.where(scale.std > 0, scale.std, 1.0)
    for shard in shards:
        for start, length in zip(shard.starts, shard.lengths, strict=True):
            rows = shard.z[int(start) : int(start) + int(length)]
            result.add_chain((rows.astype(np.float64) - scale.mean) / safe)
    return result


# --------------------------------------------------------------- summaries


def spectrum_summary(eigenvalues: np.ndarray) -> dict:
    """How many directions carry a covariance, in four standard ways."""
    lam = np.sort(np.clip(np.asarray(eigenvalues, dtype=np.float64), 0, None))[::-1]
    total = float(lam.sum())
    if total <= 0:
        raise ValueError("a covariance with no variance")
    p = lam / total
    nz = p[p > 0]
    cumulative = np.cumsum(p)
    return {
        "dim": len(lam),
        "participation_ratio": float(total**2 / (lam**2).sum()),
        "entropy_rank": float(np.exp(-(nz * np.log(nz)).sum())),
        "components_for": {
            f"{q:.2f}": int(np.searchsorted(cumulative, q) + 1)
            for q in (0.9, 0.95, 0.99)
        },
        "top_shares": [float(v) for v in p[:5]],
        "condition": float(lam[0] / lam[-1]) if lam[-1] > 0 else float("inf"),
    }


def _channel_quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {k: float("nan") for k in ("min", "p05", "median", "p95", "max")}
    q = np.quantile(values, [0.0, 0.05, 0.5, 0.95, 1.0])
    return dict(zip(("min", "p05", "median", "p95", "max"), (float(v) for v in q)))


def lag_autocorrelation(num: dict, den: dict) -> dict:
    out = {}
    for k in LAGS:
        r = num[k] / np.where(den[k] > 0, den[k], np.nan)
        out[str(k)] = {
            "mean": float(np.nanmean(r)),
            **_channel_quantiles(r[np.isfinite(r)]),
        }
    return out


def edge_profile(chain_pass: ChainPass) -> dict:
    """Mean squared norm, per channel, at each offset from an end, against the interior."""
    d = chain_pass.dim
    n = int(chain_pass.edge_n[0])
    if n == 0 or chain_pass.interior_n == 0:
        return {"note": "no chain long enough for an edge reading"}
    interior = chain_pass.interior_sq / (chain_pass.interior_n * d)
    out = {"interior_mean_sq": interior, "chains": n}
    for end in ("N", "C"):
        out[end] = {
            "mean_sq_ratio": [
                float(chain_pass.edge_sq[end][j] / (n * d) / interior)
                for j in range(EDGE)
            ],
            "mean_vector_norm": [
                float(np.linalg.norm(chain_pass.edge_vec[end][j] / n))
                for j in range(EDGE)
            ],
        }
    return out


@dataclass
class TerminalMeans:
    """The mean feature at each offset from an end, and in the interior, in raw units.

    The mean of a chain's first rows is not the mean of its middle: the end rows
    carry a position-fixed component. Any control that replaces a feature by "the
    mean" has to respect that, or it removes the terminal signal along with the
    content it means to remove.
    """

    n_terminus: np.ndarray
    c_terminus: np.ndarray
    interior: np.ndarray


def terminal_means(chain_pass: ChainPass, scale: ChannelScale) -> TerminalMeans:
    n = int(chain_pass.edge_n[0])
    if n == 0 or chain_pass.interior_n == 0:
        raise ValueError("no chain long enough for a terminal reading")
    return TerminalMeans(
        n_terminus=chain_pass.edge_vec["N"] / n * scale.std + scale.mean,
        c_terminus=chain_pass.edge_vec["C"] / n * scale.std + scale.mean,
        interior=chain_pass.interior_vec / chain_pass.interior_n * scale.std
        + scale.mean,
    )


def positional_mean(length: int, means: TerminalMeans) -> np.ndarray:
    """The mean single for a chain of ``length``: ``[length, D]``.

    Rows within ``EDGE`` of an end take that offset's terminal mean, the rest the
    interior mean. Needs ``length > 2 * EDGE`` so the two ends do not overlap.
    """
    edge = len(means.n_terminus)
    if length <= 2 * edge:
        raise ValueError(f"length {length} is not above twice the edge ({edge})")
    out = np.tile(means.interior, (length, 1))
    out[:edge] = means.n_terminus
    out[length - edge :] = means.c_terminus[::-1]
    return out


def length_dependence(chain_pass: ChainPass) -> dict:
    means = np.array(chain_pass.chain_means)
    lengths = np.array(chain_pass.chain_lengths, dtype=np.float64)
    centred_length = lengths - lengths.mean()
    centred_means = means - means.mean(axis=0)
    denom = np.sqrt((centred_means**2).sum(axis=0) * (centred_length**2).sum())
    r = (centred_means * centred_length[:, None]).sum(axis=0) / np.where(
        denom > 0, denom, np.nan
    )
    bins = {}
    for i, lo in enumerate(LENGTH_BINS):
        if chain_pass.bin_n[i] > 0:
            bins[f"{lo}+"] = float(chain_pass.bin_sq[i] / chain_pass.bin_n[i])
    return {
        "abs_correlation_of_chain_mean_with_length": _channel_quantiles(
            np.abs(r[np.isfinite(r)])
        ),
        "mean_sq_by_length_bin": bins,
    }


def identity_share(pooled: dict) -> dict | None:
    """Share of the variance the residue type fixes, from the pooled sums.

    ``sum_by_aa`` and ``count_by_aa`` are per-residue-type sums over the same
    rows; the between-type sum of squares over the total is the quantity the
    earlier reading reported as 0.061.
    """
    needed = ("sum_z", "sum_zz", "n_residues", "sum_by_aa", "count_by_aa")
    if any(key not in pooled for key in needed):
        return None
    n = float(pooled["n_residues"])
    mean = pooled["sum_z"] / n
    total = float(np.trace(pooled["sum_zz"]) / n - (mean**2).sum())
    counts = pooled["count_by_aa"].astype(np.float64)
    keep = counts > 0
    type_means = pooled["sum_by_aa"][keep] / counts[keep, None]
    between = float((counts[keep] * ((type_means - mean) ** 2).sum(axis=1)).sum() / n)
    return {"between_type_share": between / total, "total_variance": total}


def cluster_overlap(
    shard_ids: list[np.ndarray],
    corpus_ids: np.ndarray,
    corpus_clusters: np.ndarray,
    corpus_splits: np.ndarray,
) -> dict:
    """How much of each shard shares a 30 % cluster with the others.

    The sequence probe holds one shard out and fits on the rest, so a number from
    it is a held-out number only for chains the fit's clusters do not contain.
    """
    lookup = {
        str(i): (int(c), str(s))
        for i, c, s in zip(corpus_ids, corpus_clusters, corpus_splits, strict=True)
    }
    clusters_of = []
    splits_seen: dict[str, int] = {}
    missing = 0
    for ids in shard_ids:
        found = []
        for i in ids:
            hit = lookup.get(str(i))
            if hit is None:
                missing += 1
                found.append(-1)
                continue
            found.append(hit[0])
            splits_seen[hit[1]] = splits_seen.get(hit[1], 0) + 1
        clusters_of.append(np.array(found, dtype=np.int64))
    per_shard = []
    for s, own in enumerate(clusters_of):
        others = (
            np.unique(np.concatenate([c for t, c in enumerate(clusters_of) if t != s]))
            if len(clusters_of) > 1
            else np.array([], dtype=np.int64)
        )
        shared = np.isin(own, others) & (own >= 0)
        per_shard.append(
            {
                "shard": s,
                "chains": len(own),
                "clusters": len(np.unique(own[own >= 0])),
                "chains_sharing_a_cluster_with_other_shards": float(shared.mean()),
            }
        )
    return {
        "per_shard": per_shard,
        "splits_of_chains": splits_seen,
        "ids_not_in_corpus": missing,
    }


def summarize_provenance(records: list[dict]) -> dict:
    """Whether the shards came from one checkpoint and one tree."""

    def pick(record, *path):
        for key in path:
            record = record.get(key) if isinstance(record, dict) else None
        return record

    out = {}
    for name, path in (
        ("weights_sha256", ("checkpoint", "weights_sha256")),
        ("foldspace", ("foldspace",)),
        ("esm", ("esm",)),
        ("esmfold2_atomworks", ("esmfold2_atomworks",)),
        ("torch", ("torch",)),
        ("gpu", ("gpu",)),
    ):
        values = sorted({str(pick(r, *path)) for r in records})
        out[name] = values
    out["one_checkpoint"] = len(out["weights_sha256"]) == 1 and out[
        "weights_sha256"
    ] != ["None"]
    return out


# ------------------------------------------------------------------- reading


@dataclass
class Reading:
    summary: dict
    mean: np.ndarray
    std: np.ndarray
    covariance: np.ndarray
    terminal: TerminalMeans


def read(targets: Path, chains: Path | None = None) -> Reading:
    shards = [load_shard(p) for p in shard_paths(targets)]
    scale = channel_scale(iter_rows(shards))
    chain_pass = run_chain_pass(shards, scale)
    n, d = chain_pass.n_rows, chain_pass.dim
    if n != scale.n:
        raise AssertionError("the two passes counted different rows")

    skew = chain_pass.m3 / n
    excess_kurtosis = chain_pass.m4 / n - 3.0
    corr = chain_pass.corr / n
    cov = corr * np.outer(scale.std, scale.std)
    gaussian_tail = {4.0: 6.3e-5, 6.0: 2.0e-9, 8.0: 1.2e-15}
    tails = {}
    for t in TAILS:
        per_channel = chain_pass.tails[t] / n
        tails[f"{t:g}"] = {
            "fraction": float(per_channel.mean()),
            "gaussian": gaussian_tail[t],
            "channels_above_100x_gaussian": int(
                (per_channel > 100 * gaussian_tail[t]).sum()
            ),
        }
    std_order = np.argsort(scale.std)[::-1]
    summary: dict = {
        "rows": int(n),
        "chains": int(chain_pass.n_chains),
        "dim": int(d),
        "provenance": summarize_provenance([s.provenance for s in shards]),
        "scale": {
            "std": _channel_quantiles(scale.std),
            "std_max_over_median": float(scale.std.max() / np.median(scale.std)),
            "largest_channels": [
                {
                    "channel": int(c),
                    "std": float(scale.std[c]),
                    "mean": float(scale.mean[c]),
                }
                for c in std_order[:5]
            ],
            "dead_channels": int((scale.std < 1e-8).sum()),
            "abs_mean_over_std": _channel_quantiles(
                np.abs(scale.mean) / np.where(scale.std > 0, scale.std, np.nan)
            ),
        },
        "shape": {
            "skewness": _channel_quantiles(skew),
            "excess_kurtosis": _channel_quantiles(excess_kurtosis),
            "tails": tails,
        },
        "spectrum": {
            "covariance": spectrum_summary(np.linalg.eigvalsh(cov)),
            "correlation_after_channel_standardisation": spectrum_summary(
                np.linalg.eigvalsh(corr)
            ),
        },
        "between_chains_share_of_variance": float(
            chain_pass.between_ss / (chain_pass.between_ss + chain_pass.within_ss)
        ),
        "autocorrelation": {
            "raw": lag_autocorrelation(chain_pass.lag_num, chain_pass.lag_den),
            "chain_mean_removed": lag_autocorrelation(
                chain_pass.lag_num_centred, chain_pass.lag_den_centred
            ),
        },
        "edges": edge_profile(chain_pass),
        "length": length_dependence(chain_pass),
    }
    pooled_path = Path(targets) / "pooled.npz"
    if pooled_path.exists():
        pooled = dict(np.load(pooled_path, allow_pickle=False))
        summary["identity"] = identity_share(pooled)
        summary["pooled_agrees_with_shards"] = {
            "max_abs_mean_difference_in_std": float(
                np.max(
                    np.abs(pooled["sum_z"] / pooled["n_residues"] - scale.mean)
                    / np.where(scale.std > 0, scale.std, 1.0)
                )
            ),
            "rows_equal": int(pooled["n_residues"]) == n,
        }
    if chains is not None:
        held = np.load(chains, allow_pickle=False)
        summary["cluster_overlap"] = cluster_overlap(
            [s.ids for s in shards], held["ids"], held["clusters"], held["splits"]
        )
    return Reading(
        summary=summary,
        mean=scale.mean,
        std=scale.std,
        covariance=cov,
        terminal=terminal_means(chain_pass, scale),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--targets", type=Path, required=True, help="directory of z_shard*.npz"
    )
    parser.add_argument(
        "--chains", type=Path, default=None, help="chains.npz, for the cluster overlap"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="where the JSON reading goes"
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="npz of mean, std and covariance (the training-split normalisation)",
    )
    args = parser.parse_args()

    reading = read(args.targets, args.chains)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reading.summary, indent=2, sort_keys=True) + "\n")
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.stats_out,
            mean=reading.mean,
            std=reading.std,
            covariance=reading.covariance,
            terminal_mean_n=reading.terminal.n_terminus,
            terminal_mean_c=reading.terminal.c_terminus,
            interior_mean=reading.terminal.interior,
            rows=reading.summary["rows"],
        )
    print(json.dumps(reading.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
