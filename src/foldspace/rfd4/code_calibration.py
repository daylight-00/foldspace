"""Generated codes mapped to the first two moments of their target: a post hoc change of a learned distribution, made to ask what the contraction of a generated latent costs.

The generated codes of the full-width pilot's chains carry a share of the latent's variance that is alike in every direction of it and falls with training. If that were the whole failure, a map that gives the generated codes the mean and the
covariance of the latent they were trained on would give back generation, and if the codes had lost more than their scale it would not. `calibration_map` is that affine map, `z' = z A + b`: the generated codes are whitened by the Cholesky factor of their own
covariance (a ridge of `RIDGE` times its mean eigenvalue keeps a direction of almost no variance from being blown up) and coloured by that of the target. It is **not** a generator and not an arm: the sampler applies it to the codes before they are decoded
(`--calibrate`) and what comes out is the performance of a changed distribution; a number from it is never counted as a result of the arm it was made from.

    PYTHONPATH=<stack>/RFD4-Proteina/src python src/foldspace/rfd4/code_calibration.py --codes <set>/samples.pt --audit audit_pilot.json --label esmc_only_seed2 --out calibration.json

`--codes` is the `samples.pt` of a set drawn from the checkpoint to be read (the same panel and seed as the draw that will apply the map), `--audit` the representation audit of the pilot's autoencoders (`mean` and `covariance` of each latent) and `--label` the latent in it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RIDGE = 1e-2


def moments_of(codes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The mean and the covariance of the pooled rows of the codes of several samples (each `[length, channels]`)."""
    pooled = np.concatenate([np.asarray(c, dtype=np.float64) for c in codes])
    return pooled.mean(0), np.cov(pooled.T)


def calibration_map(
    mean_from: np.ndarray,
    cov_from: np.ndarray,
    mean_to: np.ndarray,
    cov_to: np.ndarray,
    ridge: float = RIDGE,
) -> tuple[np.ndarray, np.ndarray]:
    """`(A, b)` with `z' = z A + b` giving rows of mean `mean_from` and covariance `cov_from` the mean `mean_to` and the covariance `cov_to` (up to the ridge on `cov_from`).

    With `cov_from = L_f L_f^T` and `cov_to = L_t L_t^T` the rows are whitened, `(z - mean_from) L_f^{-T}`, and coloured, `. L_t^T`: `A = L_f^{-T} L_t^T` and `b = mean_to - mean_from A`.
    """
    cov_from = np.asarray(cov_from, dtype=np.float64)
    cov_to = np.asarray(cov_to, dtype=np.float64)
    width = len(cov_from)
    floor = ridge * float(np.trace(cov_from)) / width
    l_from = np.linalg.cholesky(cov_from + floor * np.eye(width))
    l_to = np.linalg.cholesky(cov_to)
    a = np.linalg.solve(l_from, np.eye(width)).T @ l_to.T
    b = (
        np.asarray(mean_to, dtype=np.float64)
        - np.asarray(mean_from, dtype=np.float64) @ a
    )
    return a, b


def apply(z: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The map on rows `z` `[..., channels]`."""
    return np.asarray(z) @ a + b


def check(
    codes: list[np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    mean_to: np.ndarray,
    cov_to: np.ndarray,
) -> dict:
    """How near the mapped codes are to the target's moments: the largest difference of a channel mean and the relative Frobenius difference of the covariances."""
    mean, cov = moments_of([apply(c, a, b) for c in codes])
    return {
        "mean_abs_max_difference": float(np.abs(mean - np.asarray(mean_to)).max()),
        "covariance_relative_difference": float(
            np.linalg.norm(cov - np.asarray(cov_to)) / np.linalg.norm(cov_to)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--codes",
        required=True,
        help="the samples.pt of a set drawn from the checkpoint",
    )
    parser.add_argument(
        "--audit",
        required=True,
        help="the representation audit (audit_pilot.json) with the moments whole",
    )
    parser.add_argument(
        "--label", required=True, help="the latent in the audit, for instance esmc_only_seed2"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--ridge", type=float, default=RIDGE)
    args = parser.parse_args(argv)

    import torch

    arrays = torch.load(args.codes, map_location="cpu", weights_only=False)
    codes = [np.asarray(v["z"], dtype=np.float32) for v in arrays.values()]
    latent = json.loads(Path(args.audit).read_text())["latents"][args.label]
    mean_to, cov_to = np.array(latent["mean"]), np.array(latent["covariance"])
    mean_from, cov_from = moments_of(codes)
    a, b = calibration_map(mean_from, cov_from, mean_to, cov_to, args.ridge)
    result = {
        "map": a.tolist(),
        "shift": b.tolist(),
        "ridge": args.ridge,
        "label": args.label,
        "codes": str(Path(args.codes).resolve()),
        "audit": str(Path(args.audit).resolve()),
        "samples": len(codes),
        "tokens": int(sum(len(c) for c in codes)),
        "total_variance": {
            "generated": float(np.trace(cov_from)),
            "target": float(np.trace(cov_to)),
        },
        "mapped_against_target": check(codes, a, b, mean_to, cov_to),
        "largest_gain": float(np.linalg.svd(a, compute_uv=False)[0]),
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(
        f"{args.label}: {result['samples']} samples, total variance {result['total_variance']['generated']:.2f} against the target's {result['total_variance']['target']:.2f}; "
        f"mapped: mean off by at most {result['mapped_against_target']['mean_abs_max_difference']:.3f}, covariance off by {result['mapped_against_target']['covariance_relative_difference']:.3f} (relative); largest gain {result['largest_gain']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
