"""Where along the sampling trajectory a generated latent loses its variance. Numpy only.

`fm.full_simulation` (the release's sampler) keeps, with `save_trajectory_every`, a snapshot of each data mode's clean prediction `x_1` and, with `dump_noisy_trajectories`, of the state `x_t` after the step. `Trace` pools them over every sample of a draw into
moments per snapshot, so that nothing but the moments is kept (a draw of 96 samples at 400 steps would otherwise hold gigabytes): for the latent the count, the mean and the covariance of the rows (every residue of every sample) of the clean prediction and of
the state, and for the `Cα` the mean step of the clean prediction (in the flow's units). A snapshot at step `k` holds the prediction made at time `ts[k]` and the state at `ts[k + 1]`, which `arrays` writes with the schedule's times.

What the variance of the state does along the trajectory is the sampler's own and not the interpolant's: for the release's flow at its inference settings the state's total variance of a healthy draw runs 31.9, 22.1, 2.4, 1.2, 3.0, 4.8, 6.9 (a U) over the
snapshots, far below the path `(1 - t)^2 v0 + t^2 v1` of the training interpolant (whose lowest value is 5.7 for those ends). A contraction is therefore read **relative to a checkpoint that generated**, step for step: `collapse_time` is the first time at which the
clean prediction's variance of a set is under half of the reference's.
"""

from __future__ import annotations

import numpy as np

FRACTION = 0.5
FROM_T = 0.05


def _as_numpy(x) -> np.ndarray:
    """A tensor (torch, on the CPU) or an array as float64 numpy."""
    return (x.float().numpy() if hasattr(x, "float") else np.asarray(x)).astype(
        np.float64
    )


class Trace:
    """The moments of a generated latent and the `Cα` step along a sampling trajectory, pooled over the batches that `add` is given."""

    def __init__(self, width: int = 32) -> None:
        self.width = width
        self.slots: dict[int, dict] = {}

    def _empty(self) -> dict:
        return {
            "n": 0,
            "sum": np.zeros(self.width),
            "outer": np.zeros((self.width, self.width)),
        }

    def _slot(self, index: int, step: int) -> dict:
        if index not in self.slots:
            self.slots[index] = {
                "step": step,
                "x1": self._empty(),
                "xt": self._empty(),
                "ca_sum": 0.0,
                "ca_n": 0,
            }
        if self.slots[index]["step"] != step:
            raise ValueError(
                f"snapshot {index} is of step {self.slots[index]['step']} in one batch and {step} in another"
            )
        return self.slots[index]

    def add(self, steps: list[dict]) -> None:
        """The snapshots of one batch (`extra["trajectory_steps"]`: each with `step`, `x_1` and `x_t`, dicts by data mode of `[samples, length, channels]` and `[samples, length, 3]`)."""
        for index, snapshot in enumerate(steps):
            slot = self._slot(index, int(snapshot["step"]))
            for name, source in (("x1", snapshot["x_1"]), ("xt", snapshot["x_t"])):
                rows = _as_numpy(source["local_latents"]).reshape(-1, self.width)
                slot[name]["n"] += len(rows)
                slot[name]["sum"] += rows.sum(0)
                slot[name]["outer"] += rows.T @ rows
            ca = _as_numpy(snapshot["x_1"]["bb_ca"])
            step = np.linalg.norm(ca[:, 1:] - ca[:, :-1], axis=-1)
            slot["ca_sum"] += float(step.sum())
            slot["ca_n"] += int(step.size)

    @staticmethod
    def moments(record: dict) -> tuple[np.ndarray, np.ndarray]:
        """The mean and the (population) covariance of the rows a record has pooled."""
        mean = record["sum"] / record["n"]
        return mean, record["outer"] / record["n"] - np.outer(mean, mean)

    def arrays(self, t_latent: np.ndarray, t_ca: np.ndarray) -> dict[str, np.ndarray]:
        """The pooled moments by snapshot, with the times of the prediction (`ts[k]`) and of the state (`ts[k + 1]`) of each modality's schedule."""
        order = sorted(self.slots)
        step = np.array([self.slots[i]["step"] for i in order])
        out = {
            "step": step,
            "t_pred_latent": np.asarray(t_latent)[step],
            "t_state_latent": np.asarray(t_latent)[step + 1],
            "t_pred_ca": np.asarray(t_ca)[step],
            "t_state_ca": np.asarray(t_ca)[step + 1],
            "tokens": np.array([self.slots[i]["x1"]["n"] for i in order]),
            "ca_step_mean": np.array(
                [self.slots[i]["ca_sum"] / max(self.slots[i]["ca_n"], 1) for i in order]
            ),
        }
        for name in ("x1", "xt"):
            moments = [self.moments(self.slots[i][name]) for i in order]
            out[f"mean_{name}"] = np.stack([m for m, _ in moments])
            out[f"cov_{name}"] = np.stack([c for _, c in moments])
        return out


def collapse_time(
    t: np.ndarray,
    reference: np.ndarray,
    observed: np.ndarray,
    fraction: float = FRACTION,
    from_t: float = FROM_T,
) -> float | None:
    """The first time, from `from_t` on, at which a set's total variance is under `fraction` of the reference set's at the same step, or `None` when it never is.

    The two are read at the same snapshots (the same schedule), so that what differs is the model and not the step: the clean prediction's variance of a collapsed checkpoint against that of the checkpoint that generated. `from_t` leaves out the first snapshot,
    where the prediction of pure noise is the data's mean and its variance across samples is almost nothing in both."""
    t, reference, observed = (
        np.asarray(x, dtype=np.float64) for x in (t, reference, observed)
    )
    below = np.flatnonzero((t >= from_t) & (observed < fraction * reference))
    return float(t[below[0]]) if len(below) else None
