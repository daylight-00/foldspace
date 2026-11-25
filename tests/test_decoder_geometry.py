"""The geometry of a decoder's reconstructions (`src/foldspace/rfd4/decoder_geometry.py`): the figures of a chain over the residues that are counted, the pooled summary, and how far a returned `Cα` lies from the one it was given. Numpy only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.offline

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "decoder_geometry", ROOT / "src/foldspace/rfd4/decoder_geometry.py"
)
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)

N_RES = 8
A, B, STEP = 1.5, 1.2, 3.8  # Cα–C, N–Cα and the Cα step of a straight toy chain


def toy_chain() -> tuple[np.ndarray, np.ndarray]:
    """A straight chain along x: Cα every 3.8 Å, C `A` after its Cα and N `B` before it (so that C to the next N is 3.8 - A - B)."""
    coords = np.zeros((N_RES, 37, 3))
    coords[:, dg.CA_SLOT, 0] = STEP * np.arange(N_RES)
    coords[:, dg.C_SLOT, 0] = coords[:, dg.CA_SLOT, 0] + A
    coords[:, dg.N_SLOT, 0] = coords[:, dg.CA_SLOT, 0] - B
    present = np.zeros((N_RES, 37), dtype=bool)
    present[:, :3] = True
    return coords, present


def test_the_figures_of_a_straight_chain_are_its_construction() -> None:
    coords, present = toy_chain()
    g = dg.geometry_of(coords, present, np.ones(N_RES, dtype=bool))
    assert len(g["ca_step"]) == N_RES - 1 and np.allclose(g["ca_step"], STEP)
    assert len(g["n_ca"]) == N_RES and np.allclose(g["n_ca"], B)
    assert np.allclose(g["ca_c"], A)
    assert len(g["c_n_next"]) == N_RES - 1 and np.allclose(g["c_n_next"], STEP - A - B)


def test_a_pair_is_measured_only_between_two_counted_residues() -> None:
    coords, present = toy_chain()
    valid = np.ones(N_RES, dtype=bool)
    valid[2] = False  # residue 2 is not scored: no step into it or out of it
    present[5, dg.C_SLOT] = False  # residue 5 has no C: it does not count either
    g = dg.geometry_of(coords, present, valid)
    # counted: 0, 1, 3, 4, 6, 7 -> pairs (0,1), (3,4), (6,7)
    assert len(g["ca_step"]) == 3 and len(g["c_n_next"]) == 3
    assert len(g["n_ca"]) == 6 and len(g["ca_c"]) == 6


def test_a_summary_pools_chains_and_reads_the_window_of_the_step() -> None:
    coords, present = toy_chain()
    stretched = coords.copy()
    stretched[:, dg.CA_SLOT, 0] *= 1.1  # a step of 4.18: outside 3.6 to 4.0
    ok = np.ones(N_RES, dtype=bool)
    pooled = dg.pool(
        [dg.geometry_of(coords, present, ok), dg.geometry_of(stretched, present, ok)]
    )
    s = dg.summarize(pooled)
    assert s["pairs"] == 2 * (N_RES - 1)
    assert s["ca_step_in_window"] == pytest.approx(0.5)
    assert s["ca_step_mean"] == pytest.approx((STEP + STEP * 1.1) / 2)
    assert s["n_ca"] == pytest.approx(
        (B + np.mean(np.linalg.norm(stretched[:, 0] - stretched[:, 1], axis=-1))) / 2
    )


def test_how_far_a_returned_ca_lies_from_the_given_one() -> None:
    given = np.zeros((4, 3))
    returned = np.array(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [9.0, 9.0, 9.0]]
    )
    d = dg.deviation_of(given, returned, np.array([True, True, True, False]))
    assert d == pytest.approx([5.0, 0.0, 2.0])
