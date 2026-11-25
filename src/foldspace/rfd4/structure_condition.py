"""The structure condition of a design: the atoms of the known residues, given to the denoiser as the release's own absolute coordinate conditions.

A masked-span design given its sequence alone leaves the backbone free, and the backbone the generator finds is not the fold of the sequence. A design is also given the
atoms of the residues it keeps, as RFD4's motif and inverse-folding tasks give theirs. The release applies a coordinate condition as **input features only** (nothing in its sampler or loss
clamps a known residue: the known tokens are noised and supervised like the generated ones, and the decoder predicts their coordinates), through three keys of the batch:

    condition_coordinate_1_atom       f32 [n, 37, 3]   the coordinates (NaN where none is given), centred on the conditioned atoms (`CenterOnCoordinateConditionedAtoms`)
    condition_CoordinateGroup_1_atom  i16 [n, 37]      0 where a coordinate is given ("known absolute"), -1 where none is
    coord_cond_mask                   bool [n]         whether the token's first atom (N) is conditioned

Here they are made from a structure file for **the known residues only**. What is under the mask is not given: the coordinates of a residue to be generated are NaN, group -1, and the centring
uses the conditioned atoms alone, since a centre taken over the whole chain would carry where the residues to be generated are (their mean position). The structure of the file at the residues
to be generated stays what it is, the native structure, for reading the result against (`fidelity`) and for nothing else, as `Design.reference` is for the sequence.

By default the backbone atoms N, CA, C and O of each known residue are given (the release's own default for a target: `C_CRD ... is_protein_backbone`), not the side chains: their identity is
the sequence condition's, and their packing is for the generator.

The reading of a structure file into atom37 needs the release's atom slot lookup (RFD4's environment); the conditions and their reading are numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: the atom37 slots of N, CA, C and O (`rfproteina.utils.coors.get_atom37_bb3o_mask`)
BACKBONE_SLOTS = (0, 1, 2, 4)
CA = 1
THREE_TO_ONE = {
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
#: the keys, as the release names them (`C_CRD.full_name`, `C_CRDGRP.full_name`, and `coord_cond_mask` of `datapipes/pipeline.py`)
KEY_COORDINATES = "condition_coordinate_1_atom"
KEY_GROUP = "condition_CoordinateGroup_1_atom"
KEY_MASK = "coord_cond_mask"


@dataclass(frozen=True)
class Structure:
    """One protein chain in atom37: the atoms it has (`mask`) at their coordinates (zero elsewhere), and its sequence."""

    coords: np.ndarray
    mask: np.ndarray
    sequence: str

    @property
    def n(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class Conditions:
    """The three keys of a design's structure condition, the shift applied to the coordinates, and which atoms are conditioned."""

    keys: dict[str, np.ndarray]
    centre: np.ndarray
    finite: np.ndarray


def read_structure(path: str | Path, chain: str | None = None) -> Structure:
    """The first protein chain of a CIF (or `chain`), the first model, in atom37 by the release's own slot lookup (`rfproteina.utils.token_encoding.build_atom37_slots`).

    Residues are taken in the order of the file, one per (residue id, insertion code); an atom the residue's slots do not have is left out, and a residue that is not one of the twenty is
    read as `X`."""
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from rfproteina.utils.token_encoding import build_atom37_slots

    cif = pdbx.CIFFile.read(str(path))
    atoms = pdbx.get_structure(cif, model=1, altloc="first", extra_fields=["occupancy"])
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if chain is not None:
        atoms = atoms[atoms.chain_id == chain]
        if len(atoms) == 0:
            raise ValueError(f"{path}: no protein chain {chain}")
    else:
        atoms = atoms[atoms.chain_id == atoms.chain_id[0]]
    if len(atoms) == 0:
        raise ValueError(f"{path}: no protein atoms")
    slots = build_atom37_slots(atoms)
    starts = struc.get_residue_starts(atoms)
    n = len(starts)
    residue_of_atom = np.zeros(len(atoms), dtype=np.int64)
    residue_of_atom[starts[1:]] = 1
    residue_of_atom = np.cumsum(residue_of_atom)
    coords = np.zeros((n, 37, 3), dtype=np.float32)
    mask = np.zeros((n, 37), dtype=bool)
    keep = slots >= 0
    coords[residue_of_atom[keep], slots[keep]] = atoms.coord[keep]
    mask[residue_of_atom[keep], slots[keep]] = True
    names = atoms.res_name[starts]
    return Structure(
        coords, mask, "".join(THREE_TO_ONE.get(str(r), "X") for r in names)
    )


def coordinate_conditions(
    coords: np.ndarray,
    mask: np.ndarray,
    known: np.ndarray,
    slots: tuple[int, ...] = BACKBONE_SLOTS,
) -> Conditions:
    """The release's absolute coordinate conditions for the atoms in `slots` of the known residues that the structure has. Nothing is read of the residues that are not known: their
    coordinates are not looked at, and the centre is the mean of the conditioned atoms."""
    coords = np.asarray(coords, dtype=np.float32)
    n = coords.shape[0]
    known = np.asarray(known, dtype=bool)
    if known.shape != (n,) or mask.shape != (n, 37):
        raise ValueError(
            f"coordinates {coords.shape}, mask {mask.shape} and known {known.shape} disagree"
        )
    selected = np.zeros(37, dtype=bool)
    selected[list(slots)] = True
    finite = known[:, None] & np.asarray(mask, dtype=bool) & selected[None, :]
    if not finite.any():
        raise ValueError("no known residue has any of the atoms to give")
    centre = coords[finite].mean(axis=0)
    given = np.full((n, 37, 3), np.nan, dtype=np.float32)
    given[finite] = coords[finite] - centre
    return Conditions(
        {
            KEY_COORDINATES: given,
            KEY_GROUP: np.where(finite, 0, -1).astype(np.int16),
            KEY_MASK: finite[:, 0].copy(),
        },
        centre.astype(np.float32),
        finite,
    )


def matches(sequence: str, masked: str) -> list[int]:
    """The positions (0-based) at which a structure's sequence disagrees with a design's known residues; a length that differs is every position."""
    if len(sequence) != len(masked):
        return list(range(max(len(sequence), len(masked))))
    return [
        i
        for i, (a, b) in enumerate(zip(sequence, masked, strict=True))
        if b != "_" and a != b
    ]


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(R, t)` with `a @ R + t` on `b` in the least-squares sense (rows are points)."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return rotation, cb - ca @ rotation


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=-1).mean()))


def fidelity(
    generated: np.ndarray,
    structure: Structure,
    conditions: Conditions,
    designed: np.ndarray,
) -> dict:
    """What a generated structure is against the structure it was given, for the known residues and for the span.

    `generated` is the decode's atom37 `[n, 37, 3]` in the conditioned frame. The conditioned atoms in the frame (no fit): their RMSD to what was given, which is how well the release's
    flow holds what it is told (it predicts the known atoms and does not copy them). After a fit of the known Cα, the Cα RMSD of the known residues and of the residues generated against
    the native structure's (the span has no condition to be held to: its distance to the native one is what a different, compatible span looks like, not an error)."""
    given = conditions.keys[KEY_COORDINATES]
    finite = conditions.finite
    out = {"known_atoms_rmsd_in_frame": rmsd(generated[finite], given[finite])}
    known = ~np.asarray(designed, dtype=bool)
    native_ca = structure.coords[:, CA] - conditions.centre
    present = structure.mask[:, CA]
    both = known & present
    rotation, translation = kabsch(generated[both, CA], native_ca[both])
    fitted = generated[:, CA] @ rotation + translation
    out["known_ca_rmsd_fitted"] = rmsd(fitted[both], native_ca[both])
    span = np.asarray(designed, dtype=bool) & present
    out["span_ca_rmsd_fitted_on_the_known"] = (
        rmsd(fitted[span], native_ca[span]) if span.any() else None
    )
    out["all_ca_rmsd_fitted_on_the_known"] = rmsd(fitted[present], native_ca[present])
    return out
