"""Batches for RFD4-Proteina's flow from the corpus shards: what the denoiser reads for an unconditional monomer, and the ESMC rows of the
same residues.

The flow is trained on `(Cα trace, latent)` pairs where the latent is the autoencoder's reading of the chain's atoms. A batch it receives
from the release pipeline for an unconditional monomer is the encoder's 14 keys, two pair features that follow from the chain's length alone,
a few index features, and defaults for every conditioning feature (`datapipes/schema/proteina.py` fills a missing key with "no condition").
This module builds that from the shards of `src/foldspace/rfd4/prepare_shards.py`:

* **crop**: a chain longer than the release's crop (256) is cut to a random window of it, as the release's contiguous crop does; the ESMC
  rows are cut to the same window (they were computed on the whole chain, which is what a reader of a cropped example has);
* **rotation and centring**: a uniform random rotation of the coordinates, then the centroid of the atoms present moved to the origin (the
  pipeline rotates before it crops and centres; a rotation about the origin and a centring commute);
* **pair features**: `token_bond_path_length` is the number of residues between two tokens (the bond graph of a single chain), `0` on the
  diagonal and `cutoff + 1` past the training cutoff of 64; `canonical_polymer_ordering` is `+1` at `[i + 1, i]` and `-1` at `[i, i + 1]`
  (the backbone linkage); the bond, order and aromatic features of a chain without ligands are all zero (the linkage is suppressed in them);
* **index features**: molecule instance 0, entity 1, `tok_idx` the position in the crop, index group 0 (`monomer_index_features`).

`src/foldspace/rfd4/flow_check.py` compares every one of these with the release pipeline's own output on chains of the corpus, so that a change in
the pipeline fails there and not in a run. The geometry is numpy; torch is imported only where a batch is made.

Batches come from an endless, seeded stream indexed by the step: step `s` takes the chains `s * batch .. (s + 1) * batch` of the epoch-wise
permutations and an augmentation stream of its own, so that two runs given the same seed read the same chains, windows and rotations, and
a run resumed at step `s` continues the stream instead of replaying it.
"""

from __future__ import annotations

import json
from pathlib import Path

import foldspace.rfd4.structure_condition as fst
import numpy as np

CROP = 256
PATH_CUTOFF = 64
SCHEMA = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"

#: the tensors of a shard that go into a batch as they are (`src/foldspace/rfd4/prepare_shards.py`)
SHARD_KEYS = (
    "coords",
    "atom37_mask",
    "residue_type",
    "token_type",
    "chain_iid",
    "charge",
    "ground_truth_element",
    "incident_bond_type_counts",
    "supports_local_latent",
    "is_in_context_conditioning",
    "is_resolved_mask",
)
TOKEN_KEYS = tuple(k for k in SHARD_KEYS if k != "coords")


def pair_features(n: int) -> tuple[np.ndarray, np.ndarray]:
    """`token_bond_path_length` (int16) and `canonical_polymer_ordering` (int8) of one unbroken chain of `n` residues."""
    idx = np.arange(n)
    path = np.minimum(np.abs(idx[:, None] - idx[None, :]), PATH_CUTOFF + 1).astype(
        np.int16
    )
    ordering = np.zeros((n, n), dtype=np.int8)
    ordering[idx[1:], idx[:-1]] = 1
    ordering[idx[:-1], idx[1:]] = -1
    return path, ordering


def monomer_index_features(n: int) -> dict[str, np.ndarray]:
    """The index features of one unbroken chain, as the release pipeline gives them (`src/foldspace/rfd4/flow_check.py` compares them with it):
    molecule instance 0, entity 1, token position in the chain, and index group 0 (`condition_index-group_1_token`, whose schema default
    is -1: a chain is one group, and the pairs of a group are the pairs the path-length feature relates)."""
    return {
        "molecule_iid": np.zeros(n, dtype=np.int16),
        "molecule_entity_id": np.ones(n, dtype=np.int16),
        "tok_idx": np.arange(n, dtype=np.int16),
        "condition_index-group_1_token": np.zeros(n, dtype=np.int16),
    }


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A rotation drawn uniformly from SO(3): the Q of a Gaussian matrix with the diagonal of R made positive, with determinant +1."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)


def crop_window(n: int, crop: int, rng: np.random.Generator) -> tuple[int, int]:
    """The residues `[start, stop)` kept: all of them, or a random contiguous window of `crop`."""
    if n <= crop:
        return 0, n
    start = int(rng.integers(0, n - crop + 1))
    return start, start + crop


def place(coords: np.ndarray, mask: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """`coords` `[n, 37, 3]` rotated about the origin, the centroid of the atoms present moved to the origin, absent slots at zero."""
    out = coords.astype(np.float32) @ rotation.T
    out = out - out[mask].mean(axis=0)
    return np.where(mask[..., None], out, 0.0).astype(np.float32)


def structure_condition(
    coords: np.ndarray, mask: np.ndarray, known: np.ndarray, rotation: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """The frame and the keys of a design given the backbone of its known residues (`src/foldspace/rfd4/structure_condition.py`): `(target coordinates, the three coordinate keys)`, or `None`
    for a chain with no known backbone atom (nothing to give).

    The chain is rotated, and **both** are expressed relative to the mean of the conditioned atoms, which are the N, CA, C and O of the known residues only: the keys, and the target
    coordinates of every atom (so that at a conditioned atom a condition and its target are one number, as the release's `CenterOnCoordinateConditionedAtoms` makes them). Nothing of the
    residues that are not known is read for the keys or the centre: a centre taken over the whole chain would carry where the residues to be generated are."""
    rotated = np.asarray(coords, dtype=np.float32) @ rotation.T
    try:
        given = fst.coordinate_conditions(rotated, mask, known)
    except ValueError:
        return None
    target = np.where(mask[..., None], rotated - given.centre, 0.0)
    return target.astype(np.float32), given.keys


def mask_token_index() -> int:
    """The unified encoding's index of the token that stands for a residue that is not given: the release's own choice (`datapipes/pipeline.py` writes it into `C_SEQ` for every
    residue a condition leaves free), `<M>` or what the encoding resolves it to."""
    from atomworks.constants import MASKED
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    name = (
        MASKED
        if MASKED in ENC.token_to_idx
        else ENC.resolve_unknown_token_name(MASKED, False)
    )
    return int(ENC.token_to_idx[name])


def sequence_conditions(
    residue_type: np.ndarray, known: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """The release's exact sequence conditioning of one chain: the whole sequence given (its `folding` task: the sequence is given and the structure is generated), or, with
    `known`, only the residues it marks (a design: the others are generated).

    `C_SEQ` holds the residue ids, and the 28 sequence-probability conditions the one-hot of each residue over the unified encoding's standard polymer tokens, which is what the
    release's `CanonicalizeSequenceProbabilityConditions` makes of an exact `C_SEQ`; a residue outside that vocabulary has all 28 at an explicit zero. The denoiser reads the
    probabilities (`sequence_probabilities` of the release recipe), the autoencoder's decode and the sequence-prediction loss read `C_SEQ`. A residue that is not `known` is what the
    release leaves free: `C_SEQ` at the mask token and every probability at NaN, and nothing of what `residue_type` holds there is read. A chain given none of these has every key at
    the schema's default, which is what an unconditional example is."""
    from atomworks.ml.conditions import C_SEQ
    from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS

    residues = np.asarray(residue_type)
    if known is None:
        out = {C_SEQ.full_name: residues.astype(np.int64)}
        for condition in SEQUENCE_PROBABILITY_CONDITIONS:
            out[condition.full_name] = (residues == condition.token_index).astype(
                np.float32
            )
        return out
    given = np.asarray(known, dtype=bool)
    if given.shape != residues.shape:
        raise ValueError(
            f"known has shape {given.shape} for a chain of {residues.shape[0]} residues"
        )
    out = {
        C_SEQ.full_name: np.where(given, residues, mask_token_index()).astype(np.int64)
    }
    for condition in SEQUENCE_PROBABILITY_CONDITIONS:
        hot = (residues == condition.token_index).astype(np.float32)
        out[condition.full_name] = np.where(given, hot, np.nan).astype(np.float32)
    return out


def chain_features(
    item: dict[str, np.ndarray],
    rng: np.random.Generator,
    crop: int = CROP,
    rotate: bool = True,
    esmc: np.ndarray | None = None,
    condition_sequence: bool = False,
    context_rows: bool = False,
    known: np.ndarray | None = None,
    masked_context: np.ndarray | None = None,
    structure: bool = False,
) -> dict[str, np.ndarray]:
    """The features of one chain as a flow batch carries them: cropped, rotated, centred, with its pair and index features.

    `esmc` holds the ESMC rows of the whole chain; the rows of the crop are returned as `esmc` (float16), so that a batch's latent
    target and its Cα trace refer to the same residues.

    `condition_sequence` gives the chain's whole sequence as a condition (`sequence_conditions`). `context_rows` adds `esmc_context`, the rows that an adapter reads as
    context: the crop's rows when the sequence is given, and zeros when it is not (an adapter must see what the denoiser is told and nothing else).

    A design: `known` (`[n]` bool over the whole chain) gives only the residues it marks, and `masked_context` the ESMC rows of the chain *read with its other residues masked*
    (the script that writes the masked rows), which the adapter then reads instead of the whole chain's rows: what the denoiser is told of the sequence and what ESMC read are the same
    residues. Both need the sequence conditioning.

    `structure` (with `known`) also gives the backbone atoms of the known residues as the release's absolute coordinate conditions (`src/foldspace/rfd4/structure_condition.py`), as inference
    does: the coordinates are those of the *rotated* chain, and the frame is centred on the conditioned atoms, **for the target too** (`coords` is shifted by the same centre, so that a
    conditioned atom's condition and its target are one number, as the release's `CenterOnCoordinateConditionedAtoms` makes them). The coordinates of the residues that are not known are not
    given and do not enter the centre. A crop with no known backbone atom has nothing to give: the example is the masked sequence alone, centred on all its atoms as before. The draws
    of the crop and of the rotation are those of the example without the structure.
    """
    if (known is not None or masked_context is not None) and not condition_sequence:
        raise ValueError(
            "a masked sequence is a condition: it needs condition_sequence"
        )
    if structure and known is None:
        raise ValueError(
            "a structure condition is for the known residues: it needs known"
        )
    n = int(item["residue_type"].shape[0])
    start, stop = crop_window(n, crop, rng)
    m = stop - start
    feats = {k: np.asarray(item[k])[start:stop] for k in TOKEN_KEYS}
    rotation = random_rotation(rng) if rotate else np.eye(3, dtype=np.float32)
    mask = feats["atom37_mask"].astype(bool)
    feats["coords"] = place(np.asarray(item["coords"])[start:stop], mask, rotation)
    if structure:
        framed = structure_condition(
            np.asarray(item["coords"])[start:stop],
            mask,
            np.asarray(known, dtype=bool)[start:stop],
            rotation,
        )
        # a crop with no known backbone atom has nothing to give: the masked sequence alone, centred on all its atoms
        if framed is not None:
            feats["coords"], keys = framed
            feats.update(keys)
    feats["token_bond_path_length"], feats["canonical_polymer_ordering"] = (
        pair_features(m)
    )
    feats.update(monomer_index_features(m))
    if esmc is not None:
        if esmc.shape[0] != n:
            raise ValueError(
                f"ESMC has {esmc.shape[0]} rows for a chain of {n} residues"
            )
        feats["esmc"] = np.asarray(esmc[start:stop], dtype=np.float16)
    if condition_sequence:
        feats.update(
            sequence_conditions(
                feats["residue_type"],
                None if known is None else np.asarray(known, dtype=bool)[start:stop],
            )
        )
    if context_rows:
        source = masked_context if masked_context is not None else esmc
        if source is None:
            raise ValueError("context rows need the ESMC rows")
        if source.shape[0] != n:
            raise ValueError(
                f"the context has {source.shape[0]} rows for a chain of {n} residues"
            )
        rows = np.asarray(source[start:stop], dtype=np.float16)
        feats["esmc_context"] = rows if condition_sequence else np.zeros_like(rows)
    return feats


class MaskedRows:
    """The ESMC rows of the masked chains of a stream's slots, as the script that writes the masked rows wrote them: `index.json` (`{"slots": {"<step>:<slot>": {"name", "start", "length",
    "spans"}}}`, the spans 0-based half-open over the whole chain) and the array of rows (`[total, 2560]` half precision) that `start` and `length` index.

    A slot is named by the step and the position in it, not by the chain: a chain that the stream meets twice is masked differently each time. The chain's name is checked, so that a
    stream that is not the one the rows were made for stops here and not in a loss."""

    def __init__(self, array: str | Path, index: str | Path) -> None:
        self.array_path = str(array)
        self.index = json.loads(Path(index).read_text())
        self._array = None

    def get(self, step: int, slot: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        """`(rows, known)` of the slot: the rows of the chain read with its masked residues, and `[n]` bool, True at the residues that are given."""
        import foldspace.rfd4.design_inputs as fdz

        entry = self.index["slots"].get(f"{step}:{slot}")
        if entry is None:
            raise KeyError(
                f"the slot {step}:{slot} ({name}) has no masked rows: they were made for another stream"
            )
        if entry["name"] != name:
            raise ValueError(
                f"the masked rows of slot {step}:{slot} are of {entry['name']}, and the stream has {name} there"
            )
        if self._array is None:
            self._array = np.load(self.array_path, mmap_mode="r")
        start, length = int(entry["start"]), int(entry["length"])
        rows = np.array(self._array[start : start + length])
        spans = [(int(a), int(b)) for a, b in entry["spans"]]
        return rows, ~fdz.mask_from_spans(spans, length)


class FlowStream:
    """An endless, seeded stream of flow batches over the chains of a shard directory; `stream[s]` is the batch of step `s`.

    `ids` name the chains (`<id>.pt` in `shards`); `esmc_rows` maps an id to `(start, length)` in `esmc_array` when the batch carries the
    ESMC rows. `replicates` adjacent copies of each chain are made by the release's own collate (the flow draws `exploration_k`
    noises for each chain and keeps the best). It is a map-style dataset of `steps` batches, so it works as a loader's dataset with
    `batch_size=None`.
    """

    def __init__(
        self,
        shards: str | Path,
        ids: list[str],
        batch: int,
        replicates: int = 4,
        seed: int = 20261111,
        steps: int = 1 << 30,
        offset: int = 0,
        crop: int = CROP,
        esmc_array: str | Path | None = None,
        esmc_rows: dict[str, tuple[int, int]] | None = None,
        buckets: list[int] | None = None,
        rotate: bool = True,
        condition_fraction: float = 0.0,
        context_rows: bool = False,
        mask_fraction: float = 0.0,
        masked_rows: MaskedRows | None = None,
        structure_fraction: float = 0.0,
    ) -> None:
        self.rotate = rotate
        self.condition_fraction = float(condition_fraction)
        self.context_rows = context_rows
        self.mask_fraction = float(mask_fraction)
        self.masked_rows = masked_rows
        self.structure_fraction = float(structure_fraction)
        self.shards = Path(shards)
        self.ids = list(ids)
        self.batch = batch
        self.replicates = replicates
        self.seed = seed
        self.steps = steps
        self.offset = offset
        self.crop = crop
        self.esmc_array = str(esmc_array) if esmc_array else None
        self.esmc_rows = esmc_rows
        self.buckets = buckets
        self._array = None
        self._epoch = (-1, None)

    def __len__(self) -> int:
        return self.steps

    def _order(self, epoch: int) -> np.ndarray:
        if self._epoch[0] != epoch:
            self._epoch = (
                epoch,
                np.random.default_rng([self.seed, 7919, epoch]).permutation(
                    len(self.ids)
                ),
            )
        return self._epoch[1]

    def chains_of_step(self, step: int) -> list[str]:
        """The ids of the chains of one step, in order; the permutation of an epoch is dealt out in sequence across steps."""
        first = step * self.batch
        out = []
        for position in range(first, first + self.batch):
            epoch, within = divmod(position, len(self.ids))
            out.append(self.ids[int(self._order(epoch)[within])])
        return out

    def conditioned(self, step: int, j: int) -> bool:
        """Whether the chain at slot `j` of step `step` is given its sequence: a coin of its own stream, so that the crops, the rotations and the noise of every other
        stream are what they are without it (an arm with a fraction of 0 is the stream of the arms before it)."""
        if self.condition_fraction <= 0.0:
            return False
        coin = np.random.default_rng([self.seed, step, j, 424242]).random()
        return bool(coin < self.condition_fraction)

    def masked(self, step: int, j: int) -> bool:
        """Whether the chain at slot `j` of step `step`, given a sequence, is given it in part (a design: its masked residues are generated) rather than whole: a coin of its own
        stream again, so that the chains, crops, rotations and which chains are conditioned at all are what they are without it."""
        if self.mask_fraction <= 0.0 or not self.conditioned(step, j):
            return False
        coin = np.random.default_rng([self.seed, step, j, 515151]).random()
        return bool(coin < self.mask_fraction)

    def structured(self, step: int, j: int) -> bool:
        """Whether the chain at slot `j` of step `step`, given its sequence in part, is also given the backbone atoms of its known residues (a design with the structure of what it
        keeps): a coin of its own stream, among the masked slots only, so that which chains are conditioned and masked are what they are without it."""
        if self.structure_fraction <= 0.0 or not self.masked(step, j):
            return False
        coin = np.random.default_rng([self.seed, step, j, 717171]).random()
        return bool(coin < self.structure_fraction)

    def masked_slots(self, steps: range) -> list[tuple[int, int, str]]:
        """`(step, slot, chain)` of every masked slot of the given steps: what the script that writes the masked rows makes rows for."""
        return [
            (step, j, name)
            for step in steps
            for j, name in enumerate(self.chains_of_step(step))
            if self.masked(step, j)
        ]

    def rows(self, name: str) -> np.ndarray | None:
        if self.esmc_array is None:
            return None
        if self._array is None:
            self._array = np.load(self.esmc_array, mmap_mode="r")
        start, length = self.esmc_rows[name]
        # a copy: the array is mapped read-only, and a batch's tensors are made from it
        return np.array(self._array[start : start + length])

    def __getitem__(self, i: int) -> dict:
        if self.mask_fraction > 0 and self.masked_rows is None:
            raise ValueError("a mask fraction above 0 needs the masked rows")
        import torch
        from rfproteina.datapipes.collate import collate_batch

        step = self.offset + int(i)
        samples = []
        for j, name in enumerate(self.chains_of_step(step)):
            shard = torch.load(self.shards / f"{name}.pt", weights_only=False)
            item = {k: shard[k].numpy() for k in SHARD_KEYS}
            design = {}
            if self.masked(step, j):
                rows, known = self.masked_rows.get(step, j, name)
                design = {
                    "known": known,
                    "masked_context": rows,
                    "structure": self.structured(step, j),
                }
            feats = chain_features(
                item,
                np.random.default_rng([self.seed, step, j]),
                crop=self.crop,
                rotate=self.rotate,
                esmc=self.rows(name),
                condition_sequence=self.conditioned(step, j),
                context_rows=self.context_rows,
                **design,
            )
            samples.append(
                {
                    "example_id": name,
                    "feats": {k: torch.from_numpy(v) for k, v in feats.items()},
                }
            )
        fixed = {"I": self.buckets} if self.buckets else None
        padded = collate_batch(
            samples,
            SCHEMA,
            fixed_dimensions=fixed,
            keep_atom_array=False,
            num_replicates=self.replicates,
        )
        # the release's collate gives a dict-like object; the module reads and writes it as a dict
        batch = dict(padded.data_samples.tensors)
        batch["padding_mask"] = padded["padding_mask"]
        batch["example_id_batch"] = padded["example_id_batch"]
        return batch


def corpus_rows(chains_npz: str | Path) -> dict[str, tuple[int, int]]:
    """`{id: (start, length)}` of every chain of the corpus in the ESMC array (rows align with `starts`, `src/foldspace/rfd4/esmc_corpus_features.py`)."""
    z = np.load(chains_npz, allow_pickle=False)
    return {
        str(i): (int(s), int(n))
        for i, s, n in zip(z["ids"], z["starts"], z["lengths"], strict=True)
    }


def split_ids(index_json: str | Path, split: str, limit: int = 0) -> list[str]:
    """The chain ids of one split of a shard directory's `index.json`, in its order (the first `limit` when given)."""
    index = json.loads(Path(index_json).read_text())
    ids = [r["id"] for r in index[split]]
    return ids[:limit] if limit else ids
