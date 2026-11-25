"""The folding substrate, opened once and driven a state at a time.

`esmfold2-atomworks` gives a structure in and a structure out. A probe needs the
middle: compute the ESMC stack once, put something else in its place, and fold
from there. Every seam used here is public in one of the two upstreams --
nothing is patched, monkey-patched or vendored, which is the rule of this repository.

The loop::

    spi                      = atom_array_to_structure_prediction_input(...)   esmfold2-atomworks
    features, chain_infos    = model.featurize(spi)                            esmfold2-atomworks
    h                        = lm_hidden_states(substrate, prepared)           esm
    out                      = model.net(**features, lm_hidden_states=h')      esm
    result                   = model.builder.decode(out, features, chain_infos) esm
    atoms                    = molecular_complex_to_atom_array(result.complex) esmfold2-atomworks
    metrics                  = fold_metrics(result)                            esmfold2-atomworks

Two properties of that loop matter more than the code.

**ESMC is computed once per sequence.** ``h`` is deterministic given the
sequence and the checkpoint, and costs a 6B forward pass; a σ sweep re-runs only
the trunk and the head. Once the stack is cached, the substrate can be opened
with ``load_esmc=False`` and the backbone is never resident.

**``lm_dropout`` must be 0 while probing.** Upstream defaults it to 0.3 and
applies it with ``training=True``, fresh mask per recycle, *on the ESMC→pair
channel* -- the exact channel a perturbation experiment is measuring. Left at
the default it would mix a 30 % random mask into every measurement of σ. The
default here is 0.0, and the scatter control has to be measured at the
same setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The substrate names its weight sets by role; foldspace names them by
#: directory, because a probe result is read against a checkpoint, not a role.
WEIGHT_SETS = {
    "ESMFold2": "standard",
    "ESMFold2-Fast": "fast",
    "ESMFold2-Experimental": "experimental",
}

#: Probing defaults. Deliberately not upstream's folding defaults: see the
#: module docstring on ``lm_dropout``.
PROBE_NUM_LOOPS: int | None = None
"""``None`` means the loaded checkpoint's own ``config.num_loops``.

It was 3 here until 2026-09-25, described as the shipped default -- and the
checkpoint `biohub/ESMFold2` publishes today declares **20**, so a fixed 3 would
now run a sixth of the schedule without saying so. The substrate hit the same
thing and fixed it the same way. Every run records what was resolved."""
PROBE_NUM_SAMPLING_STEPS = 100
PROBE_LM_DROPOUT = 0.0


@dataclass
class Substrate:
    """A folding model held open, plus the upstream pieces the probe calls.

    Attributes
    ----------
    model
        ``esmfold2_atomworks.model.AtomWorksESMFold2``.
    checkpoint
        What was asked for, for the run manifest.
    has_esmc
        Whether the ESMC backbone is resident. False means every fold must be
        given ``lm_hidden_states``; the model would otherwise silently produce a
        fold with no LM pair prior at all.
    """

    model: Any
    checkpoint: str
    weights: Any = None
    """Where the label resolved to. A checkpoint label goes through the
    substrate's own path discovery, so the label alone does not say which
    directory was read -- and provenance that cannot name the weights cannot
    identify the run."""
    has_esmc: bool = True
    #: Where the backbone came from: ``"bundled"`` in the checkpoint itself,
    #: the path it was attached from, or ``"none"`` when it is not resident.
    esmc_source: str = "none"

    @property
    def net(self) -> Any:
        """The native ``EsmFold2Model``."""
        return self.model.net

    @property
    def esmc(self) -> Any:
        """The ESMC backbone, or None -- the substrate's own seam.

        foldspace read this off ``net`` directly until `esmfold2-atomworks`
        0.3 made the seam truthful; it now goes through the accessor, which is
        what the seam is for.
        """
        return self.model.esmc

    @property
    def device(self) -> Any:
        return self.model.device

    def dims(self) -> dict[str, int]:
        """Live widths, from the substrate's own helper.

        foldspace read the migrated config names itself while
        ``representation_dims()`` still looked for the pre-migration ones and
        answered zeros. That is fixed upstream, so the helper is the seam again;
        the LM widths it does not report are added here.
        """
        cfg = self.net.config
        out = dict(self.model.representation_dims())
        out["lm_d_model"] = int(getattr(cfg, "lm_d_model", 0))
        out["lm_num_layers"] = int(getattr(cfg, "lm_num_layers", 0))
        return out


def open_substrate(
    checkpoint: str = "ESMFold2",
    *,
    device: Any = None,
    load_esmc: bool = True,
    esmc_precision: str = "bf16",
    chunk_size: int | None = 64,
) -> Substrate:
    """Load a folding checkpoint.

    ``load_esmc=False`` skips the ~12 GB backbone entirely. Valid only when
    every fold will be handed a cached ``lm_hidden_states``.

    The backbone is resolved by the substrate, not by ``config.esmc_id`` as
    written -- `esmfold2-atomworks` 0.3 constructs without one and calls
    ``attach_esmc``, which goes through ``paths.resolve_esmc`` and prefers the
    local mirror. foldspace carried its own version of that until the fix
    landed upstream; this now just reads back where the backbone came from.
    """
    from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2
    from esmfold2_atomworks.paths import ESMFOLD2_WEIGHTS

    weights = ESMFOLD2_WEIGHTS.resolve(WEIGHT_SETS.get(checkpoint, checkpoint))
    model = AtomWorksESMFold2(
        weights=weights,
        device=device,
        load_esmc=load_esmc,
        esmc_precision=esmc_precision,
        chunk_size=chunk_size,
    )
    return Substrate(
        model=model,
        checkpoint=checkpoint,
        weights=weights,
        has_esmc=model.esmc is not None,
        esmc_source=model.esmc_source,
    )


@dataclass
class Prepared:
    """One system, featurized and ready to fold repeatedly."""

    features: dict
    chain_infos: list
    spi: Any
    report: Any = None
    meta: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return int(self.features["res_type"].shape[1])


def prepare(
    substrate: Substrate,
    atoms: Any,
    *,
    chain_info: dict | None = None,
    seed: int | None = None,
    adapter_kwargs: dict | None = None,
) -> Prepared:
    """AtomArray -> batched features on the model device.

    ``chain_info`` is the dict from the same ``atomworks.io.parse`` call. It is
    strongly preferred: without it the adapter reads sequences off the modelled
    residues, silently deleting unresolved loops and folding a different
    construct (``esmfold2-atomworks`` D-003).
    """
    from esmfold2_atomworks.data.atomworks_to_esm import (
        AdapterReport,
        atom_array_to_structure_prediction_input,
    )

    report = AdapterReport()
    kwargs = dict(adapter_kwargs or {})
    kwargs.setdefault("report", report)
    spi = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, **kwargs
    )
    features, chain_infos = substrate.model.featurize(spi, seed=seed)
    return Prepared(features=features, chain_infos=chain_infos, spi=spi, report=report)


def prepare_sequences(
    substrate: Substrate,
    sequences: dict[str, str] | str,
    *,
    seed: int | None = None,
) -> Prepared:
    """Sequences -> batched features, with no structure anywhere.

    ``lm_hidden_states`` reads five features — ``input_ids``, ``asym_id``,
    ``residue_index``, ``mol_type`` and ``token_attention_mask`` — and every
    one of them is a function of the sequence. ``mixed_single`` is then a
    learned mixture over the stack. So the ESM-derived latent this repository
    calls a candidate generative coordinate needs no coordinates to compute,
    and a corpus for it needs no structures.

    Built from ``StructurePredictionInput`` and ``ProteinInput``, which is the
    upstream's own sequence-only entry point, through the same
    ``model.featurize`` as :func:`prepare`. Nothing about featurization is
    reimplemented here, which is what makes the two paths comparable.

    What this does **not** carry is the structure-derived side of the
    conditioning: no MSA drawn from the deposited construct, no reference
    conformers, no coordinates. That is fine for the LM stack and wrong for a
    fold, so this returns a ``Prepared`` for measuring the latent and not one
    to hand to :func:`fold`.
    """
    from esm.utils.structure.input_builder import (
        ProteinInput,
        StructurePredictionInput,
    )

    items = {"A": sequences} if isinstance(sequences, str) else dict(sequences)
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id=chain, sequence=sequence)
            for chain, sequence in items.items()
        ]
    )
    features, chain_infos = substrate.model.featurize(spi, seed=seed)
    return Prepared(features=features, chain_infos=chain_infos, spi=spi, report=None)


def lm_hidden_states(
    substrate: Substrate, prepared: Prepared, *, lm_mask_pct: float = 0.0
):
    """The ESMC stack the folding model would compute, ``[B, L, n_layers+1, d_model]``.

    Deterministic given the sequence and the checkpoint when ``lm_mask_pct`` is
    0 -- which is why it is worth caching, and why the cache key need not
    include anything else.

    Uses ``net._compute_lm_hidden_states``, upstream-private, because it carries
    the fp8 precision context and the CPU-offload handling that a bare call to
    ``compute_lm_hidden_states`` would drop.
    ``tests/test_upstream_contract.py`` fails loudly on a rename upstream, rather than letting it quietly change the
    numbers.
    """
    if substrate.esmc is None:
        raise RuntimeError(
            "this substrate was opened with load_esmc=False; there is no "
            "backbone to compute hidden states with"
        )
    f = prepared.features
    return substrate.net._compute_lm_hidden_states(
        f["input_ids"],
        f["asym_id"],
        f["residue_index"],
        f["mol_type"],
        # forward's own `tok_mask` is exactly this feature (model.py:1064).
        f["token_attention_mask"],
        lm_mask_pct=lm_mask_pct,
    )


@dataclass
class FoldOutput:
    """One fold, decoded."""

    result: Any
    """``MolecularComplexResult`` -- the raw upstream object, kept because the
    inference engine discards it and the probe needs its per-token arrays."""

    raw: dict
    """``EsmFold2Model.forward``'s output dict."""

    def atoms(self):
        """The prediction as an AtomArray, through the substrate's round trip.

        Not via `to_mmcif()` + `parse()`: that route reconciles a SMILES ligand
        against the real CCD entry and has been seen to reduce 19 ligand atoms
        to 8 carbons (seen with `esmfold2-atomworks`).
        """
        from esmfold2_atomworks.data.molecular_complex import (
            molecular_complex_to_atom_array,
        )

        return molecular_complex_to_atom_array(self.result.complex)

    def metrics(self) -> dict[str, float]:
        from esmfold2_atomworks.metrics import fold_metrics

        return fold_metrics(self.result)


def fold(
    substrate: Substrate,
    prepared: Prepared,
    *,
    lm_hidden_states: Any = None,
    seed: int | None = None,
    num_loops: int = PROBE_NUM_LOOPS,
    num_sampling_steps: int = PROBE_NUM_SAMPLING_STEPS,
    lm_dropout: float = PROBE_LM_DROPOUT,
    include_embeddings: bool = False,
    complex_id: str = "probe",
) -> FoldOutput:
    """Fold, optionally from a substituted ESMC stack.

    This is upstream's ``ESMFold2InputBuilder.fold`` with one difference: it
    forwards ``lm_hidden_states``, which that method's fixed keyword list does
    not carry. Everything else -- seeding, the dropout context, decoding --
    is upstream's own, called directly.
    """
    import torch
    from esm.models.esmfold2.processor import _lm_dropout_context, _seed_context

    if lm_hidden_states is None and not substrate.has_esmc:
        raise RuntimeError(
            "no lm_hidden_states given and no ESMC backbone is loaded; the "
            "fold would silently run with no LM pair prior"
        )

    from contextlib import nullcontext

    net = substrate.net
    with (
        torch.no_grad(),
        _seed_context(seed) if seed is not None else nullcontext(),
        _lm_dropout_context(net, lm_dropout),
    ):
        raw = net(
            **prepared.features,
            lm_hidden_states=lm_hidden_states,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=1,
            include_embeddings=include_embeddings,
        )

    result = substrate.model.builder.decode(
        raw,
        prepared.features,
        prepared.chain_infos,
        num_diffusion_samples=1,
        complex_id=complex_id,
    )
    return FoldOutput(result=result, raw=raw)
