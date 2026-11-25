"""The upstream seams this repository is built on, asserted.

foldspace reaches into two upstreams for things their high-level APIs do not
carry: the ``lm_hidden_states`` injection point, the feature names the ESMC call
needs, and two private helpers of upstream's folding loop. None of that is
patched or vendored -- but all of it can be renamed upstream without breaking
any test they run.

So it is asserted here. These cost no weights, no GPU and no network; they fail
loudly on an upstream rename rather than silently changing what a probe
measures.
"""

from __future__ import annotations

import inspect

import pytest

esm = pytest.importorskip("esm", reason="needs the esm source tree")


def test_forward_accepts_injected_lm_hidden_states():
    """The seam the whole design rests on."""
    from esm.models.esmfold2.model import EsmFold2Model

    params = inspect.signature(EsmFold2Model.forward).parameters
    assert "lm_hidden_states" in params
    assert params["lm_hidden_states"].default is None, (
        "an injected stack must be optional, or an ordinary fold would break"
    )


def test_esmc_is_skipped_when_a_stack_is_supplied():
    """``load_esmc=False`` is only safe because forward guards the ESMC call."""
    from esm.models.esmfold2 import model as model_mod

    source = inspect.getsource(model_mod.EsmFold2Model.forward)
    assert "lm_hidden_states is None" in source, (
        "forward no longer guards the ESMC call on a missing stack; a cached "
        "stack would be recomputed, or an unguarded call would fail"
    )


def test_the_lm_path_is_still_detached():
    """If this stops holding, a gradient path opens."""
    from esm.models.esmfold2 import model as model_mod

    source = inspect.getsource(model_mod.EsmFold2Model.forward)
    assert "lm_hidden_states.detach()" in source, (
        "the LM->pair projection is no longer detached: re-check whether a "
        "structure loss now reaches an ESM-derived latent"
    )


def test_the_high_level_fold_still_cannot_carry_a_stack():
    """Why the probe calls ``net(...)`` rather than ``fold_atom_array(...)``."""
    from esm.models.esmfold2.processor import ESMFold2InputBuilder

    params = inspect.signature(ESMFold2InputBuilder.fold).parameters
    assert "lm_hidden_states" not in params, (
        "upstream's fold() now carries lm_hidden_states; foldspace can use the "
        "supported path instead of calling forward directly"
    )


def test_decode_is_public():
    from esm.models.esmfold2.processor import ESMFold2InputBuilder

    assert callable(ESMFold2InputBuilder.decode)
    params = inspect.signature(ESMFold2InputBuilder.decode).parameters
    for name in ("output", "features", "chain_infos"):
        assert name in params


def test_upstream_helpers_the_probe_borrows_still_exist():
    from esm.models.esmfold2 import processor

    assert callable(processor._seed_context)
    assert callable(processor._lm_dropout_context)


#: Whether `_lm_dropout_context(net, 0)` applies the request, by `esm.__version__`. Under 3.4.1.post1 a request
#: of 0 is a no-op and a probe that asks for it runs at the checkpoint's own 0.25; from 3.4.2 it switches the dropout off for the call. A
#: version that is not listed here is a move of the pin that has not been read: the test fails until it has and the entry is added, so that
#: a move is a decision and not a dependency bump. Records made under the two behaviours are not comparable for that reason alone.
ZERO_IS_APPLIED = {"3.4.1.post1": False, "3.4.2": True, "3.4.3": True}


def test_a_request_of_zero_does_what_the_dropout_context_of_the_pins_esm_does():
    """The request of 0 is ignored under esm 3.4.1.post1 and applied from 3.4.2; an esm not in `ZERO_IS_APPLIED` fails."""
    from types import SimpleNamespace

    from esm.models.esmfold2.processor import _lm_dropout_context

    assert esm.__version__ in ZERO_IS_APPLIED, (
        f"esm {esm.__version__} is not a revision whose dropout context has been "
        "read: read it, add it to ZERO_IS_APPLIED and record why"
    )
    applied = ZERO_IS_APPLIED[esm.__version__]

    def release_model():
        encoder = SimpleNamespace(lm_dropout=0.25, per_loop_lm_dropout=True)
        return SimpleNamespace(
            config=SimpleNamespace(lm_encoder=encoder, type="release")
        ), encoder

    model, encoder = release_model()
    with _lm_dropout_context(model, 0.0):
        assert (encoder.lm_dropout, encoder.per_loop_lm_dropout) == (
            0.0 if applied else 0.25,
            True,
        )
    assert (encoder.lm_dropout, encoder.per_loop_lm_dropout) == (0.25, True)
    model, encoder = release_model()
    with _lm_dropout_context(model, 0.1):
        assert (encoder.lm_dropout, encoder.per_loop_lm_dropout) == (0.1, True)
    assert (encoder.lm_dropout, encoder.per_loop_lm_dropout) == (0.25, True)


def test_the_esmc_call_takes_the_feature_names_we_pass():
    from esm.models.esmfold2.model import EsmFold2Model

    params = list(inspect.signature(EsmFold2Model._compute_lm_hidden_states).parameters)
    assert params[1:6] == [
        "input_ids",
        "asym_id",
        "residue_index",
        "mol_type",
        "tok_mask",
    ]


def test_prepared_features_carry_what_the_esmc_call_needs():
    """The featurizer's key names, against the ESMC call's argument list.

    ``tok_mask`` is forward's local name for ``token_attention_mask``
    (model.py:1064); the other four are feature keys verbatim.
    """
    pytest.importorskip("atomworks", reason="needs the atomworks source tree")
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput

    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="MKTAYIAKQRQISFVKSHFSRQ")]
    )
    features, chain_infos = ESMFold2InputBuilder().prepare_input(spi)

    for key in ("input_ids", "asym_id", "residue_index", "mol_type", "res_type"):
        assert key in features, f"featurizer no longer emits {key!r}"
    assert "token_attention_mask" in features, (
        "the mask forward aliases as tok_mask was renamed"
    )
    assert len(chain_infos) == 1


def test_the_layer_stack_is_what_the_mixture_expects():
    """``compute_lm_hidden_states`` returns a stack, not a state."""
    from esm.models.esmfold2.layers import LanguageModelShim, compute_lm_hidden_states

    assert "n_layers+1" in (compute_lm_hidden_states.__doc__ or "").replace(
        " ", ""
    ) or ("N=81" in (compute_lm_hidden_states.__doc__ or ""))
    shim = LanguageModelShim(d_z=8, d_model=16, num_layers=4)
    assert shim.base_z_combine.shape == (5,), (
        "the mixture is over n_layers + 1 entries; layer_mixing assumes it"
    )


def test_every_substrate_import_resolves():
    """The imports `substrate.py` defers until a GPU is already allocated.

    They live inside functions so that `offline` tests need no upstream stack —
    which also means a wrong path is invisible until a job is running. One of
    them was wrong (`esmfold2_atomworks.model` re-exports nothing), and the
    failure surfaced on a compute node. It costs milliseconds to check here.
    """
    pytest.importorskip("atomworks", reason="needs the atomworks source tree")

    from esmfold2_atomworks.data.atomworks_to_esm import (  # noqa: F401
        AdapterReport,
        atom_array_to_structure_prediction_input,
    )
    from esmfold2_atomworks.data.molecular_complex import (  # noqa: F401
        molecular_complex_to_atom_array,
    )
    from esmfold2_atomworks.metrics import fold_metrics  # noqa: F401
    from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2

    for name in ("featurize", "fold", "fold_atom_array", "config"):
        assert hasattr(AtomWorksESMFold2, name), f"the wrapper lost `{name}`"

    # `net`, `builder` and `device` are set in __init__, so they are not class
    # attributes; the probe reads all three off an instance.
    source = inspect.getsource(AtomWorksESMFold2.__init__)
    for name in ("self.net", "self.builder", "self.device"):
        assert name in source, f"the wrapper no longer assigns `{name}`"


def test_the_two_model_classes_disagree_about_load_esmc():
    """Why `open_substrate` asks the signature instead of assuming one.

    The release `load_esmc(path, precision=...)` casts and freezes through
    `set_esmc_precision`; the experimental `load_esmc(path)` hardcodes bf16 and
    calls `finalize_esmc()`. Passing `precision` to the second is a TypeError,
    and it surfaced on a compute node.
    """
    from esm.models.esmfold2.experimental import EsmFold2ExperimentalModel
    from esm.models.esmfold2.model import EsmFold2Model

    release = inspect.signature(EsmFold2Model.load_esmc).parameters
    experimental = inspect.signature(EsmFold2ExperimentalModel.load_esmc).parameters

    assert "precision" in release
    assert "precision" not in experimental, (
        "the experimental model now takes `precision`; open_substrate can stop "
        "branching on the signature"
    )


def test_the_experimental_forward_carries_the_gradient_gate():
    """The gradient route rests on this argument existing and defaulting to None."""
    from esm.models.esmfold2.experimental import EsmFold2ExperimentalModel

    params = inspect.signature(EsmFold2ExperimentalModel.forward).parameters
    assert "res_type_soft" in params
    assert params["res_type_soft"].default is None
    assert "lm_hidden_states" in params

    source = inspect.getsource(EsmFold2ExperimentalModel.forward)
    assert "set_grad_enabled(res_type_soft is not None)" in source.replace(
        "torch.", ""
    ), "the gradient gate moved; re-check it"


def test_the_soft_sequence_width_is_taken_from_the_model():
    """A soft sequence of the wrong width fails as a matmul shape error.

    It reaches the input embedder as both `res_type_oh` and `profile`, so being
    one class short shows up as a two-column mismatch far from the cause.
    """
    from esm.models.esmfold2.layers import NUM_RES_TYPES

    assert isinstance(NUM_RES_TYPES, int)
    assert NUM_RES_TYPES > 20, "fewer classes than there are amino acids"


def _sample_call_bindings():
    """Every keyword ``structure_head.sample`` is called with, and its source.

    Parsed rather than pattern-matched, so a renamed local or a reordered call
    does not quietly pass. Returns ``(bindings, computed)``: keyword -> the
    local name it is bound to, and the subset of those locals that a
    ``self.<module>(...)`` call produced.
    """
    import ast
    import textwrap

    from esm.models.esmfold2 import model as model_mod

    source = textwrap.dedent(inspect.getsource(model_mod.EsmFold2Model.forward))
    tree = ast.parse(source)
    forward = tree.body[0]
    params = {a.arg for a in forward.args.args} | {
        a.arg for a in forward.args.kwonlyargs
    }

    def _calls_a_module(node) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "self"
            for n in ast.walk(node)
        )

    from_module: set[str] = set()
    for node in ast.walk(forward):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign) and node.value is not None
            else []
        )
        if not targets or not _calls_a_module(node.value):
            continue
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    from_module.add(name.id)

    call = next(
        n
        for n in ast.walk(forward)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "sample"
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "structure_head"
    )
    bindings, computed = {}, set()
    for keyword in call.keywords:
        names = [n.id for n in ast.walk(keyword.value) if isinstance(n, ast.Name)]
        bindings[keyword.arg] = names
        # Either bound to a local a module produced, or calling one inline.
        if any(name in from_module for name in names) or _calls_a_module(keyword.value):
            computed.add(keyword.arg)
        elif any(name not in params and name not in from_module for name in names):
            # A local derived from the features by pure ops. Nothing to hook,
            # and identical across two folds that share one `Prepared`.
            pass
    return bindings, computed


def test_only_three_of_the_samplers_inputs_are_produced_by_a_module():
    """What ``captured_head_conditioning`` is allowed to conclude from.

    The capture records three tensors and the localisation then treats every
    other input to ``structure_head.sample`` as fixed by the ``Prepared`` the
    two folds share. That holds only while the rest of the conditioning is the
    features or pure tensor ops on them. If upstream routes a fourth learned
    module into the sampler, two folds could differ through a channel nothing
    is watching and the diagnostic would still report "all identical".
    """
    _, computed = _sample_call_bindings()
    assert computed == {"z_trunk", "s_inputs", "relative_position_encoding"}, (
        "the set of sampler inputs a module computes has changed; "
        "`captured_head_conditioning` hooks parcae_coda, inputs_embedder and "
        f"rel_pos and would now miss one. Computed: {sorted(computed)}"
    )


def test_the_samplers_keyword_set_is_the_one_the_capture_accounts_for():
    """A new conditioning argument has to fail loudly, not be ignored."""
    bindings, _ = _sample_call_bindings()
    assert set(bindings) == {
        # produced by a module -- captured
        "z_trunk",
        "s_inputs",
        "relative_position_encoding",
        # passed literally
        "s_trunk",
        "return_atom_repr",
        # the features, or pure tensor ops on them
        "ref_pos",
        "ref_charge",
        "ref_mask",
        "ref_element",
        "ref_atom_name_chars",
        "ref_space_uid",
        "tok_idx",
        "asym_id",
        "residue_index",
        "entity_id",
        "token_index",
        "sym_id",
        "token_attention_mask",
        # sampler settings, fixed per call by the probe
        "num_diffusion_samples",
        "num_sampling_steps",
        "noise_scale",
        "step_scale",
        "max_inference_sigma",
    }, f"structure_head.sample's signature moved: {sorted(bindings)}"


def test_the_single_representation_is_still_not_passed():
    """``s_trunk=None`` is what makes candidate B the pair state and only it."""
    bindings, _ = _sample_call_bindings()
    assert bindings["s_trunk"] == [], (
        "the structure head now takes a single representation; candidate B is "
        "no longer 'the pair state and only that' (representations/trunk.py)"
    )
    assert bindings["s_inputs"] == ["x_inputs"]


def test_the_two_seams_each_run_once_per_forward():
    """Per-fold accounting in the capture assumes exactly one call each."""
    from esm.models.esmfold2 import model as model_mod

    source = inspect.getsource(model_mod.EsmFold2Model.forward)
    assert source.count("self.inputs_embedder(") == 1
    assert source.count("self.rel_pos(") == 1
    assert source.count("self.parcae_coda(") == 1, (
        "parcae_coda is called more than once per forward, so `pair[0]` is no "
        "longer the state the structure head read"
    )
