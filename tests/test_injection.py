"""The injection paths of `foldspace.esmfold2.interface` agree with the shim they bypass.

The GPU run (`foldspace.esmfold2.validate_interface parity`) is the experiment; this is the same claim at tensor
level, with random weights and no checkpoint: the pair prior a stack makes is the one
the hook substitutes for its mixed single, a placeholder stack is discarded by the hook,
and the no-prior and native paths refuse a model they do not fit. Needs torch and the
esm source tree, so it is not `offline`.
"""

from __future__ import annotations

import dataclasses
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("esm", reason="needs the esm source tree")

from esm.models.esmfold2.layers import LanguageModelShim

from foldspace.esmfold2 import validate_interface
from foldspace.esmfold2 import interface as I
from foldspace.esmfold2 import inject

D_Z, D_MODEL, N_LAYERS, L = 8, 16, 4, 7


@pytest.fixture
def shim():
    torch.manual_seed(0)
    module = LanguageModelShim(d_z=D_Z, d_model=D_MODEL, num_layers=N_LAYERS).eval()
    with torch.no_grad():
        module.base_z_combine.copy_(torch.tensor([-2.0, 0.5, 1.5, -1.0, 0.0]))
    return module


@pytest.fixture
def stand_in(shim):
    return types.SimpleNamespace(net=shim, has_esmc=True), types.SimpleNamespace(
        n_tokens=L
    )


def test_the_pair_prior_from_a_stack_is_the_one_the_hook_substitutes(shim, stand_in):
    substrate, _ = stand_in
    stack = torch.randn(1, L, N_LAYERS + 1, D_MODEL)
    single = inject.mixed_single(shim, stack)
    via_stack = I.pair_prior(substrate, stack=stack)
    via_single = I.pair_prior(substrate, single=single)
    assert I.tensor_diff(via_single, via_stack)["max_abs"] < 1e-6
    # a float32 round trip through float64, as the cache stores it, changes nothing
    rounded = single.double().float()
    assert torch.equal(rounded, single)
    with pytest.raises(ValueError):
        I.pair_prior(substrate)


def test_observing_the_shim_reads_its_single_and_changes_nothing(shim):
    stack = torch.randn(1, L, N_LAYERS + 1, D_MODEL)
    with torch.no_grad():
        plain = shim(stack)
        with I.observe_single(shim) as seen:
            observed = shim(stack)
        expected = inject.mixed_single(shim, stack)
    assert torch.equal(observed, plain)
    assert seen["stack_dtype"] == "torch.float32"
    assert seen["output_dtype"] == "torch.float32"
    assert seen["autocast"] is False
    assert I.tensor_diff(seen["single"], expected)["max_abs"] < 1e-6
    # the hooks are gone on exit
    assert not shim._forward_hooks and not shim._forward_pre_hooks
    assert not shim.base_z_mlp._forward_pre_hooks


def test_the_model_single_is_the_single_entering_base_z_mlp_in_the_model_context(
    shim, stand_in
):
    """Without a CUDA device the context is a no-op, so the two definitions coincide; on one
    the model's is bfloat16 and the float32 outside it is not (the GPU run reads that)."""
    if torch.cuda.is_available():
        pytest.skip(
            "on a CUDA device model_single is bfloat16 and differs from float32"
        )
    substrate, _ = stand_in
    stack = torch.randn(1, L, N_LAYERS + 1, D_MODEL)
    with I.observe_single(shim) as seen, torch.no_grad():
        shim(stack)
    made = I.model_single(substrate, stack)
    assert made.dtype == torch.float32
    assert torch.equal(made, seen["single"])
    # the same arithmetic with autograd on can differ in the last place, not more
    assert torch.allclose(made, inject.mixed_single(shim, stack), atol=1e-6)


def test_the_model_context_changes_nothing_off_a_cuda_device():
    if torch.cuda.is_available():
        pytest.skip("this checks the no-device branch")
    before = torch.is_autocast_enabled()
    with I.model_context():
        assert torch.is_autocast_enabled() == before
    assert torch.is_autocast_enabled() == before


def test_a_placeholder_stack_has_the_shims_shape_and_is_discarded_by_the_hook(
    shim, stand_in
):
    substrate, prepared = stand_in
    placeholder = I.placeholder_stack(substrate, prepared)
    assert placeholder.shape == (1, L, N_LAYERS + 1, D_MODEL)
    with torch.no_grad():
        assert torch.isfinite(shim(placeholder)).all()
        single = torch.randn(1, L, D_Z)
        with inject.substituted_single(shim, single):
            injected = shim(placeholder)
        expected = shim.base_z_mlp(single)
    assert torch.equal(injected, expected)
    # and the hook is gone on exit
    assert not torch.equal(shim(placeholder), expected)


@dataclasses.dataclass
class _Substrate:
    """The two fields of a Substrate that dropping a backbone reads and changes."""

    net: object
    has_esmc: bool = True
    esmc_source: str = "bundled"

    @property
    def esmc(self):
        return self.net.esmc


def test_dropping_the_backbone_removes_it_and_says_so():
    """`load_esmc=False` leaves a bundled backbone resident; it has to be taken out."""
    net = types.SimpleNamespace(esmc=torch.nn.Linear(2, 2))
    resident = _Substrate(net)
    assert resident.esmc is not None

    lite, info = I.drop_backbone(resident)

    assert info["present_before"] is True and info["present_after"] is False
    assert lite.has_esmc is False and lite.esmc is None
    assert lite.esmc_source == "none"
    # the model itself lost it (a shallow copy shares it), not just the wrapper's flag
    assert net.esmc is None and resident.esmc is None


def test_dropping_a_backbone_that_cannot_be_removed_is_refused():
    class Stubborn:
        """A model whose backbone cannot be unset: assigning to it is ignored."""

        @property
        def esmc(self):
            return torch.nn.Linear(1, 1)

        @esmc.setter
        def esmc(self, value):
            pass

    with pytest.raises(RuntimeError, match="still reachable"):
        I.drop_backbone(_Substrate(Stubborn()))


def test_a_fold_without_a_prior_needs_a_model_without_a_backbone():
    resident = types.SimpleNamespace(has_esmc=True)
    with pytest.raises(RuntimeError, match="load_esmc=False"):
        I.fold_without_prior(resident, None, seed=0)
    absent = types.SimpleNamespace(has_esmc=False)
    with pytest.raises(RuntimeError, match="backbone resident"):
        I.fold_native(absent, None, seed=0)


def _cache_fixture(tmp_path, length=80):
    import numpy as np

    z = np.random.default_rng(0).standard_normal((length, 256)).astype(np.float32)
    np.savez(
        tmp_path / "z_shard0.npz",
        ids=np.array(["x1"]),
        lengths=np.array([length]),
        starts=np.array([0]),
        z=z,
    )
    tsv = tmp_path / "train_sequences.tsv"
    tsv.write_text("seq_id\tsequence\nx1\t" + "A" * length + "\n")
    args = types.SimpleNamespace(
        cache_check=1, train_sequences=tsv, targets=tmp_path, lo=70, hi=220
    )
    return z, args


def test_the_cache_check_reads_a_single_that_carries_a_graph(tmp_path):
    """The shim's weights require grad; a single made from them cannot be `.numpy()`-ed as is."""
    z, args = _cache_fixture(tmp_path)
    graphed = lambda net, stack: torch.tensor(z[None], requires_grad=True)
    sub_stub = types.SimpleNamespace(
        prepare_sequences=lambda substrate, sequence: object(),
        lm_hidden_states=lambda substrate, prepared: torch.zeros(1, 80, 2, 3),
    )
    inject_stub = types.SimpleNamespace(mixed_single=graphed)

    out = validate_interface._cache_check(
        args, types.SimpleNamespace(net=None), sub_stub, inject_stub, 1e-6
    )

    assert out["all_within"] is True
    assert out["chains"][0]["exact"] is True


def test_the_cache_check_flags_a_single_that_differs_from_the_cache(tmp_path):
    z, args = _cache_fixture(tmp_path)
    shifted = lambda net, stack: torch.tensor(z[None] + 0.5, requires_grad=True)
    sub_stub = types.SimpleNamespace(
        prepare_sequences=lambda substrate, sequence: object(),
        lm_hidden_states=lambda substrate, prepared: torch.zeros(1, 80, 2, 3),
    )
    out = validate_interface._cache_check(
        args,
        types.SimpleNamespace(net=None),
        sub_stub,
        types.SimpleNamespace(mixed_single=shifted),
        1e-6,
    )
    assert out["all_within"] is False
    assert out["chains"][0]["max_abs"] == pytest.approx(0.5, rel=1e-3)


def test_the_confidence_keys_are_the_ones_the_substrate_emits():
    """A key that is not emitted is read as None without any error: a run records no
    confidence at all and nothing says so (pLDDT was None in every record once)."""
    metrics = pytest.importorskip("esmfold2_atomworks.metrics")
    assert validate_interface.PLDDT in metrics.SCALAR_METRIC_SOURCES
    assert validate_interface.PTM in metrics.SCALAR_METRIC_SOURCES


def test_tensor_tolerance_follows_the_shim_dtype():
    assert validate_interface.tensor_tolerance("float32") == pytest.approx(
        4 * torch.finfo(torch.float32).eps
    )
    assert validate_interface.tensor_tolerance("bfloat16") > validate_interface.tensor_tolerance("float32")
