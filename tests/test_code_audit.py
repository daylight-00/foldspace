"""The analysis of the representation audit (`src/foldspace/rfd4/code_audit.py`) on synthetic residues whose answer is known: a code that is the release's identity centroids under a similarity transform is aligned and
carries no context, an unrelated code is not aligned, a deterministic code has no residual to read, a learned posterior reports its regularizers and its quiet channels, and the release's own channel that depends on the
chain's fold has a held-out R² on the Cα geometry. The part of the audit that needs torch and the release's autoencoder is read on the stored artifacts."""

from __future__ import annotations

import numpy as np
import pytest
import foldspace.rfd4.code_audit as audit

pytestmark = pytest.mark.offline

CHAINS, LENGTH = 120, 80


def ideal(n: int, helical: bool) -> np.ndarray:
    k = np.arange(n)
    if helical:
        angle = np.deg2rad(100.0) * k
        return np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * k], axis=1)
    return np.stack([3.3 * k, 0.9 * (k % 2), np.zeros(n)], axis=1)


def synthetic(seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((20, 2)) * 2.0
    types, chain, position, chain_inputs = [], [], [], []
    for c in range(CHAINS):
        t = rng.integers(0, 20, LENGTH)
        ca = ideal(LENGTH, helical=c % 2 == 0) + 0.05 * rng.standard_normal((LENGTH, 3))
        chain_inputs.append((t, ca, np.ones(LENGTH, dtype=bool)))
        types.append(t)
        chain.append(np.full(LENGTH, c))
        position.append(np.arange(LENGTH))
    t, chain = np.concatenate(types), np.concatenate(chain)
    n = len(t)
    helical = (chain % 2 == 0).astype(float)
    native = rng.standard_normal((n, 32))
    native[:, 0] = (
        centroids[t, 0] + 0.3 * rng.standard_normal(n) + 1.0 * helical
    )  # channel 0 depends on the fold
    native[:, 10] = centroids[t, 1] + 0.3 * rng.standard_normal(
        n
    )  # channel 10 does not
    angle = 0.9
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    slots = [3, 7]

    def code(identity: np.ndarray) -> dict:
        mean = rng.standard_normal((n, 18))
        mean[:, slots] = identity
        return {"mean": mean, "mu": mean.copy(), "sigma": None}

    aligned = code(1.7 * centroids[t] @ rotation + 0.3 * rng.standard_normal((n, 2)))
    random_table = rng.standard_normal((20, 2)) * 2.0
    unrelated = code(random_table[t] + 0.3 * rng.standard_normal((n, 2)))
    # a deterministic code: a table of the type in every channel, so its residual within types is zero
    table18 = rng.standard_normal((20, 18)) * 2.0
    fixed = {"mean": table18[t], "mu": table18[t].copy(), "sigma": None}
    learned = code(1.7 * centroids[t] @ rotation + 0.3 * rng.standard_normal((n, 2)))
    sigma = np.full((n, 18), 0.99)
    sigma[:, slots] = 0.25
    learned["sigma"] = sigma
    learned["mu"] = learned["mean"] / np.sqrt(
        1 - sigma**2
    )  # a code whose posterior mean is `mean`
    data = {
        "native": native,
        "types": t,
        "chain": chain,
        "position": np.concatenate(position),
        "chain_inputs": chain_inputs,
        "cache": {},
        "adapters": {
            "aligned": aligned,
            "unrelated": unrelated,
            "fixed": fixed,
            "learned": learned,
        },
    }
    adapters = {
        "aligned": {"slots": slots, "learned": False, "source": "esmc"},
        "unrelated": {"slots": slots, "learned": False, "source": "esmc"},
        "fixed": {"slots": slots, "learned": False, "source": "fixed"},
        "learned": {"slots": slots, "learned": True, "source": "esmc"},
    }
    return data, adapters


@pytest.fixture(scope="module")
def report() -> dict:
    data, adapters = synthetic()
    return audit.analyse(data, adapters, perm=200, seed=1)


def test_the_release_encoders_own_channels_are_read_for_context_and_for_their_ceiling(
    report,
) -> None:
    assert report["residues"] == CHAINS * LENGTH and report["chains"] == CHAINS
    assert report["usable_for_context"] == CHAINS * (
        LENGTH - 6
    )  # the chain's ends are left out
    native = report["native"]
    assert np.median(native["split_half_ceiling"]) > 0.95
    both = native["context"]["both"]
    # channel 0 depends on whether the chain is helical, which the Cα geometry says; channel 10 depends on nothing; a shuffled geometry says nothing
    assert both[0] > 0.1 and both[1] < 0.02
    assert native["context"]["structure"][0] > 0.1
    assert max(native["context"]["both_shuffled"]) < 0.02


def test_a_similarity_transformed_code_is_aligned_with_no_context_and_an_unrelated_one_is_not(
    report,
) -> None:
    aligned, unrelated = report["adapters"]["aligned"], report["adapters"]["unrelated"]
    assert aligned["alignment"]["similarity_r2"] > 0.95
    assert aligned["alignment"]["similarity_r2"] > aligned["alignment"]["null_p99"]
    assert unrelated["alignment"]["similarity_r2"] < unrelated["alignment"]["null_p99"]
    assert aligned["alignment"]["affine_r2"] >= aligned["alignment"]["similarity_r2"] - 1e-9
    assert (
        max(aligned["context"]["both"]) < 0.02
    )  # the code has no context in it: no held-out R² above zero
    assert aligned["kind"] == "fixed-sigma-or-deterministic"
    # the identity slots are two channels fixed by the type: nearly all of their variance is between types, and the rest is noise in 2 directions
    shares = [aligned["within_type_share"][s] for s in aligned["slots"]]
    assert all(s < 0.1 for s in shares)
    assert 1.5 < aligned["participation_ratio_identity"] <= 2.0


def test_a_deterministic_code_has_no_residual_to_read(report) -> None:
    fixed = report["adapters"]["fixed"]
    assert fixed["kind"] == "fixed-code" and "context" not in fixed
    assert fixed["identity_within_type_variance"] == pytest.approx(0.0, abs=1e-12)
    assert max(fixed["within_type_share"]) < 1e-9


def test_a_learned_posterior_reports_its_quiet_channels_and_its_regularizers(
    report,
) -> None:
    learned = report["adapters"]["learned"]
    assert learned["kind"] == "learned"
    assert learned["channels_below_0.9"] == [3, 7]
    r4 = learned["regularizers"]
    # sigma is 0.99 in 16 channels and 0.25 in two: the rate term is -ln sigma summed, and the exact KL adds what it leaves out
    assert r4["all"]["rate"] == pytest.approx(
        -16 * np.log(0.99) - 2 * np.log(0.25), rel=1e-9
    )
    assert r4["all"]["exact"] - r4["all"]["rate"] == pytest.approx(
        r4["all"]["omitted"], rel=1e-9
    )
    assert r4["identity_slots"]["rate"] == pytest.approx(-2 * np.log(0.25), rel=1e-9)
    groups = ("identity_slots", "other_16")
    assert sum(r4[g]["omitted"] for g in groups) == pytest.approx(
        r4["all"]["omitted"], rel=1e-9
    )
    assert "participation_ratio_below_0.9" in learned


def test_the_donor_swap_removes_the_context_and_keeps_the_type_geometry_and_the_variance() -> (
    None
):
    """The type-shuffled control on residues whose answer is known: a code that depends on its residue's chain (helical or not) and the same code with each residue's replaced by a donor of its type."""
    import foldspace.rfd4.code_bank as bk

    data, _ = synthetic()
    rng = np.random.default_rng(7)
    t, chain = data["types"], data["chain"]
    n = len(t)
    helical = (chain % 2 == 0).astype(float)
    centroids = rng.standard_normal((20, 2)) * 2.0
    slots = [3, 7]
    mean = rng.standard_normal((n, 18))
    identity = centroids[t] + 0.3 * rng.standard_normal((n, 2))
    identity[:, 0] += 1.0 * helical  # the residue's own context
    mean[:, slots] = identity
    layout = bk.bank_layout(t, mean, 20, min_count=50)
    swapped = bk.draw_donors(layout, t, np.random.default_rng(8)).astype(np.float64)
    data["adapters"] = {
        "context": {"mean": mean, "mu": mean.copy(), "sigma": None},
        "swap": {"mean": swapped, "mu": swapped.copy(), "sigma": None},
    }
    adapters = {
        "context": {"slots": slots, "learned": False, "source": "esmc"},
        "swap": {
            "slots": slots,
            "learned": False,
            "source": "fixed",
            "kind": "swapped-code",
        },
    }
    result = audit.analyse(data, adapters, perm=200, seed=1)
    context, swap = result["adapters"]["context"], result["adapters"]["swap"]
    assert context["context"]["both"][0] > 0.1  # the code depends on the chain's fold
    assert max(abs(x) for x in swap["context"]["both"]) < 0.02  # the swapped code does not
    assert swap["kind"] == "swapped-code"
    assert abs(context["alignment"]["similarity_r2"] - swap["alignment"]["similarity_r2"]) < 0.02
    for s in slots:
        assert (
            abs(context["within_type_share"][s] - swap["within_type_share"][s]) < 0.01
        )
    assert bk.verdict(result, "swap", "context")["passes"]
    assert not bk.verdict(result, "context", "swap")["no_context"]["passes"]
