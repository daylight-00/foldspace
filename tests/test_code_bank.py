"""The code bank of the control `type_shuffled_hybrid` (`src/foldspace/rfd4/code_bank.py`): the layout of the donors by residue type, the draw of a donor, and the verdict on the two checks of the control. Numpy only."""

from __future__ import annotations

import json

import numpy as np
import pytest
import foldspace.rfd4.code_bank as bk

pytestmark = pytest.mark.offline

ROWS = 12


def synthetic(
    seed: int = 0, own: tuple[int, ...] = (1, 2, 3), few: dict[int, int] | None = None
):
    """Codes of 18 channels of types that have 700 donors each, and of a few types that have too few to have donors of their own."""
    rng = np.random.default_rng(seed)
    types = np.concatenate(
        [np.full(700, t) for t in own]
        + [np.full(n, t) for t, n in (few or {7: 10}).items()]
    )
    rng.shuffle(types)
    centres = rng.normal(size=(ROWS, 18)) * 3.0
    codes = centres[types] + rng.normal(size=(len(types), 18)) * 0.5
    return types, codes


def test_the_bank_is_sorted_by_type_and_each_type_knows_where_its_donors_are() -> None:
    types, codes = synthetic()
    layout = bk.bank_layout(types, codes, ROWS, min_count=500)
    assert np.array_equal(layout["bank_types"], np.sort(types))
    for t in (1, 2, 3):
        start, count = layout["bank_start"][t], layout["bank_count"][t]
        assert count == 700 and set(layout["bank_types"][start : start + count]) == {t}
        assert np.allclose(layout["mean"][t], codes[types == t].mean(0), atol=1e-5)
    # the rows of the bank are the residues' codes, once each
    assert layout["code_bank"].shape == codes.shape
    assert np.allclose(
        np.sort(layout["code_bank"][:, 0]), np.sort(codes[:, 0]), atol=1e-5
    )


def test_a_type_with_too_few_residues_has_all_of_them_for_donors_and_their_pooled_mean() -> (
    None
):
    types, codes = synthetic()
    layout = bk.bank_layout(types, codes, ROWS, min_count=500)
    total = len(types)
    for t in (0, 4, 7, 11):  # 7 has 10 residues, the others none
        assert layout["pooled"][t]
        assert layout["bank_start"][t] == 0 and layout["bank_count"][t] == total
        assert np.allclose(layout["mean"][t], codes.mean(0), atol=1e-5)
    assert not layout["pooled"][[1, 2, 3]].any()


def test_a_draw_is_a_donor_of_the_residues_type_and_every_donor_is_drawn() -> None:
    types, codes = synthetic()
    layout = bk.bank_layout(types, codes, ROWS, min_count=500)
    rng = np.random.default_rng(1)
    wanted = np.full(20000, 2)
    drawn = bk.draw_donors(layout, wanted, rng)
    start, count = layout["bank_start"][2], layout["bank_count"][2]
    donors = layout["code_bank"][start : start + count]
    # each draw is a row of the type's donors (the bank's rows are distinct)
    index = {tuple(np.round(row, 5)): i for i, row in enumerate(donors)}
    picked = [index[tuple(np.round(row, 5))] for row in drawn]
    assert (
        len(set(picked)) > 0.99 * count
    )  # 20,000 draws of 700 donors reach all but a handful
    counts = np.bincount(picked, minlength=count)
    assert (
        counts.min() >= 1 and counts.max() < 3 * counts.mean()
    )  # no donor is favoured
    # a type without donors of its own draws from every type
    pooled = bk.draw_donors(layout, np.full(20000, 11), np.random.default_rng(2))
    nearest = np.argmin(
        ((pooled[:, None, :] - layout["mean"][[1, 2, 3]][None]) ** 2).sum(-1), axis=1
    )
    assert set(nearest) == {0, 1, 2}


def test_the_draws_are_reproducible_and_independent_between_residues() -> None:
    types, codes = synthetic()
    layout = bk.bank_layout(types, codes, ROWS, min_count=500)
    a = bk.draw_donors(layout, np.full(500, 3), np.random.default_rng(5))
    b = bk.draw_donors(layout, np.full(500, 3), np.random.default_rng(5))
    c = bk.draw_donors(layout, np.full(500, 3), np.random.default_rng(6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert (
        len({tuple(row) for row in a}) > 300
    )  # 500 draws of 700 donors: most are different residues


def test_a_bank_that_does_not_make_sense_is_refused() -> None:
    types, codes = synthetic()
    with pytest.raises(ValueError, match="one row a residue"):
        bk.bank_layout(types[:-1], codes, ROWS)
    with pytest.raises(ValueError, match="the table has"):
        bk.bank_layout(types + 100, codes, ROWS)
    with pytest.raises(ValueError, match="fewer than two"):
        bk.bank_layout(types[:1], codes[:1], ROWS)


def report(
    control_context=(0.002, -0.001),
    control_alignment=0.49,
    share=(0.155, 0.125),
    ref_share=(0.155, 0.125),
) -> dict:
    def entry(context, r1, within):
        return {
            "slots": [0, 7],
            "alignment": {"similarity_r2": r1},
            "context": {"both": list(context)},
            "within_type_share": [within[0]] + [0.1] * 6 + [within[1]] + [0.1] * 10,
        }

    return {
        "adapters": {
            "contextual_hybrid": entry((0.117, 0.098), 0.486, ref_share),
            "type_shuffled_hybrid": entry(control_context, control_alignment, share),
        }
    }


def test_the_control_passes_when_its_code_has_no_context_and_keeps_the_type_geometry_and_the_variance() -> (
    None
):
    result = bk.verdict(report(), "type_shuffled_hybrid", "contextual_hybrid")
    assert result["passes"] and result["no_context"]["passes"] and result["type_structure_kept"]["passes"]
    assert result["no_context"]["reference_context_r2_both"] == [0.117, 0.098]


@pytest.mark.parametrize(
    "change, failing",
    [
        ({"control_context": (0.05, 0.001)}, "no_context"),
        ({"control_context": (0.001, 0.02)}, "no_context"),
        ({"control_alignment": 0.53}, "type_structure_kept"),
        ({"share": (0.18, 0.125)}, "type_structure_kept"),
    ],
)
def test_the_control_fails_each_check_on_its_own(change: dict, failing: str) -> None:
    result = bk.verdict(report(**change), "type_shuffled_hybrid", "contextual_hybrid")
    assert not result["passes"] and not result[failing]["passes"]
    other = "type_structure_kept" if failing == "no_context" else "no_context"
    assert result[other]["passes"]


def test_a_held_out_r2_below_zero_is_not_context() -> None:
    """A code with no context has a held-out R² a little below zero (the ridge overfits): that is the control passing, not failing."""
    assert bk.verdict(report(control_context=(-0.03, -0.02)), "type_shuffled_hybrid", "contextual_hybrid")["no_context"][
        "passes"
    ]


def test_a_control_without_variation_within_types_has_no_context_reading_to_check() -> (
    None
):
    broken = report()
    del broken["adapters"]["type_shuffled_hybrid"]["context"]
    with pytest.raises(ValueError, match="no context reading"):
        bk.verdict(broken, "type_shuffled_hybrid", "contextual_hybrid")


def test_the_check_command_returns_the_verdict_as_its_exit_status(
    tmp_path, capsys
) -> None:
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(report()))
    bad.write_text(json.dumps(report(control_context=(0.2, 0.0))))
    assert bk.main(["--check", str(good)]) == 0
    assert json.loads(capsys.readouterr().out)["passes"] is True
    assert bk.main(["--check", str(bad)]) == 1
    assert json.loads(capsys.readouterr().out)["no_context"]["passes"] is False
