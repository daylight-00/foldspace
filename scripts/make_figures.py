"""Draw docs/figures/results.svg from the committed records.

    python scripts/make_figures.py [OUT.svg]        # from the root of the repository; standard library only

Every number in the figure is read from a record under experiments/, so the figure cannot drift from them; a test redraws it and
compares it with the committed file. Each panel has its own colour: A purple, B green, C yellow (purple and green after the RFD4-Proteina site); grey marks references.
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
W = 880


def load(path: str):
    return json.loads((EXP / path).read_text())


def data() -> dict:
    adaptation = "rfd4/latent_adaptation"
    replacement = load("rfd4/latent_replacement/latent_statistics.json")["latents"]
    generation = load("rfd4/latent_replacement/generation_1600.json")
    hybrid = load("rfd4/hybrid_latents/unconditional_generation.json")["usable_of_192_at_scale_1"]
    rmsd = lambda name: load(f"{adaptation}/decoder_adaptation/{name}/summary.json")["end"]["complete"]["per_residue_rmsd"]  # noqa: E731
    reading = lambda name: replacement[name]["recorded_reading"]["per_residue_rmsd"]  # noqa: E731
    mean = lambda a, b: (a + b) / 2  # noqa: E731
    reconstruction = [
        ("Native (release encoder)", load(f"{adaptation}/reading_adapters/native_encoder/summary.json")["dev"]["complete"]["per_residue_rmsd"], "ctrl"),
        ("ESMC 32 channels", mean(reading("esmc_only_seed1"), reading("esmc_only_seed2")), "dec"),
        ("Type-only 32 channels", mean(reading("type_only_seed1"), reading("type_only_seed2")), "dec"),
        ("ESMC 24 + 8 residual", rmsd("esmc_residual8"), "dec"),
        ("Type-only 24 + 8 residual", rmsd("type_only_residual8"), "dec"),
    ]
    seeds = generation["seeds"]
    usable = [
        ("Native continuation", [hybrid["native_continuation"]]),
        ("Contextual hybrid", [hybrid["contextual_hybrid"], hybrid["contextual_hybrid_seed2"]]),
        ("Type-shuffled hybrid", [hybrid["type_shuffled_hybrid"], hybrid["type_shuffled_hybrid_seed2"]]),
        ("ESMC 32 channels", [seeds["s1"]["esmc_only_unconditional_usable_at_1"], seeds["s2"]["esmc_only_unconditional_usable_at_1"]]),
        ("Type-only 32 channels", [seeds["s1"]["type_only_unconditional_usable_at_1"], seeds["s2"]["type_only_unconditional_usable_at_1"]]),
    ]
    base = "esmfold2/latent_generation/results"
    summary = lambda path: load(f"{base}/{path}/summary.json")  # noqa: E731
    sets = []
    reference = summary("pilot_seed0/standardized/reencoded_fold")
    sets.append(("Natural reference chains", round(reference["reference_success_share"] * reference["references"]), None, reference["references"], "ctrl"))
    for label, path in (
        ("Standardized", "pilot_seed0/standardized/reencoded_fold"),
        ("Whitened", "pilot_seed0/whitened/reencoded_fold"),
        ("Whitened + sequence loss", "pilot_seed0/whitened_with_sequence_loss/reencoded_fold"),
        ("Standardized, second seed", "pilot_seed1/standardized/reencoded_fold"),
        ("ProtGPT2", "controls/protgpt2"),
    ):
        s = summary(path)
        sets.append((label, s["success"]["count"], s["credited"]["count"], s["samples"], "gen"))
    return {"reconstruction": reconstruction, "usable": usable, "sets": sets}


# ------------------------------------------------------------------------------------------------------------ drawing
STYLE = """
text { font-family: "Helvetica Neue", Helvetica, Arial, "Liberation Sans", sans-serif; fill: #262626; font-size: 13px; }
.m { fill: #5B6068; font-size: 12px; } .n { font-size: 12px; font-variant-numeric: tabular-nums; }
.h { font-size: 14px; font-weight: 700; }
.grid { stroke: #E6E8EC; stroke-width: 1; }
.bar { stroke-width: 1; }
.bar.dec { fill: #BFA5D8; stroke: #8D52AD; } .bar.grn { fill: #8AD3C0; stroke: #2E9C88; } .bar.grn2 { fill: #CBEBE2; stroke: #4DB6A2; }
.bar.gen { fill: #F2CD6A; stroke: #C99A25; } .bar.ctrl { fill: #C3C8D0; stroke: #8E95A1; }
.hollow { fill: #FDF5DA; stroke: #C99A25; stroke-width: 1.5; } .hollow.ctrl { fill: #EEF0F3; stroke: #8E95A1; }
.zero { stroke-width: 3; stroke-linecap: round; } .zero.dec { stroke: #8D52AD; } .zero.grn, .zero.grn2 { stroke: #2E9C88; } .zero.gen { stroke: #C99A25; } .zero.ctrl { stroke: #8E95A1; }
"""


def t(x, y, s, cls="", anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'


def bar(x0, yc, width, value, cls, height=14, kind="bar"):
    if value <= 0:
        return f'<line x1="{x0}" y1="{yc - height / 2:.1f}" x2="{x0}" y2="{yc + height / 2:.1f}" class="zero {cls}"/>'
    return f'<rect x="{x0}" y="{yc - height / 2:.1f}" width="{width:.1f}" height="{height}" class="{kind} {cls}"/>'


def panel_axis(y, rows_height, hi, ticks, fmt, title, x0=230, w=470):
    out = []
    for tick in ticks:
        x = x0 + tick / hi * w
        out.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + rows_height}" class="grid"/>')
        out.append(t(x, y + rows_height + 15, fmt(tick), "m n", "middle"))
    out.append(t(x0 + w / 2, y + rows_height + 34, title, "m", "middle"))
    return "".join(out)


def figure(d: dict) -> str:
    x0, w = 230, 470
    out, y = [], 0
    # A: reconstruction
    rows = d["reconstruction"]
    out.append(t(10, y + 20, "A. RFD4 latent reconstruction, given the true Cα trace", "h"))
    top, rowh = y + 34, 30
    out.append(panel_axis(top, len(rows) * rowh, 1.4, [0, 0.35, 0.7, 1.05, 1.4], lambda v: f"{v:g}", "per-residue RMSD of complete residues (Å, lower is better)"))
    for i, (label, value, cls) in enumerate(rows):
        yc = top + i * rowh + rowh / 2
        out.append(t(200, yc + 4, label, "", "end"))
        out.append(bar(x0, yc, value / 1.4 * w, value, cls))
        out.append(t(x0 + value / 1.4 * w + 8, yc + 4, f"{value:.3f}", "n"))
    y = top + len(rows) * rowh + 62
    # B: unconditional generation
    rows = d["usable"]
    out.append(t(10, y + 20, "B. RFD4 unconditional generation", "h"))
    top, rowh = y + 34, 40
    out.append(panel_axis(top, len(rows) * rowh, 192, [0, 48, 96, 144, 192], lambda v: f"{v:g}", "usable samples of 192 (first and second training seed)"))
    for i, (label, values) in enumerate(rows):
        yc = top + i * rowh + rowh / 2
        out.append(t(200, yc + 4, label, "", "end"))
        for j, value in enumerate(values):
            yy = yc - 7 + 14 * j if len(values) == 2 else yc
            out.append(bar(x0, yy, value / 192 * w, value, "grn" if j == 0 else "grn2", 11))
            out.append(t(x0 + value / 192 * w + 8, yy + 4, str(value), "n"))
    ly = top + len(rows) * rowh + 52
    out.append(f'<rect x="{x0}" y="{ly - 9}" width="11" height="11" class="bar grn"/>' + t(x0 + 17, ly, "first seed", "m"))
    out.append(f'<rect x="{x0 + 100}" y="{ly - 9}" width="11" height="11" class="bar grn2"/>' + t(x0 + 117, ly, "second seed", "m"))
    out.append(t(x0 + 220, ly, "native: one seed; ESMC and type-only at 1,600 updates", "m"))
    y = ly + 34
    # C: ESMFold2
    rows = d["sets"]
    out.append(t(10, y + 20, "C. ESMFold2 latent generation, re-encoded route", "h"))
    top, rowh = y + 34, 30
    out.append(panel_axis(top, len(rows) * rowh, 1.0, [0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:g}", "share of the samples of a set (48 each; 24 natural chains)"))
    for i, (label, success, credited, n, cls) in enumerate(rows):
        yc = top + i * rowh + rowh / 2
        out.append(t(200, yc + 4, label, "", "end"))
        if credited is None:
            out.append(bar(x0, yc, success / n * w, success, "ctrl", 14, "hollow"))
            out.append(t(x0 + success / n * w + 8, yc + 4, f"{success} of {n} (success rule only)", "n"))
        else:
            out.append(bar(x0, yc - 5, success / n * w, success, "gen", 9, "hollow"))
            out.append(bar(x0, yc + 5, credited / n * w, credited, "gen", 9))
            out.append(t(x0 + max(success, credited) / n * w + 8, yc + 4, f"{success} success, {credited} credited of {n}", "n"))
    ly = top + len(rows) * rowh + 52
    out.append(f'<rect x="{x0}" y="{ly - 9}" width="11" height="11" class="hollow"/>' + t(x0 + 17, ly, "success", "m"))
    out.append(f'<rect x="{x0 + 100}" y="{ly - 9}" width="11" height="11" class="bar gen"/>' + t(x0 + 117, ly, "credited", "m"))
    height = ly + 22
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height:.0f}" width="{W}" height="{height:.0f}" role="img" '
        'aria-label="Key results: RFD4 reconstruction, RFD4 unconditional generation, ESMFold2 latent generation">'
        f"<style>{STYLE}</style><rect width=\"{W}\" height=\"{height:.0f}\" fill=\"#FFFFFF\"/>"
    )
    return head + "".join(out) + "</svg>\n"


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "figures" / "results.svg"
    out.write_text(figure(data()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
