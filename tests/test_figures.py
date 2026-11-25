"""The results figure is drawn from the committed records; a redraw has to equal the committed file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_results_figure_is_what_the_records_give(tmp_path: Path) -> None:
    out = tmp_path / "results.svg"
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_figures.py"), str(out)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert out.read_text() == (ROOT / "docs" / "figures" / "results.svg").read_text()
