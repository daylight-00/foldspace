"""Marker policy.

`pyproject` declares `offline`, `gpu`, `weights` and `slow`. A `gpu` test skips when there is no CUDA device: without one,
`open_substrate(device=None)` falls back to CPU and loads 25 GB into RAM, which looks like a slow test and not like a test
running somewhere it was never meant to. The other markers are selectors, not preconditions: a `weights` test reads a
checkpoint that is either present or an honest failure, and `slow` says only what it costs.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    del config  # the policy does not depend on invocation
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001 -- no torch is also no CUDA
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="needs a CUDA device; see tests/conftest.py")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
