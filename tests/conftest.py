"""Fixtures for the trace viewer backend tests (docs/trace-format.md section 9).

``demo_root`` writes the synthetic trace of scripts/make_demo_trace.py into a
temporary directory once per session. Helpers live in trace_viewer_testlib.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from trace_viewer_testlib import load_demo_module


@pytest.fixture(scope="session")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A trace root holding the demo run (regenerated once per test session)."""
    root = tmp_path_factory.mktemp("traces_demo")
    module = load_demo_module()
    assert module.main(["--out", str(root)]) == 0
    return root
