"""Make `actdim` importable when the package has not been installed.

`python -m pytest tests/` from this directory should work on a fresh clone, with no
`pip install -e .` first. Pytest puts the test file's directory on the path, not the
project root, so the root goes on here.

Tests also run against a scratch output tree rather than the real `runs/`, so a test can
never overwrite a result that took three hours to produce.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def run_root(tmp_path):
    """A throwaway `runs/` directory for a test that needs to write one."""
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def context(tmp_path):
    """A `Context` writing into a temporary directory."""
    from actdim.runtime.context import build

    return build("test.experiment", device="cpu", jobs=1, seed=0, root=tmp_path / "runs")


@pytest.fixture(scope="session")
def archive():
    """The archived tree, for tests that check the port against the published results."""
    path = ROOT.parent / "archived_code"
    if not path.exists():
        pytest.skip("../archived_code is not present")
    return path
