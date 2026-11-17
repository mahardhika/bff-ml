"""Shared fixtures for the bff-ml test suite.

The suite spawns the CLI as ``python -m ml ...``. That import only
resolves to the real package when the working directory is the ml
package root: from a parent folder that also holds a ``ml`` directory,
Python binds ``ml.data`` to that folder instead of ``ml/ml/data`` and
the CLI dies with ``No module named 'ml.data.load'``.

So the session runs from the ml package root - the same root the other
ml suites already use.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:  # pragma: no cover - import style depends on the tests/ layout
    from _bff_paths import repo_root
except ImportError:  # pragma: no cover
    from tests._bff_paths import repo_root


@pytest.fixture(autouse=True, scope="session")
def _run_from_ml_package_root():
    """Run the suite from the ml package root."""
    root = Path(repo_root())
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)
