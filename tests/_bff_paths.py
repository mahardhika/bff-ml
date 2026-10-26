"""Runtime path resolution for the bff-ml test suite.

The tests were written when they lived inside ``<pkg>/tests/``; in this
repo they sit at ``<repo>/tests/``, one level up. Resolving the repo root
at runtime keeps both layouts working.
"""

from __future__ import annotations

import os
from pathlib import Path

PKG = "ml"


def repo_root() -> Path:
    """Nearest ancestor that holds pyproject.toml and the package dir."""
    override = os.getenv("BFF_REPO_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / PKG).is_dir():
            return parent
    return here.parents[1]


def ops_root() -> Path | None:
    """The bff-ops repo, if it sits next to this one (client handover pack)."""
    override = os.getenv("BFF_OPS_DIR")
    if override:
        path = Path(override).resolve()
        return path if path.is_dir() else None
    root = repo_root()
    for candidate in (root / "MD ref", root.parent, root.parent / "bff-ops", root.parent / "Camera-AI"):
        if (candidate / "MD ref").is_dir():
            return candidate
    return None


def asset(*parts: str) -> Path | None:
    """First existing path for ``parts`` under this repo or the ops repo."""
    for base in (repo_root(), ops_root()):
        if base is None:
            continue
        path = base.joinpath(*parts)
        if path.exists():
            return path
    return None
