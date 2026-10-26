"""Environment loading for the `ml` package.

Mirrors the backend convention (`backend/app.py::_load_env_files`) but
keeps the shell environment authoritative, because `ml` is a CLI:

    shell env        explicit export  -> highest priority
    <root>/ml/.env   package-local    -> closest wins
    <root>/.env      machine-wide
    <root>/.main-env shared secrets    -> lowest priority

Every file is loaded with ``override=False``, so the first value seen
for a key wins. The order above is therefore also the precedence order.

Without this, `python -m ml train` can never reach Supabase: the
credentials live in `.env` files that the process has not read yet, so
every sync would silently defer to the offline fallback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_LOADED = False


def repo_root() -> Path:
    """Repository root (the parent of the `ml/` package)."""
    return Path(__file__).resolve().parent.parent

def workspace_root() -> Path:
    """Workspace root: the folder that holds `.main-env`, `.env`, `Makefile`."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Makefile").is_file() and (parent / "scripts").is_dir():
            return parent
    return repo_root()


def load_env_files(force: bool = False) -> None:
    """Load the `.env` chain into `os.environ`. Idempotent."""
    global _LOADED
    if _LOADED and not force:
        return

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a dependency
        log.debug("python-dotenv not installed; skipping .env loading.")
        _LOADED = True
        return

    workspace = workspace_root()
    repo = repo_root()
    # Highest priority first: with override=False the first writer wins.
    candidates = [
        (workspace / "ml" / ".env", "ml/.env"),
        (repo / "ml" / ".env", "ml/.env"),
        (workspace / ".env", ".env"),
        (repo / ".env", ".env"),
        (workspace / ".main-env", ".main-env"),
        (repo / ".main-env", ".main-env"),
    ]
    for path, label in candidates:
        if not path.exists():
            continue
        load_dotenv(path, override=False)
        log.info("Loaded %s", label)

    _LOADED = True
