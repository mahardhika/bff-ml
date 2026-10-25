"""Entry point: ``python -m ml`` or one of the ``bff-ml-*`` console scripts.

Subcommands: `train`, `eval`, `export-onnx`, `sync`, `sync-results`, `list`.
"""

from __future__ import annotations

import argparse
import logging
import sys


_DISPATCH = {
    "train": ("ml.train", "main"),
    "eval": ("ml.eval.run", "main"),
    "export-onnx": ("ml.export_onnx", "main"),
    "export_onnx": ("ml.export_onnx", "main"),
    "sync": ("ml.sync", "main"),
    "sync-results": ("ml.sync_results", "main"),
    "sync_results": ("ml.sync_results", "main"),
    "list": ("ml.sync", "list_models"),
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in _DISPATCH:
        print(f"unknown subcommand: {cmd!r}. Use one of: {sorted(_DISPATCH)}",
              file=sys.stderr)
        sys.exit(2)
    # Load .env before dispatch so `train`/`sync` can reach Supabase.
    # A help request must not touch the environment at all - it is the
    # canary for unguarded credential lookups.
    if not ({"-h", "--help"} & set(rest)):
        from ml.env import load_env_files
        load_env_files()
    mod_name, attr = _DISPATCH[cmd]
    import importlib
    mod = importlib.import_module(mod_name)
    getattr(mod, attr)(rest)


def train(argv: list[str] | None = None) -> None:
    from ml.train import main as _train
    _train(argv)


def eval(argv: list[str] | None = None) -> None:
    from ml.eval.run import main as _eval
    _eval(argv)


def export_onnx(argv: list[str] | None = None) -> None:
    from ml.export_onnx import main as _export
    _export(argv)


def sync_results(argv: list[str] | None = None) -> None:
    from ml.sync import main as _sync
    _sync(argv)


def list_models(argv: list[str] | None = None) -> None:
    from ml.sync import list_models as _list
    _list(argv)


def _print_help() -> None:
    print("Usage: bff-ml <subcommand> [args]")
    print("Subcommands:")
    print("  train        Train a model")
    print("  eval         Evaluate a model")
    print("  export-onnx  Export a checkpoint to ONNX")
    print("  sync         Sync a run to Supabase")
    print("  sync-results Replay deferred syncs to Supabase")
    print("  list         List registered models")


if __name__ == "__main__":
    main()