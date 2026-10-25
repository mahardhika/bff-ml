"""Push deferred training results to Supabase.

`ml/sync.py` writes a record to ``data/experiments/<run_name>.json`` when
a sync is skipped (offline mode) or fails (Supabase unreachable). This
module replays those records once connectivity is back:

    python -m ml.sync_results --run-name=yolox-s-finetune-20261019-1042
    python -m ml.sync_results --list
    python -m ml.sync_results --run-name=... --all

The run directory is resolved from the deferred record's ``run_dir``
field, falling back to ``<runs-dir>/<run_name>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from ml.sync import (
    build_experiment_record,
    experiments_dir,
    sync_experiment,
)

log = logging.getLogger(__name__)

#: Where training runs are written by ``ml/train.py``.
RUNS_DIR_ENV = "BFF_RUNS_DIR"


def runs_dir() -> Path:
    """Directory holding training run folders."""
    return Path(os.getenv(RUNS_DIR_ENV, "./runs"))


def list_deferred() -> list[dict]:
    """Return every deferred experiment record, oldest first."""
    d = experiments_dir()
    if not d.exists():
        return []
    out: list[dict] = []
    for f in sorted(d.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Unreadable deferred record %s: %s", f, exc)
            continue
        rec["_file"] = str(f)
        out.append(rec)
    return out


def resolve_run_dir(record: dict, run_name: str) -> Path | None:
    """Find the run directory for a deferred record.

    Prefers the recorded ``run_dir`` (it survives a moved runs folder),
    then falls back to ``<runs-dir>/<run_name>``.
    """
    recorded = record.get("run_dir")
    if recorded:
        p = Path(recorded)
        if p.exists():
            return p
    candidate = runs_dir() / run_name
    if candidate.exists():
        return candidate
    return None


def _sync_one(record: dict, run_name: str) -> bool:
    """Replay one deferred record. Returns True when it reached Supabase."""
    run_dir = resolve_run_dir(record, run_name)
    if run_dir is None:
        log.error(
            "Run dir for %r not found (tried %r and %r). "
            "Artifacts may have been moved or deleted.",
            run_name, record.get("run_dir"), str(runs_dir() / run_name),
        )
        return False

    ok = sync_experiment(
        run_dir=run_dir,
        model_name=record.get("model_name", "model"),
        version=record.get("version", "0.0.0"),
        kind=record.get("kind", "detector"),
        run_name=run_name,
        dataset_ids=record.get("dataset_ids") or [],
    )
    return ok


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="ml.sync_results",
        description="Replay deferred training-result syncs to Supabase.",
    )
    p.add_argument("--run-name", default=None,
                   help="run to sync (the deferred record's run_name)")
    p.add_argument("--list", action="store_true",
                   help="list deferred records and exit")
    p.add_argument("--all", action="store_true",
                   help="sync every deferred record")
    p.add_argument("--runs-dir", default=None,
                   help=f"training runs dir (default: {runs_dir()})")
    args = p.parse_args(argv)

    from ml.env import load_env_files
    load_env_files()

    if args.runs_dir:
        os.environ[RUNS_DIR_ENV] = args.runs_dir

    if args.list:
        records = list_deferred()
        if not records:
            print("(no deferred syncs)")
            return
        print(f"Deferred syncs in {experiments_dir()}:")
        for r in records:
            print(f"  {r.get('run_name', '?'):40s} "
                  f"{r.get('model_name', '?')} v{r.get('version', '?')} "
                  f"deferred_at={r.get('deferred_at', '?')}")
        return

    if not args.all and not args.run_name:
        p.error("one of --run-name, --all or --list is required")

    if args.all:
        records = list_deferred()
        if not records:
            print("(no deferred syncs)")
            return
        ok = 0
        for r in records:
            name = r.get("run_name")
            if not name:
                continue
            if _sync_one(r, name):
                ok += 1
        print(f"Synced {ok}/{len(records)} deferred run(s).")
        return

    # Single run. Accept either an existing deferred record or a bare
    # run directory that was never deferred.
    record = next(
        (r for r in list_deferred() if r.get("run_name") == args.run_name),
        None,
    )
    if record is None:
        candidate = runs_dir() / args.run_name
        if not candidate.exists():
            print(f"No deferred record or run dir named {args.run_name!r}.",
                  file=sys.stderr)
            raise SystemExit(1)
        record = build_experiment_record(
            candidate, "model", "0.0.0", "detector", args.run_name, [],
        )

    if _sync_one(record, args.run_name):
        print(f"Synced {args.run_name} to Supabase.")
    else:
        print(f"Still deferred: {args.run_name} (record kept at "
              f"{experiments_dir() / (args.run_name + '.json')}).")
        raise SystemExit(2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
