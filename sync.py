"""Push training results to Supabase.

- Upload ONNX + MODEL_CARD to S3 bucket `models-public/<name>/<version>/`.
- Upsert `bff_models` row with the metrics + path.
- Insert `bff_experiments` row with the hyperparams + dataset_ids.

Skip conditions:
- `DATABASE_MODE=offline` -> keep run on disk, no network.
- `--no-sync` flag.
- Supabase unreachable -> graceful degradation + warning.
- Required env vars unset (e.g. self-hosted with empty URL) -> graceful skip.

When a sync is skipped or fails, the experiment record is written to
``data/experiments/<run_name>.json`` so it can be pushed later with
``python -m ml.sync_results --run-name=...``.

Usage:
    python -m ml sync --run-dir=./runs/<run_name> --name=bucket-detector --version=0.1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

#: Where deferred experiment records are parked when a sync is skipped.
#: Matches the `EXPERIMENTS_DIR` key in the project `.env`; the
#: `BFF_`-prefixed alias is accepted so scripts can override it.
EXPERIMENTS_DIR_ENV = "EXPERIMENTS_DIR"
EXPERIMENTS_DIR_ENV_ALIAS = "BFF_EXPERIMENTS_DIR"


def experiments_dir() -> Path:
    """Directory holding deferred (not-yet-synced) experiment records."""
    return Path(
        os.getenv(EXPERIMENTS_DIR_ENV)
        or os.getenv(EXPERIMENTS_DIR_ENV_ALIAS)
        or "./data/experiments"
    )

# ---------------------------------------------------------------------------
# Database mode helpers.
# ---------------------------------------------------------------------------
# Recognised values for DATABASE_MODE:
#   - "offline"      : skip Supabase entirely, use local artifacts only.
#   - "public"       : talk to the public Supabase project (default for prod).
#   - "self_hosted"  : talk to a self-hosted Supabase (DB-2).
# Anything else (or unset) collapses to "offline" so the package degrades
# gracefully when no .env has been provisioned.
_VALID_MODES = ("offline", "public", "self_hosted")


def db_mode() -> str:
    """Return the active DATABASE_MODE (normalized to one of ``_VALID_MODES``).

    Unset / empty / unknown values collapse to ``"offline"`` so the
    pipeline never crashes on a fresh checkout with no .env.
    """
    raw = os.getenv("DATABASE_MODE", "offline").strip().lower()
    return raw if raw in _VALID_MODES else "offline"


def _is_offline() -> bool:
    return db_mode() == "offline"


def _supabase_url() -> str | None:
    """Return the configured Supabase project URL for the current mode, or ``None``."""
    mode = db_mode()
    if mode == "self_hosted":
        return os.getenv("SELF_HOSTED_SUPABASE_URL") or None
    if mode == "public":
        return os.getenv("SUPABASE_URL") or None
    return None


def _supabase_key() -> str | None:
    """Return the configured Supabase service-role key, or ``None``."""
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY") or None


# ---------------------------------------------------------------------------
# Local runs walker (offline mode list backend).
# ---------------------------------------------------------------------------
_CARD_HEADER_RE = re.compile(r"^#\s+(?P<name>\S+)\s+v(?P<version>\S+)\s*$", re.MULTILINE)


def _parse_model_card(card_path: Path) -> tuple[str | None, str | None]:
    """Pull ``name`` and ``version`` from a MODEL_CARD.md title line.

    Returns ``(None, None)`` if the title line is missing or unreadable.
    """
    try:
        text = card_path.read_text(encoding="utf-8")
    except OSError:
        return None, None
    m = _CARD_HEADER_RE.search(text)
    if not m:
        return None, None
    return m.group("name"), m.group("version")


def list_local_runs(runs_dir: Path) -> list[dict]:
    """Walk ``runs_dir`` and return one row per subdir that looks like a run.

    Each row has the keys the remote listing exposes (``name``, ``version``,
    ``kind``, ``is_active``, ``metrics``, ``updated_at``) plus ``path`` /
    ``size_bytes`` so the CLI can render a useful local-mode table.
    """
    rows: list[dict] = []
    if not runs_dir.exists():
        return rows
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        card = sub / "MODEL_CARD.md"
        name, version = _parse_model_card(card) if card.exists() else (None, None)
        # Fall back to the run-dir name pattern "<name>-<version>-<ts>".
        if name is None or version is None:
            m = re.match(r"^(?P<name>[^-]+)-(?P<version>[^-]+)-\d+$", sub.name)
            if m:
                name = name or m.group("name")
                version = version or m.group("version")
            else:
                # Skip dirs that don't look like runs.
                continue
        onnx = next(sub.glob("*.onnx"), None)
        size = onnx.stat().st_size if onnx else 0
        rows.append({
            "name": name,
            "version": version,
            "kind": "detector",  # local listing can't tell; classifier runs are uncommon.
            "is_active": True,
            "updated_at": datetime.fromtimestamp(sub.stat().st_mtime).isoformat(timespec="seconds"),
            "path": str(onnx) if onnx else "",
            "size_bytes": size,
            "source": "local",
        })
    # Newest first to match the remote listing.
    rows.sort(key=lambda r: r["updated_at"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Subcommand: `python -m ml sync --run-dir=... --name=... --version=...`
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sync a training run to Supabase.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--name", required=True, help="model name")
    p.add_argument("--version", required=True, help="model version")
    p.add_argument("--kind", default="detector", choices=["detector", "classifier"])
    p.add_argument("--run-name", default=None,
                   help="name in bff_experiments (default: run-dir basename)")
    p.add_argument("--dataset-ids", default="",
                   help="comma-separated bff_datasets IDs to attach")
    args = p.parse_args(argv)

    sync_experiment(
        run_dir=Path(args.run_dir),
        model_name=args.name,
        version=args.version,
        kind=args.kind,
        run_name=args.run_name,
        dataset_ids=[int(x) for x in args.dataset_ids.split(",") if x.strip()],
    )


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning ``{}`` when missing or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_experiment_record(run_dir: Path, model_name: str, version: str,
                            kind: str, run_name: str | None,
                            dataset_ids: list[int] | None) -> dict:
    """Assemble the row payload for ``bff_experiments`` from a run dir."""
    hyperparams = _read_json(run_dir / "hparams.json")
    metrics = _read_json(run_dir / "metrics.json")
    canonical = "detector.onnx" if kind == "detector" else "classifier.onnx"
    onnx = sorted(run_dir.glob("*.onnx"))
    first_onnx = canonical if (run_dir / canonical).exists() else (
        onnx[0].name if onnx else canonical
    )
    from bff_core.storage import model_object_key, models_bucket

    bucket = models_bucket()
    return {
        "model_name": model_name,
        "version": version,
        "kind": kind,
        "run_name": run_name or run_dir.name,
        "run_dir": str(run_dir),
        "hyperparams": hyperparams,
        "metrics": metrics,
        "split": os.getenv("SPLIT", "train"),
        "dataset_ids": dataset_ids or [],
        "path": f"{bucket}/{model_object_key(model_name, version, first_onnx)}",
        "model_card_url": f"{bucket}/{model_object_key(model_name, version, 'MODEL_CARD.md')}",
        "started_at": hyperparams.get("started_at"),
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }


def save_deferred(record: dict, run_name: str | None = None) -> Path:
    """Persist an experiment record locally so it can be synced later.

    This is the offline / unreachable fallback described in
    `system-map-arsitektur.MD` §5.22.3. The record lands in
    ``data/experiments/<run_name>.json`` and is picked up later by
    ``python -m ml.sync_results --run-name=<run_name>``.
    """
    target_dir = experiments_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or record.get("run_name") or "unknown"
    target = target_dir / f"{name}.json"
    record = dict(record)
    record["deferred_at"] = datetime.utcnow().isoformat() + "Z"
    target.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log.warning("Deferred sync saved locally: %s", target)
    return target


def clear_deferred(run_name: str) -> None:
    """Remove a deferred record once it has been synced successfully."""
    target = experiments_dir() / f"{run_name}.json"
    try:
        target.unlink()
        log.info("Cleared deferred sync record: %s", target)
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - best effort cleanup
        log.warning("Could not remove deferred record %s: %s", target, exc)


def sync_experiment(run_dir: Path, model_name: str, version: str,
                    kind: str = "detector", run_name: str | None = None,
                    dataset_ids: list[int] | None = None) -> bool:
    """Push artifacts + rows to Supabase.

    Returns ``True`` when the run reached Supabase, ``False`` when it was
    deferred to a local record instead. Never raises for network problems
    - a failed sync must not destroy the training artifacts.
    """
    record = build_experiment_record(
        run_dir, model_name, version, kind, run_name, dataset_ids
    )
    effective_run_name = record["run_name"]

    if _is_offline():
        log.info("Offline mode: skipping Supabase sync, artifacts on disk at %s", run_dir)
        save_deferred(record, effective_run_name)
        return False

    url = _supabase_url()
    key = _supabase_key()
    if not url or not key:
        # Self-hosted mode with an empty SELF_HOSTED_SUPABASE_URL, or a fresh
        # checkout that only set DATABASE_MODE=public but never exported creds.
        log.warning(
            "Supabase creds missing for DATABASE_MODE=%s "
            "(url=%r, key=%r); skipping sync, artifacts on disk at %s",
            db_mode(), bool(url), bool(key), run_dir,
        )
        save_deferred(record, effective_run_name)
        return False

    try:
        from supabase import create_client  # type: ignore
        from bff_core.storage import (
            StorageClient, model_object_key, models_bucket, models_prefix,
        )
    except ImportError as exc:
        log.error("Required deps missing: %s", exc)
        save_deferred(record, effective_run_name)
        return False

    try:
        sb = create_client(url, key)
        storage = StorageClient.from_env()
        if storage is None:
            raise RuntimeError("storage is not configured")
        bucket = models_bucket()
        prefix = models_prefix(model_name, version)
        canonical = "detector.onnx" if kind == "detector" else "classifier.onnx"

        # 1. Upload artifacts under the canonical names the backend loads.
        files_uploaded = []
        onnx_files = list(run_dir.glob("*.onnx"))
        canonical_src = run_dir / canonical
        if not canonical_src.exists() and onnx_files:
            # Older runs named the file after the model; copy into place.
            canonical_src.write_bytes(onnx_files[0].read_bytes())
        for p in run_dir.glob("*.onnx"):
            dest_name = canonical if p.name == canonical_src.name or p == canonical_src else p.name
            if p.suffix == ".onnx" and dest_name != canonical and p.stem != Path(canonical).stem:
                dest_name = canonical
            storage.upload_file(p, bucket, model_object_key(model_name, version, dest_name))
            files_uploaded.append(dest_name)
            sidecar = Path(str(p) + ".data")
            if sidecar.exists():
                storage.upload_file(
                    sidecar, bucket,
                    model_object_key(model_name, version, sidecar.name),
                )
        if (run_dir / "MODEL_CARD.md").exists():
            storage.upload_file(
                run_dir / "MODEL_CARD.md", bucket,
                model_object_key(model_name, version, "MODEL_CARD.md"),
            )
        if (run_dir / "train.log").exists():
            storage.upload_file(
                run_dir / "train.log", bucket,
                model_object_key(model_name, version, "train.log"),
            )
        log.info("Uploaded %d files to %s/%s/", len(files_uploaded), bucket, prefix)

        # 2. Upsert bff_models.
        metrics = record["metrics"]
        sb.table("bff_models").upsert({
            "name": model_name,
            "version": version,
            "kind": kind,
            "path": record["path"],
            "metrics": metrics,
            "model_card_url": record["model_card_url"],
            "is_active": True,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }, on_conflict="name,version").execute()

        # 3. Read back the model id (so we can link the experiment).
        model_row = sb.table("bff_models").select("id").eq("name", model_name).eq(
            "version", version).limit(1).execute()
        model_id = (model_row.data or [{}])[0].get("id")

        # 4. Insert bff_experiments.
        sb.table("bff_experiments").insert({
            "model_id": model_id,
            "run_name": effective_run_name,
            "hyperparams": record["hyperparams"],
            "metrics": metrics,
            "split": record["split"],
            "dataset_ids": record["dataset_ids"],
            "started_at": record["started_at"],
            "finished_at": record["finished_at"],
        }).execute()
    except Exception as exc:  # noqa: BLE001
        # Network outage, DNS failure, RLS rejection, S3 error... none of
        # these should cost us the trained artifacts.
        log.warning("Supabase sync failed (%s) - run preserved on disk at %s",
                    exc, run_dir)
        save_deferred(record, effective_run_name)
        return False

    log.info("Synced %s v%s to Supabase (model_id=%s)", model_name, version, model_id)
    clear_deferred(effective_run_name)
    return True


# ---------------------------------------------------------------------------
# Subcommand: `python -m ml list [--runs-dir=./runs]`
# ---------------------------------------------------------------------------

def list_models(argv: list[str] | None = None) -> None:
    """Print the current ``bff_models`` rows, or the local ``runs/`` directory.

    Resolution order:
      1. If DATABASE_MODE=offline OR no Supabase creds -> walk ``runs/``.
      2. Otherwise, hit Supabase. If that raises, log a warning and fall
         back to the local listing so the command always returns something.
    """
    p = argparse.ArgumentParser(
        description="List registered models (Supabase) or local runs (offline).",
    )
    p.add_argument("--runs-dir", default="./runs",
                   help="local runs dir to walk in offline mode (default: ./runs)")
    args = p.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    remote_ok = not _is_offline() and bool(_supabase_url()) and bool(_supabase_key())

    if remote_ok:
        rows = _fetch_remote_models()
        if rows is None:
            log.warning("Remote listing failed; falling back to local runs at %s", runs_dir)
            rows = list_local_runs(runs_dir)
        else:
            print(f"Remote bff_models (DATABASE_MODE={db_mode()})")
    else:
        if not _is_offline():
            log.warning(
                "DATABASE_MODE=%s but Supabase creds missing "
                "(url=%s key=%s); using local listing.",
                db_mode(), bool(_supabase_url()), bool(_supabase_key()),
            )
        print(f"Local MLflow-style runs (offline mode) at {runs_dir}")
        rows = list_local_runs(runs_dir)

    _print_model_rows(rows)


def _fetch_remote_models() -> list[dict] | None:
    """Try to fetch the bff_models table. Return ``None`` on any failure."""
    try:
        from supabase import create_client  # type: ignore
    except ImportError as exc:
        log.warning("supabase SDK not installed (%s); using local listing.", exc)
        return None
    url, key = _supabase_url(), _supabase_key()
    if not url or not key:
        return None
    try:
        sb = create_client(url, key)
        res = sb.table("bff_models").select(
            "name, version, kind, is_active, metrics, updated_at"
        ).order("updated_at", desc=True).execute()
        return res.data or []
    except Exception as exc:  # noqa: BLE001
        log.warning("Supabase list_models failed: %s", exc)
        return None


def _print_model_rows(rows: list[dict]) -> None:
    if not rows:
        print("(no models registered)")
        return
    is_local = any(r.get("source") == "local" for r in rows)
    for r in rows:
        active = "ACTIVE" if r.get("is_active") else "      "
        kind = (r.get("kind") or "detector")[:11]
        ts = r.get("updated_at", "")
        if is_local and r.get("size_bytes") is not None:
            size_kb = r["size_bytes"] / 1024
            print(f"{active} {r['name']:30s} v{str(r['version'])[:8]:8s} "
                  f"{kind:11s} {ts}  ({size_kb:.1f} KB  {r.get('path','')})")
        else:
            print(f"{active} {r['name']:30s} v{str(r['version'])[:8]:8s} "
                  f"{kind:11s} {ts}")


if __name__ == "__main__":
    from ml.env import load_env_files
    load_env_files()
    logging.basicConfig(level=logging.INFO)
    main()