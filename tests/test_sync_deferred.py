"""Deferred sync behavior for `ml.sync`.

`system-map-arsitektur.MD` §5.22.3 specifies three skip conditions,
and every one of them must preserve the experiment somewhere:

* `DATABASE_MODE=offline`   -> park it in `data/experiments/<run>.json`
* missing Supabase creds   -> same
* Supabase unreachable      -> same, with a warning

Before this, a skipped sync simply returned. The run stayed on disk
but nothing recorded *why* it was not published and nothing made it
replayable, so the run was effectively lost.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from ml import sync as ml_sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populate_run(run_dir: Path, name: str = "bucket-detector",
                  version: str = "0.1.0") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "hparams.json").write_text(
        json.dumps({"epochs": 3, "lr": 0.001}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"mAP50": 0.5, "fill_acc": 0.9}), encoding="utf-8")
    (run_dir / "MODEL_CARD.md").write_text(
        f"# {name} v{version}\n", encoding="utf-8")
    (run_dir / f"{name}.onnx").write_bytes(b"\x00\x01")
    return run_dir


@pytest.fixture()
def deferred_dir(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the deferred-record directory into tmp_path."""
    d = tmp_path / "experiments"
    monkeypatch.setenv(ml_sync.EXPERIMENTS_DIR_ENV, str(d))
    return d


def _fake_modules(monkeypatch, *, s3_raises: bool = False,
                  sb_raises: bool = False) -> None:
    """Install fake `boto3` / `supabase` modules in sys.modules."""

    class _FakeS3:
        def upload_file(self, *a, **k):  # pragma: no cover - not asserted
            return None

    class _FakeTable:
        def __init__(self, name: str) -> None:
            self.name = name

        def upsert(self, *a, **k):
            return self

        def select(self, *a, **k):
            return self

        def eq(self, *a, **k):
            return self

        def limit(self, *a, **k):
            return self

        def insert(self, *a, **k):
            return self

        def execute(self):
            if sb_raises:
                raise RuntimeError("supabase unreachable")
            return types.SimpleNamespace(data=[{"id": 7}])

    class _FakeClient:
        def __init__(self, *a, **k) -> None:
            pass

        def table(self, name: str) -> _FakeTable:
            return _FakeTable(name)

    def _create_client(*a, **k):
        if sb_raises:
            raise RuntimeError("supabase unreachable")
        return _FakeClient()

    boto3_mod = types.ModuleType("boto3")
    if s3_raises:
        def _client(*a, **k):
            raise RuntimeError("s3 unreachable")
        boto3_mod.client = _client
    else:
        boto3_mod.client = lambda *a, **k: _FakeS3()

    sb_mod = types.ModuleType("supabase")
    sb_mod.create_client = _create_client

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "supabase", sb_mod)

    class _FakeStorage:
        transport = "s3"

        def upload_file(self, *a, **k):
            if s3_raises:
                raise RuntimeError("s3 unreachable")

    monkeypatch.setattr(
        "bff_core.storage.StorageClient.from_env",
        lambda *a, **k: _FakeStorage(),
    )


# ---------------------------------------------------------------------------
# 1) Offline mode parks the experiment locally.
# ---------------------------------------------------------------------------

def test_offline_writes_deferred_record(tmp_path: Path, deferred_dir: Path,
                                        monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_MODE", "offline")
    run = _populate_run(tmp_path / "runs" / "run-1")

    ok = ml_sync.sync_experiment(run_dir=run, model_name="bucket-detector",
                                 version="0.1.0")

    assert ok is False, "offline sync must report that it did not publish"
    target = deferred_dir / "run-1.json"
    assert target.exists(), "offline run must be parked for later sync"
    rec = json.loads(target.read_text(encoding="utf-8"))
    assert rec["model_name"] == "bucket-detector"
    assert rec["version"] == "0.1.0"
    assert rec["run_name"] == "run-1"
    assert rec["metrics"]["mAP50"] == 0.5
    assert rec["hyperparams"]["epochs"] == 3
    assert rec["path"].endswith("bucket-detector.onnx")
    assert "deferred_at" in rec


# ---------------------------------------------------------------------------
# 2) Missing creds park the experiment locally.
# ---------------------------------------------------------------------------

def test_missing_creds_writes_deferred_record(tmp_path: Path,
                                               deferred_dir: Path,
                                               monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_MODE", "public")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    run = _populate_run(tmp_path / "runs" / "run-2")

    ok = ml_sync.sync_experiment(run_dir=run, model_name="m", version="0.0.1")

    assert ok is False
    assert (deferred_dir / "run-2.json").exists()


# ---------------------------------------------------------------------------
# 3) An unreachable Supabase parks the experiment locally.
# ---------------------------------------------------------------------------

def test_unreachable_supabase_writes_deferred_record(tmp_path: Path,
                                                     deferred_dir: Path,
                                                     monkeypatch, caplog) -> None:
    monkeypatch.setenv("DATABASE_MODE", "public")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-key")
    _fake_modules(monkeypatch, sb_raises=True)
    run = _populate_run(tmp_path / "runs" / "run-3")

    ok = ml_sync.sync_experiment(run_dir=run, model_name="m", version="0.0.2")

    assert ok is False, "a failed sync must not raise, and must report False"
    assert (deferred_dir / "run-3.json").exists()
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "preserved on disk" in joined, joined


# ---------------------------------------------------------------------------
# 4) A successful sync leaves nothing behind locally.
# ---------------------------------------------------------------------------

def test_successful_sync_clears_deferred_record(tmp_path: Path,
                                                deferred_dir: Path,
                                                monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_MODE", "public")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-key")
    _fake_modules(monkeypatch)
    run = _populate_run(tmp_path / "runs" / "run-4")

    # Pre-seed a stale deferred record for the same run.
    deferred_dir.mkdir(parents=True, exist_ok=True)
    (deferred_dir / "run-4.json").write_text("{}", encoding="utf-8")

    ok = ml_sync.sync_experiment(run_dir=run, model_name="m", version="0.0.3")

    assert ok is True
    assert not (deferred_dir / "run-4.json").exists(), (
        "a successful sync must remove the stale deferred record")


# ---------------------------------------------------------------------------
# 5) The deferred record carries everything the replay needs.
# ---------------------------------------------------------------------------

def test_deferred_record_has_replay_fields(tmp_path: Path,
                                           deferred_dir: Path,
                                           monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_MODE", "offline")
    run = _populate_run(tmp_path / "runs" / "run-5")

    ml_sync.sync_experiment(run_dir=run, model_name="bucket-detector",
                            version="0.2.0", kind="classifier",
                            run_name="run-5", dataset_ids=[3, 4])

    rec = json.loads((deferred_dir / "run-5.json").read_text(encoding="utf-8"))
    assert rec["kind"] == "classifier"
    assert rec["dataset_ids"] == [3, 4]
    assert rec["run_dir"] == str(run)
    assert rec["model_card_url"].endswith("MODEL_CARD.md")