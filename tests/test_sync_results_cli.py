"""`python -m ml.sync_results` CLI behavior.

`system-map-arsitektur.MD` §5.22.4 documents the replay command:

    python -m ml.sync_results --run-name=yolox-s-finetune-20261019-1042

It must be reachable both as a module and through the `ml` dispatcher,
and it must never crash on a clean environment (no .env loaded).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ml import __main__ as ml_main
from ml import sync_results as sr

EXPERIMENTS_ENV = "BFF_EXPERIMENTS_DIR"
#: The canonical key in the project `.env`; it wins over the alias, so
#: both must be redirected for a test to be deterministic.
EXPERIMENTS_ENV_CANONICAL = "EXPERIMENTS_DIR"


def _point_experiments_dir(monkeypatch, path) -> None:
    monkeypatch.setenv(EXPERIMENTS_ENV_CANONICAL, str(path))
    monkeypatch.setenv(EXPERIMENTS_ENV, str(path))


# ---------------------------------------------------------------------------
# Dispatcher wiring
# ---------------------------------------------------------------------------

def test_sync_results_is_dispatchable() -> None:
    assert "sync-results" in ml_main._DISPATCH
    assert ml_main._DISPATCH["sync-results"] == ("ml.sync_results", "main")
    # Snake-case alias for parity with `export_onnx`.
    assert ml_main._DISPATCH["sync_results"] == ("ml.sync_results", "main")


@pytest.mark.parametrize("argv", [
    [sys.executable, "-m", "ml", "sync-results", "--help"],
    [sys.executable, "-m", "ml.sync_results", "--help"],
])
def test_cli_help_exits_clean(argv: list[str]) -> None:
    env = {k: v for k, v in os.environ.items()
           if k not in {"SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
                        "SELF_HOSTED_SUPABASE_URL", "DATABASE_MODE"}}
    proc = subprocess.run(argv, capture_output=True, text=True,
                          timeout=60, env=env)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "KeyError" not in proc.stderr, proc.stderr[-400:]


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_list_reports_empty(tmp_path: Path, monkeypatch, capsys) -> None:
    _point_experiments_dir(monkeypatch, str(tmp_path / "experiments"))
    sr.main(["--list"])
    out = capsys.readouterr().out
    assert "no deferred syncs" in out, out


def test_list_shows_deferred_records(tmp_path: Path, monkeypatch, capsys) -> None:
    d = tmp_path / "experiments"
    d.mkdir()
    (d / "run-a.json").write_text(json.dumps({
        "run_name": "run-a", "model_name": "bucket-detector",
        "version": "0.1.0", "deferred_at": "2026-10-20T00:00:00Z",
    }), encoding="utf-8")
    _point_experiments_dir(monkeypatch, str(d))

    sr.main(["--list"])
    out = capsys.readouterr().out
    assert "run-a" in out, out
    assert "bucket-detector" in out, out


def test_list_deferred_skips_unreadable(tmp_path: Path, monkeypatch,
                                        capsys) -> None:
    d = tmp_path / "experiments"
    d.mkdir()
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    (d / "good.json").write_text(json.dumps({
        "run_name": "good", "model_name": "m", "version": "0.0.1",
    }), encoding="utf-8")
    _point_experiments_dir(monkeypatch, str(d))

    sr.main(["--list"])
    out = capsys.readouterr().out
    assert "good" in out, out
    assert "broken" not in out, "an unreadable record must not be listed"


# ---------------------------------------------------------------------------
# Replay resolution
# ---------------------------------------------------------------------------

def test_resolve_run_dir_prefers_recorded_path(tmp_path: Path) -> None:
    recorded = tmp_path / "moved" / "run-x"
    recorded.mkdir(parents=True)
    rec = {"run_dir": str(recorded)}
    assert sr.resolve_run_dir(rec, "run-x") == recorded


def test_resolve_run_dir_falls_back_to_runs_dir(tmp_path: Path,
                                                monkeypatch) -> None:
    runs = tmp_path / "runs"
    (runs / "run-y").mkdir(parents=True)
    monkeypatch.setenv(sr.RUNS_DIR_ENV, str(runs))
    assert sr.resolve_run_dir({"run_dir": "/nonexistent"},
                               "run-y") == runs / "run-y"


def test_resolve_run_dir_returns_none_when_missing(tmp_path: Path,
                                                    monkeypatch) -> None:
    monkeypatch.setenv(sr.RUNS_DIR_ENV, str(tmp_path / "runs"))
    assert sr.resolve_run_dir({"run_dir": "/nonexistent"}, "ghost") is None


# ---------------------------------------------------------------------------
# Replay end-to-end (offline: stays deferred, does not raise)
# ---------------------------------------------------------------------------

def test_replay_of_deferred_run(tmp_path: Path, monkeypatch, capsys) -> None:
    """Replaying an offline run must not raise and must report the deferral."""
    runs = tmp_path / "runs" / "run-z"
    runs.mkdir(parents=True)
    (runs / "hparams.json").write_text("{}", encoding="utf-8")
    (runs / "metrics.json").write_text("{}", encoding="utf-8")
    (runs / "m.onnx").write_bytes(b"\x00")

    d = tmp_path / "experiments"
    d.mkdir()
    (d / "run-z.json").write_text(json.dumps({
        "run_name": "run-z", "model_name": "m", "version": "0.0.1",
        "kind": "detector", "run_dir": str(runs), "dataset_ids": [],
    }), encoding="utf-8")

    _point_experiments_dir(monkeypatch, str(d))
    monkeypatch.setenv(sr.RUNS_DIR_ENV, str(tmp_path / "runs"))
    monkeypatch.setenv("DATABASE_MODE", "offline")

    with pytest.raises(SystemExit) as exc:
        sr.main(["--run-name", "run-z"])
    assert exc.value.code == 2, "a still-deferred run exits 2"
    out = capsys.readouterr().out
    assert "Still deferred" in out, out


def test_replay_unknown_run_exits_1(tmp_path: Path, monkeypatch) -> None:
    _point_experiments_dir(monkeypatch, str(tmp_path / "experiments"))
    monkeypatch.setenv(sr.RUNS_DIR_ENV, str(tmp_path / "runs"))
    with pytest.raises(SystemExit) as exc:
        sr.main(["--run-name", "does-not-exist"])
    assert exc.value.code == 1


def test_cli_requires_a_selector(tmp_path: Path, monkeypatch) -> None:
    _point_experiments_dir(monkeypatch, str(tmp_path / "experiments"))
    with pytest.raises(SystemExit):
        sr.main([])