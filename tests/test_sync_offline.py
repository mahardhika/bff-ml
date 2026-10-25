"""Offline-mode + remote-fallback behavior for `ml.sync.list_models`.

Three contracts under test:
  * `list_models` works with no Supabase creds at all (offline / env unset).
  * If the Supabase call fails for any reason, we fall back to the local
    `runs/` listing rather than crashing.
  * Every subcommand in the `ml.__main__._DISPATCH` table can be invoked
    with the bare minimum args without raising `KeyError` on missing env.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ml import __main__ as ml_main
from ml.sync import list_local_runs, list_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populate_run(run_dir: Path, *, name: str, version: str) -> Path:
    """Create one fake run subdir with a MODEL_CARD.md + a 4-byte .onnx stub."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "MODEL_CARD.md").write_text(
        f"# {name} v{version}\n\nTrained for the offline-mode test.\n",
        encoding="utf-8",
    )
    (run_dir / f"{name}.onnx").write_bytes(b"\x00\x00\x00\x00")
    return run_dir


# ---------------------------------------------------------------------------
# 1) Offline listing returns every local run, never raises KeyError.
# ---------------------------------------------------------------------------

def test_list_models_offline_mode(tmp_path: Path, monkeypatch, capsys) -> None:
    # Strip every Supabase var from the environment; force offline.
    for var in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
                "SELF_HOSTED_SUPABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATABASE_MODE", "offline")

    runs_dir = tmp_path / "runs"
    _populate_run(runs_dir / "bucket-detector-0.1.0-1700000000",
                  name="bucket-detector", version="0.1.0")
    _populate_run(runs_dir / "fill-classifier-0.2.0-1700000111",
                  name="fill-classifier", version="0.2.0")

    list_models(["--runs-dir", str(runs_dir)])

    out = capsys.readouterr().out
    assert "Local MLflow-style runs" in out, out
    assert "bucket-detector" in out, out
    assert "fill-classifier" in out, out
    # Both runs should be present.
    assert out.count(".onnx") >= 2, out


# ---------------------------------------------------------------------------
# 2) Remote failure falls back to the local listing instead of crashing.
# ---------------------------------------------------------------------------

def test_list_models_remote_falls_back_to_local(tmp_path: Path, monkeypatch,
                                                capsys, caplog) -> None:
    # Pretend we have valid Supabase creds so we attempt the remote call...
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-service-role-key")
    monkeypatch.setenv("DATABASE_MODE", "public")

    runs_dir = tmp_path / "runs"
    _populate_run(runs_dir / "bucket-detector-0.1.0-1700000000",
                  name="bucket-detector", version="0.1.0")

    # ...but make the remote fetch helper return the "failed" sentinel so
    # `list_models` exercises its local fallback path.
    import ml.sync as ml_sync
    monkeypatch.setattr(ml_sync, "_fetch_remote_models",
                        lambda: None)  # type: ignore[assignment]

    import logging
    with caplog.at_level(logging.WARNING, logger="ml.sync"):
        list_models(["--runs-dir", str(runs_dir)])

    out = capsys.readouterr().out
    assert "bucket-detector" in out, out
    # The warning that triggered the fallback must be present in the log.
    joined_logs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "falling back" in joined_logs, joined_logs
    assert "Remote listing failed" in joined_logs, joined_logs


# ---------------------------------------------------------------------------
# 3) None of the `ml` subcommands raises KeyError on missing env.
# ---------------------------------------------------------------------------

# Every entry in the dispatcher must survive a clean env (no .env loaded)
# when invoked with --help.  --help short-circuits before any network /
# credential work, so this is a precise canary for unguarded os.environ[].
_DISPATCH_KEYS = sorted(ml_main._DISPATCH.keys())


@pytest.mark.parametrize("subcmd", _DISPATCH_KEYS)
def test_ml_subcommands_handle_missing_env(subcmd: str, monkeypatch) -> None:
    for var in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
                "SELF_HOSTED_SUPABASE_URL", "DATABASE_MODE"):
        monkeypatch.delenv(var, raising=False)

    proc = subprocess.run(
        [sys.executable, "-m", "ml", subcmd, "--help"],
        capture_output=True, text=True, timeout=60,
        env={k: v for k, v in os.environ.items()
             if k not in {"SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
                          "SELF_HOSTED_SUPABASE_URL", "DATABASE_MODE"}},
    )
    assert proc.returncode == 0, (
        f"`ml {subcmd} --help` raised with a clean env.\n"
        f"STDOUT: {proc.stdout[-400]}\nSTDERR: {proc.stderr[-400]}"
    )
    assert "KeyError" not in proc.stderr, proc.stderr


# ---------------------------------------------------------------------------
# Bonus: list_local_runs is importable and returns parsed cards.
# ---------------------------------------------------------------------------

def test_list_local_runs_parses_model_cards(tmp_path: Path) -> None:
    _populate_run(tmp_path / "a-0.1.0-1", name="a", version="0.1.0")
    _populate_run(tmp_path / "b-0.2.0-2", name="b", version="0.2.0")
    rows = list_local_runs(tmp_path)
    names = {r["name"] for r in rows}
    assert names == {"a", "b"}, rows
    for r in rows:
        assert r["source"] == "local"
        assert r["size_bytes"] == 4