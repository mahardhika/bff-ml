"""Train CLI smoke (no GPU, no real training)."""

import os
import subprocess
import sys
from pathlib import Path


def test_train_help():
    out = subprocess.run(
        [sys.executable, "-m", "ml", "train", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0
    assert "--data-source" in out.stdout
    assert "--epochs" in out.stdout


def test_train_dry(tmp_path: Path):
    env = os.environ.copy()
    env.setdefault("INFERENCE_PROVIDER", "cpu")
    env.setdefault("DATABASE_MODE", "offline")
    env.setdefault("LOG_LEVEL", "WARNING")
    out = subprocess.run(
        [sys.executable, "-m", "ml", "train",
         "--data-source", "local",
         "--data-path", str(tmp_path),
         "--epochs", "1",
         "--device", "cpu",
         "--no-sync",
         "--output-dir", str(tmp_path / "runs")],
        capture_output=True, text=True, timeout=120,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-400:]
    assert (tmp_path / "runs").exists()