"""Device-picker behavior.

These tests pin the contract of `ml.train._pick_device()`:

* When `torch_directml` is importable on Windows, we return the raw
  `torch.device` it gives us (a `privateuseone:0`-style object).
  We must NOT prefix it with `"dml:"` — `model.to(...)` rejects that.
* When `torch_directml` is missing and `INFERENCE_PROVIDER=dml`, we
  fall back to `"cpu"` with a log line so users aren't silently
  dropped to a different backend.
* Whatever we return must be re-parseable via `torch.device(...)`,
  even if a future caller passes through some unrecognised backend
  (`xpu`, `hpu`, …).

These are unit tests for `_pick_device()` plus a subprocess smoke
run of `python -m ml train --device cpu` to confirm end-to-end the
device string survives the train pipeline.
"""

from __future__ import annotations

import builtins
import importlib
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

try:  # pragma: no cover - import style depends on the tests/ layout
    from _bff_paths import repo_root
except ImportError:  # pragma: no cover
    from tests._bff_paths import repo_root

torch = pytest.importorskip("torch", reason="torch required for device tests")


@pytest.fixture
def fresh_train_module(monkeypatch: pytest.MonkeyPatch):
    """Reload `ml.train` with a clean `INFERENCE_PROVIDER` env."""
    def _load(provider: str | None) -> object:
        if provider is None:
            monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)
        else:
            monkeypatch.setenv("INFERENCE_PROVIDER", provider)
        sys.modules.pop("ml.train", None)
        return importlib.import_module("ml.train")

    return _load


def _block_dml_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `import torch_directml` raise inside _pick_device()."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch_directml":
            raise ImportError("torch_directml not present (test stub)")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _install_fake_directml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake torch_directml module that returns a privateuseone device."""
    fake = mock.MagicMock()
    fake.device = mock.MagicMock(return_value=torch.device("privateuseone", 0))
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch_directml":
            return fake
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


# ---------------------------------------------------------------------------
# Unit: _pick_device() contract — always returns something torch parses.
# ---------------------------------------------------------------------------


def test_pick_device_returns_torch_device(fresh_train_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock torch_directml so it returns `torch.device("privateuseone", 0)`.

    Assert that `_pick_device()` returns the device object directly
    (not a `"dml:..."` prefixed string) and that `torch.device(...)`
    round-trips without raising.
    """
    if os.name != "nt":
        pytest.skip("DML is Windows-only")

    _install_fake_directml(monkeypatch)
    mod = fresh_train_module("dml")
    picked = mod._pick_device()

    # Contract 1: not the broken `"dml:..."` string.
    assert not (isinstance(picked, str) and picked.startswith("dml:")), (
        f"_pick_device() returned the broken prefixed string {picked!r}; "
        f"`model.to(...)` rejects this with `Invalid device string`."
    )

    # Contract 2: torch.device(...) round-trips.
    rt = torch.device(picked) if not isinstance(picked, torch.device) else picked
    assert isinstance(rt, torch.device)
    assert rt.type == "privateuseone"
    assert rt.index == 0


def test_train_resolves_dml_on_windows_then_cpu(
    fresh_train_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Without torch_directml installed, INFERENCE_PROVIDER=dml must
    downgrade to cpu (parseable)."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "dml")
    _block_dml_import(monkeypatch)
    mod = fresh_train_module("dml")
    picked = mod._pick_device()

    # Contract: always parseable.
    rt = torch.device(picked) if not isinstance(picked, torch.device) else picked
    assert rt.type == "cpu", f"expected cpu fallback, got {picked!r}"

    # End-to-end smoke: synthesize a tiny coco dataset (the trainer
    # refuses an empty data dir before reaching device logic), then
    # run `ml train --device cpu` and confirm it completes.
    import importlib.util
    helper_path = Path(__file__).parent / "test_train_dry.py"
    spec = importlib.util.spec_from_file_location("_tcd_helper", helper_path)
    helper_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(helper_mod)  # type: ignore[union-attr]
    build_synthetic_coco_dataset = helper_mod.build_synthetic_coco_dataset  # noqa: N806

    data_dir = build_synthetic_coco_dataset(tmp_path / "data")
    env = os.environ.copy()
    env.setdefault("DATABASE_MODE", "offline")
    env.setdefault("LOG_LEVEL", "WARNING")
    out_dir = tmp_path / "runs"
    out = subprocess.run(
        [
            sys.executable, "-m", "ml", "train",
            "--data-source", "local",
            "--data-path", str(data_dir),
            "--epochs", "1",
            "--device", "cpu",
            "--no-sync",
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True, timeout=120,
        env=env,
        cwd=str(repo_root()),  # repo root
    )
    assert out.returncode == 0, (
        f"`ml train --device cpu` failed.\nSTDOUT(last 400):\n"
        f"{out.stdout[-400:]}\nSTDERR(last 400):\n{out.stderr[-400:]}"
    )


def test_train_default_when_no_dml_no_cuda(
    fresh_train_module, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if env hints at dml but the package is missing, _pick_device
    must never return an unparseable string."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "dml")
    _block_dml_import(monkeypatch)
    mod = fresh_train_module("dml")
    picked = mod._pick_device()

    try:
        rt = torch.device(picked) if not isinstance(picked, torch.device) else picked
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"_pick_device() returned unparseable value {picked!r}: {exc}")
    assert rt.type in {"cpu", "cuda", "privateuseone"}, (
        f"unexpected device type {rt.type!r} from {picked!r}"
    )
