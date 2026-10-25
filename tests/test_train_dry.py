"""Full ML training dry-run on synthetic CVAT-style COCO data.

End-to-end exercise of the `ml train` pipeline WITHOUT:
  - a live CVAT server (we synthesize the COCO export)
  - real images (PIL Image.new() produces tiny PNGs)
  - a GPU (cpu + 1 epoch)
  - Supabase sync (--no-sync)

The dry-run must produce the canonical run-dir artifacts:
  - hparams.json
  - metrics.json
  - MODEL_CARD.md
  - <model_name>.onnx (possibly empty when torch export path is unavailable)
  - the run-dir name embeds the model name + version + a unix timestamp

If this test ever breaks on a clean machine, the trainer is no longer
a usable smoke gate for CI / new contributors.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# We import the helper below for unit-level checks; the trainer subprocess
# runs `python -m ml train` against the same fixtures.
from ml.data.load import LocalDataset

# Pillow is a hard dep of ml[notebook] and most ml[*] extras, so the
# import here should always succeed - but guard it so a missing
# optional dep doesn't break test collection.
PIL = pytest.importorskip("PIL.Image", reason="Pillow is required for synthetic CVAT fixtures")


# ---------------------------------------------------------------------------
# Fixture builder: synthesize a CVAT-style COCO dataset on disk.
# ---------------------------------------------------------------------------

CLASSES = ["bucket", "truck", "person"]
IMG_SIZE = (96, 64)


def build_synthetic_coco_dataset(root: Path) -> Path:
    """Create a minimal CVAT-style COCO folder with 2 images and 4 bboxes.

    Returns the root path (with annotations.json + images/ subdir).
    """
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Two PNG placeholders.
    img_specs = [
        ("frame_001.png", IMG_SIZE, "red"),
        ("frame_002.png", IMG_SIZE, "blue"),
    ]
    file_names = []
    for i, (name, size, color) in enumerate(img_specs, start=1):
        img = PIL.new("RGB", size, color=color)
        img.save(images_dir / name)
        file_names.append(name)

    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1,
         "bbox": [10, 10, 30, 25], "iscrowd": 0},  # bucket
        {"id": 2, "image_id": 1, "category_id": 2,
         "bbox": [50, 20, 25, 25], "iscrowd": 0},  # truck
        {"id": 3, "image_id": 2, "category_id": 1,
         "bbox": [12, 8, 28, 30], "iscrowd": 0},   # bucket
        {"id": 4, "image_id": 2, "category_id": 3,
         "bbox": [60, 30, 18, 22], "iscrowd": 0},  # person
    ]
    coco = {
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)],
        "images": [
            {"id": i + 1, "file_name": f"images/{f}",
             "width": IMG_SIZE[0], "height": IMG_SIZE[1]}
            for i, f in enumerate(file_names)
        ],
        "annotations": annotations,
    }
    (root / "annotations.json").write_text(
        json.dumps(coco), encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# Helper: run `python -m ml train ...` and surface the run-dir.
# ---------------------------------------------------------------------------

def _run_train(data_dir: Path, output_dir: Path, *,
               model_name: str = "bucket-detector",
               version: str = "0.1.0",
               epochs: int = 1,
               batch_size: int = 2) -> Path:
    env = os.environ.copy()
    env.setdefault("INFERENCE_PROVIDER", "cpu")
    env.setdefault("DATABASE_MODE", "offline")
    env.setdefault("LOG_LEVEL", "WARNING")
    proc = subprocess.run(
        [
            sys.executable, "-m", "ml", "train",
            "--data-source", "local",
            "--data-path", str(data_dir),
            "--model-name", model_name,
            "--version", version,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--device", "cpu",
            "--no-sync",
            "--output-dir", str(output_dir),
        ],
        capture_output=True, text=True, timeout=300,
        env=env,
    )
    assert proc.returncode == 0, (
        f"`ml train` failed (rc={proc.returncode}).\n"
        f"STDOUT (last 400 chars):\n{proc.stdout[-400:]}\n"
        f"STDERR (last 400 chars):\n{proc.stderr[-400:]}"
    )
    # The trainer creates one subdir per run, named
    # `<model-name>-<version>-<unix-ts>`. There should be exactly one.
    runs = sorted(p for p in output_dir.iterdir() if p.is_dir())
    assert len(runs) == 1, (
        f"expected 1 run dir under {output_dir}, found: "
        f"{[p.name for p in runs]}"
    )
    return runs[0]


# ---------------------------------------------------------------------------
# Unit-level: dataset loader works on the synthetic fixture.
# ---------------------------------------------------------------------------

def test_synthetic_loader_iterates_records(tmp_path: Path) -> None:
    root = build_synthetic_coco_dataset(tmp_path / "data")
    ds = LocalDataset(path=str(root)).load()
    assert ds.kind == "coco"
    assert set(ds.classes) == set(CLASSES)
    records = list(ds)
    assert len(records) == 2, f"expected 2 image records, got {len(records)}"
    img_path, label = records[0]
    assert img_path.exists()
    assert len(label["boxes"]) >= 1


# ---------------------------------------------------------------------------
# End-to-end: ml train dry-run on synthetic CVAT data.
# ---------------------------------------------------------------------------

def test_train_dry_run_emits_canonical_artifacts(tmp_path: Path) -> None:
    data_dir = build_synthetic_coco_dataset(tmp_path / "data")
    runs_dir = tmp_path / "runs"

    run_dir = _run_train(data_dir, runs_dir,
                         model_name="bucket-detector",
                         version="0.1.0",
                         epochs=1, batch_size=2)

    # --- hparams.json ---
    hparams_path = run_dir / "hparams.json"
    assert hparams_path.exists(), f"missing {hparams_path}"
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    assert hparams["epochs"] == 1
    assert hparams["batch_size"] == 2
    assert hparams["device"] == "cpu"
    assert hparams["data_source"] == "local"
    assert hparams["data_path"] == str(data_dir)

    # --- metrics.json ---
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists(), f"missing {metrics_path}"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for k in ("train_loss", "val_loss", "device"):
        assert k in metrics, f"metrics.json missing {k!r}"
    # device was honored
    assert metrics["device"] == "cpu"

    # --- MODEL_CARD.md ---
    card_path = run_dir / "MODEL_CARD.md"
    assert card_path.exists(), f"missing {card_path}"
    card = card_path.read_text(encoding="utf-8")
    assert "# bucket-detector v0.1.0" in card
    assert "epochs: 1" in card

    # --- ONNX (canonical name the backend inference worker loads) ---
    onnx_path = run_dir / "detector.onnx"
    assert onnx_path.exists(), f"missing {onnx_path}"
    # ONNX may be an empty placeholder if torch export path failed.
    # That's still acceptable for a smoke run.
    assert onnx_path.is_file()

    # --- run-dir naming convention ---
    name = run_dir.name
    assert name.startswith("bucket-detector-0.1.0-"), (
        f"unexpected run-dir name: {name!r}"
    )
    assert re.search(r"bucket-detector-0\.1\.0-\d+$", name)


def test_train_dry_run_with_classifier_kind(tmp_path: Path) -> None:
    """`--kind=classifier` produces a 5-class fill classifier instead of
    the 3-class detector. The run-dir artifacts must still be complete."""
    data_dir = build_synthetic_coco_dataset(tmp_path / "data")
    runs_dir = tmp_path / "runs"
    env = os.environ.copy()
    env.setdefault("INFERENCE_PROVIDER", "cpu")
    env.setdefault("DATABASE_MODE", "offline")
    env.setdefault("LOG_LEVEL", "WARNING")
    proc = subprocess.run(
        [
            sys.executable, "-m", "ml", "train",
            "--data-source", "local",
            "--data-path", str(data_dir),
            "--kind", "classifier",
            "--model-name", "fill-classifier",
            "--version", "0.1.0",
            "--epochs", "1",
            "--batch-size", "2",
            "--device", "cpu",
            "--no-sync",
            "--output-dir", str(runs_dir),
        ],
        capture_output=True, text=True, timeout=300,
        env=env,
    )
    assert proc.returncode == 0, (
        f"`ml train --kind=classifier` failed.\n"
        f"STDERR (last 400 chars):\n{proc.stderr[-400:]}"
    )
    runs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "hparams.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "classifier.onnx").exists()
    assert (run_dir / "MODEL_CARD.md").exists()
    assert "fill-classifier v0.1.0" in (run_dir / "MODEL_CARD.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Failure mode: a missing data path must abort with a clear error.
# ---------------------------------------------------------------------------

def test_train_missing_data_dir_aborts(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("INFERENCE_PROVIDER", "cpu")
    env.setdefault("DATABASE_MODE", "offline")
    env.setdefault("LOG_LEVEL", "WARNING")
    proc = subprocess.run(
        [
            sys.executable, "-m", "ml", "train",
            "--data-source", "local",
            "--data-path", str(tmp_path / "does-not-exist"),
            "--epochs", "1",
            "--device", "cpu",
            "--no-sync",
            "--output-dir", str(tmp_path / "runs"),
        ],
        capture_output=True, text=True, timeout=60,
        env=env,
    )
    # LocalDataset raises FileNotFoundError before any artifacts are
    # written, so the trainer subprocess should exit non-zero.
    assert proc.returncode != 0
    # No partial run-dir should be left behind.
    assert not (tmp_path / "runs").exists() or \
        list((tmp_path / "runs").iterdir()) == [], (
        f"partial run-dir left behind after failed train: "
        f"{list((tmp_path / 'runs').iterdir())}"
    )