"""LocalDataset smoke test (no actual training involved)."""

import json
from pathlib import Path

import pytest

from ml.data.load import LocalDataset


def test_local_coco_loader(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "bucket"}, {"id": 2, "name": "truck"}],
        "annotations": [],
    }))
    # No image file on disk, loader should skip rather than crash.
    ds = LocalDataset(path=str(tmp_path)).load()
    assert ds.kind == "coco"
    assert "bucket" in ds.classes