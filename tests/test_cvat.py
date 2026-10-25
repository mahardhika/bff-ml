"""Smoke test for `ml.data.cvat.coco_to_yolo` without a live CVAT server."""

from __future__ import annotations

import json
from pathlib import Path

from ml.data.cvat import coco_to_yolo


def _sample_coco(tmp: Path) -> Path:
    data = {
        "categories": [
            {"id": 1, "name": "bucket"}, {"id": 2, "name": "truck"},
            {"id": 3, "name": "person"},
        ],
        "images": [
            {"id": 1, "file_name": "frame_001.jpg", "width": 1920, "height": 1080},
            {"id": 2, "file_name": "frame_002.jpg", "width": 1920, "height": 1080},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 200, 400, 300], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [1500, 400, 200, 200], "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [200, 250, 350, 280], "iscrowd": 0},
        ],
    }
    p = tmp / "annotations.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_coco_to_yolo_writes_classes_and_labels(tmp_path: Path) -> None:
    coco = _sample_coco(tmp_path)
    out = tmp_path / "yolo"
    coco_to_yolo(coco, out)

    classes = (out / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes == ["bucket", "truck", "person"]

    files = sorted(p.name for p in (out / "labels").glob("*.txt"))
    assert files == ["frame_001.txt", "frame_002.txt"]

    body = (out / "labels" / "frame_001.txt").read_text(encoding="utf-8").splitlines()
    assert len(body) == 2, body
    # Each line: cls cx cy w h.
    parts = body[0].split()
    assert len(parts) == 5
    assert int(parts[0]) == 0  # bucket
    assert 0 <= float(parts[1]) <= 1


def test_coco_to_yolo_ignores_iscrowd(tmp_path: Path) -> None:
    coco = tmp_path / "a.json"
    coco.write_text(json.dumps({
        "categories": [{"id": 1, "name": "bucket"}],
        "images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [0, 0, 10, 10], "iscrowd": 1}],
    }), encoding="utf-8")
    out = tmp_path / "yolo"
    coco_to_yolo(coco, out)
    assert list((out / "labels").glob("*.txt")) == [], "iscrowd should be skipped"