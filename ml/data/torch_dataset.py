"""torch Dataset wrapping a resolved ``ml.data.load.Dataset``."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ml.data.augment import augment_detection
from ml.data.load import Dataset as BffDataset


class DetectionFrames:
    """Iterable of (tensor CHW float32, boxes xyxy, cls ids)."""

    def __init__(self, dataset: BffDataset, input_size: int = 640,
                 train: bool = True) -> None:
        try:
            self.samples = list(dataset)
        except FileNotFoundError:
            self.samples = []
        self.input_size = input_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        path, label = self.samples[idx]
        image = _read_bgr(path)
        boxes, cls = _boxes_from_label(label)
        image, boxes = augment_detection(
            image, boxes, input_size=self.input_size, train=self.train,
        )
        tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return {
            "image": tensor,
            "boxes": torch.from_numpy(boxes).float(),
            "cls": torch.from_numpy(cls).long(),
            "path": str(path),
        }


def collate_detections(batch: list[dict]) -> dict:
    import torch

    images = torch.stack([b["image"] for b in batch], dim=0)
    return {
        "image": images,
        "boxes": [b["boxes"] for b in batch],
        "cls": [b["cls"] for b in batch],
        "path": [b["path"] for b in batch],
    }


def _read_bgr(path: Path) -> np.ndarray:
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def _boxes_from_label(label: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = label.get("boxes") or []
    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    boxes = []
    cls = []
    for row in rows:
        bbox = row.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        # Accept COCO xywh if the box looks inverted.
        if x2 < x1 or y2 < y1:
            x2 = x1 + abs(x2)
            y2 = y1 + abs(y2)
        boxes.append([x1, y1, x2, y2])
        cls.append(int(row.get("category_id") or 0))
    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(cls, dtype=np.int64),
    )
