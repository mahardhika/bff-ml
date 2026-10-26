"""torch Dataset wrapping a resolved ``ml.data.load.Dataset``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.data.augment import augment_detection
from ml.data.load import Dataset as BffDataset


class DetectionFrames:
    """Iterable of (tensor CHW float32, boxes xyxy, cls ids)."""

    def __init__(self, dataset: BffDataset, input_size: int = 640,
                 train: bool = True, limit: int = 0) -> None:
        try:
            self.samples = list(dataset)
        except FileNotFoundError:
            self.samples = []
        if limit and limit > 0 and len(self.samples) > limit:
            # Even spread over the whole list: the first N rows of a
            # COCO export are usually one camera batch.
            stride = len(self.samples) / float(limit)
            picked = [self.samples[int(i * stride)] for i in range(limit)]
            self.samples = picked
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


class ActivityFrames:
    """Frames with a multi-hot activity label.

    ``samples`` are :class:`ml.data.activity.ActivitySample` rows; the
    image is read from ``root`` and letterboxed to ``input_size``.
    """

    def __init__(self, root: Path, samples, num_classes: int,
                 input_size: int = 224, train: bool = True,
                 split: str = "train") -> None:
        self.root = Path(root)
        self.samples = list(samples)
        self.num_classes = num_classes
        self.input_size = input_size
        self.train = train
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        from ml.data.augment import letterbox

        sample = self.samples[idx]
        path = self._resolve(sample.file_name)
        image = _read_bgr(path)
        canvas, _ = letterbox(image, input_size=self.input_size)
        if self.train:
            canvas = canvas[:, ::-1].copy() if idx % 2 else canvas
        tensor = (torch.from_numpy(canvas[:, :, ::-1].copy())
                  .permute(2, 0, 1).float() / 255.0)
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for value in sample.labels:
            if 0 <= value < self.num_classes:
                label[value] = 1.0
        return {"image": tensor, "label": label, "path": str(path)}

    def _resolve(self, file_name: str) -> Path:
        """Find the image: split folder first, then the dataset root."""
        candidates = [
            self.root / self.split / file_name,
            self.root / file_name,
            self.root / self.split / file_name.replace("images/", ""),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]


def collate_activity(batch: list[dict]) -> dict:
    import torch

    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
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
