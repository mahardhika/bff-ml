"""Letterbox / flip keep boxes in canvas."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ml.data.augment import augment_detection, letterbox


def test_letterbox_square() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    canvas, scale = letterbox(image, input_size=640)
    assert canvas.shape == (640, 640, 3)
    assert scale == pytest.approx(640 / 200)


def test_augment_keeps_boxes_inside() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 40.0, 40.0]], dtype=np.float32)
    canvas, out = augment_detection(image, boxes, input_size=64, train=True)
    assert canvas.shape == (64, 64, 3)
    assert (out >= 0).all()
    assert (out[:, [0, 2]] <= 64).all()
    assert (out[:, [1, 3]] <= 64).all()
