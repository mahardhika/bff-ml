"""Lightweight detection augmentations matching the 640 letterbox used at inference."""

from __future__ import annotations

import random

import numpy as np


def letterbox(image: np.ndarray, input_size: int = 640,
              pad_value: int = 114) -> tuple[np.ndarray, float]:
    """Resize with aspect ratio preserved, pad to a square canvas."""
    h0, w0 = image.shape[:2]
    scale = input_size / max(h0, w0)
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    import cv2
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), pad_value, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale


def scale_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    out = boxes.astype(np.float32).copy()
    out[:, :4] *= scale
    return out


def random_hflip(image: np.ndarray, boxes: np.ndarray, p: float = 0.5
                 ) -> tuple[np.ndarray, np.ndarray]:
    if random.random() >= p:
        return image, boxes
    image = image[:, ::-1].copy()
    if boxes.size:
        w = image.shape[1]
        x1 = w - boxes[:, 2]
        x2 = w - boxes[:, 0]
        boxes = boxes.copy()
        boxes[:, 0] = x1
        boxes[:, 2] = x2
    return image, boxes


def hsv_jitter(image: np.ndarray, hgain: float = 0.015, sgain: float = 0.7,
               vgain: float = 0.4) -> np.ndarray:
    import cv2
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat),
                         cv2.LUT(val, lut_val)))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def augment_detection(image: np.ndarray, boxes: np.ndarray,
                      input_size: int = 640, train: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Letterbox + optional flip/HSV. Boxes are xyxy in pixel coords."""
    canvas, scale = letterbox(image, input_size=input_size)
    boxes = scale_boxes(boxes, scale)
    if train:
        canvas, boxes = random_hflip(canvas, boxes)
        canvas = hsv_jitter(canvas)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_size)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_size)
    return canvas, boxes
