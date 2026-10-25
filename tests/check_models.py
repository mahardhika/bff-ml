"""Smoke check: build models, run forward pass, count params."""

from __future__ import annotations

import sys

import torch

from ml.models.yolox import build_yolox_s
from ml.models.classifier import build_fill_classifier, CLASSES
from ml.models.loss import YOLOXLoss


def main() -> int:
    det = build_yolox_s(num_classes=3, input_size=640)
    det.eval()
    with torch.no_grad():
        out = det(torch.zeros(1, 3, 640, 640))
    print(f"YOLOX-S: {len(out)} feature maps, shapes {[tuple(o.shape) for o in out]}")
    n_params = sum(p.numel() for p in det.parameters())
    print(f"YOLOX-S params: {n_params/1e6:.2f}M")

    cls = build_fill_classifier(num_classes=5, input_size=224)
    cls.eval()
    with torch.no_grad():
        y = cls(torch.zeros(1, 3, 224, 224))
    print(f"FillClassifier: {tuple(y.shape)}, classes: {CLASSES}")
    n_params_cls = sum(p.numel() for p in cls.parameters())
    print(f"FillClassifier params: {n_params_cls/1e3:.1f}K")

    loss_fn = YOLOXLoss(num_classes=3)
    losses = loss_fn(out, None)
    total = sum(v.item() for v in losses.values())
    print(f"Loss total: {total:.4f} (keys={list(losses)})")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())