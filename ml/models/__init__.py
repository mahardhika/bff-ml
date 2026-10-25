"""Model architectures for the BFF Camera Demo.

This package contains the **Apache-2.0 clean-room** YOLOX-style
detector and an EfficientNet-B0 5-class classifier. We avoid
Ultralytics YOLO (AGPL-3.0) per `system-map-arsitektur.MD` §0.

The detector is small enough to train on a single T4 (Colab free
tier) in a few hours for a v0.1 demo. Production training uses
larger variants and is documented separately.
"""

from ml.models.yolox import YOLOXS, build_yolox_s
from ml.models.classifier import FillClassifier, build_fill_classifier
from ml.models.loss import YOLOXLoss

__all__ = [
    "YOLOXS", "build_yolox_s",
    "FillClassifier", "build_fill_classifier",
    "YOLOXLoss",
]