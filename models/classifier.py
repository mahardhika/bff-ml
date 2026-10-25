"""5-class fill classifier.

Uses a custom lightweight CNN (BFFNet-5) instead of EfficientNet-B0
to keep the ONNX small and CPU-friendly. Architecture:
  - 3x conv-bn-relu blocks (32 -> 64 -> 128) with maxpool
  - Global average pool
  - FC(128 -> 5) with softmax

5 classes: empty, partial, full, over, not_visible
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

CLASSES: tuple[str, ...] = ("empty", "partial", "full", "over", "not_visible")


@dataclass
class ClassifierConfig:
    num_classes: int = 5
    input_size: int = 224
    base_channels: int = 32
    dropout: float = 0.2


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FillClassifier(nn.Module):
    """Lightweight 5-class CNN classifier.

    ~150k params; trains in minutes on a small labeled set.
    """

    def __init__(self, cfg: ClassifierConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ClassifierConfig()
        self.cfg = cfg
        c = cfg.base_channels
        self.stem = nn.Sequential(
            ConvBNAct(3, c), nn.MaxPool2d(2),       # 224 -> 112
            ConvBNAct(c, c * 2), nn.MaxPool2d(2),   # 112 -> 56
            ConvBNAct(c * 2, c * 4), nn.MaxPool2d(2),  # 56 -> 28
            ConvBNAct(c * 4, c * 4),                 # 28
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(c * 4, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stem(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[int, float, Sequence[float]]:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
        return idx, float(probs[idx]), [float(p) for p in probs]


def build_fill_classifier(num_classes: int = 5, input_size: int = 224) -> FillClassifier:
    return FillClassifier(ClassifierConfig(num_classes=num_classes, input_size=input_size))


def export_onnx(model: FillClassifier, onnx_path: str, input_size: int = 224,
                opset: int = 13) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size)
    torch.onnx.export(
        model, dummy, onnx_path, opset_version=opset,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    log.info("Exported FillClassifier to %s (opset=%d, input=%dx%d)",
             onnx_path, opset, input_size, input_size)