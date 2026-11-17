"""The bucket-fill head.

The head is a small ImageNet-pretrained EfficientNet-B0 with a linear
fill classifier on top. Two details matter for the deployment:

* **The ONNX input is byte-identical to what the backend sends.** The
  backend feeds ``letterbox_bgr`` + ``to_blob`` (RGB, 0..1, NCHW). The
  ImageNet mean/std normalisation is a *graph node* inside the model,
  so the exported ONNX still takes the raw 0..1 blob and the backend
  preprocessing does not have to change.
* **The class list lives next to the weights.** ``labels.json`` holds
  the class names in head-index order; the backend reads it and maps
  them onto the product vocabulary.

When torchvision is not installed the builder falls back to the repo's
own BFFNet-5 (``ml.models.classifier``), which trains from scratch and
needs no extra dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

#: ImageNet channel statistics (the pretrained backbone expects them).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class FillHeadConfig:
    num_classes: int = 2
    input_size: int = 224
    pretrained: bool = True
    dropout: float = 0.2


class EfficientNetFillHead(nn.Module):
    """EfficientNet-B0 backbone + linear fill head.

    ``forward`` takes the 0..1 RGB blob the backend produces and
    normalises it inside the graph, so the exported ONNX keeps the
    backend's preprocessing contract.
    """

    def __init__(self, cfg: FillHeadConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or FillHeadConfig()
        self.cfg = cfg
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        net = efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Linear(in_features, cfg.num_classes)
        if cfg.dropout > 0:
            net.classifier = nn.Sequential(nn.Dropout(cfg.dropout), net.classifier)
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.net(x)


def build_fill_head(num_classes: int = 2, input_size: int = 224,
                    pretrained: bool = True) -> nn.Module:
    """The fill head. Falls back to BFFNet-5 without torchvision."""
    try:
        return EfficientNetFillHead(FillHeadConfig(
            num_classes=num_classes, input_size=input_size,
            pretrained=pretrained))
    except ImportError as exc:
        log.warning("torchvision not available (%s) - falling back to BFFNet-5.",
                    exc)
        from ml.models.classifier import build_fill_classifier
        return build_fill_classifier(num_classes=num_classes,
                                     input_size=input_size)


def export_fill_head_onnx(model: nn.Module, onnx_path: str,
                          input_size: int = 224, opset: int = 13) -> None:
    """Export the fill head to ONNX (input: 0..1 RGB NCHW)."""
    import copy

    model.eval()
    export_model = copy.deepcopy(model)
    try:
        export_model = export_model.to("cpu")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not move the fill head to the CPU for export: %s", exc)
    dummy = torch.zeros(1, 3, input_size, input_size)
    torch.onnx.export(
        export_model, dummy, onnx_path, opset_version=opset,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    log.info("Exported fill head to %s (opset=%d, input=%dx%d)",
             onnx_path, opset, input_size, input_size)
