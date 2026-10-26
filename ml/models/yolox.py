"""YOLOX-S style detector (Apache-2.0 clean-room).

Architecture:
- Backbone: CSPDarkNet-S (depth=0.33, width=0.50).
- Neck: PAFPN (Path Aggregation Feature Pyramid Network).
- Head: decoupled 1x1 conv branches for cls + reg + obj.

The shape follows the original YOLOX paper / open-source release,
but the code here is a clean implementation under Apache-2.0; we do
not vendor any GPL/AGPL code from Ultralytics.

The model is small enough (~9M params) to train on a T4 with
batch=16 at 640x640. Inference is <10ms per frame on a 1080p clip
sampled at 1 fps.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# --- helpers ---


def _silu():
    return nn.SiLU(inplace=True)


class ConvBNAct(nn.Module):
    """Conv + BatchNorm + SiLU. The default block in YOLOX."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1,
                 p: int | None = None, g: int = 1) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = _silu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Residual bottleneck: 1x1 reduce -> 3x3 expand -> add."""

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True) -> None:
        super().__init__()
        mid = out_ch
        self.cv1 = ConvBNAct(in_ch, mid, k=1)
        self.cv2 = ConvBNAct(mid, out_ch, k=3)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        if self.add:
            out = out + x
        return out


class CSPLayer(nn.Module):
    """Cross-Stage-Partial layer with N bottlenecks."""

    def __init__(self, in_ch: int, out_ch: int, n: int = 1,
                 shortcut: bool = True) -> None:
        super().__init__()
        mid = out_ch
        self.cv1 = ConvBNAct(in_ch, mid, k=1)
        self.cv2 = ConvBNAct(in_ch, mid, k=1)
        self.m = nn.Sequential(*(Bottleneck(mid, mid, shortcut) for _ in range(n)))
        self.cv3 = ConvBNAct(2 * mid, out_ch, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling: 3 parallel max-pools, then concat + 1x1."""

    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: Sequence[int] = (5, 9, 13)) -> None:
        super().__init__()
        mid = in_ch // 2
        self.cv1 = ConvBNAct(in_ch, mid, k=1)
        self.cv2 = ConvBNAct(mid * (len(kernel_sizes) + 1), out_ch, k=1)
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernel_sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [p(x) for p in self.pools], dim=1))


# --- backbone: CSPDarkNet-S ---


class Focus(nn.Module):
    """Slice + concat (Focus). YOLOX uses this for the first stem block.
    Equivalent to a stride-2 conv on 4x the channels.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3) -> None:
        super().__init__()
        self.conv = ConvBNAct(in_ch * 4, out_ch, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2x downsample by slicing.
        return self.conv(torch.cat([
            x[..., ::2, ::2], x[..., ::2, 1::2],
            x[..., 1::2, ::2], x[..., 1::2, 1::2],
        ], dim=1))


class CSPDarkNet(nn.Module):
    """YOLOX-S backbone. Outputs strides 8, 16, 32."""

    def __init__(self, depth: float = 0.33, width: float = 0.50) -> None:
        super().__init__()
        def ch(c: int) -> int: return max(8, int(c * width))
        def n(d: float) -> int: return max(1, int(round(d)))

        self.stem = Focus(3, ch(64), k=3)
        # Stage 1: stride 2.
        self.dark2 = nn.Sequential(
            ConvBNAct(ch(64), ch(128), k=3, s=2),
            CSPLayer(ch(128), ch(128), n=n(depth * 3)),
        )
        # Stage 2: stride 4.
        self.dark3 = nn.Sequential(
            ConvBNAct(ch(128), ch(256), k=3, s=2),
            CSPLayer(ch(256), ch(256), n=n(depth * 9)),
        )
        # Stage 3: stride 8.
        self.dark4 = nn.Sequential(
            ConvBNAct(ch(256), ch(512), k=3, s=2),
            CSPLayer(ch(512), ch(512), n=n(depth * 9)),
        )
        # Stage 4: stride 16.
        self.dark5 = nn.Sequential(
            ConvBNAct(ch(512), ch(1024), k=3, s=2),
            SPPBottleneck(ch(1024), ch(1024)),
            CSPLayer(ch(1024), ch(1024), n=n(depth * 3), shortcut=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.dark2(x)
        # Stride 8 feature.
        feat_s8 = self.dark3(x)
        # Stride 16 feature.
        feat_s16 = self.dark4(feat_s8)
        # Stride 32 feature.
        feat_s32 = self.dark5(feat_s16)
        return feat_s8, feat_s16, feat_s32


# --- neck: PAFPN ---


class PAFPN(nn.Module):
    """Path Aggregation FPN. Top-down + bottom-up fusion of 3 levels.

    `in_channels` should be the channel counts as the backbone emits
    them - i.e. *already* width-scaled. The caller (YOLOXS) does
    the width math and passes raw channel sizes here.
    """

    def __init__(self, in_channels: Sequence[int], depth: float = 0.33,
                 width: float = 0.50) -> None:
        super().__init__()
        c2, c3, c4 = in_channels
        n = max(1, int(round(depth * 3)))

        # Top-down
        self.up1 = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.csp1 = CSPLayer(c4 + c3, c3, n=n, shortcut=False)
        self.up2 = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.csp2 = CSPLayer(c3 + c2, c2, n=n, shortcut=False)

        # Bottom-up
        self.down1 = ConvBNAct(c2, c2, k=3, s=2)
        self.csp3 = CSPLayer(c2 + c3, c3, n=n, shortcut=False)
        self.down2 = ConvBNAct(c3, c3, k=3, s=2)
        self.csp4 = CSPLayer(c3 + c4, c4, n=n, shortcut=False)

        self.out_channels = (c2, c3, c4)

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s8, s16, s32 = feats
        # Top-down
        up1 = self.up1(s32)
        cat1 = torch.cat([up1, s16], dim=1)
        td_16 = self.csp1(cat1)
        up2 = self.up2(td_16)
        cat2 = torch.cat([up2, s8], dim=1)
        td_8 = self.csp2(cat2)
        # Bottom-up
        dn1 = self.down1(td_8)
        cat3 = torch.cat([dn1, td_16], dim=1)
        bu_16 = self.csp3(cat3)
        dn2 = self.down2(bu_16)
        cat4 = torch.cat([dn2, s32], dim=1)
        bu_32 = self.csp4(cat4)
        return td_8, bu_16, bu_32


# --- head: decoupled ---


class YOLOXHead(nn.Module):
    """Decoupled head: cls + reg + obj. 1 anchor-free per pixel."""

    def __init__(self, num_classes: int, in_channels: Sequence[int],
                 strides: Sequence[int] = (8, 16, 32)) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.n_anchors = 1  # anchor-free

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for ch_in in in_channels:
            self.cls_convs.append(nn.Sequential(
                ConvBNAct(ch_in, ch_in, k=3), ConvBNAct(ch_in, ch_in, k=3)))
            self.reg_convs.append(nn.Sequential(
                ConvBNAct(ch_in, ch_in, k=3), ConvBNAct(ch_in, ch_in, k=3)))
            self.cls_preds.append(nn.Conv2d(ch_in, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(ch_in, 4, 1))
            self.obj_preds.append(nn.Conv2d(ch_in, 1, 1))

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> torch.Tensor:
        outputs = []
        for cls_conv, reg_conv, cls_p, reg_p, obj_p, x in zip(
            self.cls_convs, self.reg_convs, self.cls_preds, self.reg_preds,
            self.obj_preds, feats
        ):
            cls_feat = cls_conv(x)
            reg_feat = reg_conv(x)
            cls_out = cls_p(cls_feat)
            reg_out = reg_p(reg_feat)
            obj_out = obj_p(reg_feat)
            # Concat per pixel: [reg(4), obj(1), cls(C)]
            out = torch.cat([reg_out, obj_out, cls_out], dim=1)
            outputs.append(out)
        return outputs


# --- assembly ---


@dataclass
class YOLOXConfig:
    num_classes: int = 3  # bucket, truck, person (operator)
    input_size: int = 640
    depth: float = 0.33
    width: float = 0.50
    strides: tuple[int, ...] = (8, 16, 32)


class YOLOXS(nn.Module):
    """YOLOX-S: CSPDarkNet-S + PAFPN + decoupled head."""

    def __init__(self, cfg: YOLOXConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or YOLOXConfig()
        self.cfg = cfg
        self.backbone = CSPDarkNet(depth=cfg.depth, width=cfg.width)
        # Backbone emits these channel counts at strides 8, 16, 32.
        # They match the per-stage widths in CSPDarkNet.
        def ch(c: int) -> int: return max(8, int(c * cfg.width))
        c2 = ch(256)   # dark3 output
        c3 = ch(512)   # dark4 output
        c4 = ch(1024)  # dark5 output
        self.neck = PAFPN(in_channels=(c2, c3, c4), depth=cfg.depth, width=cfg.width)
        self.head = YOLOXHead(cfg.num_classes, self.neck.out_channels, cfg.strides)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)

    def load_backbone_from_pretrained(self, weights_path: str) -> None:
        """Optional: load ImageNet-pretrained backbone weights if available."""
        sd = torch.load(weights_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        own = self.backbone.state_dict()
        loaded = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        own.update(loaded)
        self.backbone.load_state_dict(own)
        log.info("Loaded %d backbone tensors from %s", len(loaded), weights_path)


def build_yolox_s(num_classes: int = 3, input_size: int = 640) -> YOLOXS:
    """Convenience builder for the S variant."""
    return YOLOXS(YOLOXConfig(num_classes=num_classes, input_size=input_size))


# --- ONNX export helper ---


def export_onnx(model: YOLOXS, onnx_path: str, input_size: int = 640,
                opset: int = 13) -> None:
    """Export the model to ONNX with the same input shape the BE expects.

    The weights are pulled back to the CPU first: a model trained on
    DirectML carries private-use-one tensors that ONNX export rejects.
    """
    import copy
    model.eval()
    export_model = copy.deepcopy(model)
    try:
        export_model = export_model.to("cpu")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not move the model to the CPU for export: %s", exc)
    dummy = torch.zeros(1, 3, input_size, input_size)
    torch.onnx.export(
        export_model, dummy, onnx_path, opset_version=opset,
        input_names=["input"],
        output_names=["stride8", "stride16", "stride32"],
        dynamic_axes={"input": {0: "batch"}},
    )
    log.info("Exported YOLOX-S to %s (opset=%d, input=%dx%d)",
             onnx_path, opset, input_size, input_size)