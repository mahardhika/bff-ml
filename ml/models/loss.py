"""YOLOX loss (IoU-aware + cls + obj BCE).

Standard YOLOX loss formula:
    L = L_obj + L_cls + L_reg + L_l1
where L_obj / L_cls are BCEWithLogits, L_reg is IoU loss, L_l1 is L1.

``forward`` expects the assignment dicts produced by
``ml.models.assign.assign`` so the loop is data-driven rather than
synthetic zeros.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def bbox_iou(b1: torch.Tensor, b2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """IoU between two sets of boxes in xyxy format."""
    x1 = torch.max(b1[..., 0], b2[..., 0])
    y1 = torch.max(b1[..., 1], b2[..., 1])
    x2 = torch.min(b1[..., 2], b2[..., 2])
    y2 = torch.min(b1[..., 3], b2[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    a2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    return inter / (a1 + a2 - inter + eps)


def _decode_xyxy(pred: torch.Tensor, stride: int) -> torch.Tensor:
    """Decode ``[B, 4, H, W]`` (cx-offset, cy-offset, w, h) to xyxy pixels."""
    _, _, height, width = pred.shape
    device = pred.device
    gy, gx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    cx = (pred[:, 0] + gx) * stride
    cy = (pred[:, 1] + gy) * stride
    w = pred[:, 2].abs() * stride
    h = pred[:, 3].abs() * stride
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


class YOLOXLoss(nn.Module):
    """Composite YOLOX loss driven by center-based assignments."""

    def __init__(self, num_classes: int, strides=(8, 16, 32),
                 loss_weight: dict | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.loss_weight = loss_weight or {
            "obj": 1.0, "cls": 1.0, "reg": 5.0, "l1": 1.0,
        }

    def forward(self, preds: list[torch.Tensor], assignments: list[dict] | None
                ) -> dict[str, torch.Tensor]:
        device = preds[0].device
        if not assignments:
            obj_loss = sum(
                self.bce(p[:, 4:5], torch.zeros_like(p[:, 4:5])) for p in preds
            ) / max(1, sum(p.shape[0] for p in preds))
            zero = torch.zeros((), device=device)
            return {"obj": obj_loss, "cls": zero, "reg": zero, "l1": zero}

        obj_acc = torch.zeros((), device=device)
        cls_acc = torch.zeros((), device=device)
        reg_acc = torch.zeros((), device=device)
        l1_acc = torch.zeros((), device=device)
        n_pos = 0
        n_obj = 0

        for pred, assigned, stride in zip(preds, assignments, self.strides):
            pos = assigned["positive_mask"]
            obj_target = pos.to(pred.dtype).unsqueeze(1)
            obj_acc = obj_acc + self.bce(pred[:, 4:5], obj_target)
            n_obj += obj_target.numel()

            if not pos.any():
                continue
            n_pos += int(pos.sum().item())
            decoded = _decode_xyxy(pred[:, :4], int(stride))
            pos_boxes = decoded.permute(0, 2, 3, 1)[pos]
            gt_boxes = assigned["target_box"][pos]
            iou = bbox_iou(pos_boxes, gt_boxes).clamp(min=1e-6)
            reg_acc = reg_acc + (1.0 - iou).sum()
            l1_acc = l1_acc + (pos_boxes - gt_boxes).abs().sum()

            cls_logits = pred[:, 5:].permute(0, 2, 3, 1)[pos]
            cls_ids = assigned["target_cls"][pos].clamp(0, self.num_classes - 1)
            cls_oh = F.one_hot(cls_ids, num_classes=self.num_classes).to(pred.dtype)
            cls_acc = cls_acc + self.bce(cls_logits, cls_oh)

        n_pos = max(1, n_pos)
        n_obj = max(1, n_obj)
        return {
            "obj": obj_acc / n_obj * self.loss_weight["obj"],
            "cls": cls_acc / n_pos * self.loss_weight["cls"],
            "reg": reg_acc / n_pos * self.loss_weight["reg"],
            "l1": l1_acc / (n_pos * 4) * self.loss_weight["l1"],
        }
