"""Anchor-free center-based assignment for YOLOX FPN levels.

For each ground-truth box we pick the grid cell that contains its
center on every FPN level where that cell is in-bounds. If two GTs
land on the same cell, the larger box wins. This is a slim stand-in
for SimOTA: enough to drive a real data-driven loop without the
full dynamic-k matching.
"""

from __future__ import annotations

from typing import Sequence

import torch


def assign(
    preds: list[torch.Tensor],
    boxes: list[torch.Tensor],
    cls: list[torch.Tensor],
    strides: Sequence[int] = (8, 16, 32),
    input_size: int = 640,
) -> list[dict[str, torch.Tensor]]:
    """Return per-level assignment dicts.

    ``preds[i]`` is ``[B, 5+C, H, W]``.
    ``boxes[b]`` is ``[N_b, 4]`` xyxy in input-canvas pixels.
    ``cls[b]`` is ``[N_b]`` class ids.
    """
    device = preds[0].device
    batch = preds[0].shape[0]
    out: list[dict[str, torch.Tensor]] = []
    for pred, stride in zip(preds, strides):
        _, _, height, width = pred.shape
        pos = torch.zeros((batch, height, width), dtype=torch.bool, device=device)
        gt_index = torch.full((batch, height, width), -1, dtype=torch.long, device=device)
        target_box = torch.zeros((batch, height, width, 4), dtype=pred.dtype, device=device)
        target_cls = torch.zeros((batch, height, width), dtype=torch.long, device=device)
        area_best = torch.zeros((batch, height, width), dtype=pred.dtype, device=device)

        for b in range(batch):
            b_boxes = boxes[b].to(device) if boxes[b] is not None else None
            b_cls = cls[b].to(device) if cls[b] is not None else None
            if b_boxes is None or b_boxes.numel() == 0:
                continue
            centers = (b_boxes[:, :2] + b_boxes[:, 2:]) / 2.0
            gx = (centers[:, 0] / stride).long()
            gy = (centers[:, 1] / stride).long()
            areas = (b_boxes[:, 2] - b_boxes[:, 0]).clamp(min=0) * (
                b_boxes[:, 3] - b_boxes[:, 1]).clamp(min=0)
            for i in range(b_boxes.shape[0]):
                x = int(gx[i].item())
                y = int(gy[i].item())
                if not (0 <= x < width and 0 <= y < height):
                    continue
                a = float(areas[i].item())
                if pos[b, y, x] and a <= float(area_best[b, y, x].item()):
                    continue
                pos[b, y, x] = True
                gt_index[b, y, x] = i
                target_box[b, y, x] = b_boxes[i]
                target_cls[b, y, x] = int(b_cls[i].item()) if b_cls is not None else 0
                area_best[b, y, x] = a
        out.append({
            "positive_mask": pos,
            "gt_index": gt_index,
            "target_box": target_box,
            "target_cls": target_cls,
            "stride": torch.tensor(stride, device=device),
        })
    return out
