"""The detector's box terms must be adjustable from the CLI.

The L1 term is in pixel units, so for a small object it dwarfs the
scale-invariant IoU term: the optimum of an L1-dominated loss is the
MEDIAN box, and the model then predicts the same box in every frame
(verified on the bucket detector - it could not overfit 200 images with
the default weights). These tests pin the defaults so the shipped
behaviour cannot change silently, and pin the new knobs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from ml.models.loss import YOLOXLoss  # noqa: E402


def test_default_weights_are_unchanged() -> None:
    """The shipped detector recipe must not change under us."""
    loss = YOLOXLoss(num_classes=2)
    assert loss.obj_pos_weight == 1000.0
    assert loss.loss_weight == {"obj": 1.0, "cls": 1.0, "reg": 5.0, "l1": 1.0}


def test_l1_weight_can_be_scaled_down() -> None:
    """A small object needs the IoU term to drive the regression."""
    loss = YOLOXLoss(num_classes=1,
                     loss_weight={"obj": 1.0, "cls": 1.0, "reg": 5.0, "l1": 0.1})
    assert loss.loss_weight["l1"] == 0.1
    assert loss.loss_weight["reg"] == 5.0


def test_obj_pos_weight_is_configurable() -> None:
    loss = YOLOXLoss(num_classes=1, obj_pos_weight=5000.0)
    assert loss.obj_pos_weight == 5000.0


def test_l1_weight_scales_the_pixel_term() -> None:
    """The L1 term is measured in pixels, so its weight decides its share.

    Note that the IoU term has no gradient while the prediction does not
    overlap the target at all, so the L1 term is what bootstraps the
    regression - but at full weight its optimum is the median box, which
    is why the detector predicted the same box in every frame.
    """
    torch.manual_seed(0)
    num_classes = 1
    stride = 8
    height = width = 80
    box = torch.tensor([[8.0, 8.0, 40.0, 56.0]])
    target_cls = torch.tensor([0])

    def total(l1_weight: float) -> float:
        loss_fn = YOLOXLoss(
            num_classes=num_classes,
            loss_weight={"obj": 1.0, "cls": 1.0, "reg": 5.0, "l1": l1_weight})
        preds = [torch.zeros((1, 5 + num_classes, height, width))]
        centers = (box[:, :2] + box[:, 2:]) / 2.0
        gx = int((centers[0, 0] / stride).long())
        gy = int((centers[0, 1] / stride).long())
        pos = torch.zeros((1, height, width), dtype=torch.bool)
        target_box = torch.zeros((1, height, width, 4))
        tgt_cls = torch.zeros((1, height, width), dtype=torch.long)
        pos[0, gy, gx] = True
        target_box[0, gy, gx] = box[0]
        tgt_cls[0, gy, gx] = int(target_cls[0])
        assignments = [{"positive_mask": pos, "target_box": target_box,
                        "target_cls": tgt_cls,
                        "stride": torch.tensor(stride)}]
        out = loss_fn(preds, assignments)
        return float(out["l1"] + out["reg"] + out["obj"] + out["cls"])

    full = total(1.0)
    scaled = total(0.1)
    assert full > scaled, "scaling the L1 term down must lower the total"
