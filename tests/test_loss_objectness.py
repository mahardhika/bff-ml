"""Regression test for the objectness collapse in YOLOXLoss.

Why this test exists
---------------------
The shipped loss supervises the objectness channel with a BCE that is
summed over every grid cell and normalised by the *total* cell count.
The construction-vehicle dataset has ~1.31 positives per image against
8,400 cells (strides 8/16/32 at 640px) - a 6,400:1 imbalance - so the
background gradient outweighs the positive one and the optimiser settles
on "predict 0 everywhere".

Measured on the shipped weights:

    earthmoving-detector@0.1.0 : obj max 0.0100, 0 anchors clear conf 0.05
    construction-vehicle 10ep   : obj max 0.0991, 0 anchors clear conf 0.40

The gradient ratio below is the mechanism: with the shipped loss the
background's objectness gradient is *larger* than the positive cell's, so
the branch is trained to say "nothing here". The fix has to invert that
ratio without letting the objectness term drown the reg/l1 terms.

Run with::

    cd ml && pytest tests/test_loss_objectness.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.models.loss import YOLOXLoss

STRIDES = (8, 16, 32)
SIZES = (80, 40, 20)
NUM_CLASSES = 8
CHANNELS = 4 + 1 + NUM_CLASSES


def _synthetic_batch(seed: int = 0):
    """One image, one positive cell, the same shape as the real head."""
    gen = torch.Generator().manual_seed(seed)
    preds, assignments = [], []
    for size in SIZES:
        p = torch.randn(1, CHANNELS, size, size, generator=gen,
                        requires_grad=True)
        pos = torch.zeros(1, size, size, dtype=torch.bool)
        pos[0, size // 2, size // 2] = True
        target_box = torch.zeros(1, size, size, 4)
        target_box[0, size // 2, size // 2] = torch.tensor([10.0, 10.0, 40.0, 40.0])
        target_cls = torch.zeros(1, size, size, dtype=torch.long)
        target_cls[0, size // 2, size // 2] = 2
        preds.append(p)
        assignments.append({"positive_mask": pos,
                            "target_box": target_box,
                            "target_cls": target_cls})
    return preds, assignments


def _objectness_gradients(loss_fn, preds, assignments):
    """Gradient of the total loss on the objectness channel.

    Returns ``(at_positive, max_at_background)``.
    """
    out = loss_fn(preds, assignments)
    total = out["obj"] + out["cls"] + out["reg"] + out["l1"]
    total.backward()
    grads = []
    for p, size in zip(preds, SIZES):
        g = p.grad[:, 4]              # objectness channel, (B, H, W)
        pos_g = g[0, size // 2, size // 2].item()
        bg = g[0].clone()
        bg[size // 2, size // 2] = 0.0
        grads.append((pos_g, bg.abs().max().item()))
    return out, grads


def test_objectness_gradient_at_positive_beats_background() -> None:
    """The positive cell must pull the objectness head harder than the
    background pushes it - otherwise the head learns "no object anywhere"."""
    preds, assignments = _synthetic_batch()
    loss_fn = YOLOXLoss(num_classes=NUM_CLASSES, strides=STRIDES)
    _out, grads = _objectness_gradients(loss_fn, preds, assignments)
    for pos_g, bg_max in grads:
        assert abs(pos_g) > bg_max, (
            f"objectness gradient at the positive cell ({pos_g:.3e}) is not "
            f"stronger than the background ({bg_max:.3e}); the head is "
            f"being trained to predict 'nothing' everywhere"
        )


def test_objectness_term_stays_in_the_same_range_as_the_other_terms() -> None:
    """A fix must not win by exploding the objectness loss: the reg and
    l1 terms would be drowned out and the boxes would stop regressing."""
    preds, assignments = _synthetic_batch()
    loss_fn = YOLOXLoss(num_classes=NUM_CLASSES, strides=STRIDES)
    out, _ = _objectness_gradients(loss_fn, preds, assignments)
    obj = float(out["obj"])
    assert 0.0 < obj < 50.0, (
        f"objectness loss {obj:.3f} is out of range; it must stay "
        f"comparable to the reg/l1 terms"
    )
    assert float(out["reg"]) > 0.0
    assert float(out["l1"]) > 0.0


def test_loss_is_finite_for_a_batch_without_positives() -> None:
    """An image with no assignment must not produce NaNs."""
    preds, assignments = _synthetic_batch()
    for a in assignments:
        a["positive_mask"] = torch.zeros_like(a["positive_mask"])
    loss_fn = YOLOXLoss(num_classes=NUM_CLASSES, strides=STRIDES)
    out = loss_fn(preds, assignments)
    for key, value in out.items():
        assert torch.isfinite(value).all(), f"{key} is not finite"
