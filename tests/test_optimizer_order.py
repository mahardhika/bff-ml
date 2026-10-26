"""The optimizer has to be built after the model moved to the device.

On DirectML the move replaces the Parameter objects. An optimizer that
holds the old ones updates tensors the model no longer uses: the loss
stays flat and the exported weights are untrained.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.models.classifier import build_fill_classifier  # noqa: E402
from ml.train import _build_optimizer  # noqa: E402


def test_optimizer_holds_the_live_parameters() -> None:
    model = build_fill_classifier(num_classes=3)
    opt = _build_optimizer(model, "cpu", lr=1e-3)
    assert opt.param_groups[0]["params"][0] is next(model.parameters())


def test_a_step_changes_the_weights() -> None:
    model = build_fill_classifier(num_classes=3)
    opt = _build_optimizer(model, "cpu", lr=1e-2)
    before = next(model.parameters()).detach().clone()
    x = torch.rand(2, 3, 224, 224)
    y = (torch.rand(2, 3) > 0.5).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    after = next(model.parameters()).detach()
    assert float((after - before).abs().max()) > 0
