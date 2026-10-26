"""Export a trained torch model to ONNX.

Usage:
    python -m ml export-onnx --ckpt ./runs/.../model.pt \
                             --out ./runs/.../model.onnx \
                             --opset 13

If the ckpt is already ONNX (`.onnx`), we copy it through.
If a fresh export fails (no torch), we emit a placeholder so the
sync step doesn't break.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--input-shape", default="1,3,640,640")
    p.add_argument("--kind", default="detector",
                   choices=["detector", "classifier", "activity"])
    p.add_argument("--num-classes", type=int, default=3,
                   help="classes in the head; must match the checkpoint")
    args = p.parse_args(argv)

    log.info("Loading checkpoint: %s", args.ckpt)
    try:
        import torch  # type: ignore
    except ImportError as exc:
        log.error("torch is required: %s", exc)
        Path(args.out).write_bytes(b"")
        return

    src = Path(args.ckpt)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".onnx":
        dst.write_bytes(src.read_bytes())
        log.info("Copied ONNX -> %s", dst)
        return

    shape = tuple(int(s) for s in args.input_shape.split(","))
    try:
        if args.kind == "detector":
            from ml.models.yolox import (YOLOXConfig, YOLOXS,
                                         export_onnx as _yolox_export)
            model = YOLOXS(YOLOXConfig(num_classes=int(args.num_classes)))
            model.load_state_dict(torch.load(src, map_location="cpu"))
            model.eval()
            _yolox_export(model, str(dst), input_size=shape[-1], opset=args.opset)
        else:
            from ml.models.classifier import (ClassifierConfig, FillClassifier,
                                               export_onnx as _cls_export)
            model = FillClassifier(ClassifierConfig(num_classes=int(args.num_classes)))
            model.load_state_dict(torch.load(src, map_location="cpu"))
            model.eval()
            _cls_export(model, str(dst), input_size=shape[-1], opset=args.opset)
    except Exception as exc:  # noqa: BLE001
        log.error("Export failed: %s", exc)
        dst.write_bytes(b"")
        log.warning("Wrote empty placeholder to %s", dst)


def _export_with_torch(run_dir, model, train_args) -> None:
    """Helper for `ml.train` to export the in-memory trained model."""
    import torch
    from pathlib import Path
    canonical = "detector.onnx" if train_args.kind == "detector" else "classifier.onnx"
    out_path = Path(run_dir) / canonical
    input_size = int(getattr(train_args, "input_size", 0) or 640)
    if train_args.kind == "detector":
        from ml.models.yolox import export_onnx as _yolox_export
        _yolox_export(model, str(out_path), input_size=input_size, opset=13)
    else:
        from ml.models.classifier import export_onnx as _cls_export
        size = int(getattr(train_args, "classifier_input_size", 0) or 224)
        _cls_export(model, str(out_path), input_size=size, opset=13)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()