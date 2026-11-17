"""Evaluate a trained ONNX model on a held-out split.

This used to be a stub: it wrote ``{"mAP50": 0.0, ...}`` for every
model and never opened the weights. It now runs the exported ONNX
through onnxruntime and computes real numbers:

* ``--kind=detector``  -> pycocotools mAP50 / mAP50:95, per-class
  precision / recall / F1 at the shipped operating point, plus the
  repo's own YOLOX validation loss (``ml.models.loss``).
* ``--kind=classifier`` (multi-label activity head) -> per-class
  precision / recall / F1, macro F1, exact-match accuracy.

* ``--kind=fill`` (binary bucket-fill head) -> per-class
  precision / recall / F1, macro / micro F1, exact-match accuracy and
  the confusion matrix over the letterboxed bucket crops.

Usage:
    python -m ml eval --model-path=./runs/.../detector.onnx \
                      --kind=detector \
                      --dataset-dir=/data/construction-vehicle/valid \
                      --device=cuda --name=eval-cv-valid

    python -m ml eval --model-path=./runs/.../classifier.onnx \
                      --kind=classifier \
                      --dataset-root=/data/earthmoving-equipment \
                      --split=valid --activity-valid-frames=1500 \
                      --device=cuda --name=eval-em-activity

``--update-run`` merges the measured values back into the training
run's ``metrics.json`` (the placeholder ``mAP50`` / ``val_loss`` are
replaced) and drops a full ``eval_metrics.json`` next to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained ONNX model.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--kind", default="detector",
                   choices=["detector", "classifier", "fill"])
    p.add_argument("--data-root", default="./data/train")
    p.add_argument("--split-file", default="./data/splits/test.txt")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--name", default=None)
    p.add_argument("--device", default="auto",
                   help="auto|cuda|cpu - onnxruntime execution provider")
    p.add_argument("--no-sync", action="store_true")
    # real-eval options
    p.add_argument("--dataset-dir", default=None,
                   help="split folder holding annotations.json + images/")
    p.add_argument("--dataset-root", default=None,
                   help="dataset root (classifier / activity kind)")
    p.add_argument("--split", default="valid",
                   help="split name for --dataset-root (train|valid|test)")
    p.add_argument("--limit", type=int, default=0,
                   help="evaluate at most N images (0 = all)")
    p.add_argument("--activity-valid-frames", type=int, default=1500,
                   help="frames sampled from the valid split for the "
                        "activity head (0 = all)")
    p.add_argument("--conf", type=float, default=0.40,
                   help="operating-point score threshold")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--val-loss-frames", type=int, default=0,
                   help="frames used for the YOLOX validation loss "
                        "(0 = skip the loss)")
    p.add_argument("--update-run", default=None,
                   help="training run dir whose metrics.json is updated "
                        "with the measured values")
    return p


def _merge_into_run(run_dir: Path, metrics: dict) -> None:
    """Replace the placeholder metrics of a training run."""
    run_dir = Path(run_dir)
    metrics_file = run_dir / "metrics.json"
    existing: dict = {}
    if metrics_file.exists():
        try:
            existing = json.loads(metrics_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    merged = dict(existing)
    for key in ("mAP50", "mAP50_95", "val_loss", "val_accuracy",
                "precision_macro", "recall_macro", "f1_macro",
                "precision_micro", "recall_micro", "f1_micro",
                "exact_match_accuracy", "subset_accuracy",
                "accuracy", "per_class", "classes", "n_frames",
                "subset_size", "confusion_matrix",
                "gt_label_distribution"):
        if key in metrics and metrics[key] is not None:
            merged[key] = metrics[key]
    merged["evaluated"] = True
    merged["eval_split"] = metrics.get("split") or metrics.get("eval_split")
    merged["eval_device"] = metrics.get("device")
    metrics_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    (run_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Updated %s and wrote %s", metrics_file,
             run_dir / "eval_metrics.json")


def main(argv: list[str] | None = None) -> None:
    p = _build_parser()
    args = p.parse_args(argv)

    name = args.name or f"eval-{int(time.time())}"
    out = Path(args.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    from ml.eval import metrics as M

    device = args.device
    if device == "auto":
        device = os.getenv("INFERENCE_PROVIDER", "auto")

    if args.kind == "detector":
        split_dir = args.dataset_dir
        if not split_dir:
            p.error("--dataset-dir is required for --kind=detector "
                    "(the split folder with annotations.json)")
        log.info("Evaluating detector %s on %s (device=%s)",
                 args.model_path, split_dir, device)
        result = M.evaluate_detector(
            args.model_path, split_dir, device=device, conf=args.conf,
            iou=args.iou, limit=args.limit,
            val_loss_frames=args.val_loss_frames)
        result["kind"] = "detector"
        result["split"] = Path(split_dir).name
    else:
        root = args.dataset_root
        if not root:
            p.error("--dataset-root is required for --kind=classifier")
        if args.kind == "fill":
            log.info("Evaluating fill classifier %s on %s/%s (device=%s)",
                     args.model_path, root, args.split, device)
            result = M.evaluate_fill_classifier(
                args.model_path, root, split=args.split, device=device,
                limit=args.limit)
        else:
            log.info("Evaluating classifier %s on %s/%s (device=%s)",
                     args.model_path, root, args.split, device)
            result = M.evaluate_classifier(
                args.model_path, root, split=args.split, device=device,
                limit=args.activity_valid_frames)
        result["kind"] = args.kind

    result["device"] = device
    result["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "metrics.json").write_text(json.dumps(result, indent=2),
                                       encoding="utf-8")
    log.info("Wrote metrics to %s", out / "metrics.json")

    if args.update_run:
        _merge_into_run(Path(args.update_run), result)

    if not args.no_sync:
        log.info("Supabase sync skipped for eval (use ml.sync for runs).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
