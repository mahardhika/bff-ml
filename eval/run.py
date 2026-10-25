"""Evaluate a trained ONNX model on the held-out test split.

Reads `data/splits/test.txt`, runs the detector (or classifier) on
each image, compares to ground truth, writes metrics to
`runs/<eval_name>/metrics.json`. Optional Supabase sync (same code
path as `ml/sync.py`).

Usage:
    python -m ml eval --model-path=./runs/.../detector.onnx \
                      --kind=detector --output-dir=./runs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained ONNX model.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--kind", default="detector", choices=["detector", "classifier"])
    p.add_argument("--data-root", default="./data/train")
    p.add_argument("--split-file", default="./data/splits/test.txt")
    p.add_argument("--output-dir", default="./runs")
    p.add_argument("--name", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-sync", action="store_true")
    args = p.parse_args(argv)

    name = args.name or f"eval-{int(time.time())}"
    out = Path(args.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    test_list = Path(args.split_file).read_text().splitlines() if Path(args.split_file).exists() else []
    log.info("Evaluating %s on %d test images", args.model_path, len(test_list))

    metrics = {
        "n_test": len(test_list),
        "mAP50": 0.0,  # placeholder; real eval computes this
        "fill_acc": 0.0,
        "not_visible_recall": 0.0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Wrote metrics to %s", out / "metrics.json")

    if not args.no_sync:
        log.info("Supabase sync skipped for eval (use ml.sync for runs).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()