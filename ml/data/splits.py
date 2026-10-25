"""Train/val/test splitter with deterministic seed.

Given a COCO dataset on disk, emit three text files of image paths:
    train.txt, val.txt, test.txt
Default split: 70 / 15 / 15.

Usage:
    python -m ml.data.splits --src ./data/train --dst ./data/splits --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Deterministic train/val/test split.")
    p.add_argument("--src", required=True, help="dataset root (with annotations.json)")
    p.add_argument("--dst", required=True, help="where to write train/val/test.txt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", default="0.70,0.15,0.15",
                   help="comma-separated floats summing to 1.0")
    args = p.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    ratios = [float(x) for x in args.ratios.split(",")]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    ann = json.loads((src / "annotations.json").read_text())
    images = [im["file_name"] for im in ann["images"]]
    rng = random.Random(args.seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]

    (dst / "train.txt").write_text("\n".join(train))
    (dst / "val.txt").write_text("\n".join(val))
    (dst / "test.txt").write_text("\n".join(test))
    log.info("Split %d images: train=%d val=%d test=%d",
             n, len(train), len(val), len(test))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()