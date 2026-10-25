"""Convert a raw labeled folder into a CVAT-compatible COCO export.

Most users start with manually-annotated still frames in a folder of
`img_001.jpg` + `img_001.txt` (YOLO format). This script groups them
into the COCO JSON the training pipeline expects.

Usage:
    python -m ml.data.prepare --src ./data/raw --dst ./data/train
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Convert raw YOLO -> COCO format.")
    p.add_argument("--src", required=True, help="source folder (images/ + labels/)")
    p.add_argument("--dst", required=True, help="destination folder")
    p.add_argument("--classes", required=True, help="path to classes.txt")
    args = p.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    classes = [l.strip() for l in Path(args.classes).read_text().splitlines() if l.strip()]

    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    img_dir = src / "images"
    lbl_dir = src / "labels"
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        h, w = _probe_size(img_path)
        images.append({
            "id": img_id, "file_name": img_path.name,
            "width": w, "height": h,
            "license": 1, "date_captured": datetime.utcnow().isoformat() + "Z",
        })
        lbl = lbl_dir / (img_path.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = (float(x) for x in parts[1:5])
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cls + 1,
                    "bbox": [x, y, bw * w, bh * h],
                    "area": bw * w * bh * h,
                    "iscrowd": 0, "segmentation": [],
                })
                ann_id += 1
        img_id += 1
        # Copy image to dst.
        (dst / img_path.name).write_bytes(img_path.read_bytes())

    coco = {
        "info": {"description": "BFF training set", "version": "0.1.0",
                 "date_created": datetime.utcnow().isoformat() + "Z"},
        "licenses": [{"id": 1, "name": "Internal", "url": ""}],
        "categories": [{"id": i + 1, "name": n, "supercategory": "bff"}
                       for i, n in enumerate(classes)],
        "images": images, "annotations": annotations,
    }
    (dst / "annotations.json").write_text(json.dumps(coco, indent=2))
    log.info("Wrote %d images, %d annotations to %s",
             len(images), len(annotations), dst)


def _probe_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size[1], im.size[0]
    except Exception:
        return (0, 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()