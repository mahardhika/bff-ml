"""Pull labeled data from a CVAT instance and convert to YOLO format.

Workflow:
1. Hit CVAT's REST endpoints to list tasks / download the COCO export
   for the project.
2. Unzip into a folder of `images/` + `labels/` (YOLO .txt per image).
3. Hand off to `ml.data.prepare` for the YOLO -> COCO step the trainer
   wants (we keep the canonical training format separate from CVAT's
   intermediate one).

Configuration via env:
    CVAT_BASE_URL   e.g. http://localhost:8080
    CVAT_USERNAME   admin
    CVAT_PASSWORD   admin
    CVAT_PROJECT    7         # project ID (or use --task below)

Usage:
    python -m ml.data.cvat --out ./data/cvat_export
    python -m ml.data.prepare --src ./data/cvat_export --dst ./data/train \
        --classes ./data/cvat_export/classes.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)


def _auth_headers() -> dict[str, str]:
    """Build the auth headers for CVAT (Basic auth via session token)."""
    base = os.getenv("CVAT_BASE_URL", "http://localhost:8080")
    user = os.getenv("CVAT_USERNAME", "admin")
    pw = os.getenv("CVAT_PASSWORD", "admin")

    # CVAT supports both Basic and session token. We use the simple
    # login flow: POST /api/auth/login with form data, keep the cookie.
    import requests  # type: ignore
    s = requests.Session()
    s.post(f"{base}/api/auth/login",
           data={"username": user, "password": pw}).raise_for_status()
    return {"session-cookie": "; ".join(
        f"{k}={v}" for k, v in s.cookies.items())}, s


def download_task_coco(task_id: int, out_dir: Path) -> Path:
    """Download a CVAT task dump as COCO JSON; returns the unzipped dir."""
    import requests  # type: ignore
    base = os.getenv("CVAT_BASE_URL", "http://localhost:8080")
    _, s = _auth_headers()

    # CVAT `GET /api/tasks/{id}/dataset?format=COCO%201.0` returns a zip.
    r = s.get(f"{base}/api/tasks/{task_id}/dataset",
              params={"format": "COCO 1.0"}, stream=True)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        for chunk in r.iter_content(chunk_size=1 << 14):
            tmp.write(chunk)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp.name) as z:
        z.extractall(out_dir)
    Path(tmp.name).unlink(missing_ok=True)
    log.info("CVAT task %d dumped to %s", task_id, out_dir)
    return out_dir


def coco_to_yolo(coco_path: Path, out_dir: Path) -> Path:
    """Convert a CVAT-exported COCO JSON to YOLO format + classes.txt.

    Side effect: writes `classes.txt` and one `.txt` per image into `out_dir`.
    """
    coco_data = json.loads(coco_path.read_text(encoding="utf-8"))
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(coco_data["categories"])}
    classes = [c["name"] for c in coco_data["categories"]]
    img_id_to_meta = {im["id"]: im for im in coco_data["images"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    label_dir = out_dir / "labels"
    label_dir.mkdir(exist_ok=True)
    for ann in coco_data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        meta = img_id_to_meta.get(ann["image_id"])
        if not meta:
            continue
        x, y, bw, bh = ann["bbox"]
        cx = (x + bw / 2) / meta["width"]
        cy = (y + bh / 2) / meta["height"]
        nw = bw / meta["width"]
        nh = bh / meta["height"]
        cls = cat_id_to_idx[ann["category_id"]]
        out_file = label_dir / f"{Path(meta['file_name']).stem}.txt"
        with out_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    log.info("Wrote %d label files (classes: %s)", len(list(label_dir.glob("*.txt"))), classes)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pull labels from CVAT and convert to YOLO.")
    p.add_argument("--out", required=True, help="output folder for images/ + labels/")
    p.add_argument("--task", type=int, default=int(os.getenv("CVAT_TASK", "0")) or None,
                   help="CVAT task id (overrides CVAT_TASK env)")
    p.add_argument("--format", default="COCO 1.0", help="CVAT dump format")
    args = p.parse_args(argv)

    if not args.task:
        log.error("No task id. Set CVAT_TASK or pass --task N.")
        return

    out_dir = Path(args.out)
    raw = download_task_coco(args.task, out_dir / "_cvat_raw")
    coco_json = next(raw.glob("*.json"), None) or next(raw.glob("annotations/*.json"), None)
    if coco_json is None:
        log.error("No COCO JSON in CVAT dump; aborting.")
        shutil.rmtree(raw, ignore_errors=True)
        return

    coco_to_yolo(coco_json, out_dir)
    shutil.rmtree(raw, ignore_errors=True)
    log.info("Done. Next: python -m ml.data.prepare --src %s --dst ... --classes %s",
             out_dir, out_dir / "classes.txt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()