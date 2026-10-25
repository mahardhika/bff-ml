"""Dataset loaders with local-cache deduplication.

`LocalDataset` reads from a CVAT COCO export on disk.
`S3Dataset` downloads a Supabase Storage prefix to `data/cache/<hash>/`.
`UrlDataset` (v0.2) downloads a public URL.

The cache is content-addressed: a stable hash of the source identifier
becomes the directory name, so two training runs on the same data
share a single download.

Cache lifetime (see `system-map-arsitektur.MD` §5.22.1): a cached copy
expires after ``CACHE_TTL_DAYS`` (default 7). Once expired the next
`sync()` re-downloads. ``force=True`` skips the cache entirely.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

log = logging.getLogger(__name__)

#: How long a downloaded cache stays valid, in days.
CACHE_TTL_DAYS = 7

#: Seconds in a day.
_SECONDS_PER_DAY = 86400.0


@dataclass
class Dataset:
    """Resolved dataset on disk. Iterable yields (image_path, label_dict)."""
    root: Path
    kind: str  # "coco" | "yolo" | "raw"
    classes: Sequence[str] = ()
    extra: dict | None = None

    def __iter__(self) -> Iterator[tuple[Path, dict]]:
        if self.kind == "coco":
            yield from _iter_coco(self.root)
        elif self.kind == "yolo":
            yield from _iter_yolo(self.root, self.classes)
        else:
            raise ValueError(f"unsupported dataset kind: {self.kind!r}")


def load_dataset(source: str, **kwargs) -> Dataset:
    """Factory: dispatch to the right loader."""
    if source == "local":
        return LocalDataset(**kwargs).load()
    if source == "s3":
        return S3Dataset(**kwargs).sync()
    if source == "db":
        return DbDataset(**kwargs).sync()
    if source == "url":
        return UrlDataset(**kwargs).sync()
    raise ValueError(f"unknown data source: {source!r}")


# --- Cache freshness -------------------------------------------------------


def cache_age_s(cache_dir: Path) -> float:
    """Age of a cache directory in seconds, or ``inf`` when it is empty.

    Measured from the newest mtime inside the directory, so a partially
    copied download does not look fresh just because its root folder was
    touched.
    """
    if not cache_dir.exists():
        return float("inf")
    newest = 0.0
    for p in cache_dir.rglob("*"):
        try:
            st = p.stat()
        except OSError:
            continue
        newest = max(newest, st.st_mtime)
    if newest == 0.0:
        return float("inf")
    return max(0.0, time.time() - newest)


def is_cache_fresh(cache_dir: Path, ttl_days: float = CACHE_TTL_DAYS) -> bool:
    """True when the cache exists, has content, and is younger than the TTL."""
    if not cache_dir.exists():
        return False
    if not any(cache_dir.iterdir()):
        return False
    if ttl_days <= 0:
        return False
    return cache_age_s(cache_dir) < ttl_days * _SECONDS_PER_DAY


def _purge(cache_dir: Path) -> None:
    """Remove a cache directory, ignoring a missing path."""
    shutil.rmtree(cache_dir, ignore_errors=True)


# --- Source classes ---


class LocalDataset:
    def __init__(self, path: str, kind: str = "coco") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.kind = kind

    def load(self) -> Dataset:
        classes = _read_classes(self.path, self.kind)
        return Dataset(root=self.path, kind=self.kind, classes=classes)


class S3Dataset:
    def __init__(self, bucket: str, prefix: str,
                 cache_dir: str = "./data/cache", kind: str = "coco",
                 force: bool = False,
                 ttl_days: float = CACHE_TTL_DAYS) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.cache = Path(cache_dir) / _hash(f"{bucket}/{self.prefix}")
        self.kind = kind
        self.force = force
        self.ttl_days = ttl_days

    def sync(self) -> Dataset:
        if self.force:
            log.info("force-download: purging cache %s", self.cache)
            _purge(self.cache)

        self.cache.mkdir(parents=True, exist_ok=True)

        if is_cache_fresh(self.cache, self.ttl_days):
            log.info("S3 cache hit (age %.1fh): %s",
                     cache_age_s(self.cache) / 3600.0, self.cache)
            return Dataset(root=self.cache, kind=self.kind,
                           classes=_read_classes(self.cache, self.kind))

        if any(self.cache.iterdir()):
            log.warning(
                "S3 cache stale (age %.1fh, ttl %.1fh) - re-downloading: %s",
                cache_age_s(self.cache) / 3600.0, self.ttl_days, self.cache,
            )
            _purge(self.cache)
            self.cache.mkdir(parents=True, exist_ok=True)

        self._download_prefix()
        return Dataset(root=self.cache, kind=self.kind,
                       classes=_read_classes(self.cache, self.kind))

    def _download_prefix(self) -> None:
        """Stream-copy a prefix from Supabase Storage to local cache."""
        from bff_core.storage import StorageClient

        client = StorageClient.from_env()
        if client is None:
            raise RuntimeError("storage is not configured; cannot download dataset")
        prefix = self.prefix.rstrip("/")
        for obj in client.list_objects(self.bucket, prefix):
            key = obj["Key"]
            rel = key[len(prefix) + 1:] if key.startswith(prefix) else key
            if not rel:
                continue
            target = self.cache / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and obj.get("Size") and target.stat().st_size == obj["Size"]:
                continue
            client.download_file(self.bucket, key, target)
        log.info("Downloaded %s/%s/ -> %s", self.bucket, self.prefix, self.cache)


class DbDataset:
    """Resolve ``bff_dataset_items`` and cache blobs locally as COCO."""

    def __init__(self, dataset_id: int, cache_dir: str = "./data/cache",
                 kind: str = "coco", force: bool = False,
                 ttl_days: float = CACHE_TTL_DAYS,
                 split: str | None = None) -> None:
        self.dataset_id = int(dataset_id)
        self.cache = Path(cache_dir) / f"dataset-{self.dataset_id}"
        self.kind = kind
        self.force = force
        self.ttl_days = ttl_days
        self.split = split

    def sync(self) -> Dataset:
        if self.force:
            log.info("force-download: purging cache %s", self.cache)
            _purge(self.cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        if is_cache_fresh(self.cache, self.ttl_days) and (self.cache / "annotations.json").exists():
            log.info("DB cache hit (age %.1fh): %s",
                     cache_age_s(self.cache) / 3600.0, self.cache)
            return Dataset(root=self.cache, kind=self.kind,
                           classes=_read_classes(self.cache, self.kind))
        self._materialize()
        return Dataset(root=self.cache, kind=self.kind,
                       classes=_read_classes(self.cache, self.kind))

    def _materialize(self) -> None:
        from bff_core.storage import StorageClient, parse_storage_path
        from supabase import create_client  # type: ignore

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY required for --data-source=db")
        sb = create_client(url, key)
        q = sb.table("bff_dataset_items").select("*").eq("dataset_id", self.dataset_id)
        if self.split:
            q = q.eq("split", self.split)
        res = q.execute()
        items = res.data or []
        if not items:
            raise FileNotFoundError(
                f"no items in bff_dataset_items for dataset_id={self.dataset_id}"
            )
        client = StorageClient.from_env()
        if client is None:
            raise RuntimeError("storage is not configured; cannot download dataset items")

        images_dir = self.cache / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        coco_images = []
        coco_anns = []
        coco_cats: dict[int, str] = {}
        ann_id = 1
        for i, item in enumerate(items, start=1):
            storage_path = item.get("storage_path") or ""
            bucket, key = parse_storage_path(storage_path)
            ext = Path(key).suffix or Path(item.get("item_key") or "frame.jpg").suffix or ".jpg"
            dest = images_dir / f"{item.get('sha256') or i}{ext}"
            if not dest.exists() or dest.stat().st_size == 0:
                client.download_file(bucket, key, dest)
            rel = dest.relative_to(self.cache).as_posix()
            width = int(item.get("width") or 0)
            height = int(item.get("height") or 0)
            coco_images.append({
                "id": i,
                "file_name": rel,
                "width": width,
                "height": height,
            })
            for box in item.get("labels") or []:
                bbox = box.get("bbox") or [0, 0, 0, 0]
                x1, y1, x2, y2 = (bbox + [0, 0, 0, 0])[:4]
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                cat_id = int(box.get("category_id") or 0)
                coco_cats.setdefault(cat_id, box.get("category") or f"class_{cat_id}")
                coco_anns.append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": cat_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1
        categories = [{"id": cid, "name": name} for cid, name in sorted(coco_cats.items())]
        if not categories:
            categories = [{"id": 0, "name": "bucket"}, {"id": 1, "name": "truck"},
                          {"id": 2, "name": "person"}]
        import json
        (self.cache / "annotations.json").write_text(json.dumps({
            "images": coco_images,
            "annotations": coco_anns,
            "categories": categories,
        }), encoding="utf-8")
        log.info("Materialized dataset %s (%d items) -> %s",
                 self.dataset_id, len(items), self.cache)


class UrlDataset:
    """v0.2 candidate - public URL download."""

    def __init__(self, url: str, cache_dir: str = "./data/cache",
                 kind: str = "coco", force: bool = False,
                 ttl_days: float = CACHE_TTL_DAYS) -> None:
        self.url = url
        self.cache = Path(cache_dir) / _hash(url)
        self.kind = kind
        self.force = force
        self.ttl_days = ttl_days

    def sync(self) -> Dataset:
        raise NotImplementedError("UrlDataset is deferred to v0.2")


# --- internals ---


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _read_classes(root: Path, kind: str) -> Sequence[str]:
    if kind == "coco":
        import json
        ann = root / "annotations.json"
        if not ann.exists():
            ann = next((p for p in root.rglob("*.json")
                        if "instances" in p.name.lower()), None)
        if ann is None:
            return ()
        data = json.loads(ann.read_text())
        cats = data.get("categories", [])
        return tuple(c["name"] for c in sorted(cats, key=lambda c: c["id"]))
    if kind == "yolo":
        names = root / "classes.txt"
        if names.exists():
            return tuple(line.strip() for line in names.read_text().splitlines() if line.strip())
    return ()


def _iter_coco(root: Path) -> Iterator[tuple[Path, dict]]:
    import json
    ann_path = root / "annotations.json"
    if not ann_path.exists():
        ann_path = next((p for p in root.rglob("*.json")
                         if "instances" in p.name.lower()), None)
    if ann_path is None:
        raise FileNotFoundError("no COCO annotations.json found")
    data = json.loads(ann_path.read_text())
    img_by_id = {im["id"]: im for im in data["images"]}
    ann_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)
    for image_id, image in img_by_id.items():
        img_path = root / image["file_name"]
        if not img_path.exists():
            img_path = (root / image["file_name"].split("/")[-1])
        if not img_path.exists():
            continue
        anns = ann_by_image.get(image_id, [])
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            labels.append({
                "category_id": a["category_id"],
                "bbox": [x, y, x + w, y + h],
            })
        yield img_path, {"boxes": labels, "width": image["width"], "height": image["height"]}


def _iter_yolo(root: Path, classes: Sequence[str]) -> Iterator[tuple[Path, dict]]:
    img_dir = root / "images"
    lbl_dir = root / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError("YOLO dataset needs images/ and labels/ dirs")
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        labels = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:5])
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                labels.append({"category_id": cls_id, "bbox": [x1, y1, x2, y2]})
        yield img_path, {"boxes": labels, "width": 0, "height": 0}