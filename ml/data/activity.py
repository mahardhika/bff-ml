"""Image-level activity labels (multi-label) for the public datasets.

Two shapes are supported.

``activity.json`` (earthmoving archive)
    The archive ships temporal action segments per object:

    .. code-block:: json

        {
          "legend": {"0": "idle", "1": "swing_bucket", ...},
          "videos": {"104154": {"excavator": [
              {"start_frame": 1, "end_frame": 32, "action": "load_bucket"}, ...]}}
        }

    A frame can carry several actions at once (the machine moves while
    the bucket swings), so the label is a multi-hot vector.

COCO categories (Roboflow EXCAVATOR export)
    That export has no timeline: the activity state is the category of
    the annotation, so one label is read from the COCO file.

The labels are read from the dataset root; the split folders hold the
images and the COCO file.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: ``<video>_<frame:05d>.jpg`` - the frame name used by the archive.
_FRAME_RE = re.compile(r"^(\d+)_(\d+)$")


@dataclass(frozen=True)
class ActivitySample:
    """One frame with the actions that are active in it."""

    file_name: str
    video_id: str
    frame_index: int
    labels: tuple[int, ...]


def resolve_root(path: Path) -> tuple[Path, str]:
    """Find the dataset root and the split name for ``path``.

    Accepts the dataset root (``.../earthmoving-equipment``) or a split
    folder (``.../earthmoving-equipment/train``).
    """
    path = Path(path)
    if (path / "annotations.json").exists():
        return path.parent, path.name
    if (path / "train" / "annotations.json").exists():
        return path, "train"
    return path, "train"


def _legend_classes(activity: dict) -> tuple[str, ...]:
    """Class names ordered by action id."""
    legend = activity.get("legend") or {}
    items = []
    for key, name in legend.items():
        try:
            items.append((int(key), str(name)))
        except (TypeError, ValueError):
            continue
    items.sort()
    return tuple(name for _, name in items)


def _temporal_samples(root: Path, split: str, limit: int
                      ) -> tuple[list[ActivitySample], tuple[str, ...]]:
    """Multi-hot labels from the temporal action segments."""
    activity = json.loads((root / "activity.json").read_text(encoding="utf-8"))
    classes = _legend_classes(activity)
    index = {name: i for i, name in enumerate(classes)}
    videos = activity.get("videos") or {}

    ann_path = root / split / "annotations.json"
    if not ann_path.exists():
        ann_path = root / "annotations.json"
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    samples: list[ActivitySample] = []
    for image in data.get("images") or []:
        name = str(image.get("file_name") or "")
        stem = Path(name).stem
        match = _FRAME_RE.match(stem)
        if match is None:
            continue
        video_id, frame_text = match.group(1), match.group(2)
        frame_index = int(frame_text)
        labels: set[int] = set()
        for segments in (videos.get(video_id) or {}).values():
            for segment in segments or []:
                start = int(segment.get("start_frame", 0))
                end = int(segment.get("end_frame", -1))
                if start <= frame_index <= end:
                    action = str(segment.get("action") or "")
                    if action in index:
                        labels.add(index[action])
        if not labels:
            # No segment covers this frame: the machine is idle.
            if "idle" in index:
                labels.add(index["idle"])
        samples.append(ActivitySample(name, video_id, frame_index,
                                      tuple(sorted(labels))))
    return _spread(samples, limit), classes


def _coco_samples(root: Path, split: str, limit: int
                  ) -> tuple[list[ActivitySample], tuple[str, ...]]:
    """Single-label samples from the COCO categories of a split."""
    ann_path = root / split / "annotations.json"
    if not ann_path.exists():
        ann_path = root / "annotations.json"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = sorted(data.get("categories") or [],
                        key=lambda c: int(c.get("id", 0)))
    classes = tuple(str(c.get("name") or f"class_{c.get('id')}") for c in categories)
    index = {int(c.get("id", i)): i for i, c in enumerate(categories)}

    per_image: dict[int, set[int]] = {}
    for ann in data.get("annotations") or []:
        raw = int(ann.get("category_id", 0))
        per_image.setdefault(int(ann.get("image_id", 0)), set()).add(
            index.get(raw, raw))

    samples: list[ActivitySample] = []
    for image in data.get("images") or []:
        labels = sorted(per_image.get(int(image.get("id", 0)), set()))
        if not labels:
            continue
        name = str(image.get("file_name") or "")
        stem = Path(name).stem
        match = _FRAME_RE.match(stem)
        video_id = match.group(1) if match else ""
        frame_index = int(match.group(2)) if match else 0
        samples.append(ActivitySample(name, video_id, frame_index, tuple(labels)))
    return _spread(samples, limit), classes


def _spread(samples: list[ActivitySample], limit: int) -> list[ActivitySample]:
    """Cut the list down to ``limit`` entries, evenly spread.

    The frames are ordered by video, so the first N of them would all
    come from one video. The stride keeps every video in the sample.
    """
    if not limit or limit <= 0 or len(samples) <= limit:
        return samples
    stride = len(samples) / float(limit)
    return [samples[int(i * stride)] for i in range(limit)]


def load_activity_samples(root: Path, split: str = "train", limit: int = 0
                          ) -> tuple[list[ActivitySample], tuple[str, ...]]:
    """Samples and class names for one split of an activity dataset."""
    root = Path(root)
    if (root / "activity.json").exists():
        samples, classes = _temporal_samples(root, split, limit)
    else:
        samples, classes = _coco_samples(root, split, limit)
    if not samples:
        log.warning("No activity samples found in %s (%s)", root, split)
    else:
        log.info("Activity dataset %s/%s: %d frames, %d classes -> %s",
                 root, split, len(samples), len(classes), list(classes))
    return samples, classes


def label_distribution(samples: Sequence[ActivitySample], num_classes: int
                       ) -> list[int]:
    """How many frames carry each class."""
    counts = [0] * num_classes
    for sample in samples:
        for label in sample.labels:
            if 0 <= label < num_classes:
                counts[label] += 1
    return counts
