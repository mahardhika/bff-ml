"""Bucket-fill labels for the earthmoving dataset.

The earthmoving archive ships two things per split: a COCO export
(``<split>/annotations.json`` with an ``excavator`` box per frame) and
``activity.json`` with the temporal action segments of the excavator.

The fill head is trained on the *detector's* crop - the ``excavator``
box letterboxed to 224x224 - so the training distribution matches the
backend exactly. The label comes from the action segments.

Label derivation (deterministic, per frame)
-------------------------------------------
The excavator's segments are walked in temporal order and a *carry
state* is kept: what the bucket is holding right now. The state starts
at ``empty`` (a fresh bucket) and changes on

* ``load_bucket`` -> the bucket is filled -> ``full``
* ``dump``       -> the contents leave the bucket -> ``empty``

``swing_bucket`` and ``idle`` do not change the state, so they inherit
it. That is the disambiguation the raw action names need: the swing
that runs from the dig face to the truck carries a load (``full``),
the swing that runs back is empty (``empty``).

For a frame that several actions cover, the priority is

1. ``dump``        -> ``empty``  (the contents are leaving the bucket)
2. ``load_bucket`` -> ``full``   (the bucket was just loaded)
3. ``swing_bucket``-> carry state of that swing segment
4. ``idle``        -> carry state of that idle segment
5. ``move``        -> the frame is dropped (``move`` has 4 segments in
   the whole archive - it is not learnable and must not be faked)

The result is a binary target ``full`` (loaded) vs ``empty`` (not
loaded). The shipped head therefore has two classes; the product
vocabulary (``empty``/``partial``/``full``/``over``/``not_visible``) is
applied by the backend, and the classes this dataset cannot support
stay unreachable at inference.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: The binary target of the fill head.
#:
#: ``not_visible`` was added for the site footage: there the camera only
#: ever sees the bucket while it swings back (empty, the teeth are
#: visible) - a loaded bucket is turned away from the camera, so the
#: honest answer there is "not visible", not a coin flip. The backend
#: maps a head class named ``not_visible`` onto the product vocabulary
#: (``_CLASS_ALIASES`` in ``backend/core/classifier.py``).
#: The fill label is the STATE ("what the bucket holds"), not a reading
#: of the visible bucket content. Two classes on purpose: the state
#: machine leaves DIG only when the fill class is "full"/"over", so a
#: third class ("not_visible") would silently stop the pass counter.
#: The backend maps an unknown class name to "not_visible", which is the
#: right answer for the product - but it is not a training label.
FILL_CLASSES: tuple[str, ...] = ("empty", "full")

#: ``<video>_<frame:05d>.jpg`` - the frame name used by the archive.
_FRAME_RE = re.compile(r"^(\d+)_(\d+)$")

#: Actions the label derivation understands. ``move`` is deliberately
#: absent: those frames are dropped instead of being guessed.
_ACTIONS: tuple[str, ...] = ("idle", "swing_bucket", "load_bucket", "dump",
                             "move")


@dataclass(frozen=True)
class FillSample:
    """One excavator crop with its fill label."""

    file_name: str
    video_id: str
    frame_index: int
    #: Index into :data:`FILL_CLASSES`.
    label: int
    #: The ``excavator`` box (xyxy) in source pixels.
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class _Segment:
    start: int
    end: int
    action: str
    #: Carry state when the segment starts.
    carry: str


class _VideoLabels:
    """Fill label per frame of one video."""

    def __init__(self, segments: Sequence[dict]) -> None:
        ordered = sorted(
            (_Segment(int(s.get("start_frame", 0)),
                      int(s.get("end_frame", -1)),
                      str(s.get("action") or ""),
                      "") for s in segments),
            key=lambda s: (s.start, s.end),
        )
        state = "empty"
        prepared: list[_Segment] = []
        for seg in ordered:
            prepared.append(_Segment(seg.start, seg.end, seg.action, state))
            if seg.action == "load_bucket":
                state = "full"
            elif seg.action == "dump":
                state = "empty"
        self._segments = prepared
        # state timeline: (frame, state) after every load/dump
        timeline: list[tuple[int, str]] = []
        state = "empty"
        for seg in prepared:
            if seg.action == "load_bucket":
                state = "full"
            elif seg.action == "dump":
                state = "empty"
            else:
                continue
            timeline.append((seg.start, state))
        self._timeline = timeline

    def _covering(self, frame: int, action: str) -> _Segment | None:
        for seg in self._segments:
            if seg.action != action:
                continue
            if seg.start <= frame <= seg.end:
                return seg
        return None

    def _state_at(self, frame: int) -> str:
        state = "empty"
        for start, value in self._timeline:
            if start > frame:
                break
            state = value
        return state

    def label(self, frame: int) -> str | None:
        """``"full"`` / ``"empty"`` for a frame, ``None`` when unusable."""
        if self._covering(frame, "dump") is not None:
            return "empty"
        if self._covering(frame, "load_bucket") is not None:
            return "full"
        swing = self._covering(frame, "swing_bucket")
        if swing is not None:
            return swing.carry
        idle = self._covering(frame, "idle")
        if idle is not None:
            return idle.carry
        if self._covering(frame, "move") is not None:
            return None
        return self._state_at(frame)


def _legend_classes(activity: dict) -> tuple[str, ...]:
    legend = activity.get("legend") or {}
    items: list[tuple[int, str]] = []
    for key, name in legend.items():
        try:
            items.append((int(key), str(name)))
        except (TypeError, ValueError):
            continue
    items.sort()
    return tuple(name for _, name in items)


def _excavator_segments(activity: dict, video_id: str) -> list[dict]:
    """Action segments of the excavator of one video."""
    videos = activity.get("videos") or {}
    objects = videos.get(video_id) or {}
    for name in ("excavator", "Excavator", "excavator 1"):
        segs = objects.get(name)
        if segs:
            return list(segs)
    # Fall back to the object with the most segments - in this archive
    # that is always the excavator.
    best: list[dict] = []
    for segs in objects.values():
        if segs and len(segs) > len(best):
            best = list(segs)
    return best


def _resolve_image(root: Path, split: str, file_name: str) -> Path:
    """Find an image in the split folder or its ``images/`` subfolder."""
    name = Path(file_name).name
    for candidate in (root / split / file_name, root / split / "images" / name,
                      root / split / "images" / file_name, root / file_name):
        if candidate.exists():
            return candidate
    return root / split / "images" / name


def load_fill_samples(root: Path, split: str = "train", limit: int = 0,
                      bucket_label: str = "excavator"
                      ) -> list[FillSample]:
    """Fill-labelled excavator crops of one split.

    Returns the samples in dataset order (sorted by video and frame).
    ``limit`` cuts the list down to an evenly spread subset.
    """
    root = Path(root)
    activity_path = root / "activity.json"
    # A per-frame label sidecar replaces the timeline: it is what the
    # vision review of the site footage produces (there the timeline
    # cannot be derived - the footage repeats a dig cycle and a single
    # "load_bucket" segment would mark the rest of the clip full).
    label_path = root / "labels.json"
    explicit: dict[str, str] = {}
    if label_path.exists():
        explicit = {str(k): str(v) for k, v in
                    json.loads(label_path.read_text(encoding="utf-8")).items()}
    if not activity_path.exists() and not explicit:
        raise FileNotFoundError(f"no activity.json under {root}")
    activity: dict = {}
    if activity_path.exists():
        activity = json.loads(activity_path.read_text(encoding="utf-8"))

    ann_path = root / split / "annotations.json"
    if not ann_path.exists():
        ann_path = root / "annotations.json"
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    categories = {int(c.get("id", i)): str(c.get("name") or "")
                  for i, c in enumerate(data.get("categories") or [])}
    bucket_ids = {cid for cid, name in categories.items()
                  if name.lower() == bucket_label.lower()}
    if not bucket_ids:
        raise ValueError(f"no category named {bucket_label!r} in {ann_path}")

    boxes: dict[int, tuple[float, float, float, float]] = {}
    for ann in data.get("annotations") or []:
        if int(ann.get("category_id", 0)) not in bucket_ids:
            continue
        x, y, w, h = (float(v) for v in ann.get("bbox") or (0, 0, 0, 0))
        boxes[int(ann.get("image_id", 0))] = (x, y, x + w, y + h)

    labels_cache: dict[str, _VideoLabels] = {}
    samples: list[FillSample] = []
    dropped_move = 0
    dropped_no_box = 0
    for image in data.get("images") or []:
        image_id = int(image.get("id", 0))
        bbox = boxes.get(image_id)
        if bbox is None:
            dropped_no_box += 1
            continue
        file_name = str(image.get("file_name") or "")
        stem = Path(file_name).stem
        match = _FRAME_RE.match(stem)
        if match is None:
            dropped_no_box += 1
            continue
        video_id, frame_text = match.group(1), match.group(2)
        frame_index = int(frame_text)
        key = f"{video_id}_{frame_index:06d}"
        if key in explicit:
            name = explicit[key]
            if name not in FILL_CLASSES:
                raise ValueError(f"label {name!r} for {key} is not one of "
                                 f"{FILL_CLASSES}")
            label = name
        else:
            video_labels = labels_cache.get(video_id)
            if video_labels is None:
                video_labels = _VideoLabels(_excavator_segments(activity, video_id))
                labels_cache[video_id] = video_labels
            label = video_labels.label(frame_index)
        if label is None:
            dropped_move += 1
            continue
        samples.append(FillSample(
            file_name=file_name,
            video_id=video_id,
            frame_index=frame_index,
            label=FILL_CLASSES.index(label),
            bbox=bbox,
        ))

    if limit and limit > 0 and len(samples) > limit:
        stride = len(samples) / float(limit)
        samples = [samples[int(i * stride)] for i in range(limit)]
    counts = fill_distribution(samples)
    log.info("Fill dataset %s/%s: %d crops %s (no bucket box: %d, "
             "move dropped: %d)", root, split, len(samples), counts,
             dropped_no_box, dropped_move)
    return samples


def fill_distribution(samples: Sequence[FillSample]) -> dict[str, int]:
    """How many crops carry each fill class."""
    counts = {name: 0 for name in FILL_CLASSES}
    for sample in samples:
        if 0 <= sample.label < len(FILL_CLASSES):
            counts[FILL_CLASSES[sample.label]] += 1
    return counts


class FillCrops:
    """torch Dataset of letterboxed excavator crops.

    The crop mirrors the backend exactly: the ``excavator`` box is cut
    out of the frame and letterboxed to ``input_size`` with the same
    padding and colour order the inference code uses
    (``ml.data.augment.letterbox`` == ``backend.core.preprocess``).
    """

    def __init__(self, root: Path, samples: Sequence[FillSample],
                 input_size: int = 224, train: bool = False,
                 split: str = "train") -> None:
        self.root = Path(root)
        self.samples = list(samples)
        self.input_size = input_size
        self.train = train
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import torch

        from ml.data.augment import letterbox

        sample = self.samples[idx]
        path = _resolve_image(self.root, self.split, sample.file_name)
        import cv2
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        x1, y1, x2, y2 = sample.bbox
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))
        crop = image[y1:y2, x1:x2]
        canvas, _scale = letterbox(crop, input_size=self.input_size)
        if self.train:
            # The old code flipped EVERY image on EVERY epoch: the model
            # saw exactly the same tensors again and again, which is why
            # the shipped head reached train_loss 0.015 after six epochs
            # and then overfit. Randomise it, and add the variations the
            # site footage actually has (dust, motion blur, exposure).
            import random

            if random.random() < 0.5:
                canvas = canvas[:, ::-1].copy()
            from ml.data.augment import hsv_jitter
            canvas = hsv_jitter(canvas, hgain=0.01, sgain=0.25, vgain=0.25)
            if random.random() < 0.25:
                k = random.choice((3, 5))
                canvas = cv2.GaussianBlur(canvas, (k, k), 0)
        tensor = (torch.from_numpy(canvas[:, :, ::-1].copy())
                  .permute(2, 0, 1).float() / 255.0)
        return {
            "image": tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "path": str(path),
        }


def collate_fill(batch: list[dict]) -> dict:
    import torch

    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
    }
