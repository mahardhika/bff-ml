"""Activity labels: temporal segments and the COCO fallback."""

import json
from pathlib import Path

from ml.data.activity import label_distribution, load_activity_samples, resolve_root


def _write_split(root: Path, split: str, frames: list[str]) -> None:
    folder = root / split
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "annotations.json").write_text(json.dumps({
        "images": [{"id": i + 1, "file_name": f"images/{name}", "width": 4,
                    "height": 4} for i, name in enumerate(frames)],
        "categories": [{"id": 1, "name": "excavator"}],
        "annotations": [],
    }), encoding="utf-8")


def test_temporal_labels_are_multi_hot(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", ["104154_00010.jpg", "104154_00020.jpg"])
    (tmp_path / "activity.json").write_text(json.dumps({
        "legend": {"0": "idle", "1": "swing_bucket", "2": "load_bucket",
                   "3": "dump", "4": "move"},
        "videos": {"104154": {"excavator": [
            {"start_frame": 1, "end_frame": 15, "action": "load_bucket"},
            {"start_frame": 16, "end_frame": 30, "action": "swing_bucket"},
        ]}},
    }), encoding="utf-8")

    samples, classes = load_activity_samples(tmp_path, "train")
    assert classes == ("idle", "swing_bucket", "load_bucket", "dump", "move")
    by_frame = {s.frame_index: s.labels for s in samples}
    assert by_frame[10] == (2,)            # load_bucket
    assert by_frame[20] == (1,)            # swing_bucket
    assert label_distribution(samples, len(classes)) == [0, 1, 1, 0, 0]


def test_frames_without_a_segment_fall_back_to_idle(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", ["104154_00099.jpg"])
    (tmp_path / "activity.json").write_text(json.dumps({
        "legend": {"0": "idle", "1": "swing_bucket"},
        "videos": {"104154": {"excavator": [
            {"start_frame": 1, "end_frame": 10, "action": "swing_bucket"},
        ]}},
    }), encoding="utf-8")

    samples, classes = load_activity_samples(tmp_path, "train")
    assert classes == ("idle", "swing_bucket")
    assert samples[0].labels == (0,)


def test_coco_categories_are_used_when_no_timeline_exists(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", ["frame_00001.jpg"])
    folder = tmp_path / "train"
    (folder / "annotations.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "images/frame_00001.jpg", "width": 4,
                    "height": 4}],
        "categories": [{"id": 1, "name": "idle"}, {"id": 2, "name": "dump"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 2,
                         "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}],
    }), encoding="utf-8")

    samples, classes = load_activity_samples(tmp_path, "train")
    assert classes == ("idle", "dump")
    assert samples[0].labels == (1,)


def test_resolve_root_accepts_the_root_or_a_split(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", ["a.jpg"])
    root, split = resolve_root(tmp_path)
    assert root == tmp_path and split == "train"
    root, split = resolve_root(tmp_path / "train")
    assert root == tmp_path and split == "train"
