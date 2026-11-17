"""Activity samples the in-app labeler writes into the database."""

from __future__ import annotations

import json

from ml.data.activity import ACTIVITY_CLASSES, load_db_activity_samples


def _write(root, items) -> None:
    (root / "activity_labels.json").write_text(
        json.dumps({"items": items}), encoding="utf-8"
    )


def test_db_samples_are_multi_hot(tmp_path) -> None:
    _write(tmp_path, [
        {"file_name": "images/a.jpg", "split": "train", "labels": ["idle", "dump"]},
        {"file_name": "images/b.jpg", "split": "train", "labels": ["swing_bucket"]},
        {"file_name": "images/c.jpg", "split": "valid", "labels": ["move"]},
    ])
    samples, classes = load_db_activity_samples(tmp_path, "train")
    assert len(samples) == 2
    assert classes[:5] == ACTIVITY_CLASSES
    # idle and dump are both on for the first frame.
    assert set(samples[0].labels) == {
        classes.index("idle"), classes.index("dump"),
    }
    assert samples[0].file_name == "images/a.jpg"


def test_db_samples_respect_the_split(tmp_path) -> None:
    _write(tmp_path, [
        {"file_name": "images/a.jpg", "split": "train", "labels": ["idle"]},
        {"file_name": "images/c.jpg", "split": "valid", "labels": ["move"]},
    ])
    valid, _ = load_db_activity_samples(tmp_path, "valid")
    assert [s.file_name for s in valid] == ["images/c.jpg"]


def test_db_samples_without_file_are_empty(tmp_path) -> None:
    samples, classes = load_db_activity_samples(tmp_path, "train")
    assert samples == []
    assert classes == ACTIVITY_CLASSES
