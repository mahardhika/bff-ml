"""The ground-truth format must be read the way the dataset stores it.

COCO stores ``bbox`` as [x, y, width, height] and declares the size of the
image; the legacy earthmoving annotations store the corners and declare no
size at all. Reading one format as the other makes the evaluator compare
against boxes that are 40% too wide - the shipped model's mAP was measured
against such boxes.
"""
from ml.eval.metrics import _gt_box


def test_coco_annotation_is_xywh():
    image = {"id": 1, "width": 1920, "height": 1080}
    a = {"bbox": [100, 50, 200, 120]}
    assert _gt_box(a, image).tolist() == [100.0, 50.0, 300.0, 170.0]


def test_legacy_annotation_is_xyxy():
    image = {"id": 1, "width": 0, "height": 0}
    a = {"bbox": [109, 26, 241, 194]}
    assert _gt_box(a, image).tolist() == [109.0, 26.0, 241.0, 194.0]


def test_missing_size_falls_back_to_corners():
    a = {"bbox": [10, 20, 30, 40]}
    assert _gt_box(a, {"id": 3}).tolist() == [10.0, 20.0, 30.0, 40.0]
