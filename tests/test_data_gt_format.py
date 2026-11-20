"""The training loader must read the box in the format the dataset uses.

COCO stores ``bbox`` as [x, y, width, height] and declares the size of the
image; the legacy earthmoving annotations store the corners and declare no
size at all. Reading one format as the other supervises the regression
with the wrong box.
"""
from ml.data.load import _annotation_box


def test_coco_annotation_is_xywh():
    image = {"id": 1, "width": 1920, "height": 1080}
    a = {"bbox": [100, 50, 200, 120]}
    assert _annotation_box(a, image) == [100.0, 50.0, 300.0, 170.0]


def test_legacy_annotation_is_xyxy():
    image = {"id": 1, "width": 0, "height": 0}
    a = {"bbox": [109, 26, 241, 194]}
    assert _annotation_box(a, image) == [109.0, 26.0, 241.0, 194.0]


def test_missing_size_falls_back_to_corners():
    a = {"bbox": [10, 20, 30, 40]}
    assert _annotation_box(a, {"id": 3}) == [10.0, 20.0, 30.0, 40.0]
