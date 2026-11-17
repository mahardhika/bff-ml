"""Real metrics for trained BFF models.

Why this module exists
----------------------
`ml.train` never validated: it hard-coded ``"val_loss": float("nan")``
and ``"mAP50": 0.0`` into ``metrics.json`` (``ml/train.py`` lines
294-297), and ``ml.eval.run`` was a stub that wrote the same
placeholder for every model (``"mAP50": 0.0,  # placeholder``).
This module computes the numbers that were missing.

detector
    * runs the exported ONNX through onnxruntime (CUDA when present),
    * decodes with the rules of ``backend.core.detector`` (per-level
      stride, score = sigmoid(obj) * max sigmoid(cls), class-agnostic
      NMS, boxes mapped back to source pixels),
    * scores the detections against the split's COCO ground truth with
      pycocotools -> mAP50, mAP50:95 and per-class AP,
    * reports precision / recall / F1 per class at the shipped
      operating point (conf 0.40, IoU 0.50),
    * computes the repo's own YOLOX validation loss
      (``ml.models.loss`` + ``ml.models.assign``) on the same split and
      on the same ONNX outputs.

Two decode variants are reported because the training loss and the
shipped backend disagree on the sign of the predicted width/height:

* ``loss_consistent`` - ``w = abs(reg_w) * stride``. This is what
  ``ml.models.loss._decode_xyxy`` uses, i.e. what the network was
  trained to produce.
* ``backend_exact`` - ``w = reg_w * stride``. This is what
  ``backend.core.detector.YOLOXDetector._decode_level`` does, so it is
  what the production app actually sees.

classifier (activity)
    * runs ``classifier.onnx`` over the valid split,
    * multi-label per-class precision / recall / F1 + macro F1,
    * exact-match accuracy (what ``train.py`` reported as
      ``val_accuracy``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Operating point the backend ships with (backend/core/detector.py).
DEFAULT_CONF = 0.40
DEFAULT_IOU = 0.45
#: Score threshold used for the mAP PR curve (COCO style: keep low).
AP_CONF = 0.001
#: COCO caps the detections per image at 100.
MAX_DETS = 100
#: Raw candidates kept per image before NMS (the head emits thousands).
PRE_NMS_TOPK = 500


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (same as backend/core/detector.py)."""
    return 1.0 / (1.0 + np.exp(-x))


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Class-agnostic NMS - a copy of the backend's ``_nms``."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        mask = iou <= iou_thresh
        order = order[1:][mask]
    return keep


# --------------------------------------------------------------------------
# ONNX session
# --------------------------------------------------------------------------

def make_session(model_path: str | Path, device: str = "auto"):
    """Open an onnxruntime session, preferring the CUDA provider."""
    import onnxruntime as ort

    want = (device or "auto").lower()
    available = ort.get_available_providers()
    providers: list[Any] = []
    if want in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    if want == "cuda" and not providers:
        log.warning("CUDAExecutionProvider not available in this onnxruntime "
                    "build - falling back to CPU. Available: %s", available)
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(model_path), providers=providers)
    log.info("ORT session %s providers=%s", Path(model_path).name,
             sess.get_providers())
    return sess


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

def load_coco_split(split_dir: str | Path) -> dict:
    """Read ``<split>/annotations.json`` into a COCO-ish dict."""
    split_dir = Path(split_dir)
    ann_path = split_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"no annotations.json under {split_dir}")
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = sorted(data.get("categories") or [], key=lambda c: int(c.get("id", 0)))
    return {
        "root": split_dir,
        "images": data.get("images") or [],
        "annotations": data.get("annotations") or [],
        "categories": cats,
        # model head index -> raw COCO category id (same order as
        # ml.data.load._coco_classes)
        "class_to_cat": {i: int(c.get("id", i)) for i, c in enumerate(cats)},
        "cat_names": [str(c.get("name") or f"class_{c.get('id')}") for c in cats],
    }


def _resolve_image(split_dir: Path, file_name: str) -> Path:
    """Find the image file, mirroring ``ml.data.load._iter_coco``."""
    p = split_dir / file_name
    if p.exists():
        return p
    return split_dir / file_name.split("/")[-1]


# --------------------------------------------------------------------------
# detector
# --------------------------------------------------------------------------

def _decode_level(raw: np.ndarray, input_size: int, conf: float,
                  abs_box: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Decode one FPN level - mirrors backend/core/detector.py._decode_level.

    ``abs_box`` reproduces ``ml.models.loss._decode_xyxy`` (True) or the
    shipped backend (False).
    """
    t = np.transpose(raw[0], (1, 2, 0))
    h, w, c = t.shape
    if c < 6:
        return None
    preds = t.reshape(-1, c)
    stride = input_size / max(h, w)
    gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    obj = _sigmoid(preds[:, 4])
    cls_scores = _sigmoid(preds[:, 5:])
    cls_ids = cls_scores.argmax(1)
    cls_conf = cls_scores.max(1)
    scores = obj * cls_conf
    keep = scores > conf
    if not np.any(keep):
        return None

    preds = preds[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]
    boxes = preds[:, :4].copy()
    if abs_box:
        boxes[:, 2] = np.abs(boxes[:, 2])
        boxes[:, 3] = np.abs(boxes[:, 3])
    boxes[:, 0] = (boxes[:, 0] + grid[keep, 0]) * stride
    boxes[:, 1] = (boxes[:, 1] + grid[keep, 1]) * stride
    boxes[:, 2] = boxes[:, 2] * stride
    boxes[:, 3] = boxes[:, 3] * stride
    return boxes, scores, cls_ids


def _decode_image(raw_list, input_size: int, conf: float, abs_box: bool):
    """Decode every FPN level of one image into canvas-space boxes."""
    boxes_all, scores_all, cls_all = [], [], []
    for level in raw_list:
        decoded = _decode_level(level, input_size, conf, abs_box)
        if decoded is None:
            continue
        b, s, c = decoded
        boxes_all.append(b)
        scores_all.append(s)
        cls_all.append(c)
    if not boxes_all:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return (np.concatenate(boxes_all, axis=0),
            np.concatenate(scores_all, axis=0),
            np.concatenate(cls_all, axis=0))


def _to_source_pixels(boxes, scale, w0, h0):
    """Canvas-space xywh-center boxes -> source-pixel xyxy boxes."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) / scale
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) / scale
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) / scale
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) / scale
    x1 = np.clip(x1, 0, w0)
    y1 = np.clip(y1, 0, h0)
    x2 = np.clip(x2, 0, w0)
    y2 = np.clip(y2, 0, h0)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def evaluate_detector(
    model_path: str | Path,
    split_dir: str | Path,
    device: str = "auto",
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    limit: int = 0,
    val_loss_frames: int = 0,
    input_size: int = 0,
) -> dict:
    """Evaluate a YOLOX-style detector ONNX on one COCO split."""
    import cv2

    from ml.data.augment import letterbox

    split_dir = Path(split_dir)
    gt = load_coco_split(split_dir)
    images = list(gt["images"])
    if limit and limit > 0 and len(images) > limit:
        stride = len(images) / float(limit)
        images = [images[int(i * stride)] for i in range(limit)]
    log.info("Detector eval: %d images from %s", len(images), split_dir)

    meta_path = Path(model_path).parent / "labels.json"
    if not input_size and meta_path.exists():
        try:
            input_size = int(json.loads(meta_path.read_text()).get("input_size") or 0)
        except Exception:  # noqa: BLE001
            input_size = 0
    input_size = input_size or 640

    sess = make_session(model_path, device)
    in_name = sess.get_inputs()[0].name

    ann_by_image: dict[int, list[dict]] = {}
    for a in gt["annotations"]:
        ann_by_image.setdefault(int(a["image_id"]), []).append(a)

    variants = {"loss_consistent": True, "backend_exact": False}
    dets: dict[str, list[dict]] = {k: [] for k in variants}
    op_dets: dict[str, list[dict]] = {k: [] for k in variants}
    gt_images: list[dict] = []
    gt_anns: list[dict] = []
    img_ids: list[int] = []
    per_image_gt: dict[int, list[tuple[int, np.ndarray]]] = {}
    score_stats: dict[str, dict] = {
        name: {"max_score": 0.0, "n_above_conf": 0, "n_above_0.01": 0}
        for name in gt["cat_names"]
    }
    n_kept = 0
    t0 = time.time()

    for n_img, im in enumerate(images, start=1):
        path = _resolve_image(split_dir, str(im.get("file_name") or ""))
        if not path.exists():
            log.warning("missing image %s", path)
            continue
        img = cv2.imread(str(path))
        if img is None:
            log.warning("unreadable image %s", path)
            continue
        h0, w0 = img.shape[:2]
        image_id = int(im.get("id", 0))

        canvas, scale = letterbox(img, input_size=input_size)
        blob = np.transpose(canvas[:, :, ::-1].astype(np.float32) / 255.0,
                            (2, 0, 1))[None]
        raw = sess.run(None, {in_name: blob})

        for variant, abs_box in variants.items():
            boxes, scores, cls_ids = _decode_image(raw, input_size, AP_CONF,
                                                   abs_box)
            # Keep the best N raw candidates before NMS: the head emits
            # thousands of anchors per image and COCO scores at most 100
            # detections per image anyway.
            if len(scores) > PRE_NMS_TOPK:
                top = scores.argsort()[::-1][:PRE_NMS_TOPK]
                boxes, scores, cls_ids = boxes[top], scores[top], cls_ids[top]
            src = _to_source_pixels(boxes, scale, w0, h0)
            keep = _nms(src, scores, iou)
            kept_boxes = src[keep] if keep else np.zeros((0, 4), dtype=np.float32)
            kept_scores = scores[keep] if keep else np.zeros((0,), dtype=np.float32)
            kept_cls = cls_ids[keep] if keep else np.zeros((0,), dtype=np.int64)
            if variant == "loss_consistent":
                n_kept += len(kept_boxes)
                for c in range(len(gt["cat_names"])):
                    mask = kept_cls == c
                    if not mask.any():
                        continue
                    sc = kept_scores[mask]
                    name = gt["cat_names"][c]
                    score_stats[name]["max_score"] = max(
                        score_stats[name]["max_score"], float(sc.max()))
                    score_stats[name]["n_above_conf"] += int((sc >= conf).sum())
                    score_stats[name]["n_above_0.01"] += int((sc >= 0.01).sum())

            order = kept_scores.argsort()[::-1][:MAX_DETS] if len(kept_scores) \
                else []
            for i in order:
                cat_id = gt["class_to_cat"].get(int(kept_cls[i]))
                if cat_id is None:
                    continue
                bx = kept_boxes[i]
                bw = float(bx[2] - bx[0])
                bh = float(bx[3] - bx[1])
                dets[variant].append({
                    "id": len(dets[variant]) + 1,
                    "image_id": image_id,
                    "category_id": int(cat_id),
                    "bbox": [float(bx[0]), float(bx[1]), bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "score": float(kept_scores[i]),
                })

            op_dets[variant].append({
                "image_id": image_id,
                # The **pre-NMS** candidates. The P/R/F1 sweep has to run
                # the NMS at the threshold it reports, because the shipped
                # detector filters by confidence *before* NMS - re-thresholding
                # an already-NMSed set measures a different pipeline.
                "boxes": src,
                "scores": scores,
                "cls": cls_ids,
            })

        # ---- ground truth (original pixels) ----
        gt_boxes: list[tuple[int, np.ndarray]] = []
        gt_list = []
        for a in ann_by_image.get(image_id, []):
            x, y, w, h = (float(v) for v in a["bbox"])
            cat_id = int(a.get("category_id", 0))
            cls_idx = next((k for k, v in gt["class_to_cat"].items()
                            if v == cat_id), None)
            if cls_idx is None:
                continue
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            gt_boxes.append((cls_idx, box))
            gt_list.append({
                "id": int(a.get("id", len(gt_list) + 1)),
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": int(a.get("iscrowd", 0)),
            })
        per_image_gt[image_id] = gt_boxes
        gt_images.append({"id": image_id, "file_name": im.get("file_name"),
                          "width": w0, "height": h0})
        gt_anns.extend(gt_list)
        img_ids.append(image_id)

        if n_img % 1000 == 0:
            log.info("  %d/%d images  %.1f img/s  dets=%d",
                     n_img, len(images), n_img / max(1e-6, time.time() - t0),
                     n_kept)

    elapsed = time.time() - t0
    log.info("Detector inference done: %d images in %.1fs (%.1f img/s), "
             "%d kept detections", len(img_ids), elapsed,
             len(img_ids) / max(1e-6, elapsed), n_kept)

    result: dict[str, Any] = {
        "n_images": len(img_ids),
        "eval_seconds": round(elapsed, 2),
        "input_size": input_size,
        "conf_thresh": conf,
        "iou_thresh": iou,
        "classes": gt["cat_names"],
        "artifact": str(model_path),
        "n_detections_total": n_kept,
        "score_stats": score_stats,
    }

    blocks = {}
    for variant in variants:
        blocks[variant] = _map_metrics(
            gt_images, gt_anns, gt["categories"], dets[variant], img_ids,
            op_dets[variant], per_image_gt, gt["cat_names"], conf=conf,
            nms_iou=iou)
    result["decode_variants"] = blocks
    # headline numbers: the decode the network was actually trained for
    primary = blocks["loss_consistent"]
    result["mAP50"] = primary["mAP50"]
    result["mAP50_95"] = primary["mAP50_95"]
    result["precision_macro"] = primary["precision_macro"]
    result["recall_macro"] = primary["recall_macro"]
    result["f1_macro"] = primary["f1_macro"]
    result["precision_micro"] = primary["precision_micro"]
    result["recall_micro"] = primary["recall_micro"]
    result["per_class"] = primary["per_class"]

    if val_loss_frames:
        vl = _detector_val_loss(sess, in_name, split_dir, images, input_size,
                                val_loss_frames, len(gt["cat_names"]))
        result["val_loss"] = vl["val_loss"]
        result["val_loss_parts"] = vl["parts"]
        result["val_loss_frames"] = vl["frames"]
        result["val_loss_batches"] = vl.get("batches")
        result["val_loss_batch_size"] = vl.get("batch_size")
    return result


def _map_metrics(gt_images, gt_anns, categories, dets, img_ids, op_dets,
                 per_image_gt, cat_names, conf: float = DEFAULT_CONF,
                 nms_iou: float = DEFAULT_IOU) -> dict:
    """mAP50 / mAP50:95 from pycocotools + operating-point P/R/F1."""
    out: dict[str, Any] = {"mAP50": None, "mAP50_95": None, "per_class": {}}
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:  # noqa: BLE001
        log.error("pycocotools unavailable: %s", exc)
        return out

    coco_gt = COCO()
    coco_gt.dataset = {"images": gt_images, "annotations": gt_anns,
                       "categories": categories}
    try:
        coco_gt.createIndex()
    except Exception as exc:  # noqa: BLE001
        log.error("coco_gt.createIndex failed: %s", exc)
    coco_dt = COCO()
    coco_dt.dataset = {"images": gt_images, "annotations": dets,
                       "categories": categories}
    try:
        coco_dt.createIndex()
    except Exception as exc:  # noqa: BLE001
        log.error("coco_dt.createIndex failed: %s", exc)

    if dets and gt_anns:
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = img_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        out["mAP50"] = float(evaluator.stats[1])
        out["mAP50_95"] = float(evaluator.stats[0])
        precision = evaluator.eval["precision"]
        iou50 = int(np.where(np.isclose(evaluator.params.iouThrs, 0.5))[0][0])
        cat_ids = list(evaluator.params.catIds)
        for k, cid in enumerate(cat_ids):
            p50 = precision[iou50, :, k, 0, -1]
            p50 = p50[p50 > -1]
            p_all = precision[:, :, k, 0, -1]
            p_all = p_all[p_all > -1]
            name = cat_names[k] if k < len(cat_names) else f"cat_{cid}"
            out["per_class"][name] = {
                "AP50": float(p50.mean()) if p50.size else None,
                "AP50_95": float(p_all.mean()) if p_all.size else None,
                "category_id": int(cid),
            }
    else:
        out["mAP50"] = 0.0
        out["mAP50_95"] = 0.0

    # ---------------- operating point P / R / F1 ----------------
    # 0.02 is the threshold the shipped earthmoving detector actually
    # wants (measured with the real threshold->NMS pipeline); the others
    # are kept for comparison.
    thresholds = [0.40, 0.10, 0.05, 0.02, 0.01]
    per_threshold = {}
    for thr in thresholds:
        per_threshold[f"{thr:.2f}"] = _prf_at(op_dets, per_image_gt, cat_names,
                                             conf=thr, nms_iou=nms_iou)
    out["operating_points"] = per_threshold
    best = _prf_at(op_dets, per_image_gt, cat_names, conf=conf, nms_iou=nms_iou)
    # keep the AP numbers from COCOeval next to the operating-point P/R/F1
    for name, pc in best["per_class"].items():
        ap = out["per_class"].get(name) or {}
        pc.update({k: v for k, v in ap.items()
                   if k in ("AP50", "AP50_95", "category_id")})
        out["per_class"][name] = pc
    out.update({k: best[k] for k in
                ("precision_macro", "recall_macro", "f1_macro",
                 "precision_micro", "recall_micro", "tp", "fp", "fn")})
    out["op_conf_thresh"] = conf
    out["op_iou_thresh"] = 0.5
    out["per_class"] = best["per_class"]
    return out


def _prf_at(op_dets, per_image_gt, cat_names, conf: float,
            iou_thresh: float = 0.5, nms_iou: float = DEFAULT_IOU) -> dict:
    """Precision / recall / F1 at one score threshold (IoU 0.5, per class).

    The candidates are the **pre-NMS** ones; the NMS runs here, at this
    threshold, because the shipped detector filters by confidence before
    it suppresses overlapping boxes. Thresholding an already-NMSed set
    would report the operating point of a pipeline that does not exist -
    that is what made "0.05 is the best F1 point" wrong in the .env.
    """
    n_classes = len(cat_names)
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    for item in op_dets:
        image_id = item["image_id"]
        gt_by_cls: dict[int, list[np.ndarray]] = {}
        for cls_idx, box in per_image_gt.get(image_id, []):
            gt_by_cls.setdefault(cls_idx, []).append(box)
        used = {c: [False] * len(v) for c, v in gt_by_cls.items()}
        scores = np.asarray(item["scores"])
        boxes = np.asarray(item["boxes"])
        cls_ids = np.asarray(item["cls"])
        sel = np.nonzero(scores >= conf)[0] if scores.size else []
        if len(sel):
            keep = _nms(boxes[sel], scores[sel], nms_iou)
            sel = sel[np.asarray(keep, dtype=int)] if keep else []
        for i in sel:
            cls_idx = int(cls_ids[i])
            box = boxes[i]
            cands = gt_by_cls.get(cls_idx, [])
            if cands:
                ious = _box_iou(box[None, :], np.stack(cands, axis=0))[0]
                j = int(ious.argmax())
                if ious[j] >= iou_thresh and not used[cls_idx][j]:
                    used[cls_idx][j] = True
                    tp[cls_idx] += 1
                    continue
            fp[cls_idx] += 1
        for cls_idx, cands in gt_by_cls.items():
            fn[cls_idx] += sum(1 for u in used[cls_idx] if not u)

    p_list, r_list, f1_list = [], [], []
    per_class = {}
    for c in range(n_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        p_list.append(prec)
        r_list.append(rec)
        f1_list.append(f1)
        name = cat_names[c] if c < len(cat_names) else f"class_{c}"
        per_class[name] = {
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c]),
            "support": int(tp[c] + fn[c]),
        }
    tp_t, fp_t, fn_t = tp.sum(), fp.sum(), fn.sum()
    return {
        "conf": conf,
        "precision_macro": float(np.mean(p_list)) if p_list else 0.0,
        "recall_macro": float(np.mean(r_list)) if r_list else 0.0,
        "f1_macro": float(np.mean(f1_list)) if f1_list else 0.0,
        "precision_micro": float(tp_t / (tp_t + fp_t)) if (tp_t + fp_t) else 0.0,
        "recall_micro": float(tp_t / (tp_t + fn_t)) if (tp_t + fn_t) else 0.0,
        "tp": int(tp_t), "fp": int(fp_t), "fn": int(fn_t),
        "per_class": per_class,
    }


def _detector_val_loss(sess, in_name: str, split_dir: Path, images: list,
                       input_size: int, max_frames: int, num_classes: int,
                       batch_size: int = 16) -> dict:
    """YOLOX validation loss computed from the ONNX outputs.

    Uses the repo's own ``ml.models.assign.assign`` and
    ``ml.models.loss.YOLOXLoss`` so the number is comparable with the
    ``train_loss`` the trainer logged. The batch size matches the
    trainer's default (16) because the repo's loss normalises the
    regression terms by the number of positive cells in the batch.
    """
    import cv2
    import torch

    from ml.data.augment import letterbox
    from ml.models.assign import assign
    from ml.models.loss import YOLOXLoss

    frames = images
    if max_frames > 0 and len(frames) > max_frames:
        stride = len(frames) / float(max_frames)
        frames = [frames[int(i * stride)] for i in range(max_frames)]

    gt = load_coco_split(split_dir)
    ann_by_image: dict[int, list[dict]] = {}
    for a in gt["annotations"]:
        ann_by_image.setdefault(int(a["image_id"]), []).append(a)

    loss_fn = YOLOXLoss(num_classes=num_classes)
    total = 0.0
    parts = {"obj": 0.0, "cls": 0.0, "reg": 0.0, "l1": 0.0}
    n_batches = 0
    n_frames = 0

    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        blobs, box_list, cls_list = [], [], []
        for im in chunk:
            path = _resolve_image(split_dir, str(im.get("file_name") or ""))
            if not path.exists():
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            canvas, scale = letterbox(img, input_size=input_size)
            blobs.append(np.transpose(
                canvas[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1)))
            boxes, cls = [], []
            for a in ann_by_image.get(int(im.get("id", 0)), []):
                x, y, w, h = (float(v) for v in a["bbox"])
                cat_id = int(a.get("category_id", 0))
                cls_idx = next((k for k, v in gt["class_to_cat"].items()
                                if v == cat_id), None)
                if cls_idx is None:
                    continue
                boxes.append([x * scale, y * scale,
                              (x + w) * scale, (y + h) * scale])
                cls.append(cls_idx)
            if boxes:
                box_list.append(torch.tensor(boxes, dtype=torch.float32))
                cls_list.append(torch.tensor(cls, dtype=torch.long))
            else:
                box_list.append(torch.zeros((0, 4), dtype=torch.float32))
                cls_list.append(torch.zeros((0,), dtype=torch.long))
        if not blobs:
            continue
        blob = np.stack(blobs)
        raw = sess.run(None, {in_name: blob})
        preds = [torch.from_numpy(np.asarray(r, dtype=np.float32)) for r in raw]
        assigned = assign(preds, box_list, cls_list, input_size=input_size)
        losses = loss_fn(preds, assigned)
        total += float(sum(losses.values()))
        for k in parts:
            parts[k] += float(losses[k])
        n_batches += 1
        n_frames += len(blobs)

    if n_batches == 0:
        return {"val_loss": None, "parts": {}, "frames": 0}
    parts = {k: v / n_batches for k, v in parts.items()}
    return {"val_loss": total / n_batches,
            "parts": {k: round(v, 6) for k, v in parts.items()},
            "frames": n_frames,
            "batches": n_batches,
            "batch_size": batch_size}


# --------------------------------------------------------------------------
# fill classifier
# --------------------------------------------------------------------------

def evaluate_fill_classifier(
    model_path: str | Path,
    dataset_root: str | Path,
    split: str = "valid",
    device: str = "auto",
    limit: int = 0,
    input_size: int = 0,
) -> dict:
    """Evaluate the binary bucket-fill head on one split.

    The crops are the same letterboxed ``excavator`` boxes the backend
    feeds the head at inference time, and the labels come from the
    archive's temporal action segments (``ml.data.fill``).
    """
    from ml.data.fill import FILL_CLASSES, load_fill_samples

    dataset_root = Path(dataset_root)
    samples = load_fill_samples(dataset_root, split, 0)
    total_available = len(samples)
    if limit and limit > 0 and len(samples) > limit:
        stride = len(samples) / float(limit)
        samples = [samples[int(i * stride)] for i in range(limit)]
    log.info("Fill eval: %d/%d crops from %s/%s", len(samples),
             total_available, dataset_root, split)

    meta_path = Path(model_path).parent / "labels.json"
    classes = list(FILL_CLASSES)
    if meta_path.exists():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            listed = raw.get("classes") or []
            if listed:
                classes = [str(c) for c in listed]
            if not input_size:
                input_size = int(raw.get("input_size") or 0)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            log.warning("Could not read %s: %s", meta_path, exc)
    input_size = input_size or 224

    sess = make_session(model_path, device)
    in_name = sess.get_inputs()[0].name

    from ml.data.fill import FillCrops, collate_fill
    from torch.utils.data import DataLoader

    frames = FillCrops(dataset_root, samples, input_size=input_size,
                       train=False, split=split)
    loader = DataLoader(frames, batch_size=64, shuffle=False,
                        collate_fn=collate_fill)

    num_classes = len(classes)
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    support = np.zeros(num_classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    exact = 0
    total = 0
    ce_sum = 0.0
    ce_n = 0
    t0 = time.time()
    for batch in loader:
        blob = batch["image"].numpy().astype(np.float32)
        logits = np.asarray(sess.run(None, {in_name: blob})[0], dtype=np.float32)
        if logits.ndim == 1:
            logits = logits[None]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        pred = probs.argmax(axis=1)
        gt = batch["label"].numpy().astype(np.int64)
        rows = np.arange(len(gt))
        np.add.at(confusion, (gt, pred), 1)
        for c in range(num_classes):
            p = pred == c
            g = gt == c
            tp[c] += float(np.sum(p & g))
            fp[c] += float(np.sum(p & ~g))
            fn[c] += float(np.sum(~p & g))
            support[c] += float(np.sum(g))
        exact += int(np.sum(pred == gt))
        p_true = probs[rows, np.clip(gt, 0, num_classes - 1)]
        ce_sum += float(-np.log(np.clip(p_true, 1e-12, 1.0)).sum())
        ce_n += len(gt)
        total += len(gt)
    elapsed = time.time() - t0

    per_class = {}
    p_list, r_list, f1_list = [], [], []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        p_list.append(prec)
        r_list.append(rec)
        f1_list.append(f1)
        per_class[classes[c] if c < len(classes) else f"class_{c}"] = {
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1": round(float(f1), 6),
            "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c]),
            "support": int(support[c]),
        }

    tp_t, fp_t, fn_t = tp.sum(), fp.sum(), fn.sum()
    micro_p = float(tp_t / (tp_t + fp_t)) if (tp_t + fp_t) else 0.0
    micro_r = float(tp_t / (tp_t + fn_t)) if (tp_t + fn_t) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                 if (micro_p + micro_r) else 0.0)
    return {
        "artifact": str(model_path),
        "split": split,
        "n_frames": total,
        "frames_available": total_available,
        "subset_size": len(samples),
        "input_size": input_size,
        "classes": classes,
        "per_class": per_class,
        "precision_macro": round(float(np.mean(p_list)), 6) if p_list else 0.0,
        "recall_macro": round(float(np.mean(r_list)), 6) if r_list else 0.0,
        "f1_macro": round(float(np.mean(f1_list)), 6) if f1_list else 0.0,
        "precision_micro": round(micro_p, 6),
        "recall_micro": round(micro_r, 6),
        "f1_micro": round(micro_f1, 6),
        "exact_match_accuracy": round(exact / total, 6) if total else None,
        "accuracy": round(exact / total, 6) if total else None,
        "confusion_matrix": confusion.tolist(),
        "gt_label_distribution": {
            classes[c] if c < len(classes) else f"class_{c}": int(support[c])
            for c in range(num_classes)
        },
        "val_loss": round(ce_sum / ce_n, 6) if ce_n else None,
        "val_loss_note": "cross entropy over the evaluated subset "
                         "(same loss as ml.train._train_fill)",
        "eval_seconds": round(elapsed, 2),
    }

# --------------------------------------------------------------------------
# activity classifier
# --------------------------------------------------------------------------

def evaluate_classifier(
    model_path: str | Path,
    dataset_root: str | Path,
    split: str = "valid",
    device: str = "auto",
    limit: int = 1500,
    threshold: float = 0.5,
    input_size: int = 0,
) -> dict:
    """Evaluate the multi-label activity classifier on one split."""
    from ml.data.activity import load_activity_samples

    dataset_root = Path(dataset_root)
    samples, classes = load_activity_samples(dataset_root, split, 0)
    total_available = len(samples)
    if limit and limit > 0 and len(samples) > limit:
        stride = len(samples) / float(limit)
        samples = [samples[int(i * stride)] for i in range(limit)]
    log.info("Activity eval: %d/%d frames from %s/%s", len(samples),
             total_available, dataset_root, split)

    gt_dist = [0] * len(classes)
    for s in samples:
        for v in s.labels:
            if 0 <= v < len(classes):
                gt_dist[v] += 1
    log.info("GT label distribution (evaluated subset): %s",
             dict(zip(classes, gt_dist)))

    meta_path = Path(model_path).parent / "labels.json"
    if not input_size and meta_path.exists():
        try:
            input_size = int(json.loads(meta_path.read_text()).get("input_size") or 0)
        except Exception:  # noqa: BLE001
            input_size = 0
    input_size = input_size or 224
    num_classes = len(classes)

    sess = make_session(model_path, device)
    in_name = sess.get_inputs()[0].name

    from ml.data.torch_dataset import ActivityFrames, collate_activity
    from torch.utils.data import DataLoader

    frames = ActivityFrames(dataset_root, samples, num_classes,
                             input_size=input_size, train=False, split=split)
    loader = DataLoader(frames, batch_size=64, shuffle=False,
                        collate_fn=collate_activity)

    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    support = np.zeros(num_classes)
    exact = 0
    total = 0
    label_hits = 0
    label_total = 0
    bce_sum = 0.0
    bce_n = 0
    t0 = time.time()
    for batch in loader:
        blob = batch["image"].numpy().astype(np.float32)
        logits = sess.run(None, {in_name: blob})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred = probs > threshold
        gt = batch["label"].numpy() > 0.5
        # repo's own loss: torch.nn.BCEWithLogitsLoss (ml/train.py)
        eps = 1e-7
        raw = np.clip(logits, -30.0, 30.0)
        bce = -(gt * np.log(1.0 / (1.0 + np.exp(-raw)))
                + (1.0 - gt) * np.log(1.0 / (1.0 + np.exp(raw))))
        bce_sum += float(bce.sum())
        bce_n += bce.size
        for c in range(num_classes):
            p = pred[:, c]
            g = gt[:, c]
            tp[c] += float(np.sum(p & g))
            fp[c] += float(np.sum(p & ~g))
            fn[c] += float(np.sum(~p & g))
            support[c] += float(np.sum(g))
        exact += int(np.all(pred == gt, axis=1).sum())
        label_hits += int((pred == gt).sum())
        label_total += gt.size
        total += gt.shape[0]
    elapsed = time.time() - t0

    per_class = {}
    p_list, r_list, f1_list = [], [], []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        p_list.append(prec)
        r_list.append(rec)
        f1_list.append(f1)
        per_class[classes[c] if c < len(classes) else f"class_{c}"] = {
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1": round(float(f1), 6),
            "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c]),
            "support": int(support[c]),
        }

    tp_t, fp_t, fn_t = tp.sum(), fp.sum(), fn.sum()
    micro_p = float(tp_t / (tp_t + fp_t)) if (tp_t + fp_t) else 0.0
    micro_r = float(tp_t / (tp_t + fn_t)) if (tp_t + fn_t) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                 if (micro_p + micro_r) else 0.0)
    # macro over the classes that actually occur in the evaluated split
    # (a class with zero positives has an undefined P/R/F1)
    sup_idx = [c for c in range(num_classes) if support[c] > 0]
    sup_p = [p_list[c] for c in sup_idx]
    sup_r = [r_list[c] for c in sup_idx]
    sup_f1 = [f1_list[c] for c in sup_idx]
    return {
        "artifact": str(model_path),
        "split": split,
        "n_frames": total,
        "frames_available": total_available,
        "subset_size": len(samples),
        "threshold": threshold,
        "input_size": input_size,
        "classes": list(classes),
        "per_class": per_class,
        "precision_macro": round(float(np.mean(p_list)), 6) if p_list else 0.0,
        "recall_macro": round(float(np.mean(r_list)), 6) if r_list else 0.0,
        "f1_macro": round(float(np.mean(f1_list)), 6) if f1_list else 0.0,
        "classes_with_support": [classes[c] if c < len(classes)
                                 else f"class_{c}" for c in sup_idx],
        "precision_macro_supported": round(float(np.mean(sup_p)), 6) if sup_p else None,
        "recall_macro_supported": round(float(np.mean(sup_r)), 6) if sup_r else None,
        "f1_macro_supported": round(float(np.mean(sup_f1)), 6) if sup_f1 else None,
        "precision_micro": round(micro_p, 6),
        "recall_micro": round(micro_r, 6),
        "f1_micro": round(micro_f1, 6),
        "exact_match_accuracy": round(exact / total, 6) if total else None,
        "subset_accuracy": round(label_hits / label_total, 6) if label_total else None,
        "gt_label_distribution": dict(zip(classes, gt_dist)),
        "val_loss": round(bce_sum / bce_n, 6) if bce_n else None,
        "val_loss_note": "BCEWithLogitsLoss over the evaluated subset "
                         "(same loss as ml/train.py _train_activity)",
        "eval_seconds": round(elapsed, 2),
    }
