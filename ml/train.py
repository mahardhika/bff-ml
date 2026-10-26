"""Training entry point.

Supports a `--device=auto|cuda|dml|cpu` flag (resolved to ORT /
torch-directml). After training, drops the weights into
`runs/<run_name>/`, calls `ml.export_onnx` to produce ONNX, and then
invokes `ml.sync` to push the artifacts to Supabase.

Real training (full YOLOX-S, 50 epochs) is heavy; this script is the
orchestrator. For a quick smoke test, set `--epochs=1 --batch-size=4`.

Usage:
    python -m ml train --data-source=local --data-path=./data/train \
                       --model-name=bucket-detector --version=0.1.0 \
                       --epochs=50 --device=auto
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from ml.data.load import CACHE_TTL_DAYS

log = logging.getLogger(__name__)


def _pick_device(want: str | None = None):
    """Decide the runtime device from env + platform.

    Returns a value `torch.device(...)` can parse: a string like
    ``"cpu"`` / ``"cuda"`` / ``"privateuseone:0"``, or a real
    ``torch.device`` instance (returned as-is when torch_directml
    supplies one). The bare ``privateuseone`` backend string — NOT a
    ``"dml:"`` prefix — is what `model.to(...)` accepts. Returning a
    prefixed string here produced `RuntimeError: Invalid device
    string: 'dml:privateuseone:0'` on Windows.
    """
    want = (want or os.getenv("INFERENCE_PROVIDER", "auto")).lower()
    if want == "cuda":
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    if want == "dml" and os.name == "nt":
        try:
            import torch_directml  # type: ignore
            d = torch_directml.device()  # type: ignore[attr-defined]
            log.info("DML device resolved: %s", d)
            return d
        except Exception:
            log.info("DML not available; falling back to cpu")
            pass
    if want == "auto" and os.name == "nt":
        try:
            import torch_directml  # type: ignore
            d = torch_directml.device()  # type: ignore[attr-defined]
            log.info("DML device resolved (auto): %s", d)
            return d
        except Exception:
            log.info("DML not available; falling back to cpu")
            pass
    return "cpu"


def _build_optimizer(model, device, lr: float):
    """Move the model to the device and build the optimizer.

    The order matters. On DirectML the move replaces the Parameter
    objects, so an optimizer that was built before the move keeps the
    old tensors and updates weights the model no longer uses: the loss
    stays flat and the run produces an untrained model.
    """
    import torch

    model.to(device)
    return torch.optim.Adam(model.parameters(), lr=lr)


def main(argv: list[str] | None = None) -> None:
    from ml.env import load_env_files
    load_env_files()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    p = argparse.ArgumentParser(description="Train a BFF model.")
    p.add_argument("--data-source", default="local", choices=["local", "s3", "db", "url"])
    p.add_argument("--data-path", default="./data/train",
                   help="local path (when --data-source=local)")
    p.add_argument("--dataset-id", type=int, default=None,
                   help="bff_datasets.id (when --data-source=db)")
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET_CLIPS", "bff-media"))
    p.add_argument("--prefix", default="")
    p.add_argument("--force-download", action="store_true",
                   help="ignore the local dataset cache and re-download")
    p.add_argument("--cache-ttl-days", type=float, default=CACHE_TTL_DAYS,
                   help="dataset cache lifetime in days (default 7)")
    p.add_argument("--model-name", default="bucket-detector")
    p.add_argument("--version", default="0.1.0")
    p.add_argument("--kind", default="detector",
                   choices=["detector", "classifier", "activity"])
    p.add_argument("--num-classes", type=int, default=0,
                   help="classes in the model head; 0 = take them from the dataset")
    p.add_argument("--input-size", type=int, default=0,
                   help="detector input size; 0 = DETECTOR_INPUT_SIZE (640)")
    p.add_argument("--classifier-input-size", type=int, default=224,
                   help="classifier input size (fill and activity heads)")
    p.add_argument("--workers", type=int, default=0,
                   help="dataloader workers; >0 keeps the GPU fed on Windows")
    p.add_argument("--max-images", type=int, default=0,
                   help="stop after N images (quick runs); 0 = whole dataset")
    p.add_argument("--limit-images", type=int, default=0,
                   help="alias of --max-images")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12,
                   help="stop when train loss has not improved for this many epochs")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-sync", action="store_true",
                   help="skip Supabase sync after training")
    p.add_argument("--output-dir", default="./runs")
    args = p.parse_args(argv)

    device = _pick_device(None if args.device == "auto" else args.device)
    # Round-trip through torch.device(...) so unknown backends (xpu,
    # hpu, …) downgrade to cpu before we try `model.to(device)`.
    try:
        import torch  # type: ignore
        device = torch.device(device) if not isinstance(device, torch.device) else device
    except Exception as exc:  # noqa: BLE001
        log.warning("Unparseable device %r (%s); downgrading to cpu.", device, exc)
        device = torch.device("cpu")
    log.info("Training start: model=%s v=%s device=%s epochs=%d bs=%d",
             args.model_name, args.version, device, args.epochs, args.batch_size)

    run_name = f"{args.model_name}-{args.version}-{int(time.time())}"

    # 1. Resolve data FIRST so we don't leave an empty run-dir behind
    # if the data path is bogus or unreadable.
    activity_root: Path | None = None
    activity_train = []
    activity_valid = []
    if args.kind == "activity":
        # The activity dataset is read from the label sidecar, not from
        # the COCO boxes, so it has its own loader.
        from ml.data.activity import load_activity_samples, resolve_root
        activity_root, _split = resolve_root(Path(args.data_path))
        activity_train, activity_classes = load_activity_samples(
            activity_root, "train", _max_images(args))
        activity_valid, _ = load_activity_samples(activity_root, "valid", 0)

        class _ActivityDataset:
            root = activity_root
            kind = "activity"
            classes = activity_classes

        ds = _ActivityDataset()
    else:
        from ml.data.load import load_dataset
        if args.data_source == "local":
            ds = load_dataset("local", path=args.data_path)
        elif args.data_source == "s3":
            ds = load_dataset("s3", bucket=args.bucket, prefix=args.prefix,
                              force=args.force_download,
                              ttl_days=args.cache_ttl_days)
        elif args.data_source == "db":
            if args.dataset_id is None:
                raise SystemExit("--dataset-id is required when --data-source=db")
            ds = load_dataset("db", dataset_id=args.dataset_id,
                              force=args.force_download,
                              ttl_days=args.cache_ttl_days)
        else:
            raise NotImplementedError("url source is v0.2")
    log.info("Dataset: root=%s kind=%s classes=%s", ds.root, ds.kind, ds.classes)

    # Class count follows the dataset (COCO categories), so one trainer
    # serves the 3-class product model and wider public datasets alike.
    if args.num_classes > 0:
        num_classes = int(args.num_classes)
    elif ds.classes:
        num_classes = len(ds.classes)
    elif args.kind == "classifier":
        num_classes = 5
    else:
        num_classes = 3
    if ds.classes and args.num_classes > 0 and args.num_classes != len(ds.classes):
        log.warning("num_classes=%d but the dataset lists %d classes: %s",
                    args.num_classes, len(ds.classes), list(ds.classes))
    class_names = list(ds.classes) or [f"class_{i}" for i in range(num_classes)]
    log.info("Model head: %d classes -> %s", num_classes, class_names)
    input_size = args.input_size or int(os.getenv("DETECTOR_INPUT_SIZE", "640"))
    classifier_input_size = int(args.classifier_input_size or 224)
    # The size the weights were built with, for labels.json.
    model_input_size = input_size if args.kind == "detector" else classifier_input_size

    out = Path(args.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    # 2. Build the model + run a (stub) training loop.
    hparams = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "device": str(device), "data_source": args.data_source,
        "data_path": (
            args.data_path if args.data_source == "local"
            else (f"db:{args.dataset_id}" if args.data_source == "db"
                  else f"s3://{args.bucket}/{args.prefix}")
        ),
        "dataset_id": args.dataset_id,
        "force_download": bool(args.force_download),
        "cache_ttl_days": float(args.cache_ttl_days),
        "num_classes": num_classes,
        "classes": class_names,
        "input_size": model_input_size,
    }
    (out / "hparams.json").write_text(json.dumps(hparams, indent=2))

    metrics: dict[str, float] = {}
    started = time.time()

    try:
        import torch  # type: ignore

        from ml.models.classifier import build_fill_classifier
        from ml.models.loss import YOLOXLoss
        from ml.models.yolox import build_yolox_s
        torch_available = True
    except ImportError:
        torch_available = False

    last_loss = float("nan")
    n_images = 0
    n_boxes = 0
    val_accuracy = float("nan")

    if args.kind == "detector":
        log.info("Building YOLOX-S detector (%d classes).", num_classes)
        if torch_available:
            model = build_yolox_s(num_classes=num_classes, input_size=input_size)
            opt = _build_optimizer(model, device, args.lr)
            loss_fn = YOLOXLoss(num_classes=num_classes)
            last_loss, n_images, n_boxes = _train_detector(
                model, loss_fn, opt, ds, args, device, input_size)
        else:
            log.warning("torch not available; running stub training loop.")
            for epoch in range(1, max(1, args.epochs) + 1):
                time.sleep(0.05)
                log.info("epoch %d/%d in 0.1s", epoch, args.epochs)
    elif args.kind == "activity":
        log.info("Building activity classifier (%d classes, multi-label).",
                 num_classes)
        if torch_available:
            model = build_fill_classifier(num_classes=num_classes,
                                          input_size=classifier_input_size)
            opt = _build_optimizer(model, device, args.lr)
            last_loss, n_images, val_accuracy = _train_activity(
                model, opt, activity_root, activity_train, activity_valid,
                args, device, classifier_input_size)
        else:
            for epoch in range(1, max(1, args.epochs) + 1):
                time.sleep(0.05)
                log.info("epoch %d/%d in 0.1s", epoch, args.epochs)
    else:
        log.info("Building FillClassifier (%d classes).", num_classes)
        if torch_available:
            model = build_fill_classifier(num_classes=num_classes,
                                          input_size=classifier_input_size)
            opt = _build_optimizer(model, device, args.lr)
            last_loss, n_images, n_boxes = _train_classifier(
                model, opt, ds, args, device)
        else:
            for epoch in range(1, max(1, args.epochs) + 1):
                time.sleep(0.05)
                log.info("epoch %d/%d in 0.1s", epoch, args.epochs)

    metrics = {
        "train_loss": last_loss if torch_available else float("nan"),
        "val_loss": float("nan"),
        "mAP50": 0.0,
        "fill_acc": 0.0,
        "not_visible_recall": 0.0,
        "val_accuracy": val_accuracy,
        "torch_available": torch_available,
        "device": str(device),
        "epochs": args.epochs,
        "n_images": n_images,
        "n_boxes": n_boxes,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 3. Export ONNX under the canonical filename the backend loads.
    canonical = "detector.onnx" if args.kind == "detector" else "classifier.onnx"
    onnx_path = out / canonical
    if torch_available:
        from ml.export_onnx import _export_with_torch
        try:
            _export_with_torch(out, model, args)
            ckpt = out / "model.pt"
            try:
                import torch as _torch
                # CPU tensors keep the checkpoint portable (a DirectML
                # state dict cannot be reloaded outside the device).
                _torch.save({k: v.detach().cpu()
                             for k, v in model.state_dict().items()}, ckpt)
            except Exception as exc:  # noqa: BLE001
                log.warning("checkpoint save failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("ONNX export failed (%s); writing empty placeholder.", exc)
            onnx_path.write_bytes(b"")
    else:
        onnx_path.write_bytes(b"")
    log.info("ONNX written: %s", onnx_path)

    # 3b. Class list next to the weights. The backend reads it to map
    # head indices to labels instead of assuming bucket/truck/person.
    labels_path = out / "labels.json"
    labels_path.write_text(json.dumps({
        "classes": class_names,
        "num_classes": num_classes,
        "kind": args.kind,
        "input_size": model_input_size,
        "multi_label": args.kind == "activity",
    }, indent=2), encoding="utf-8")
    log.info("labels.json written: %s", labels_path)

    # 4. Model card.
    card = out / "MODEL_CARD.md"
    card.write_text(_model_card(args, metrics, started))
    log.info("MODEL_CARD written: %s", card)

    # 5. Sync unless told not to.
    if not args.no_sync:
        from ml.sync import sync_experiment
        try:
            sync_experiment(run_dir=out, model_name=args.model_name,
                            version=args.version, kind=args.kind,
                            dataset_ids=[args.dataset_id] if args.dataset_id else [])
        except Exception as exc:  # noqa: BLE001
            log.warning("Supabase sync failed (run preserved on disk): %s", exc)

    log.info("Run finished in %.1fs. Artifacts at %s", time.time() - started, out)


def _max_images(args) -> int:
    """Image cap for quick runs: ``--max-images`` (or its alias)."""
    return int(getattr(args, "max_images", 0) or getattr(args, "limit_images", 0) or 0)


def _train_detector(model, loss_fn, opt, ds, args, device, input_size: int
                    ) -> tuple[float, int, int]:
    import torch
    from torch.utils.data import DataLoader

    from ml.data.torch_dataset import DetectionFrames, collate_detections
    from ml.models.assign import assign

    frames = DetectionFrames(ds, input_size=input_size, train=True,
                             limit=_max_images(args))
    n_images = len(frames)
    n_boxes = 0
    last_loss = 0.0
    if n_images == 0:
        log.warning("Dataset is empty - running a synthetic warmup so export still works.")
        for epoch in range(1, max(1, args.epochs) + 1):
            epoch_t = time.time()
            model.train()
            x = torch.zeros(1, 3, input_size, input_size,
                            device=next(model.parameters()).device)
            logits = model(x)
            losses = loss_fn(logits, None)
            opt.zero_grad()
            total = sum(losses.values())
            total.backward()
            opt.step()
            last_loss = float(total.detach().cpu())
            log.info("epoch %d/%d loss=%.4f in %.1fs",
                     epoch, args.epochs, last_loss, time.time() - epoch_t)
        return last_loss, 0, 0

    loader = DataLoader(
        frames, batch_size=max(1, min(args.batch_size, n_images)),
        shuffle=True, collate_fn=collate_detections,
        num_workers=max(0, int(getattr(args, "workers", 0) or 0)),
        persistent_workers=bool(int(getattr(args, "workers", 0) or 0)),
    )
    best = float("inf")
    stale = 0
    patience = max(1, int(getattr(args, "patience", 12) or 12))
    for epoch in range(1, max(1, args.epochs) + 1):
        epoch_t = time.time()
        model.train()
        running = 0.0
        steps = 0
        epoch_boxes = 0
        batch_t = time.time()
        seen_total = 0
        for bi, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            boxes = [b.to(device) for b in batch["boxes"]]
            cls = [c.to(device) for c in batch["cls"]]
            epoch_boxes += sum(int(b.shape[0]) for b in boxes)
            preds = model(images)
            assigned = assign(preds, boxes, cls, input_size=input_size)
            losses = loss_fn(preds, assigned)
            opt.zero_grad()
            total = sum(losses.values())
            total.backward()
            opt.step()
            running += float(total.detach().cpu())
            steps += 1
            if bi % 25 == 0 or bi == len(loader):
                now = time.time()
                window = now - batch_t
                done = bi * images.shape[0]
                rate = (done - seen_total) / max(1e-6, window)
                log.info("  batch %d/%d  %.2f img/s  loss=%.4f",
                         bi, len(loader), rate,
                         running / max(1, steps))
                batch_t = now
                seen_total = done
        last_loss = running / max(1, steps)
        n_boxes = epoch_boxes
        log.info("epoch %d/%d loss=%.4f in %.1fs",
                 epoch, args.epochs, last_loss, time.time() - epoch_t)
        if last_loss + 1e-4 < best:
            best = last_loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d (best loss=%.4f)", epoch, best)
                last_loss = best
                break
    return last_loss, n_images, n_boxes


def _train_activity(model, opt, root, train_samples, valid_samples,
                    args, device, input_size: int = 224
                    ) -> tuple[float, int, float]:
    """Train the multi-label activity head.

    The loss is a binary cross entropy over the activity classes: a
    frame can carry several actions at once, so the head uses one
    sigmoid per class instead of a softmax.
    """
    import torch
    from torch.utils.data import DataLoader

    from ml.data.torch_dataset import ActivityFrames, collate_activity

    num_classes = int(getattr(model.cfg, "num_classes", 0) or 1)
    frames = ActivityFrames(root, train_samples, num_classes,
                             input_size=input_size, train=True, split="train")
    n_images = len(frames)
    last_loss = float("nan")
    if n_images == 0:
        log.warning("No activity frames - running a synthetic warmup so export works.")
        return last_loss, 0, float("nan")

    loader = DataLoader(
        frames, batch_size=max(1, min(args.batch_size, n_images)),
        shuffle=True, collate_fn=collate_activity,
        num_workers=max(0, int(getattr(args, "workers", 0) or 0)),
        persistent_workers=bool(int(getattr(args, "workers", 0) or 0)),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best = float("inf")
    stale = 0
    patience = max(1, int(getattr(args, "patience", 12) or 12))
    for epoch in range(1, max(1, args.epochs) + 1):
        epoch_t = time.time()
        model.train()
        running = 0.0
        steps = 0
        batch_t = time.time()
        seen_total = 0
        for bi, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            steps += 1
            if bi % 50 == 0 or bi == len(loader):
                now = time.time()
                done = bi * images.shape[0]
                rate = (done - seen_total) / max(1e-6, now - batch_t)
                log.info("  batch %d/%d  %.2f img/s  loss=%.4f",
                         bi, len(loader), rate, running / max(1, steps))
                batch_t = now
                seen_total = done
        last_loss = running / max(1, steps)
        log.info("epoch %d/%d loss=%.4f in %.1fs",
                 epoch, args.epochs, last_loss, time.time() - epoch_t)
        if last_loss + 1e-4 < best:
            best = last_loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d (best loss=%.4f)", epoch, best)
                last_loss = best
                break
    accuracy = _activity_accuracy(model, root, valid_samples, num_classes,
                                   input_size, device)
    return last_loss, n_images, accuracy


def _activity_accuracy(model, root, samples, num_classes: int,
                       input_size: int, device) -> float:
    """Exact-match accuracy of the multi-label head on the valid split."""
    import torch
    from torch.utils.data import DataLoader

    from ml.data.torch_dataset import ActivityFrames, collate_activity

    if not samples:
        return float("nan")
    frames = ActivityFrames(root, samples, num_classes,
                             input_size=input_size, train=False, split="valid")
    loader = DataLoader(frames, batch_size=32, shuffle=False,
                         collate_fn=collate_activity)
    model.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            probs = torch.sigmoid(model(images))
            predicted = (probs > 0.5).float()
            hits += int((predicted == labels).all(dim=1).sum().item())
            total += int(labels.shape[0])
    return hits / max(1, total)


def _train_classifier(model, opt, ds, args, device) -> tuple[float, int, int]:
    import torch
    from torch.utils.data import DataLoader

    from ml.data.torch_dataset import DetectionFrames, collate_detections

    frames = DetectionFrames(ds, input_size=224, train=True,
                             limit=_max_images(args))
    n_images = len(frames)
    last_loss = 0.0
    if n_images == 0:
        log.warning("Dataset is empty - running a synthetic warmup so export still works.")
        for epoch in range(1, max(1, args.epochs) + 1):
            epoch_t = time.time()
            model.train()
            x = torch.zeros(args.batch_size, 3, 224, 224,
                            device=next(model.parameters()).device)
            y = torch.zeros(args.batch_size, dtype=torch.long,
                            device=next(model.parameters()).device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
            log.info("epoch %d/%d loss=%.4f in %.1fs",
                     epoch, args.epochs, last_loss, time.time() - epoch_t)
        return last_loss, 0, 0

    loader = DataLoader(
        frames, batch_size=max(1, min(args.batch_size, n_images)),
        shuffle=True, collate_fn=collate_detections,
        num_workers=max(0, int(getattr(args, "workers", 0) or 0)),
        persistent_workers=bool(int(getattr(args, "workers", 0) or 0)),
    )
    best = float("inf")
    stale = 0
    patience = max(1, int(getattr(args, "patience", 12) or 12))
    for epoch in range(1, max(1, args.epochs) + 1):
        epoch_t = time.time()
        model.train()
        running = 0.0
        steps = 0
        for batch in loader:
            images = batch["image"].to(device)
            # Weak label: class of the first box, else 0.
            y = torch.tensor(
                [int(c[0].item()) if c.numel() else 0 for c in batch["cls"]],
                dtype=torch.long, device=device,
            )
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            steps += 1
        last_loss = running / max(1, steps)
        log.info("epoch %d/%d loss=%.4f in %.1fs",
                 epoch, args.epochs, last_loss, time.time() - epoch_t)
        if last_loss + 1e-4 < best:
            best = last_loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d (best loss=%.4f)", epoch, best)
                last_loss = best
                break
    return last_loss, n_images, 0


def _model_card(args, metrics: dict, started: float) -> str:
    return f"""# {args.model_name} v{args.version}

Trained by BFF Camera Demo v0.1.

## Hyperparameters
- epochs: {args.epochs}
- batch_size: {args.batch_size}
- lr: {args.lr}
- device: {args.device}
- data: {args.data_path}

## Metrics
- train_loss: {metrics.get("train_loss", "n/a")}
- val_loss: {metrics.get("val_loss", "n/a")}
- mAP50: {metrics.get("mAP50", "n/a")}
- fill_acc: {metrics.get("fill_acc", "n/a")}
- not_visible_recall: {metrics.get("not_visible_recall", "n/a")}

## Notes
- Wall time: {time.time() - started:.1f}s
- Run on: {os.uname().sysname if hasattr(os, "uname") else os.name}
- Generated: {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
"""


if __name__ == "__main__":
    from ml.env import load_env_files
    load_env_files()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
