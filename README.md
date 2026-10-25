# bff-ml

Training pipeline for the BFF Camera Demo: YOLOX-style bucket detector
(`bucket` / `truck` / `person`), 5-class fill classifier
(`empty` / `partial` / `full` / `over` / `not_visible`), ONNX export, and
Supabase Storage sync.

## Install

```bash
pip install "bff-core @ git+https://github.com/bff-demo/bff-core.git@v0.1.0"
pip install -e ".[cuda,dev]"
```

## Train

```bash
python -m ml train --data-source=local \
    --data-path=./data/train/reference-screenshots/detector \
    --kind=detector --model-name=bucket-detector --version=0.1.1 \
    --epochs=80 --batch-size=8 --device=cuda --no-sync
```

Output lands in `runs/<model>-<version>-<timestamp>/`:
`detector.onnx`, `model.pt`, `metrics.json`, `hparams.json`, `MODEL_CARD.md`.

## Colab

`train.ipynb` and `demo_colab.ipynb` are the cloud path. They clone
`bff-core`, `bff-backend`, `bff-frontend` and `bff-ml` separately - see
`docs/train-guide/TUTORIAL_COLAB.md` in `bff-ops`.
