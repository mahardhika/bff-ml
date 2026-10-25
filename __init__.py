"""BFF Camera Demo ML package.

Training pipeline (YOLOX-style detector + 5-class fill classifier) plus
the Supabase sync that publishes the resulting ONNX weights and metrics
to the `models-public` bucket and the `bff_models` / `bff_experiments`
tables.

Run via:
    bff-ml train --data-source=local --data-path=./data/train
    bff-ml sync-results --run-name=...
"""

__version__ = "0.1.0"
__all__ = ["__version__"]