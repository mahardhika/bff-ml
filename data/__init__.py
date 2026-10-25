"""Data loading utilities for training.

Three sources (see `system-map-arsitektur.MD` §5.22.1):
- `local` (CVAT COCO export on local disk)
- `s3` (Supabase Storage, downloaded to local cache)
- `url` (public dataset URL - deferred to v0.2)
"""

from ml.data.load import Dataset, LocalDataset, S3Dataset, UrlDataset, load_dataset

__all__ = ["Dataset", "LocalDataset", "S3Dataset", "UrlDataset", "load_dataset"]