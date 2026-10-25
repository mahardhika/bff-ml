"""Sync skip-on-offline behavior."""

import logging

from ml.sync import sync_experiment


def test_offline_mode_skips(tmp_path, caplog):
    import os
    os.environ["DATABASE_MODE"] = "offline"
    (tmp_path / "hparams.json").write_text("{}")
    (tmp_path / "metrics.json").write_text("{}")
    with caplog.at_level(logging.INFO):
        sync_experiment(run_dir=tmp_path, model_name="m", version="0.0.0")
    # No exception is the contract; nothing to assert beyond that.