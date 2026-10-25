"""Dataset cache TTL + force-download behavior.

`system-map-arsitektur.MD` §5.22.1 specifies a local cache under
`data/cache/<source_hash>/` with a 7-day TTL and a `--force-download`
override. Before this, the cache never expired: once a prefix was
downloaded it was reused forever, so a re-annotated dataset could never
be picked up without manually deleting the cache directory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ml.data.load import (
    CACHE_TTL_DAYS,
    S3Dataset,
    cache_age_s,
    is_cache_fresh,
)


# ---------------------------------------------------------------------------
# cache_age_s / is_cache_fresh
# ---------------------------------------------------------------------------

def test_cache_age_missing_dir_is_infinite(tmp_path: Path) -> None:
    assert cache_age_s(tmp_path / "nope") == float("inf")


def test_cache_age_empty_dir_is_infinite(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert cache_age_s(tmp_path / "empty") == float("inf")


def test_cache_age_of_populated_dir(tmp_path: Path) -> None:
    d = tmp_path / "cache"
    d.mkdir()
    (d / "a.txt").write_text("x")
    age = cache_age_s(d)
    assert 0 <= age < 60


def test_is_cache_fresh_requires_content(tmp_path: Path) -> None:
    d = tmp_path / "cache"
    d.mkdir()
    # Exists but empty -> not fresh (nothing was downloaded).
    assert not is_cache_fresh(d)


def test_is_cache_fresh_rejects_missing(tmp_path: Path) -> None:
    assert not is_cache_fresh(tmp_path / "nope")


def test_is_cache_fresh_true_for_recent(tmp_path: Path) -> None:
    d = tmp_path / "cache"
    d.mkdir()
    (d / "a.txt").write_text("x")
    assert is_cache_fresh(d)


def test_is_cache_fresh_false_when_expired(tmp_path: Path) -> None:
    d = tmp_path / "cache"
    d.mkdir()
    (d / "a.txt").write_text("x")
    # 8 days old > 7 day default TTL.
    old = time.time() - (CACHE_TTL_DAYS + 1) * 86400
    os.utime(d / "a.txt", (old, old))
    assert not is_cache_fresh(d)


def test_is_cache_fresh_zero_ttl_always_false(tmp_path: Path) -> None:
    d = tmp_path / "cache"
    d.mkdir()
    (d / "a.txt").write_text("x")
    assert not is_cache_fresh(d, ttl_days=0)


# ---------------------------------------------------------------------------
# S3Dataset cache lifecycle
# ---------------------------------------------------------------------------

def _make_cache(tmp_path: Path, bucket: str = "b", prefix: str = "p") -> Path:
    """Create the cache dir S3Dataset would use, with one stale file."""
    ds = S3Dataset(bucket=bucket, prefix=prefix, cache_dir=str(tmp_path / "cache"))
    ds.cache.mkdir(parents=True, exist_ok=True)
    (ds.cache / "old.txt").write_text("old")
    old = time.time() - (CACHE_TTL_DAYS + 1) * 86400
    os.utime(ds.cache / "old.txt", (old, old))
    return ds.cache


def test_stale_cache_triggers_redownload(tmp_path: Path, monkeypatch) -> None:
    """An expired cache must be purged and re-fetched."""
    cache = _make_cache(tmp_path)
    calls: list[str] = []

    def fake_download(self) -> None:
        calls.append(str(self.cache))
        (self.cache / "new.txt").write_text("new")

    monkeypatch.setattr(S3Dataset, "_download_prefix", fake_download)

    ds = S3Dataset(bucket="b", prefix="p", cache_dir=str(tmp_path / "cache"))
    ds.sync()

    assert calls == [str(cache)], "expired cache must be re-downloaded"
    assert not (cache / "old.txt").exists(), "stale files must be purged"
    assert (cache / "new.txt").exists()


def test_fresh_cache_skips_download(tmp_path: Path, monkeypatch) -> None:
    """A cache inside the TTL must be reused without any network call."""
    ds = S3Dataset(bucket="b", prefix="p", cache_dir=str(tmp_path / "cache"))
    ds.cache.mkdir(parents=True, exist_ok=True)
    (ds.cache / "a.txt").write_text("a")

    def boom(self) -> None:
        raise AssertionError("fresh cache must not trigger a download")

    monkeypatch.setattr(S3Dataset, "_download_prefix", boom)
    ds.sync()
    assert (ds.cache / "a.txt").exists()


def test_force_download_purges_and_refetches(tmp_path: Path, monkeypatch) -> None:
    """--force-download must ignore a perfectly fresh cache."""
    ds = S3Dataset(bucket="b", prefix="p", cache_dir=str(tmp_path / "cache"))
    ds.cache.mkdir(parents=True, exist_ok=True)
    (ds.cache / "a.txt").write_text("a")

    calls: list[str] = []

    def fake_download(self) -> None:
        calls.append("download")
        (self.cache / "b.txt").write_text("b")

    monkeypatch.setattr(S3Dataset, "_download_prefix", fake_download)

    forced = S3Dataset(bucket="b", prefix="p", cache_dir=str(tmp_path / "cache"),
                       force=True)
    forced.sync()

    assert calls == ["download"]
    assert not (ds.cache / "a.txt").exists(), "force must purge the old cache"
    assert (ds.cache / "b.txt").exists()


def test_custom_ttl_is_respected(tmp_path: Path, monkeypatch) -> None:
    """A 1-day TTL must expire a 2-day-old cache."""
    ds = S3Dataset(bucket="b", prefix="p", cache_dir=str(tmp_path / "cache"))
    ds.cache.mkdir(parents=True, exist_ok=True)
    (ds.cache / "a.txt").write_text("a")
    two_days = time.time() - 2 * 86400
    os.utime(ds.cache / "a.txt", (two_days, two_days))

    calls: list[str] = []

    def fake_download(self) -> None:
        calls.append("download")

    monkeypatch.setattr(S3Dataset, "_download_prefix", fake_download)

    S3Dataset(bucket="b", prefix="p", cache_dir=str(tmp_path / "cache"),
              ttl_days=1).sync()
    assert calls == ["download"], "2-day-old cache must expire under a 1-day TTL"


def test_zero_ttl_forces_redownload(tmp_path: Path, monkeypatch) -> None:
    """ttl_days=0 means "never trust the cache"."""
    ds = S3Dataset(bucket="b", prefix="p", cache_dir=str(tmp_path / "cache"))
    ds.cache.mkdir(parents=True, exist_ok=True)
    (ds.cache / "a.txt").write_text("a")

    calls: list[str] = []
    monkeypatch.setattr(
        S3Dataset, "_download_prefix",
        lambda self: calls.append("download"),
    )
    S3Dataset(bucket="b", prefix="p", cache_dir=str(tmp_path / "cache"),
              ttl_days=0).sync()
    assert calls == ["download"]