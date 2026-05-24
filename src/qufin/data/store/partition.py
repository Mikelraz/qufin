"""Hive-style partition path helpers for the store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def partition_path(root: Path, symbol: str, interval: str, year: int) -> Path:
    """Return the parquet file for one (symbol, interval, year) partition."""
    return root / symbol / interval / f"year={year}" / "bars.parquet"


def symbol_root(root: Path, symbol: str, interval: str) -> Path:
    """Return the directory containing all year partitions for (symbol, interval)."""
    return root / symbol / interval


def years_in_range(start: datetime, end: datetime) -> range:
    """Inclusive year range spanned by [start, end]."""
    if end < start:
        raise ValueError("end must be >= start")
    return range(start.year, end.year + 1)
