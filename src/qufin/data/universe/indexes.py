"""
Loaders for index-membership CSV files.

The expected file format is one row per (symbol, index, start, end) tuple
with ISO-8601 timestamps. ``end`` may be empty for currently-active members,
in which case it is filled with ``datetime.max`` (UTC).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .pit import MEMBERSHIP_SCHEMA

UTC = ZoneInfo("UTC")
_SENTINEL_END = datetime(2200, 1, 1, tzinfo=UTC)


def load_membership_csv(path: Path | str) -> pl.DataFrame:
    """Load a CSV with columns ``(symbol, index, start, end)``.

    ``start`` and ``end`` are parsed as UTC. An empty ``end`` is treated as
    "currently a member" and filled with the year-9999 sentinel.
    """
    df = pl.read_csv(path, try_parse_dates=False)
    missing = {"symbol", "index", "start", "end"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df = df.with_columns(
        pl.col("start").str.to_datetime(time_zone="UTC", strict=True),
        pl.col("end")
        .str.to_datetime(time_zone="UTC", strict=False)
        .fill_null(_SENTINEL_END),
    )
    return df.select(
        *(pl.col(name).cast(dtype) for name, dtype in MEMBERSHIP_SCHEMA.items())
    )
