"""
Generic CSV / Parquet ingester.

Loads bars from a directory of files keyed by symbol, e.g.
``<root>/<symbol>.csv`` or ``<root>/<symbol>.parquet``. Columns are
case-normalised; ``date`` / ``datetime`` are accepted in place of
``timestamp``. Timestamps without a timezone are treated as UTC.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from .._types import BAR_SCHEMA, OHLCV

_RENAMES: dict[str, str] = {
    "Date": "timestamp",
    "Datetime": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


@dataclass(slots=True)
class CsvOHLC:
    """``OHLCSource``-conforming loader for user-provided CSV / Parquet files."""

    root: Path
    extension: str = ".csv"

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def fetch(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        interval: str,  # noqa: ARG002 - retained for protocol compatibility
    ) -> OHLCV:
        path = self.root / f"{symbol}{self.extension}"
        if not path.exists():
            raise FileNotFoundError(f"no data file for {symbol!r} at {path}")
        if self.extension == ".parquet":
            df = pl.read_parquet(path)
        else:
            df = pl.read_csv(path, try_parse_dates=True)
        df = df.rename({k: v for k, v in _RENAMES.items() if k in df.columns})
        missing = set(BAR_SCHEMA) - set(df.columns)
        if missing:
            raise ValueError(f"file {path} is missing columns: {sorted(missing)}")
        ts = pl.col("timestamp")
        if df.schema["timestamp"] == pl.Datetime("ns"):
            ts = ts.dt.replace_time_zone("UTC")
        df = df.with_columns(
            ts.cast(pl.Datetime("ns", time_zone="UTC")),
            *(pl.col(c).cast(pl.Float64()) for c in ("open", "high", "low", "close", "volume")),
        )
        df = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
        df = df.sort("timestamp")
        return OHLCV.from_records(df, symbol=symbol)

    def fetch_many(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> dict[str, OHLCV]:
        return {
            sym: self.fetch(sym, start=start, end=end, interval=interval) for sym in symbols
        }
