"""
Parquet-backed local cache for market data.

Files are named ``{symbol}__{interval}__{start}__{end}.parquet`` under the
cache root and contain a polars frame matching ``BAR_SCHEMA``. The cache is
content-addressable on those four keys; callers wanting cache invalidation
should simply ``cache.invalidate(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from ...wyckoff._types import BAR_SCHEMA, OHLCV


@dataclass(slots=True)
class ParquetCache:
    """Tiny on-disk cache for OHLC frames."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str, start: date, end: date) -> Path:
        return self.root / f"{symbol}__{interval}__{start.isoformat()}__{end.isoformat()}.parquet"

    def get(
        self, symbol: str, interval: str, start: date, end: date
    ) -> OHLCV | None:
        path = self._path(symbol, interval, start, end)
        if not path.exists():
            return None
        frame = pl.read_parquet(path)
        return OHLCV.from_records(frame, symbol=symbol)

    def put(self, ohlcv: OHLCV, interval: str, start: date, end: date) -> Path:
        path = self._path(ohlcv.symbol, interval, start, end)
        ohlcv.data.write_parquet(path)
        return path

    def invalidate(self, symbol: str, interval: str, start: date, end: date) -> None:
        path = self._path(symbol, interval, start, end)
        path.unlink(missing_ok=True)


__all__ = ["BAR_SCHEMA", "ParquetCache"]
