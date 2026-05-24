"""
Canonical schemas and core containers for the qufin data layer.

Conventions
-----------
* Array-backed containers store ``np.ndarray`` of ``float64``.
* OHLCV frames carry a fixed polars schema (``BAR_SCHEMA``).
* Timestamps are timezone-aware UTC (``zoneinfo.ZoneInfo('UTC')``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import polars as pl

BAR_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ns", time_zone="UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
}

TICK_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ns", time_zone="UTC"),
    "price": pl.Float64(),
    "size": pl.Float64(),
}


def to_numpy_1d(x: Any) -> np.ndarray:
    """Coerce a 1-D polars Series / numpy array / array-like to float64 numpy."""
    match x:
        case np.ndarray():
            arr = x
        case pl.Series():
            arr = x.to_numpy()
        case _:
            arr = np.asarray(x)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D input, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float64)


@dataclass(slots=True)
class OHLCV:
    """
    Polars-backed OHLCV bar sequence.

    Attributes
    ----------
    data       Long-format polars DataFrame with the schema in ``BAR_SCHEMA``.
    symbol     Ticker symbol (informational).

    The frame must be in monotonically non-decreasing timestamp order; the
    constructor enforces this.
    """

    data: pl.DataFrame
    symbol: str = ""

    def __post_init__(self) -> None:
        missing = set(BAR_SCHEMA) - set(self.data.columns)
        if missing:
            raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")
        if self.data.height >= 2:
            ts = self.data["timestamp"]
            if not ts.is_sorted():
                raise ValueError("OHLCV data must be sorted by timestamp ascending.")

    @classmethod
    def from_records(cls, records: pl.DataFrame, *, symbol: str = "") -> Self:
        """Coerce a DataFrame to ``BAR_SCHEMA`` and construct an ``OHLCV``."""
        missing = set(BAR_SCHEMA) - set(records.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing columns: {sorted(missing)}")
        coerced = records.select(*(pl.col(name).cast(dtype) for name, dtype in BAR_SCHEMA.items()))
        return cls(data=coerced, symbol=symbol)

    def __len__(self) -> int:
        return self.data.height

    @property
    def n_bars(self) -> int:
        return self.data.height

    def timestamps(self) -> np.ndarray:
        return self.data["timestamp"].to_numpy()

    def open(self) -> np.ndarray:
        return self.data["open"].to_numpy().astype(np.float64, copy=False)

    def high(self) -> np.ndarray:
        return self.data["high"].to_numpy().astype(np.float64, copy=False)

    def low(self) -> np.ndarray:
        return self.data["low"].to_numpy().astype(np.float64, copy=False)

    def close(self) -> np.ndarray:
        return self.data["close"].to_numpy().astype(np.float64, copy=False)

    def volume(self) -> np.ndarray:
        return self.data["volume"].to_numpy().astype(np.float64, copy=False)

    def slice_bars(self, start: int, end: int) -> Self:
        """Return a new OHLCV with rows [start, end) — half-open."""
        return type(self)(data=self.data.slice(start, end - start), symbol=self.symbol)
