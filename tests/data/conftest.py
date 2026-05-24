"""Fixtures for qufin.data tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from qufin.data import BAR_SCHEMA, OHLCV


def make_ohlcv(
    n: int,
    *,
    start: datetime | None = None,
    step: timedelta = timedelta(days=1),
    symbol: str = "TEST",
    seed: int = 0,
) -> OHLCV:
    """Build a synthetic OHLCV frame with ``n`` bars on a UTC grid."""
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    opens = closes + rng.standard_normal(n) * 0.1
    highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n)) * 0.2
    lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n)) * 0.2
    volumes = (1_000_000 + rng.standard_normal(n) * 100_000).clip(min=1.0)
    ts = [start + step * i for i in range(n)]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        schema={name: dtype for name, dtype in BAR_SCHEMA.items()},
    )
    return OHLCV(data=df, symbol=symbol)
