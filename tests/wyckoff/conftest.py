"""Shared fixtures for Wyckoff tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from qufin.wyckoff import BAR_SCHEMA, OHLCV


def make_ohlcv(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    *,
    start: datetime | None = None,
    symbol: str = "TEST",
) -> OHLCV:
    """Build an OHLCV frame with a daily UTC timestamp grid."""
    n = opens.shape[0]
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
    ts = [start + timedelta(days=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": opens.astype(np.float64),
            "high": highs.astype(np.float64),
            "low": lows.astype(np.float64),
            "close": closes.astype(np.float64),
            "volume": volumes.astype(np.float64),
        },
        schema={name: dtype for name, dtype in BAR_SCHEMA.items()},
    )
    return OHLCV(data=df, symbol=symbol)


def synthetic_drift(
    n: int,
    *,
    drift: float = 0.0,
    sigma: float = 0.01,
    base: float = 100.0,
    seed: int = 0,
) -> OHLCV:
    """Random-walk OHLCV with controllable drift and volatility."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, sigma, size=n)
    closes = base * np.exp(np.cumsum(log_rets))
    opens = np.concatenate(([base], closes[:-1]))
    bar_noise = rng.uniform(0.0, 0.5 * sigma, size=n) * closes
    highs = np.maximum(opens, closes) + bar_noise
    lows = np.minimum(opens, closes) - bar_noise
    volumes = rng.lognormal(mean=10.0, sigma=0.4, size=n)
    return make_ohlcv(opens, highs, lows, closes, volumes)
