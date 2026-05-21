"""Shared fixtures for trading tests."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _load_dotenv_once() -> None:
    """Populate ``os.environ`` from the project-root ``.env`` if present.

    Lets credential-gated tests (e.g. live Alpaca paper) pick up the keys
    without requiring users to ``source`` the file themselves. The file is
    gitignored so this is safe.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv_once()


@pytest.fixture
def synthetic_bars() -> dict[str, pl.DataFrame]:
    """Deterministic ramping bars for one symbol over 50 trading days.

    Close starts near 100 and increases by 1 each day plus light noise so a
    buy-and-hold target-weight strategy has a known equity curve. Useful
    for invariant checks (parity, lookahead).
    """
    n = 50
    rng = np.random.default_rng(0)
    base = np.arange(100, 100 + n, dtype=np.float64)
    noise = rng.normal(scale=0.05, size=n)
    close = base + noise
    open_ = close - 0.1
    high = close + 0.3
    low = close - 0.3
    start = datetime(2024, 1, 2, tzinfo=UTC)
    timestamps = pl.datetime_range(
        start=start,
        end=start + timedelta(days=n - 1),
        interval="1d",
        eager=True,
        time_zone="UTC",
    )
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    ).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", time_zone="UTC")),
        pl.col("open").cast(pl.Float64()),
        pl.col("high").cast(pl.Float64()),
        pl.col("low").cast(pl.Float64()),
        pl.col("close").cast(pl.Float64()),
        pl.col("volume").cast(pl.Float64()),
    )
    return {"AAA": frame}
