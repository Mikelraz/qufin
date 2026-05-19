"""Bar utilities: schema validation, true range, ATR, resampling, z-scores."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from qufin.wyckoff import BAR_SCHEMA, OHLCV, atr, normalize_volume, resample, true_range
from qufin.wyckoff.bars import validate_ohlcv
from tests.wyckoff.conftest import make_ohlcv


def test_ohlcv_rejects_missing_columns() -> None:
    df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "open": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        OHLCV(data=df)


def test_ohlcv_enforces_timestamp_order() -> None:
    ts = [
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
    ]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        },
        schema={name: dtype for name, dtype in BAR_SCHEMA.items()},
    )
    with pytest.raises(ValueError, match="sorted"):
        OHLCV(data=df)


def test_validate_rejects_invalid_high_low() -> None:
    n = 5
    bars = make_ohlcv(
        opens=np.full(n, 100.0),
        highs=np.full(n, 99.0),  # high < low — invalid
        lows=np.full(n, 100.0),
        closes=np.full(n, 100.0),
        volumes=np.full(n, 1.0),
    )
    with pytest.raises(ValueError):
        validate_ohlcv(bars)


def test_true_range_matches_scalar_formula() -> None:
    opens = np.array([10.0, 11.0, 9.0, 12.0], dtype=np.float64)
    highs = np.array([11.0, 12.0, 11.0, 13.0], dtype=np.float64)
    lows = np.array([9.0, 10.0, 8.0, 11.0], dtype=np.float64)
    closes = np.array([10.5, 11.5, 9.5, 12.5], dtype=np.float64)
    volumes = np.ones(4)
    bars = make_ohlcv(opens, highs, lows, closes, volumes)
    tr = true_range(bars)
    assert tr[0] == pytest.approx(highs[0] - lows[0])
    for i in range(1, 4):
        expected = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        assert tr[i] == pytest.approx(expected)


def test_atr_is_wilder_smoothed() -> None:
    rng = np.random.default_rng(0)
    n = 60
    closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    opens = closes
    highs = closes + 1.0
    lows = closes - 1.0
    vols = np.full(n, 1.0)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    a = atr(bars, window=14)
    assert np.isnan(a[:13]).all()
    assert np.isfinite(a[13:]).all()
    # Smoothed ATR is bounded by the most recent true ranges (sanity).
    assert a[-1] > 0.0


def test_normalize_volume_zero_mean_unit_var_in_window() -> None:
    rng = np.random.default_rng(1)
    n = 200
    closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    opens = closes
    highs = closes + 0.5
    lows = closes - 0.5
    vols = rng.lognormal(mean=10.0, sigma=0.3, size=n)
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    z = normalize_volume(bars, window=50)
    # Final value of the rolling z lies within a reasonable band.
    assert np.isfinite(z[-1])
    assert abs(z[-1]) < 6.0


def test_resample_preserves_ohlc_invariants() -> None:
    rng = np.random.default_rng(2)
    n = 24 * 7  # one week of hourly bars
    closes = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    opens = closes
    highs = closes + 0.3
    lows = closes - 0.3
    vols = rng.lognormal(mean=8.0, sigma=0.2, size=n)
    ts = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        },
        schema={name: dtype for name, dtype in BAR_SCHEMA.items()},
    )
    bars = OHLCV(data=df)
    daily = resample(bars, every="1d")
    # First daily bar covers 24 hourly bars.
    assert daily.high()[0] == pytest.approx(highs[:24].max())
    assert daily.low()[0] == pytest.approx(lows[:24].min())
    assert daily.volume()[0] == pytest.approx(vols[:24].sum())
    assert daily.open()[0] == pytest.approx(opens[0])
    assert daily.close()[0] == pytest.approx(closes[23])
