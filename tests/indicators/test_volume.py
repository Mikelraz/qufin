"""Tests for volume indicators."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.indicators import (
    accumulation_distribution,
    cmf,
    mfi,
    obv,
    rolling_vwap,
    vwap,
)


def test_obv_known_pattern() -> None:
    close = np.array([10.0, 11.0, 10.5, 10.5, 12.0])
    volume = np.array([100.0, 200.0, 150.0, 80.0, 120.0])
    out = obv(close, volume)
    assert out[0] == 0.0
    assert out[1] == 200.0
    assert out[2] == 50.0
    assert out[3] == 50.0
    assert out[4] == 170.0


def test_vwap_in_price_range(synthetic_ohlcv) -> None:
    highs, lows, closes, volumes = synthetic_ohlcv
    out = vwap(highs, lows, closes, volumes)
    assert (out >= lows.min() - 1e-6).all()
    assert (out <= highs.max() + 1e-6).all()


def test_rolling_vwap_matches_global_when_window_eq_n(synthetic_ohlcv) -> None:
    highs, lows, closes, volumes = synthetic_ohlcv
    n = closes.shape[0]
    rv = rolling_vwap(highs, lows, closes, volumes, window=n)
    full = vwap(highs, lows, closes, volumes)
    assert rv[-1] == pytest.approx(full[-1])


def test_mfi_in_range(synthetic_ohlcv) -> None:
    highs, lows, closes, volumes = synthetic_ohlcv
    out = mfi(highs, lows, closes, volumes, window=14)
    valid = out[~np.isnan(out)]
    assert (valid >= 0.0).all() and (valid <= 100.0).all()


def test_cmf_in_range(synthetic_ohlcv) -> None:
    highs, lows, closes, volumes = synthetic_ohlcv
    out = cmf(highs, lows, closes, volumes, window=20)
    valid = out[~np.isnan(out)]
    assert (valid >= -1.0).all() and (valid <= 1.0).all()


def test_ad_line_runs(synthetic_ohlcv) -> None:
    highs, lows, closes, volumes = synthetic_ohlcv
    out = accumulation_distribution(highs, lows, closes, volumes)
    assert out.shape == closes.shape
    assert np.isfinite(out).all()
