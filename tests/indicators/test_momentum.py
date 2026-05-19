"""Tests for momentum oscillators."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.indicators import cci, macd, roc, rsi, stochastic, williams_r


def test_rsi_monotonic_up_is_100() -> None:
    x = np.arange(1.0, 50.0)
    r = rsi(x, window=14)
    assert np.allclose(r[14:], 100.0)


def test_rsi_monotonic_down_is_0() -> None:
    x = np.arange(50.0, 1.0, -1.0)
    r = rsi(x, window=14)
    assert np.allclose(r[14:], 0.0)


def test_rsi_in_unit_range(synthetic_close: np.ndarray) -> None:
    r = rsi(synthetic_close, window=14)
    valid = r[~np.isnan(r)]
    assert (valid >= 0.0).all() and (valid <= 100.0).all()


def test_macd_hist_is_macd_minus_signal(synthetic_close: np.ndarray) -> None:
    result = macd(synthetic_close, fast=12, slow=26, signal=9)
    valid = ~np.isnan(result.hist)
    assert np.allclose(result.hist[valid], result.macd[valid] - result.signal[valid])


def test_macd_fast_lt_slow_required() -> None:
    with pytest.raises(ValueError):
        macd(np.zeros(50), fast=26, slow=12)


def test_stochastic_in_unit_range(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    s = stochastic(highs, lows, closes, k_window=14, d_window=3)
    k_valid = s.k[~np.isnan(s.k)]
    d_valid = s.d[~np.isnan(s.d)]
    assert (k_valid >= 0.0).all() and (k_valid <= 100.0).all()
    assert (d_valid >= 0.0).all() and (d_valid <= 100.0).all()


def test_roc_known_value() -> None:
    x = np.array([100.0, 101.0, 110.0])
    r = roc(x, window=2)
    assert r[2] == pytest.approx(10.0)
    assert np.isnan(r[1])


def test_williams_r_in_range(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    wr = williams_r(highs, lows, closes, window=14)
    valid = wr[~np.isnan(wr)]
    assert (valid >= -100.0).all() and (valid <= 0.0).all()


def test_cci_runs(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    out = cci(highs, lows, closes, window=20)
    assert out.shape == closes.shape
    assert np.isfinite(out[-50:]).all()
