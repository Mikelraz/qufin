"""Tests for volatility indicators."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.indicators import (
    atr,
    bollinger_bands,
    donchian_channels,
    keltner_channels,
    true_range,
)


def test_true_range_first_bar_is_hl(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    tr = true_range(highs, lows, closes)
    assert tr[0] == pytest.approx(highs[0] - lows[0])


def test_true_range_ge_high_low(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    tr = true_range(highs, lows, closes)
    assert (tr >= (highs - lows) - 1e-12).all()


def test_atr_non_negative_and_finite(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    a = atr(highs, lows, closes, window=14)
    valid = a[~np.isnan(a)]
    assert (valid > 0.0).all() and np.isfinite(valid).all()


def test_bollinger_envelope_contains_middle(synthetic_close: np.ndarray) -> None:
    bb = bollinger_bands(synthetic_close, window=20, n_std=2.0)
    valid = ~np.isnan(bb.middle)
    assert (bb.upper[valid] >= bb.middle[valid]).all()
    assert (bb.lower[valid] <= bb.middle[valid]).all()


def test_bollinger_percent_b_in_range(synthetic_close: np.ndarray) -> None:
    bb = bollinger_bands(synthetic_close, window=20, n_std=2.0)
    pb = bb.percent_b[~np.isnan(bb.percent_b)]
    assert np.isfinite(pb).all()


def test_keltner_envelope(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    kc = keltner_channels(highs, lows, closes, window=20, atr_window=10, atr_mult=2.0)
    valid = ~np.isnan(kc.middle) & ~np.isnan(kc.upper)
    assert (kc.upper[valid] > kc.middle[valid]).all()
    assert (kc.lower[valid] < kc.middle[valid]).all()


def test_donchian_bounds_actual_prices(synthetic_ohlc) -> None:
    highs, lows, _ = synthetic_ohlc
    dc = donchian_channels(highs, lows, window=20)
    valid = ~np.isnan(dc.upper)
    assert (dc.upper[valid] >= highs[valid]).all()
    assert (dc.lower[valid] <= lows[valid]).all()
