"""Tests for moving averages."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.indicators import dema, ema, hma, kama, sma, tema, wma


def test_sma_matches_manual_mean() -> None:
    x = np.arange(1, 11, dtype=np.float64)
    out = sma(x, window=3)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[-1] == pytest.approx(9.0)


def test_sma_constant_input_is_constant() -> None:
    x = np.full(50, 7.5)
    out = sma(x, window=10)
    assert np.allclose(out[9:], 7.5)


def test_ema_seed_is_sma_of_first_window() -> None:
    x = np.arange(1, 21, dtype=np.float64)
    out = ema(x, window=5)
    assert out[4] == pytest.approx(3.0)  # mean of 1..5
    # Each subsequent value moves toward the latest input.
    assert out[5] > out[4]


def test_ema_constant_input_is_constant() -> None:
    x = np.full(50, 3.14)
    out = ema(x, window=10)
    assert np.allclose(out[9:], 3.14)


def test_wma_weights_most_recent() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = wma(x, window=3)
    expected = (1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0) / 6.0
    assert out[3] == pytest.approx(expected)


def test_dema_tracks_linear_trend(synthetic_close: np.ndarray) -> None:
    # On a slow trend DEMA should sit close to the source.
    trend = np.linspace(100.0, 200.0, 200)
    d = dema(trend, window=10)
    err = np.abs(d[-50:] - trend[-50:])
    assert err.max() < 2.0


def test_tema_runs_without_nans_past_warmup(synthetic_close: np.ndarray) -> None:
    t = tema(synthetic_close, window=10)
    assert np.isfinite(t[-50:]).all()


def test_hma_length_and_warmup(synthetic_close: np.ndarray) -> None:
    h = hma(synthetic_close, window=16)
    assert h.shape == synthetic_close.shape
    # Warm-up: WMA(16) needs 15 bars, then WMA(sqrt(16)=4) needs 3 more.
    assert np.isnan(h[:18]).all()
    assert np.isfinite(h[18:]).all()


def test_kama_within_input_range(synthetic_close: np.ndarray) -> None:
    k = kama(synthetic_close, window=10, fast=2, slow=30)
    valid = k[~np.isnan(k)]
    lo, hi = synthetic_close.min(), synthetic_close.max()
    assert (valid >= lo - 1e-6).all() and (valid <= hi + 1e-6).all()


def test_invalid_windows_raise() -> None:
    with pytest.raises(ValueError):
        sma(np.zeros(5), window=0)
    with pytest.raises(ValueError):
        ema(np.zeros(5), window=0)
    with pytest.raises(ValueError):
        hma(np.zeros(5), window=1)
    with pytest.raises(ValueError):
        kama(np.zeros(5), window=10, fast=30, slow=2)
