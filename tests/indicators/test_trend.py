"""Tests for trend / directional indicators."""

from __future__ import annotations

import numpy as np

from qufin.indicators import adx, aroon, ichimoku, parabolic_sar, supertrend


def test_adx_components_bounded(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    res = adx(highs, lows, closes, window=14)
    pdi = res.plus_di[~np.isnan(res.plus_di)]
    mdi = res.minus_di[~np.isnan(res.minus_di)]
    adx_v = res.adx[~np.isnan(res.adx)]
    assert (pdi >= 0.0).all() and (pdi <= 100.0).all()
    assert (mdi >= 0.0).all() and (mdi <= 100.0).all()
    assert (adx_v >= 0.0).all() and (adx_v <= 100.0).all()


def test_aroon_bounds(synthetic_ohlc) -> None:
    highs, lows, _ = synthetic_ohlc
    res = aroon(highs, lows, window=25)
    up = res.up[~np.isnan(res.up)]
    down = res.down[~np.isnan(res.down)]
    assert (up >= 0.0).all() and (up <= 100.0).all()
    assert (down >= 0.0).all() and (down <= 100.0).all()


def test_aroon_uptrend_saturates_up() -> None:
    highs = np.arange(1.0, 100.0)
    lows = highs - 0.5
    res = aroon(highs, lows, window=25)
    assert np.allclose(res.up[25:], 100.0)
    assert np.allclose(res.down[25:], 0.0)


def test_parabolic_sar_flips_direction_on_reversal() -> None:
    up = np.linspace(100.0, 120.0, 20)
    down = np.linspace(120.0, 100.0, 20)
    closes = np.concatenate([up, down])
    highs = closes + 0.5
    lows = closes - 0.5
    res = parabolic_sar(highs, lows)
    assert (res.direction == -1.0).any()
    assert (res.direction == 1.0).any()


def test_supertrend_direction_is_pm_one(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    res = supertrend(highs, lows, closes, window=10, multiplier=3.0)
    d = res.direction[~np.isnan(res.direction)]
    assert set(np.unique(d)).issubset({-1.0, 1.0})


def test_ichimoku_shapes(synthetic_ohlc) -> None:
    highs, lows, closes = synthetic_ohlc
    res = ichimoku(highs, lows, closes)
    n = closes.shape[0]
    assert res.tenkan.shape == (n,)
    assert res.kijun.shape == (n,)
    assert res.chikou.shape == (n,)
    assert res.senkou_a.shape == (n + 26,)
    assert res.senkou_b.shape == (n + 26,)
