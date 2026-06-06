"""Momentum factors: trailing return, TSMOM, cross-sectional momentum."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.analysis import (
    MomentumFactorResult,
    cross_sectional_momentum,
    time_series_momentum,
    trailing_return,
    volatility_scaled_signal,
)


def test_trailing_return_simple_and_log() -> None:
    prices = np.array([1.0, 2.0, 4.0, 8.0])
    simple = trailing_return(prices, 1, log=False)
    assert np.isnan(simple[0])
    np.testing.assert_allclose(simple[1:], 1.0)
    log_r = trailing_return(prices, 2, log=True)
    np.testing.assert_allclose(log_r[2:], np.log([4.0, 4.0]))


def test_volatility_scaled_signal_targets_vol() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=1000)
    r_in = np.concatenate([[np.nan], r[1:]])
    signal = np.ones(1000)
    pos = volatility_scaled_signal(signal, r_in, vol_window=60, target_vol=0.10, leverage_cap=100.0)
    realised = np.nanstd(pos[200:] * r[200:]) * np.sqrt(252.0)
    assert realised == pytest.approx(0.10, rel=0.25)


def test_tsmom_long_in_uptrend() -> None:
    rng = np.random.default_rng(1)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.005, size=1500)))
    res = time_series_momentum(prices, lookback=252, vol_window=60)
    assert isinstance(res, MomentumFactorResult)
    assert res.kind == "time_series"
    assert np.nanmean(res.signal[400:]) > 0.8
    assert res.cumulative_return > 0.0


def test_tsmom_profits_in_downtrend_by_shorting() -> None:
    rng = np.random.default_rng(2)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(-0.001, 0.005, size=1500)))
    res = time_series_momentum(prices, lookback=252, vol_window=60)
    assert np.nanmean(res.signal[400:]) < -0.8
    assert res.cumulative_return > 0.0  # short position profits as price falls


def test_cross_sectional_weights_dollar_neutral() -> None:
    t_len, n = 600, 6
    drifts = np.linspace(-0.0015, 0.0015, n)  # asset 0 falls, asset 5 rises
    rng = np.random.default_rng(3)
    log_p = np.cumsum(drifts + rng.normal(0.0, 0.004, size=(t_len, n)), axis=0)
    prices = 100.0 * np.exp(log_p)
    res = cross_sectional_momentum(prices, lookback=120, skip=5, holding=20, n_quantiles=5)
    assert res.kind == "cross_sectional"
    assert res.weights.shape == (t_len, n)
    # Each rebalanced row is dollar-neutral.
    row_sums = res.weights[130:].sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-9)
    # The strongest riser is long, the strongest faller short.
    assert res.weights[-1, 5] > 0.0
    assert res.weights[-1, 0] < 0.0
    assert res.cumulative_return > 0.0


def test_cross_sectional_requires_2d() -> None:
    with pytest.raises(ValueError):
        cross_sectional_momentum(np.ones(100))


def test_tsmom_rejects_short_series() -> None:
    with pytest.raises(ValueError):
        time_series_momentum(np.linspace(1.0, 2.0, 100), lookback=252, vol_window=60)
