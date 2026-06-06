"""Spread utilities: hedge ratio, spread, half-life, z-scores."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qufin.models import half_life, hedge_ratio, rolling_zscore, spread, zscore


def _ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=n)
    s = np.empty(n)
    s[0] = 0.0
    for t in range(1, n):
        s[t] = phi * s[t - 1] + eps[t]
    return s


def test_hedge_ratio_ols_recovers_slope() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=2000)
    y = 2.0 * x + rng.normal(0.0, 0.01, size=2000)
    assert hedge_ratio(y, x, method="ols") == pytest.approx(2.0, abs=0.01)


def test_hedge_ratio_tls_and_kalman_recover_slope() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=2000)
    y = 1.5 * x + rng.normal(0.0, 0.02, size=2000)
    assert hedge_ratio(y, x, method="tls") == pytest.approx(1.5, abs=0.05)
    assert hedge_ratio(y, x, method="kalman", delta=1e-3) == pytest.approx(1.5, abs=0.2)


def test_spread_definition() -> None:
    y = np.array([3.0, 5.0, 7.0])
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(spread(y, x, beta=2.0, alpha=0.5), y - 2.0 * x - 0.5)


def test_half_life_recovers_ar1_mean_reversion() -> None:
    phi = 0.9
    s = _ar1(phi, 20_000)
    expected = math.log(2.0) / (1.0 - phi)
    assert half_life(s) == pytest.approx(expected, rel=0.2)


def test_half_life_infinite_when_not_mean_reverting() -> None:
    explosive = 1.01 ** np.arange(100, dtype=np.float64)
    assert half_life(explosive) == math.inf


def test_zscore_standardises() -> None:
    z = zscore(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert np.std(z) == pytest.approx(1.0)


def test_rolling_zscore_causal_and_warmup() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, size=200)
    z = rolling_zscore(x, window=20, min_periods=5)
    assert np.all(np.isnan(z[:4]))  # fewer than min_periods
    assert np.isfinite(z[50])
    # Last value uses only the trailing window ending at t (causal): 20 elements.
    win = x[180:200]
    assert z[-1] == pytest.approx((x[-1] - win.mean()) / win.std())
