"""Range-based realized-volatility estimators: Parkinson, GK, RS, Yang-Zhang."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.indicators import garman_klass, parkinson, rogers_satchell, yang_zhang


def sim_ohlc(
    n: int, daily_sigma: float, seed: int, *, steps: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate OHLC bars whose intraday path has the given daily volatility."""
    rng = np.random.default_rng(seed)
    o = np.empty(n)
    h = np.empty(n)
    low = np.empty(n)
    c = np.empty(n)
    price = 100.0
    step_sigma = daily_sigma / np.sqrt(steps)
    for t in range(n):
        o[t] = price
        path = price * np.exp(np.cumsum(rng.normal(0.0, step_sigma, size=steps)))
        h[t] = max(path.max(), price)
        low[t] = min(path.min(), price)
        c[t] = path[-1]
        price = c[t]
    return o, h, low, c


ESTIMATORS = [parkinson, garman_klass, rogers_satchell, yang_zhang]


@pytest.mark.parametrize("est", ESTIMATORS)
def test_shape_and_warmup(est) -> None:
    o, h, low, c = sim_ohlc(300, 0.01, seed=0)
    args = (h, low) if est is parkinson else (o, h, low, c)
    out = est(*args, window=20)
    assert out.shape == (300,)
    assert np.isnan(out[0])
    assert np.all(out[100:] > 0.0)


@pytest.mark.parametrize("est", ESTIMATORS)
def test_recovers_order_of_magnitude(est) -> None:
    sigma = 0.012
    o, h, low, c = sim_ohlc(600, sigma, seed=1)
    args = (h, low) if est is parkinson else (o, h, low, c)
    out = est(*args, window=30, trading_periods=252.0)
    estimate = float(np.nanmean(out))
    assert estimate == pytest.approx(sigma * np.sqrt(252.0), rel=0.4)


@pytest.mark.parametrize("est", ESTIMATORS)
def test_sensitivity_to_volatility(est) -> None:
    lo = sim_ohlc(400, 0.008, seed=2)
    hi = sim_ohlc(400, 0.020, seed=2)
    a_lo = (lo[1], lo[2]) if est is parkinson else lo
    a_hi = (hi[1], hi[2]) if est is parkinson else hi
    assert np.nanmean(est(*a_hi, window=30)) > np.nanmean(est(*a_lo, window=30))
