"""Realized variance / volatility, bipower variation, and HAR-RV."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.timeseries import (
    HARRV,
    bipower_variation,
    realized_variance,
    realized_volatility,
)
from qufin.timeseries._types import ForecastResult


def test_realized_variance_scalar_and_rolling() -> None:
    r = np.array([0.01, -0.02, 0.015, -0.005])
    assert realized_variance(r) == pytest.approx(float(np.sum(r**2)))
    roll = realized_variance(r, window=2)
    assert isinstance(roll, np.ndarray)
    assert np.isnan(roll[0])
    assert roll[1] == pytest.approx(r[0] ** 2 + r[1] ** 2)


def test_realized_volatility_annualize() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=2520)  # 10y of daily returns
    vol = realized_volatility(r, annualize=True, periods=252.0)
    assert vol == pytest.approx(0.01 * np.sqrt(252.0), rel=0.1)


def test_bipower_close_to_rv_without_jumps() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, size=5000)
    rv = realized_variance(r)
    bv = bipower_variation(r)
    assert bv / rv == pytest.approx(1.0, abs=0.1)


def test_bipower_robust_to_jump() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.01, size=2000)
    r[1000] += 0.5  # a large jump
    rv = realized_variance(r)
    bv = bipower_variation(r)
    assert rv > bv  # RV absorbs the jump, BV largely ignores it


def test_harrv_fit_and_forecast() -> None:
    rng = np.random.default_rng(3)
    # Persistent positive realized-variance series (log-AR(1)).
    log_rv = np.empty(1000)
    log_rv[0] = -9.0
    for t in range(1, 1000):
        log_rv[t] = -9.0 + 0.95 * (log_rv[t - 1] + 9.0) + rng.normal(0.0, 0.3)
    rv = np.exp(log_rv)

    res = HARRV().fit(rv)
    assert res.beta.shape == (4,)
    assert 0.0 <= res.r_squared <= 1.0
    assert np.isfinite(res.aic) and np.isfinite(res.bic)

    fc = res.forecast(10)
    assert isinstance(fc, ForecastResult)
    assert fc.mean.shape == (10,)
    assert np.all(fc.mean >= 0.0)

    fc_int = res.forecast(5, alpha=0.05)
    assert fc_int.lower is not None and fc_int.upper is not None
    assert np.all(fc_int.upper >= fc_int.mean)
    assert np.all(fc_int.lower >= 0.0)


def test_harrv_requires_enough_data() -> None:
    with pytest.raises(ValueError):
        HARRV().fit(np.ones(10))
