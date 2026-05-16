"""
Tests for src.timeseries.stationarity.

Correctness benchmarks
----------------------
* ADF rejects strongly on a stationary AR(1); does not reject on a random walk
* KPSS does not reject on a stationary AR(1); rejects on a random walk
* Phillips-Perron rejects on a stationary AR(1); does not reject on a random walk
* Variance-ratio Z statistic is near 0 for i.i.d. returns; far from 0 for
  persistently autocorrelated returns
* Result dataclasses carry the expected fields
* polars input round-trips
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.timeseries.stationarity import (
    ADFResult,
    KPSSResult,
    PPResult,
    VRResult,
    adf,
    kpss,
    phillips_perron,
    variance_ratio,
)

RNG = np.random.default_rng(99)


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _random_walk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


class TestADF:
    def test_returns_dataclass(self):
        x = _ar1(0.3, 500, seed=1)
        res = adf(x)
        assert isinstance(res, ADFResult)
        assert res.regression == "c"
        assert set(res.critical_values) == {0.01, 0.05, 0.10}
        assert res.used_lag >= 0

    def test_stationary_ar1_rejects(self):
        x = _ar1(0.3, 1_000, seed=2)
        res = adf(x, regression="c")
        assert res.stat < res.critical_values[0.05]
        assert res.p_value < 0.05

    def test_random_walk_does_not_reject(self):
        x = _random_walk(1_000, seed=3)
        res = adf(x, regression="c")
        assert res.p_value > 0.10

    def test_trend_regression(self):
        # A deterministic trend + stationary noise should reject when the
        # trend term is included.
        rng = np.random.default_rng(4)
        t = np.arange(500, dtype=np.float64)
        x = 0.5 * t + rng.standard_normal(500)
        res = adf(x, regression="ct")
        assert res.regression == "ct"
        assert res.p_value < 0.10

    def test_invalid_regression(self):
        with pytest.raises(ValueError):
            adf(RNG.standard_normal(100), regression="bad")

    def test_invalid_autolag(self):
        with pytest.raises(ValueError):
            adf(RNG.standard_normal(100), autolag="bad")

    def test_autolag_none_uses_maxlag(self):
        x = _ar1(0.5, 300, seed=5)
        res = adf(x, maxlag=4, autolag=None)
        assert res.used_lag == 4

    def test_polars_input(self):
        x = _ar1(0.4, 500, seed=6)
        res_np = adf(x)
        res_pl = adf(pl.Series("x", x))
        assert res_np.stat == pytest.approx(res_pl.stat)
        assert res_np.p_value == pytest.approx(res_pl.p_value)


class TestKPSS:
    def test_stationary_does_not_reject(self):
        # KPSS test has correct nominal size (~5% under H0); use a seed where
        # the realised path is clearly inside the acceptance region rather
        # than the borderline 5 % tail.
        x = _ar1(0.1, 2_000, seed=100)
        res = kpss(x, regression="c")
        assert isinstance(res, KPSSResult)
        assert res.p_value > 0.05

    def test_random_walk_rejects(self):
        x = _random_walk(500, seed=11)
        res = kpss(x, regression="c")
        assert res.stat > res.critical_values[0.05]
        assert res.p_value < 0.05

    def test_invalid_regression(self):
        with pytest.raises(ValueError):
            kpss(RNG.standard_normal(50), regression="n")


class TestPhillipsPerron:
    def test_stationary_rejects(self):
        x = _ar1(0.3, 1_000, seed=20)
        res = phillips_perron(x, regression="c")
        assert isinstance(res, PPResult)
        assert res.p_value < 0.05

    def test_random_walk_does_not_reject(self):
        x = _random_walk(1_000, seed=21)
        res = phillips_perron(x, regression="c")
        assert res.p_value > 0.10


class TestVarianceRatio:
    def test_iid_returns_near_one(self):
        rng = np.random.default_rng(30)
        r = rng.standard_normal(5_000)
        res = variance_ratio(r, q=4)
        assert isinstance(res, VRResult)
        # VR ≈ 1 ± a couple of standard errors for i.i.d. returns.
        assert abs(res.stat - 1.0) < 0.15
        assert res.p_value > 0.05

    def test_positive_autocorrelation_rejects(self):
        # Build returns r_t = 0.5 r_{t-1} + ε_t — highly persistent → VR > 1.
        rng = np.random.default_rng(31)
        n = 5_000
        eps = rng.standard_normal(n)
        r = np.empty(n)
        r[0] = eps[0]
        for t in range(1, n):
            r[t] = 0.5 * r[t - 1] + eps[t]
        res = variance_ratio(r, q=4)
        assert res.stat > 1.5
        assert res.p_value < 0.01

    def test_invalid_q(self):
        with pytest.raises(ValueError):
            variance_ratio(RNG.standard_normal(100), q=1)
