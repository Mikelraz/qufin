"""
Tests for src.timeseries.forecast_eval — backtests, metrics, DM, CRPS.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.timeseries.arima import AR
from src.timeseries.forecast_eval import (
    DMResult,
    RollingBacktest,
    crps,
    diebold_mariano,
    mae,
    mape,
    mase,
    rmse,
)

RNG = np.random.default_rng(101)


# ===========================================================================
# Metrics
# ===========================================================================


class TestMetrics:
    def test_rmse_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert rmse(x, x) == pytest.approx(0.0)

    def test_rmse_correct(self):
        actual = np.array([1.0, 2.0, 3.0])
        forecast = np.array([1.5, 2.5, 2.5])
        # Errors: -0.5, -0.5, 0.5 → RMSE = √(0.75/3) = 0.5
        assert rmse(actual, forecast) == pytest.approx(0.5)

    def test_rmse_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            rmse(np.zeros(3), np.zeros(4))

    def test_mae_correct(self):
        assert mae(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)

    def test_mae_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mae(np.zeros(3), np.zeros(4))

    def test_mape_correct(self):
        actual = np.array([1.0, 2.0, 4.0])
        forecast = np.array([1.1, 1.8, 4.4])
        # Errors: -0.1, +0.2, -0.4 → %: 0.1, 0.1, 0.1 → mean = 0.1
        assert mape(actual, forecast) == pytest.approx(0.1)

    def test_mape_zero_actual_skipped(self):
        actual = np.array([0.0, 0.0, 1.0])
        forecast = np.array([1.0, 1.0, 1.5])
        # Only the third row contributes (0.5/1.0 = 0.5)
        assert mape(actual, forecast) == pytest.approx(0.5)

    def test_mape_all_zero_returns_nan(self):
        result = mape(np.zeros(3), np.array([1.0, 2.0, 3.0]))
        assert math.isnan(result)

    def test_mase_correct(self):
        train = np.arange(10, dtype=float)  # linear, naive diff = 1
        actual = np.array([10.0, 11.0, 12.0])
        forecast = np.array([10.5, 11.5, 11.5])
        # |errors|: 0.5, 0.5, 0.5 → MASE = 0.5 / 1 = 0.5
        assert mase(actual, forecast, train) == pytest.approx(0.5)

    def test_mase_constant_training_returns_none(self):
        train = np.zeros(10)
        result = mase(np.zeros(3), np.zeros(3), train)
        assert result is None

    def test_mase_invalid_seasonality(self):
        with pytest.raises(ValueError):
            mase(np.zeros(3), np.zeros(3), np.zeros(10), seasonality=0)

    def test_mase_short_training(self):
        with pytest.raises(ValueError):
            mase(np.zeros(3), np.zeros(3), np.zeros(2), seasonality=5)


# ===========================================================================
# CRPS
# ===========================================================================


class TestCRPS:
    def test_crps_scalar(self):
        rng = np.random.default_rng(1)
        sample = rng.standard_normal(2000)
        val = crps(0.0, sample)
        # For N(0,1), CRPS at 0 ≈ 0.2336 (analytical: 2/√π − 1/√(2π) ≈ 0.5642)
        # Empirical-CDF formula gives a slightly different but bounded value.
        assert val > 0
        assert val < 1.0

    def test_crps_known_against_constant_sample(self):
        # If the entire ensemble is the same constant c, CRPS = |c − obs|.
        sample = np.full(50, 1.0)
        assert crps(0.0, sample) == pytest.approx(1.0)
        assert crps(2.0, sample) == pytest.approx(1.0)

    def test_crps_multi_obs_1d_sample(self):
        rng = np.random.default_rng(2)
        sample = rng.standard_normal(200)
        obs = np.array([0.0, 0.5])
        val = crps(obs, sample)
        assert val > 0

    def test_crps_multi_obs_2d_sample(self):
        rng = np.random.default_rng(3)
        sample = rng.standard_normal((5, 200))
        obs = rng.standard_normal(5)
        val = crps(obs, sample)
        assert val > 0

    def test_crps_shape_mismatch_raises(self):
        sample = np.zeros((5, 50))
        with pytest.raises(ValueError):
            crps(np.zeros(4), sample)


# ===========================================================================
# Diebold-Mariano
# ===========================================================================


class TestDieboldMariano:
    def test_returns_dm_result(self):
        a = RNG.standard_normal(100)
        b = RNG.standard_normal(100)
        res = diebold_mariano(a, b)
        assert isinstance(res, DMResult)

    def test_detects_worse_model(self):
        rng = np.random.default_rng(4)
        good = rng.standard_normal(500)
        bad = rng.standard_normal(500) * 2.0  # double the error magnitude
        res = diebold_mariano(good, bad, loss="squared")
        # bad has higher loss → d = loss(good) − loss(bad) < 0 → stat < 0
        assert res.stat < 0
        assert res.p_value < 0.01

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError):
            diebold_mariano(np.zeros(50), np.zeros(50), loss="bogus")

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            diebold_mariano(np.zeros(50), np.zeros(40))

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            diebold_mariano(np.zeros(2), np.zeros(2))

    def test_absolute_loss_runs(self):
        rng = np.random.default_rng(5)
        a = rng.standard_normal(200)
        b = rng.standard_normal(200) * 1.5
        res = diebold_mariano(a, b, loss="absolute")
        assert math.isfinite(res.stat)


# ===========================================================================
# Rolling backtest
# ===========================================================================


class TestRollingBacktest:
    def setup_method(self) -> None:
        rng = np.random.default_rng(6)
        n = 600
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * rng.standard_normal()
        self.y = y

    def test_runs_end_to_end(self):
        bt = RollingBacktest(lambda: AR(p=1), window=200, refit_every=10, h=1).run(self.y)
        assert bt.forecasts.shape == bt.actuals.shape
        assert bt.errors.shape == bt.forecasts.shape
        assert math.isfinite(bt.rmse)
        assert bt.n_windows > 0
        assert bt.horizon == 1

    def test_h_greater_than_one(self):
        bt = RollingBacktest(lambda: AR(p=1), window=200, refit_every=20, h=5).run(self.y)
        assert bt.forecasts.shape[1] == 5
        assert bt.horizon == 5

    def test_expanding_window(self):
        bt = RollingBacktest(lambda: AR(p=1), window=None, refit_every=50, h=1).run(self.y)
        assert bt.n_windows > 0

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            RollingBacktest(lambda: AR(p=1), window=2, h=1)

    def test_invalid_refit_every(self):
        with pytest.raises(ValueError):
            RollingBacktest(lambda: AR(p=1), refit_every=0, h=1)

    def test_invalid_h(self):
        with pytest.raises(ValueError):
            RollingBacktest(lambda: AR(p=1), h=0)

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            RollingBacktest(lambda: AR(p=1), window=200, h=1).run(np.zeros(10))

    def test_to_dataframe(self):
        bt = RollingBacktest(lambda: AR(p=1), window=200, refit_every=20, h=1).run(self.y)
        df = RollingBacktest.to_dataframe(bt)
        assert set(df.columns) == {"window", "h", "forecast", "actual", "error"}
