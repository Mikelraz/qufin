"""
Tests for qufin.timeseries.garch — GARCH, EGARCH, GJR, EWMA.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from qufin.timeseries.garch import (
    EGARCH,
    EWMA,
    GARCH,
    GJR,
    EGARCHFitResult,
    EWMAResult,
    GARCHFitResult,
    GJRFitResult,
)

RNG = np.random.default_rng(2026)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _simulate_garch11(omega: float, alpha: float, beta: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    s2 = np.full(n, omega / max(1.0 - alpha - beta, 1e-6))
    z = rng.standard_normal(n)
    for t in range(1, n):
        s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
        eps[t] = math.sqrt(s2[t]) * z[t]
    return eps


def _simulate_gjr11(
    omega: float, alpha: float, gamma: float, beta: float, n: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    s2 = np.full(n, omega / max(1.0 - alpha - 0.5 * gamma - beta, 1e-6))
    z = rng.standard_normal(n)
    for t in range(1, n):
        prev = eps[t - 1]
        v = omega + alpha * prev * prev + beta * s2[t - 1]
        if prev < 0:
            v += gamma * prev * prev
        s2[t] = v
        eps[t] = math.sqrt(s2[t]) * z[t]
    return eps


# ===========================================================================
# GARCH
# ===========================================================================


class TestGARCHConstruction:
    def test_defaults(self):
        m = GARCH()
        assert m.p == 1 and m.q == 1 and m.mean == "constant"

    def test_negative_orders_raise(self):
        with pytest.raises(ValueError):
            GARCH(p=-1)
        with pytest.raises(ValueError):
            GARCH(q=-1)

    def test_zero_orders_raise(self):
        with pytest.raises(ValueError):
            GARCH(p=0, q=0)

    def test_invalid_mean(self):
        with pytest.raises(ValueError):
            GARCH(mean="bogus")

    def test_result_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            _ = GARCH().result


class TestGARCHFit:
    def setup_method(self) -> None:
        self.eps = _simulate_garch11(0.05, 0.1, 0.85, n=4000, seed=1)
        self.model = GARCH(p=1, q=1, mean="zero")
        self.res = self.model.fit(self.eps)

    def test_returns_result_type(self):
        assert isinstance(self.res, GARCHFitResult)

    def test_shapes(self):
        assert self.res.sigma2.shape == (4000,)
        assert self.res.residuals.shape == (4000,)
        assert self.res.std_residuals.shape == (4000,)
        assert self.res.alpha.shape == (1,)
        assert self.res.beta.shape == (1,)

    def test_parameter_recovery(self):
        # 15% tolerance on each coefficient
        assert self.res.omega == pytest.approx(0.05, rel=0.5)
        assert self.res.alpha[0] == pytest.approx(0.1, abs=0.05)
        assert self.res.beta[0] == pytest.approx(0.85, abs=0.05)

    def test_persistence_below_one(self):
        assert self.res.persistence < 1.0

    def test_log_lik_finite(self):
        assert math.isfinite(self.res.log_lik)

    def test_std_residuals_unit_variance(self):
        assert np.var(self.res.std_residuals) == pytest.approx(1.0, abs=0.15)

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            GARCH().fit(np.array([1.0, 2.0, 3.0]))

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError):
            GARCH().fit(np.zeros(100))

    def test_polars_input(self):
        series = pl.Series("r", self.eps)
        res = GARCH(mean="zero").fit(series)
        assert isinstance(res, GARCHFitResult)


class TestGARCHForecast:
    def setup_method(self) -> None:
        eps = _simulate_garch11(0.05, 0.1, 0.85, n=2000, seed=2)
        self.model = GARCH(p=1, q=1, mean="zero")
        self.model.fit(eps)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            self.model.forecast(0)

    def test_deterministic_shape(self):
        f = self.model.forecast(10)
        assert f.shape == (10,)
        assert np.all(np.isfinite(f))

    def test_converges_to_unconditional(self):
        f = self.model.forecast(200)
        unc = self.model.result.unconditional_var
        assert abs(f[-1] - unc) < 0.05 * abs(unc) + 0.05

    def test_monte_carlo_shape(self):
        f = self.model.forecast(5, n_paths=100, seed=11)
        assert f.shape == (5,)
        assert np.all(np.isfinite(f))


class TestGARCHSimulate:
    def setup_method(self) -> None:
        eps = _simulate_garch11(0.05, 0.1, 0.85, n=1500, seed=3)
        self.model = GARCH(mean="zero")
        self.model.fit(eps)

    def test_shape(self):
        sim = self.model.simulate(500, seed=21)
        assert sim.shape == (500,)

    def test_seed_reproducibility(self):
        a = self.model.simulate(200, seed=42)
        b = self.model.simulate(200, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_invalid_t_total_raises(self):
        with pytest.raises(ValueError):
            self.model.simulate(0)


class TestGARCHDataFrame:
    def test_to_dataframe(self):
        eps = _simulate_garch11(0.05, 0.1, 0.85, n=500, seed=4)
        res = GARCH(p=1, q=1, mean="zero").fit(eps)
        df = res.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        # mu + omega + 1 alpha + 1 beta = 4
        assert df.shape[0] == 4


# ===========================================================================
# EGARCH
# ===========================================================================


class TestEGARCH:
    def setup_method(self) -> None:
        # Use a GARCH(1,1) series — EGARCH should fit it without exploding.
        self.eps = _simulate_garch11(0.05, 0.1, 0.85, n=3000, seed=5)

    def test_construction_invalid(self):
        with pytest.raises(ValueError):
            EGARCH(p=-1)
        with pytest.raises(ValueError):
            EGARCH(p=0, q=0)
        with pytest.raises(ValueError):
            EGARCH(mean="bad")

    def test_fit_returns_result(self):
        m = EGARCH(p=1, q=1, mean="zero")
        res = m.fit(self.eps)
        assert isinstance(res, EGARCHFitResult)
        assert res.sigma2.shape == self.eps.shape
        assert math.isfinite(res.log_lik)

    def test_persistence_below_one(self):
        m = EGARCH(p=1, q=1, mean="zero")
        res = m.fit(self.eps)
        assert res.persistence < 1.0

    def test_invalid_horizon_raises(self):
        m = EGARCH(mean="zero")
        m.fit(self.eps)
        with pytest.raises(ValueError):
            m.forecast(0)

    def test_forecast_shape(self):
        m = EGARCH(mean="zero")
        m.fit(self.eps)
        f = m.forecast(5, n_paths=200, seed=7)
        assert f.shape == (5,)

    def test_simulate_shape(self):
        m = EGARCH(mean="zero")
        m.fit(self.eps)
        sim = m.simulate(300, seed=8)
        assert sim.shape == (300,)

    def test_to_dataframe(self):
        m = EGARCH(mean="zero")
        res = m.fit(self.eps)
        df = res.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        # mu + omega + alpha + gamma + beta = 5
        assert df.shape[0] == 5


# ===========================================================================
# GJR
# ===========================================================================


class TestGJR:
    def setup_method(self) -> None:
        # Inject a real leverage effect: γ = 0.2.
        self.eps = _simulate_gjr11(0.05, 0.05, 0.2, 0.7, n=4000, seed=9)

    def test_construction_invalid(self):
        with pytest.raises(ValueError):
            GJR(p=-1)
        with pytest.raises(ValueError):
            GJR(p=0, q=0)
        with pytest.raises(ValueError):
            GJR(mean="bad")

    def test_fit_recovers_leverage(self):
        m = GJR(p=1, q=1, mean="zero")
        res = m.fit(self.eps)
        assert isinstance(res, GJRFitResult)
        # Leverage coefficient should be positive when the truth has γ > 0.
        assert res.gamma[0] > 0.05

    def test_log_lik_finite(self):
        m = GJR(p=1, q=1, mean="zero")
        res = m.fit(self.eps)
        assert math.isfinite(res.log_lik)

    def test_forecast_shape(self):
        m = GJR(mean="zero")
        m.fit(self.eps)
        f = m.forecast(5, n_paths=200, seed=10)
        assert f.shape == (5,)
        assert np.all(np.isfinite(f))

    def test_simulate_shape(self):
        m = GJR(mean="zero")
        m.fit(self.eps)
        sim = m.simulate(300, seed=11)
        assert sim.shape == (300,)

    def test_to_dataframe(self):
        m = GJR(mean="zero")
        res = m.fit(self.eps)
        df = res.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 5  # mu, omega, alpha, gamma, beta


# ===========================================================================
# EWMA
# ===========================================================================


class TestEWMA:
    def setup_method(self) -> None:
        self.eps = _simulate_garch11(0.05, 0.1, 0.85, n=2000, seed=12)

    def test_invalid_lam(self):
        with pytest.raises(ValueError):
            EWMA(lam=0.0)
        with pytest.raises(ValueError):
            EWMA(lam=1.0)
        with pytest.raises(ValueError):
            EWMA(lam=-0.1)

    def test_fit_returns_result(self):
        res = EWMA(lam=0.94).fit(self.eps)
        assert isinstance(res, EWMAResult)
        assert res.sigma2.shape == self.eps.shape
        assert res.lam == 0.94

    def test_forecast_constant(self):
        m = EWMA(lam=0.94)
        m.fit(self.eps)
        f = m.forecast(5)
        assert f.shape == (5,)
        assert np.all(f == f[0])

    def test_invalid_horizon_raises(self):
        m = EWMA(lam=0.94)
        m.fit(self.eps)
        with pytest.raises(ValueError):
            m.forecast(0)

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError):
            EWMA().fit(np.zeros(100))

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            EWMA().fit(np.array([1.0]))

    def test_result_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            _ = EWMA().result

    def test_to_dataframe(self):
        m = EWMA(lam=0.94)
        res = m.fit(self.eps)
        df = res.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"t", "sigma2", "std_residual"}
        assert df.shape[0] == self.eps.shape[0]

    def test_sigma2_tracks_variance(self):
        # Average EWMA variance should be close to sample variance.
        m = EWMA(lam=0.94)
        res = m.fit(self.eps)
        assert res.sigma2.mean() == pytest.approx(np.var(res.residuals), rel=0.3)
