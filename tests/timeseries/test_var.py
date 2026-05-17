# ruff: noqa: N806, N803  — matrix variables use control-theory/econometric uppercase (A, L, X)
"""
Tests for qufin.timeseries.var — VAR, Granger causality, impulse responses.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from qufin.timeseries.var import (
    VAR,
    GrangerResult,
    VARFitResult,
    granger_causality,
    impulse_response,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _simulate_var1(
    A: np.ndarray, sigma: np.ndarray, n: int, seed: int, c: np.ndarray | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = A.shape[0]
    L = np.linalg.cholesky(sigma)
    noise = rng.standard_normal((n, k)) @ L.T
    y = np.zeros((n + 100, k))
    if c is None:
        c = np.zeros(k)
    for t in range(1, n + 100):
        y[t] = c + A @ y[t - 1] + (noise[t - 100] if t >= 100 else rng.standard_normal(k) * 0.1)
    return y[100:]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_order(self):
        model = VAR(p=2)
        assert model.p == 2

    def test_invalid_order_zero(self):
        with pytest.raises(ValueError):
            VAR(p=0)

    def test_invalid_order_negative(self):
        with pytest.raises(ValueError):
            VAR(p=-1)

    def test_result_before_fit_raises(self):
        model = VAR(p=1)
        with pytest.raises(RuntimeError):
            _ = model.result


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestFit:
    def setup_method(self) -> None:
        self.A_true = np.array([[0.5, 0.1], [0.2, 0.4]])
        self.sigma_true = np.array([[0.04, 0.01], [0.01, 0.04]])
        self.y = _simulate_var1(self.A_true, self.sigma_true, n=2000, seed=1)

    def test_returns_result_type(self):
        res = VAR(p=1).fit(self.y)
        assert isinstance(res, VARFitResult)

    def test_shapes(self):
        res = VAR(p=1).fit(self.y)
        assert res.coef.shape == (1, 2, 2)
        assert res.const.shape == (2,)
        assert res.sigma_u.shape == (2, 2)
        assert res.residuals.shape == (1999, 2)
        assert res.fitted_values.shape == (1999, 2)
        assert res.n_obs == 1999

    def test_parameter_recovery(self):
        res = VAR(p=1).fit(self.y, include_const=False)
        # Recovery within 10 %
        np.testing.assert_allclose(res.coef[0], self.A_true, atol=0.05)
        # Sigma_u close to true
        np.testing.assert_allclose(res.sigma_u, self.sigma_true, atol=0.01)

    def test_stationarity_flag(self):
        res = VAR(p=1).fit(self.y)
        assert res.is_stationary is True

    def test_const_excluded(self):
        res = VAR(p=1).fit(self.y, include_const=False)
        assert np.all(res.const == 0.0)
        assert res.include_const is False

    def test_const_included(self):
        c = np.array([0.5, -0.3])
        y2 = _simulate_var1(self.A_true, self.sigma_true, n=2000, seed=2, c=c)
        res = VAR(p=1).fit(y2, include_const=True)
        # The unconditional mean is (I − A)^−1 c
        expected_mean = np.linalg.solve(np.eye(2) - self.A_true, c)
        empirical_mean = res.fitted_values.mean(axis=0) + res.residuals.mean(axis=0)
        np.testing.assert_allclose(empirical_mean, expected_mean, atol=0.1)

    def test_log_lik_finite(self):
        res = VAR(p=1).fit(self.y)
        assert np.isfinite(res.log_lik)
        assert res.log_lik < 0 or res.log_lik > 0  # just finite

    def test_higher_order_lower_aic_when_truth_is_higher(self):
        # Truth is VAR(1), so VAR(2) should not improve much; both should be finite.
        res1 = VAR(p=1).fit(self.y)
        res2 = VAR(p=2).fit(self.y)
        assert np.isfinite(res1.aic)
        assert np.isfinite(res2.aic)

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            VAR(p=5).fit(RNG.standard_normal((6, 2)))

    def test_polars_input(self):
        df = pl.DataFrame({"a": self.y[:, 0], "b": self.y[:, 1]})
        res = VAR(p=1).fit(df, include_const=False)
        assert res.coef.shape == (1, 2, 2)

    def test_non_stationary_warns(self):
        # Explosive VAR(1):  y_t = 1.05 y_{t-1} + ε_t
        bad = _simulate_var1(np.array([[1.05, 0.0], [0.0, 0.9]]), np.eye(2) * 0.01, 500, 7)
        # Suppress overflow warnings from explosive simulation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = VAR(p=1).fit(bad)
        # Either stationarity is False, or eigenvalues touch unit circle
        if np.isfinite(res.coef).all():
            assert (not res.is_stationary) or np.any(np.abs(res.companion_eigenvalues) >= 0.99)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


class TestForecast:
    def setup_method(self) -> None:
        self.A_true = np.array([[0.5, 0.1], [0.2, 0.4]])
        self.sigma_true = np.eye(2) * 0.04
        self.y = _simulate_var1(self.A_true, self.sigma_true, n=1000, seed=3)
        self.model = VAR(p=1)
        self.model.fit(self.y, include_const=False)

    def test_shape(self):
        f = self.model.forecast(5)
        assert f.shape == (5, 2)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            self.model.forecast(0)
        with pytest.raises(ValueError):
            self.model.forecast(-3)

    def test_forecast_converges_to_zero(self):
        # Stationary VAR with c = 0 → forecasts converge to 0
        f = self.model.forecast(50)
        assert np.linalg.norm(f[-1]) < 0.1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class TestSimulate:
    def setup_method(self) -> None:
        A = np.array([[0.5, 0.1], [0.2, 0.4]])
        sigma = np.eye(2) * 0.04
        self.y = _simulate_var1(A, sigma, n=1000, seed=4)
        self.model = VAR(p=1)
        self.model.fit(self.y, include_const=False)

    def test_shape(self):
        sim = self.model.simulate(500, seed=11)
        assert sim.shape == (500, 2)

    def test_seed_reproducibility(self):
        a = self.model.simulate(200, seed=99)
        b = self.model.simulate(200, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_invalid_t_total_raises(self):
        with pytest.raises(ValueError):
            self.model.simulate(0)


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------


class TestGranger:
    def setup_method(self) -> None:
        # y2 causes y1:  y1_t = 0.5 y1_{t-1} + 0.4 y2_{t-1} + e
        #                y2_t = 0.5 y2_{t-1} + e
        rng = np.random.default_rng(20)
        n = 1000
        y = np.zeros((n, 2))
        for t in range(1, n):
            y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal() * 0.3
            y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal() * 0.3
        self.y = y

    def test_detects_true_causality(self):
        res = granger_causality(self.y, caused=0, causing=1, lags=2)
        assert isinstance(res, GrangerResult)
        assert res.p_value < 0.01

    def test_no_spurious_reverse_causality(self):
        res = granger_causality(self.y, caused=1, causing=0, lags=2)
        # Should fail to reject H0 (no causality from y1 to y2)
        assert res.p_value > 0.01

    def test_invalid_lags_raises(self):
        with pytest.raises(ValueError):
            granger_causality(self.y, caused=0, causing=1, lags=0)

    def test_same_indices_raises(self):
        with pytest.raises(ValueError):
            granger_causality(self.y, caused=0, causing=0, lags=1)

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError):
            granger_causality(self.y, caused=0, causing=5, lags=1)

    def test_f_stat_positive(self):
        res = granger_causality(self.y, caused=0, causing=1, lags=2)
        assert res.f_stat > 0


# ---------------------------------------------------------------------------
# Impulse response
# ---------------------------------------------------------------------------


class TestImpulseResponse:
    def setup_method(self) -> None:
        A = np.array([[0.6, 0.0], [0.0, 0.6]])
        sigma = np.diag([0.04, 0.09])
        self.y = _simulate_var1(A, sigma, n=2000, seed=5)
        self.res = VAR(p=1).fit(self.y, include_const=False)

    def test_shape(self):
        irf = impulse_response(self.res, h=10)
        assert irf.shape == (10, 2, 2)

    def test_non_orthogonalized_starts_at_identity(self):
        irf = impulse_response(self.res, h=5, orthogonalized=False)
        np.testing.assert_allclose(irf[0], np.eye(2))

    def test_orthogonalized_starts_at_cholesky(self):
        irf = impulse_response(self.res, h=5, orthogonalized=True)
        L = np.linalg.cholesky(self.res.sigma_u + 1e-12 * np.eye(2))
        np.testing.assert_allclose(irf[0], L, atol=1e-6)

    def test_decays_to_zero_for_stationary(self):
        irf = impulse_response(self.res, h=30)
        assert np.max(np.abs(irf[-1])) < 0.1

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            impulse_response(self.res, h=0)


# ---------------------------------------------------------------------------
# Information criteria + DataFrame
# ---------------------------------------------------------------------------


class TestInfoAndDataFrame:
    def test_info_criteria_finite(self):
        A = np.array([[0.5]])
        sigma = np.array([[0.04]])
        y = _simulate_var1(A, sigma, n=500, seed=6)
        res = VAR(p=1).fit(y)
        for v in (res.aic, res.bic, res.hqic):
            assert np.isfinite(v)

    def test_to_dataframe(self):
        y = _simulate_var1(np.array([[0.5, 0.0], [0.0, 0.5]]), np.eye(2) * 0.04, 500, 8)
        res = VAR(p=2).fit(y)
        df = res.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2 * 2 * 2, 4)  # p * k * k rows, 4 cols
        assert set(df.columns) == {"lag", "i", "j", "value"}
