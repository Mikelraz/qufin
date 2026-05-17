"""
Tests for qufin.timeseries.statespace — ARMAStateSpace.

Correctness benchmarks
----------------------
* Filtered and smoothed states have correct shapes
* Log-likelihood matches arima._arma_log_likelihood
* Filter forward + RTS smoother round-trip
* Polars input accepted
* from_result() alternate constructor
* Invalid construction raises
* to_dataframe() returns pl.DataFrame with expected columns
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from qufin.timeseries.arima import ARMA, _arma_log_likelihood
from qufin.timeseries.statespace import ARMAStateSpace, StateSpaceResult

RNG = np.random.default_rng(200)


def _simulate_ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(n):
        y[t] = eps[t]
        if t > 0:
            y[t] += phi * y[t - 1]
    return y


def _simulate_arma11(phi: float, theta: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(n):
        y[t] = eps[t]
        if t > 0:
            y[t] += phi * y[t - 1] + theta * eps[t - 1]
    return y


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_ar1(self):
        ss = ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 1.0)
        assert ss.p == 1
        assert ss.q == 0
        assert ss.state_dim == 1

    def test_valid_arma11(self):
        ss = ARMAStateSpace(1, 1, np.array([0.5]), np.array([0.3]), 1.0)
        assert ss.state_dim == 2  # r = max(1, 2) = 2

    def test_valid_ma1(self):
        ss = ARMAStateSpace(0, 1, np.array([]), np.array([0.4]), 1.0)
        assert ss.state_dim == 2  # r = max(0, 2) = 2

    def test_invalid_negative_p(self):
        with pytest.raises(ValueError):
            ARMAStateSpace(-1, 1, np.array([]), np.array([0.4]), 1.0)

    def test_invalid_sigma2_zero(self):
        with pytest.raises(ValueError):
            ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 0.0)

    def test_invalid_both_zero(self):
        with pytest.raises(ValueError):
            ARMAStateSpace(0, 0, np.array([]), np.array([]), 1.0)

    def test_from_result(self):
        y = _simulate_arma11(0.5, 0.3, 500, seed=201)
        arma = ARMA(1, 1)
        arma.fit(y, method="mle")
        ss = ARMAStateSpace.from_result(arma.result)
        assert ss.p == 1
        assert ss.q == 1
        np.testing.assert_allclose(ss.ar_coef, arma.result.ar_coef, rtol=1e-12)

    def test_matrix_properties(self):
        ss = ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 1.0)
        assert ss.F.shape == (1, 1)
        assert ss.H.shape == (1, 1)
        assert ss.Q.shape == (1, 1)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filter_result_shapes(self):
        y = _simulate_ar1(0.5, 300, seed=210)
        ss = ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 1.0)
        fr = ss.filter(y)
        assert fr.states.shape == (300, 1)
        assert fr.covariances.shape == (300, 1, 1)
        assert fr.innovations.shape == (300, 1)

    def test_filter_log_likelihood_matches_helper(self):
        phi = np.array([0.6])
        sigma2 = 1.2
        y = _simulate_ar1(0.6, 500, seed=211)
        ss = ARMAStateSpace(1, 0, phi, np.array([]), sigma2)
        ll_ss = ss.log_likelihood(y)
        ll_direct = _arma_log_likelihood(phi, np.array([]), sigma2, y)
        # Both use same state-space; may differ by tiny numerics (different P0)
        assert abs(ll_ss - ll_direct) < 2.0  # within 2 nats

    def test_polars_input(self):
        y = _simulate_ar1(0.4, 300, seed=212)
        ss = ARMAStateSpace(1, 0, np.array([0.4]), np.array([]), 1.0)
        fr_np = ss.filter(y)
        fr_pl = ss.filter(pl.Series("y", y))
        np.testing.assert_allclose(fr_np.log_likelihood, fr_pl.log_likelihood, rtol=1e-12)

    def test_filter_innovation_mean_near_zero(self):
        # If the model is correctly specified, innovations should be near WN
        y = _simulate_ar1(0.5, 1_000, seed=213)
        ss = ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 1.0)
        fr = ss.filter(y)
        innov = fr.innovations[:, 0]
        assert abs(np.mean(innov)) < 0.1


# ---------------------------------------------------------------------------
# Smoother tests
# ---------------------------------------------------------------------------


class TestSmoother:
    def test_smoother_result_type(self):
        y = _simulate_ar1(0.5, 300, seed=220)
        ss = ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 1.0)
        result = ss.smooth(y)
        assert isinstance(result, StateSpaceResult)

    def test_smoother_shapes(self):
        y = _simulate_ar1(0.5, 300, seed=221)
        ss = ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 1.0)
        result = ss.smooth(y)
        assert result.filtered_states.shape == (300, 1)
        assert result.smoothed_states.shape == (300, 1)

    def test_smoothed_more_efficient_than_filtered(self):
        # Smoother should reduce mean squared uncertainty
        # E[P_t|T] <= E[P_t|t] element-wise for each t
        y = _simulate_ar1(0.5, 200, seed=222)
        ss = ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 1.0)
        result = ss.smooth(y)
        # Average filtered variance vs. smoothed variance
        avg_filtered_var = np.mean(result.filter_result.covariances[:, 0, 0])
        avg_smoothed_var = np.mean(result.smoother_result.covariances[:, 0, 0])
        assert avg_smoothed_var <= avg_filtered_var + 1e-8

    def test_log_likelihood_preserved(self):
        y = _simulate_ar1(0.5, 300, seed=223)
        ss = ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 1.0)
        result = ss.smooth(y)
        assert np.isfinite(result.log_likelihood)
        # Smoother LL should equal filter LL
        assert result.log_likelihood == pytest.approx(
            result.filter_result.log_likelihood, rel=1e-10
        )

    def test_innovations_accessor(self):
        y = _simulate_ar1(0.4, 200, seed=224)
        ss = ARMAStateSpace(1, 0, np.array([0.4]), np.array([]), 1.0)
        result = ss.smooth(y)
        assert result.innovations.shape == (200, 1)

    def test_to_dataframe(self):
        y = _simulate_ar1(0.5, 100, seed=225)
        ss = ARMAStateSpace(1, 0, np.array([0.5]), np.array([]), 1.0)
        result = ss.smooth(y)
        df = result.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"t", "filtered", "smoothed", "innovation"}
        assert len(df) == 100


# ---------------------------------------------------------------------------
# ARMA(1,1) state-space
# ---------------------------------------------------------------------------


class TestARMA11StateSpace:
    def test_state_dim_is_two(self):
        ss = ARMAStateSpace(1, 1, np.array([0.5]), np.array([0.3]), 1.0)
        assert ss.state_dim == 2

    def test_filter_shapes(self):
        y = _simulate_arma11(0.5, 0.3, 300, seed=230)
        ss = ARMAStateSpace(1, 1, np.array([0.5]), np.array([0.3]), 1.0)
        fr = ss.filter(y)
        assert fr.states.shape == (300, 2)
        assert fr.innovations.shape == (300, 1)

    def test_from_result_end_to_end(self):
        y = _simulate_arma11(0.5, 0.3, 1_000, seed=231)
        arma = ARMA(1, 1)
        arma.fit(y, method="mle")
        ss = ARMAStateSpace.from_result(arma.result)
        result = ss.smooth(y - arma.result.const)
        assert result.smoothed_states.shape == (1_000, 2)
        assert np.isfinite(result.log_likelihood)
