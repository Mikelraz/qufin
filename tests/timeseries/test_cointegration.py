"""
Tests for src.timeseries.cointegration — Engle-Granger, Johansen, VECM.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.timeseries.cointegration import (
    EngleGrangerResult,
    JohansenResult,
    VECMResult,
    engle_granger,
    johansen,
    vecm,
)

# ruff: noqa: N806  — X is a matrix variable (econometric convention)

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _cointegrated_pair(
    beta: float = 1.2, alpha_: float = 0.5, sigma_u: float = 0.4, n: int = 500, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """y_t = alpha + beta x_t + u_t with x random walk and u stationary AR(1)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + rng.standard_normal() * sigma_u
    y = alpha_ + beta * x + u
    return y, x


def _independent_random_walks(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, k)), axis=0)


def _johansen_system(n: int, seed: int) -> np.ndarray:
    """3-variate system with exactly one cointegration relation: y3 ≈ y1."""
    rng = np.random.default_rng(seed)
    mat = np.zeros((n, 3))
    for t in range(1, n):
        mat[t, 0] = mat[t - 1, 0] + rng.standard_normal() * 0.2
        mat[t, 1] = mat[t - 1, 1] + rng.standard_normal() * 0.2
        # Strong mean-reverting component anchored to y1 (cointegration relation).
        mat[t, 2] = 0.5 * mat[t - 1, 2] + 0.5 * mat[t - 1, 0] + rng.standard_normal() * 0.2
    return mat


# ===========================================================================
# Engle-Granger
# ===========================================================================


class TestEngleGrangerBasic:
    def test_rejects_for_cointegrated(self):
        y, x = _cointegrated_pair(seed=1)
        res = engle_granger(y, x)
        assert isinstance(res, EngleGrangerResult)
        assert res.cointegrated is True
        assert res.p_value < 0.05

    def test_recovers_beta(self):
        y, x = _cointegrated_pair(beta=1.2, alpha_=0.5, n=2000, seed=2)
        res = engle_granger(y, x)
        assert res.beta.shape == (1,)
        assert res.beta[0] == pytest.approx(1.2, abs=0.05)
        assert res.alpha == pytest.approx(0.5, abs=0.5)

    def test_no_rejection_for_independent_rw(self):
        y = _independent_random_walks(500, 1, seed=10).ravel()
        x = _independent_random_walks(500, 1, seed=11).ravel()
        res = engle_granger(y, x)
        assert res.cointegrated is False
        assert res.p_value > 0.05

    def test_residuals_shape(self):
        y, x = _cointegrated_pair(n=300, seed=3)
        res = engle_granger(y, x)
        assert res.residuals.shape == (300,)

    def test_multiple_regressors_shape(self):
        rng = np.random.default_rng(4)
        x1 = np.cumsum(rng.standard_normal(500))
        x2 = np.cumsum(rng.standard_normal(500))
        u = np.zeros(500)
        for t in range(1, 500):
            u[t] = 0.5 * u[t - 1] + rng.standard_normal() * 0.4
        y = 1.0 + 0.5 * x1 - 0.7 * x2 + u
        X = np.column_stack([x1, x2])
        res = engle_granger(y, X)
        assert res.beta.shape == (2,)
        assert res.n_regressors == 2


class TestEngleGrangerValidation:
    def test_invalid_trend(self):
        with pytest.raises(ValueError):
            engle_granger(np.zeros(50), np.zeros(50), trend="bogus")

    def test_invalid_autolag(self):
        with pytest.raises(ValueError):
            engle_granger(np.zeros(50), np.zeros(50), autolag="bogus")

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            engle_granger(np.zeros(50), np.zeros(50), alpha=1.5)

    def test_invalid_maxlag(self):
        with pytest.raises(ValueError):
            engle_granger(np.zeros(50), np.zeros(50), maxlag=-1, autolag=None)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            engle_granger(np.zeros(50), np.zeros(40))

    def test_short_series(self):
        with pytest.raises(ValueError):
            engle_granger(np.zeros(5), np.zeros(5))


# ===========================================================================
# Johansen
# ===========================================================================


class TestJohansenBasic:
    def test_returns_result_type(self):
        mat = _johansen_system(500, seed=20)
        res = johansen(mat, k_ar_diff=1)
        assert isinstance(res, JohansenResult)

    def test_shapes(self):
        mat = _johansen_system(400, seed=21)
        res = johansen(mat, k_ar_diff=1)
        k = 3
        assert res.eigenvalues.shape == (k,)
        assert res.eigenvectors.shape == (k, k)
        assert res.trace_stats.shape == (k,)
        assert res.max_eig_stats.shape == (k,)
        assert res.trace_crits.shape == (k, 3)

    def test_estimates_rank_one(self):
        mat = _johansen_system(800, seed=22)
        res = johansen(mat, k_ar_diff=1)
        assert res.rank_trace == 1
        assert res.rank_max_eig == 1

    def test_rank_zero_for_random_walks(self):
        mat = _independent_random_walks(500, 3, seed=23)
        res = johansen(mat, k_ar_diff=1)
        assert res.rank_trace == 0

    def test_eigenvalues_sorted_descending(self):
        mat = _johansen_system(400, seed=24)
        res = johansen(mat, k_ar_diff=1)
        assert np.all(np.diff(res.eigenvalues) <= 0)

    def test_eigenvalues_in_unit_interval(self):
        mat = _johansen_system(400, seed=25)
        res = johansen(mat, k_ar_diff=1)
        assert np.all(res.eigenvalues > 0)
        assert np.all(res.eigenvalues < 1)


class TestJohansenValidation:
    def test_invalid_det_order(self):
        with pytest.raises(ValueError):
            johansen(np.zeros((100, 2)), det_order=2)

    def test_invalid_k_ar_diff(self):
        with pytest.raises(ValueError):
            johansen(np.zeros((100, 2)), k_ar_diff=-1)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            johansen(np.zeros((100, 2)), alpha=0.0)

    def test_too_many_variables(self):
        with pytest.raises(ValueError):
            johansen(np.zeros((100, 13)))


# ===========================================================================
# VECM
# ===========================================================================


class TestVECMBasic:
    def setup_method(self) -> None:
        self.mat = _johansen_system(800, seed=30)
        self.res = vecm(self.mat, coint_rank=1, k_ar_diff=1)

    def test_returns_result_type(self):
        assert isinstance(self.res, VECMResult)

    def test_shapes(self):
        k = 3
        assert self.res.alpha.shape == (k, 1)
        assert self.res.beta.shape == (k, 1)
        assert self.res.pi.shape == (k, k)
        assert self.res.gamma.shape == (1, k, k)
        assert self.res.sigma_u.shape == (k, k)
        assert self.res.residuals.shape[1] == k
        assert self.res.fitted_values.shape == self.res.residuals.shape

    def test_beta_normalised(self):
        # First row of beta normalised to identity
        assert self.res.beta[0, 0] == pytest.approx(1.0)

    def test_beta_recovers_relationship(self):
        # True relation: y3 − y1 stationary, so beta should have y1 ~ −1, y3 ~ +1 (or normalised)
        beta_norm = self.res.beta[:, 0] / self.res.beta[0, 0]
        # After normalisation first entry is 1; y3 entry should be approx −1
        assert abs(beta_norm[2]) > 0.5  # y3 entry meaningful
        assert abs(beta_norm[1]) < 0.3  # y2 entry small (it's a free random walk)

    def test_alpha_nonzero(self):
        # Adjustment to disequilibrium should be non-trivial
        assert np.linalg.norm(self.res.alpha) > 0.01

    def test_log_lik_finite(self):
        assert np.isfinite(self.res.log_lik)

    def test_to_dataframe(self):
        df = self.res.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        # k * r alpha rows + k * r beta rows = 2 * 3 * 1 = 6
        assert df.shape[0] == 6
        assert set(df.columns) == {"component", "i", "r", "value"}


class TestVECMValidation:
    def test_invalid_rank_zero(self):
        mat = _johansen_system(200, seed=40)
        with pytest.raises(ValueError):
            vecm(mat, coint_rank=0)

    def test_invalid_rank_too_high(self):
        mat = _johansen_system(200, seed=41)
        with pytest.raises(ValueError):
            vecm(mat, coint_rank=3)  # k=3, so r must be ≤ 2

    def test_invalid_det_order(self):
        mat = _johansen_system(200, seed=42)
        with pytest.raises(ValueError):
            vecm(mat, coint_rank=1, det_order=5)

    def test_invalid_k_ar_diff(self):
        mat = _johansen_system(200, seed=43)
        with pytest.raises(ValueError):
            vecm(mat, coint_rank=1, k_ar_diff=-1)


class TestVECMWithUnrestrictedConst:
    def test_unrestricted_const_fits(self):
        mat = _johansen_system(600, seed=50)
        res = vecm(mat, coint_rank=1, k_ar_diff=1, det_order=1)
        assert res.det_order == 1
        assert res.const.shape == (3,)
        assert np.all(np.isfinite(res.const))
