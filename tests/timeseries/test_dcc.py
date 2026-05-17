# ruff: noqa: NPY002, N806  — random-noise inputs in negative-path tests; T is an econometric sample-size variable
"""
Tests for qufin.timeseries.garch.dcc — Engle (2002) DCC-GARCH.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from qufin.timeseries.garch import DCC, GARCH, DCCFitResult

RNG = np.random.default_rng(13)


def _simulate_garch11(omega: float, a: float, b: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s2 = np.full(n, omega / max(1 - a - b, 1e-6))
    eps = np.zeros(n)
    z = rng.standard_normal(n)
    for t in range(1, n):
        s2[t] = omega + a * eps[t - 1] ** 2 + b * s2[t - 1]
        eps[t] = math.sqrt(s2[t]) * z[t]
    return eps


def _correlated_pair(n: int, rho: float, seed: int) -> np.ndarray:
    r0 = _simulate_garch11(0.05, 0.1, 0.85, n, seed)
    r1 = _simulate_garch11(0.05, 0.1, 0.85, n, seed + 1)
    return np.column_stack([r0 + rho * r1, rho * r0 + r1])


class TestDCCBasic:
    def setup_method(self) -> None:
        self.returns = _correlated_pair(n=1500, rho=0.4, seed=10)
        self.res = DCC().fit(self.returns)

    def test_result_type(self):
        assert isinstance(self.res, DCCFitResult)

    def test_shapes(self):
        T, k = self.returns.shape
        assert self.res.sigma2.shape == (T, k)
        assert self.res.std_residuals.shape == (T, k)
        assert self.res.R.shape == (T, k, k)
        assert self.res.H.shape == (T, k, k)
        assert self.res.Q_bar.shape == (k, k)

    def test_persistence_below_one(self):
        assert self.res.persistence < 1.0

    def test_correlations_in_unit_interval(self):
        # Off-diagonal of every R_t must be in [-1, 1]; diagonal must equal 1
        offdiag = self.res.R[:, 0, 1]
        assert np.all(offdiag >= -1.0)
        assert np.all(offdiag <= 1.0)
        assert np.allclose(self.res.R[:, 0, 0], 1.0)
        assert np.allclose(self.res.R[:, 1, 1], 1.0)

    def test_log_lik_finite(self):
        assert math.isfinite(self.res.log_lik)

    def test_to_dataframe(self):
        df = self.res.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        T, k, _ = self.res.R.shape
        assert df.shape[0] == T * k * k


class TestDCCValidation:
    def test_too_few_series_raises(self):
        with pytest.raises(ValueError):
            DCC().fit(np.random.randn(100, 1))

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            DCC().fit(np.random.randn(10, 2))

    def test_wrong_garch_spec_length(self):
        with pytest.raises(ValueError):
            DCC(garch_specs=[GARCH()]).fit(_correlated_pair(500, 0.2, 1))


class TestDCCCustomSpecs:
    def test_custom_specs_run(self):
        returns = _correlated_pair(800, rho=0.3, seed=20)
        specs = [GARCH(p=1, q=1, mean="zero"), GARCH(p=1, q=1, mean="zero")]
        res = DCC(garch_specs=specs).fit(returns)
        assert res.k == 2
        assert math.isfinite(res.log_lik)
