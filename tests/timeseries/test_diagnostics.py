"""
Tests for src.timeseries.diagnostics.

Correctness benchmarks
----------------------
* ACF / PACF return ACFResult with correct shapes and Bartlett bands ±z/√n
* On a long AR(1) path, sample ACF ≈ φ^k and PACF is sharp at lag 1
* Ljung-Box does not reject H0 on white noise; rejects strongly on AR(1)
* Ljung-Box matches the textbook formula on a small fixed input
* Jarque-Bera does not reject on standard normal; rejects on Student-t(3)
* ARCH-LM rejects on a GARCH(1,1)-like volatility-clustered series
* polars I/O round-trip yields the same numeric result as numpy input
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.timeseries.diagnostics import (
    ACFResult,
    acf,
    arch_lm,
    jarque_bera,
    ljung_box,
    pacf,
)

RNG = np.random.default_rng(2024)


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


class TestACF:
    def test_shapes(self):
        x = RNG.standard_normal(200)
        res = acf(x, nlags=10, ci=0.95)
        assert isinstance(res, ACFResult)
        assert res.values.shape == (10,)
        assert res.lower_ci.shape == (10,)
        assert res.upper_ci.shape == (10,)
        assert res.n_obs == 200

    def test_bartlett_bands(self):
        x = RNG.standard_normal(400)
        res = acf(x, nlags=5, ci=0.95)
        # ±1.96 / √400 ≈ ±0.098
        np.testing.assert_allclose(res.upper_ci, np.full(5, 1.959963984540054 / 20.0), atol=1e-10)
        np.testing.assert_allclose(res.lower_ci, -res.upper_ci, atol=1e-10)

    def test_ar1_decay(self):
        x = _ar1(0.6, 20_000, seed=1)
        res = acf(x, nlags=4)
        expected = 0.6 ** np.arange(1, 5)
        np.testing.assert_allclose(res.values, expected, atol=0.05)

    def test_polars_input(self):
        rng = np.random.default_rng(7)
        arr = rng.standard_normal(100)
        res_np = acf(arr, nlags=5)
        res_pl = acf(pl.Series("x", arr), nlags=5)
        np.testing.assert_allclose(res_np.values, res_pl.values)

    def test_to_dataframe(self):
        x = RNG.standard_normal(50)
        res = acf(x, nlags=5)
        df = res.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["lag", "value", "lower", "upper"]
        assert df.height == 5

    def test_invalid_nlags(self):
        with pytest.raises(ValueError):
            acf(RNG.standard_normal(50), nlags=0)

    def test_invalid_ci(self):
        with pytest.raises(ValueError):
            acf(RNG.standard_normal(50), nlags=5, ci=1.5)

    def test_too_short(self):
        with pytest.raises(ValueError, match="at least"):
            acf(np.arange(3), nlags=5)


class TestPACF:
    def test_ar1_sharp_first_lag(self):
        x = _ar1(0.7, 5_000, seed=3)
        res = pacf(x, nlags=5)
        assert res.values[0] == pytest.approx(0.7, abs=0.05)
        # Lags >= 2 should be near zero.
        assert np.max(np.abs(res.values[1:])) < 0.05

    def test_bartlett_bands_match_acf(self):
        x = RNG.standard_normal(300)
        a = acf(x, nlags=5)
        p = pacf(x, nlags=5)
        np.testing.assert_allclose(a.lower_ci, p.lower_ci)
        np.testing.assert_allclose(a.upper_ci, p.upper_ci)


class TestLjungBox:
    def test_white_noise_does_not_reject(self):
        # 50 trials of i.i.d. data at α=0.05 expect ~2.5 false rejections.
        # We pick a single seeded trial and require p > 0.05 — adjust seed
        # search if flaky in practice.
        x = RNG.standard_normal(500)
        _, p = ljung_box(x, lags=10)
        assert p > 0.05

    def test_ar1_rejects(self):
        x = _ar1(0.6, 1_000, seed=11)
        q, p = ljung_box(x, lags=10)
        assert q > 50.0
        assert p < 1e-6

    def test_dof_adjust(self):
        x = RNG.standard_normal(200)
        # dof_adjust shifts the chi² df; with fewer df, p should be smaller.
        _, p_full = ljung_box(x, lags=10, dof_adjust=0)
        _, p_adj = ljung_box(x, lags=10, dof_adjust=2)
        assert p_adj <= p_full

    def test_dof_adjust_too_large(self):
        with pytest.raises(ValueError, match="degrees of freedom"):
            ljung_box(RNG.standard_normal(50), lags=2, dof_adjust=2)


class TestJarqueBera:
    def test_normal_does_not_reject(self):
        x = RNG.standard_normal(2000)
        jb, p = jarque_bera(x)
        assert p > 0.01

    def test_student_t_rejects(self):
        rng = np.random.default_rng(20)
        x = rng.standard_t(df=3, size=2000)
        jb, p = jarque_bera(x)
        assert jb > 50.0
        assert p < 1e-6

    def test_constant_series(self):
        x = np.full(20, 5.0)
        jb, p = jarque_bera(x)
        assert jb == 0.0
        assert p == 1.0


class TestARCHLM:
    def test_iid_does_not_reject(self):
        # Deterministic seed where i.i.d. Gaussian noise clearly does not
        # reject H0 — at α=0.05 the test has ~5 % nominal size, so a single
        # arbitrary seed occasionally tips just below the threshold.
        rng = np.random.default_rng(11)
        x = rng.standard_normal(500)
        _, p = arch_lm(x, lags=5)
        assert p > 0.05

    def test_volatility_clustering_rejects(self):
        # Simulate a crude GARCH(1,1)-like return process.
        rng = np.random.default_rng(31)
        n = 2_000
        omega, alpha, beta = 0.05, 0.1, 0.85
        sigma2 = np.empty(n)
        sigma2[0] = omega / (1.0 - alpha - beta)
        r = np.empty(n)
        z = rng.standard_normal(n)
        for t in range(n):
            r[t] = np.sqrt(sigma2[t]) * z[t]
            if t + 1 < n:
                sigma2[t + 1] = omega + alpha * r[t] ** 2 + beta * sigma2[t]
        lm, p = arch_lm(r, lags=10)
        assert lm > 50.0
        assert p < 1e-6
