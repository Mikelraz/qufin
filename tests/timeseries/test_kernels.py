"""
Tests for qufin.timeseries._kernels — numba-jitted numerical kernels.

Correctness benchmarks
----------------------
* sample_acf agrees with the slow O(n²) naive implementation on a fixed sample
* sample_acf of i.i.d. noise is close to zero at every positive lag
* sample_acovf at lag 0 equals the biased sample variance
* durbin_levinson on AR(1) autocovariances recovers φ and σ² exactly
* durbin_levinson on AR(2) autocovariances recovers the true coefficients
* lag_matrix has the documented (T-p, p) shape and most-recent-lag-first layout
* yule_walker_solve returns AR coefs + innovation variance
"""

from __future__ import annotations

import numpy as np
import pytest

from qufin.timeseries._kernels import (
    durbin_levinson,
    lag_matrix,
    sample_acf,
    sample_acovf,
    yule_walker_solve,
)

RNG = np.random.default_rng(0)


def _simulate_ar1(phi: float, sigma: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * sigma
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _naive_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    n = x.shape[0]
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    out = np.zeros(nlags)
    for k in range(1, nlags + 1):
        s = 0.0
        for i in range(n - k):
            s += (x[i] - mean) * (x[i + k] - mean)
        out[k - 1] = s / (n * var)
    return out


class TestSampleAcf:
    def test_matches_naive(self):
        x = RNG.standard_normal(500)
        kernel = sample_acf(x, 10)
        naive = _naive_acf(x, 10)
        np.testing.assert_allclose(kernel, naive, rtol=1e-12, atol=1e-12)

    def test_white_noise_small_values(self):
        x = RNG.standard_normal(5000)
        rho = sample_acf(x, 10)
        # With n=5000 the SE of each sample autocorrelation is ~1/√n ≈ 0.014.
        # 10 lags × 5σ headroom keeps the test deterministic.
        assert np.max(np.abs(rho)) < 0.1

    def test_ar1_decays_geometrically(self):
        x = _simulate_ar1(phi=0.7, sigma=1.0, n=20_000, seed=1)
        rho = sample_acf(x, 5)
        expected = 0.7 ** np.arange(1, 6)
        np.testing.assert_allclose(rho, expected, atol=0.05)

    def test_zero_variance_returns_zeros(self):
        x = np.full(20, 3.0)
        rho = sample_acf(x, 5)
        np.testing.assert_array_equal(rho, np.zeros(5))


class TestSampleAcovf:
    def test_lag_zero_is_biased_variance(self):
        x = RNG.standard_normal(200)
        acov = sample_acovf(x, 5)
        biased_var = float(np.mean((x - x.mean()) ** 2))
        assert acov[0] == pytest.approx(biased_var, rel=1e-12)

    def test_shape(self):
        x = RNG.standard_normal(50)
        acov = sample_acovf(x, 7)
        assert acov.shape == (8,)


class TestDurbinLevinson:
    def test_ar1_recovery(self):
        phi = 0.6
        sigma2 = 2.0
        # AR(1) theoretical autocovariances.
        gamma_0 = sigma2 / (1.0 - phi * phi)
        gammas = np.array([gamma_0 * phi**k for k in range(4)])
        ar, pacf, var = durbin_levinson(gammas)
        assert ar.shape == (3,)
        assert ar[0] == pytest.approx(phi, rel=1e-10)
        # PACF of an AR(1) is φ at lag 1, ~0 at higher lags.
        assert pacf[0] == pytest.approx(phi, rel=1e-10)
        assert abs(pacf[1]) < 1e-8
        assert var[-1] == pytest.approx(sigma2, rel=1e-8)

    def test_ar2_recovery(self):
        phi1, phi2 = 0.5, -0.3
        sigma2 = 1.0
        # Stationary AR(2) autocovariances:
        #   γ_0 = σ² · [1 - φ_1² (1+φ_2)/(1-φ_2) - φ_2²]^{-1}
        #   γ_1 = φ_1 γ_0 / (1 - φ_2)
        #   γ_k = φ_1 γ_{k-1} + φ_2 γ_{k-2}   for k ≥ 2
        gamma = np.zeros(5)
        denom = 1.0 - phi1 * phi1 * (1.0 + phi2) / (1.0 - phi2) - phi2 * phi2
        gamma[0] = sigma2 / denom
        gamma[1] = phi1 * gamma[0] / (1.0 - phi2)
        for k in range(2, 5):
            gamma[k] = phi1 * gamma[k - 1] + phi2 * gamma[k - 2]
        ar, _, _ = durbin_levinson(gamma)
        np.testing.assert_allclose(ar[:2], [phi1, phi2], atol=1e-8)

    def test_degenerate_zero_variance(self):
        gamma = np.zeros(4)
        ar, pacf, var = durbin_levinson(gamma)
        np.testing.assert_array_equal(ar, np.zeros(3))


class TestLagMatrix:
    def test_shape_and_content(self):
        x = np.arange(10, dtype=np.float64)
        x_design, y = lag_matrix(x, 3)
        assert x_design.shape == (7, 3)
        assert y.shape == (7,)
        # First row should be [x[2], x[1], x[0]] (most recent lag first), y[0] = x[3].
        np.testing.assert_array_equal(x_design[0], [2.0, 1.0, 0.0])
        assert y[0] == 3.0
        # Last row's target is the final element of x.
        assert y[-1] == 9.0

    def test_p_equal_length_returns_empty(self):
        x = np.arange(5, dtype=np.float64)
        x_design, y = lag_matrix(x, 5)
        assert x_design.shape == (0, 5)
        assert y.shape == (0,)


class TestYuleWalkerSolve:
    def test_ar1(self):
        phi = 0.4
        sigma2 = 0.7
        gamma_0 = sigma2 / (1.0 - phi * phi)
        gammas = np.array([gamma_0 * phi**k for k in range(3)])
        ar, var = yule_walker_solve(gammas)
        assert ar[0] == pytest.approx(phi, rel=1e-10)
        assert var == pytest.approx(sigma2, rel=1e-8)
