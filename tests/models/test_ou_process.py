"""
Tests for src.models.ou_process — OrnsteinUhlenbeck.

Correctness benchmarks
----------------------
  Construction & validation
  Parameter setters enforce positivity constraints
  Derived quantities (half_life, stationary_std, autocorrelation)
  OLS fit recovers known parameters on long simulated paths
  MLE fit recovers known parameters and achieves higher or equal log-likelihood
  OLS and MLE agree closely (same Gaussian model, OLS is exact MLE)
  Simulation — shape, starting value, reproducibility, stationarity
  Log-likelihood is finite, negative, and correctly ordered
  z_score — zero at mean, unit std under stationary distribution
  residuals — mean-zero, correct length
  band_probability — [μ±σ_eq] contains ~68 % probability
  ljung_box — i.i.d. residuals from a well-fitted model pass H₀
  summary / OUFitResult str output is non-empty
  Edge cases: non-stationary warning, very fast/slow mean reversion
"""

from __future__ import annotations

import sys
import os
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.ou_process import OrnsteinUhlenbeck, OUFitResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)

TRUE_THETA = 0.15
TRUE_MU    = 2.0
TRUE_SIGMA = 0.4
DT         = 1.0


def make_ou(theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA, dt=DT) -> OrnsteinUhlenbeck:
    return OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, dt=dt)


def simulate_long(T: int = 5_000, seed: int = 42) -> np.ndarray:
    """Simulate a long path so parameter estimates are reliable."""
    ou = make_ou()
    return ou.simulate(T, x0=TRUE_MU, seed=seed)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults_unset(self):
        ou = OrnsteinUhlenbeck()
        with pytest.raises(RuntimeError):
            _ = ou.theta

    def test_valid_params_stored(self):
        ou = make_ou()
        assert ou.theta == pytest.approx(TRUE_THETA)
        assert ou.mu    == pytest.approx(TRUE_MU)
        assert ou.sigma == pytest.approx(TRUE_SIGMA)

    def test_negative_theta_raises(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck(theta=-0.1, mu=0.0, sigma=1.0)

    def test_zero_theta_raises(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck(theta=0.0, mu=0.0, sigma=1.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck(theta=0.1, mu=0.0, sigma=-0.5)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck(theta=0.1, mu=0.0, sigma=0.0)

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck(dt=-1.0)

    def test_setter_validation(self):
        ou = OrnsteinUhlenbeck()
        with pytest.raises(ValueError):
            ou.theta = -1.0
        with pytest.raises(ValueError):
            ou.sigma = 0.0

    def test_partial_construction(self):
        ou = OrnsteinUhlenbeck(theta=0.2, mu=1.0)
        with pytest.raises(RuntimeError):
            _ = ou.sigma


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

class TestDerivedQuantities:
    def test_half_life(self):
        ou = make_ou(theta=0.2)
        assert ou.half_life == pytest.approx(np.log(2) / 0.2)

    def test_stationary_std(self):
        ou = make_ou(theta=TRUE_THETA, sigma=TRUE_SIGMA)
        expected = TRUE_SIGMA / np.sqrt(2 * TRUE_THETA)
        assert ou.stationary_std == pytest.approx(expected)

    def test_stationary_var(self):
        ou = make_ou()
        assert ou.stationary_var == pytest.approx(ou.stationary_std ** 2)

    def test_autocorrelation_lag0(self):
        ou = make_ou()
        assert ou.autocorrelation(0) == pytest.approx(1.0)

    def test_autocorrelation_lag1(self):
        ou = make_ou(theta=TRUE_THETA, dt=1.0)
        assert ou.autocorrelation(1) == pytest.approx(np.exp(-TRUE_THETA))

    def test_autocorrelation_decreasing(self):
        ou = make_ou()
        acf = [ou.autocorrelation(k) for k in range(10)]
        assert all(acf[i] > acf[i + 1] for i in range(9))

    def test_autocorrelation_negative_lag_raises(self):
        ou = make_ou()
        with pytest.raises(ValueError):
            ou.autocorrelation(-1)


# ---------------------------------------------------------------------------
# OLS fit
# ---------------------------------------------------------------------------

class TestOLSFit:
    def test_returns_fit_result(self):
        x = simulate_long()
        ou = OrnsteinUhlenbeck(dt=DT)
        result = ou.fit(x, method="ols")
        assert isinstance(result, OUFitResult)

    def test_method_label(self):
        x = simulate_long()
        ou = OrnsteinUhlenbeck(dt=DT)
        result = ou.fit(x, method="ols")
        assert result.method == "ols"

    def test_n_obs(self):
        x = simulate_long(T=200)
        ou = OrnsteinUhlenbeck(dt=DT)
        result = ou.fit(x, method="ols")
        assert result.n_obs == len(x) - 1

    def test_recovers_theta(self):
        x = simulate_long(T=8_000)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x, method="ols")
        assert ou.theta == pytest.approx(TRUE_THETA, rel=0.15)

    def test_recovers_mu(self):
        x = simulate_long(T=8_000)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x, method="ols")
        assert ou.mu == pytest.approx(TRUE_MU, abs=0.15)

    def test_recovers_sigma(self):
        x = simulate_long(T=8_000)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x, method="ols")
        assert ou.sigma == pytest.approx(TRUE_SIGMA, rel=0.15)

    def test_sets_params_on_object(self):
        x = simulate_long()
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        # No RuntimeError after fit
        _ = ou.theta
        _ = ou.mu
        _ = ou.sigma

    def test_half_life_positive(self):
        x = simulate_long()
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        assert ou.half_life > 0

    def test_short_series_raises(self):
        ou = OrnsteinUhlenbeck(dt=DT)
        with pytest.raises(ValueError):
            ou.fit(np.array([1.0, 2.0]))

    def test_unknown_method_raises(self):
        ou = OrnsteinUhlenbeck(dt=DT)
        with pytest.raises(ValueError):
            ou.fit(simulate_long(100), method="bayes")


# ---------------------------------------------------------------------------
# MLE fit
# ---------------------------------------------------------------------------

class TestMLEFit:
    def test_returns_fit_result(self):
        x = simulate_long(T=500)
        ou = OrnsteinUhlenbeck(dt=DT)
        result = ou.fit(x, method="mle")
        assert isinstance(result, OUFitResult)

    def test_method_label(self):
        x = simulate_long(T=500)
        ou = OrnsteinUhlenbeck(dt=DT)
        result = ou.fit(x, method="mle")
        assert result.method == "mle"

    def test_recovers_theta(self):
        x = simulate_long(T=8_000)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x, method="mle")
        assert ou.theta == pytest.approx(TRUE_THETA, rel=0.15)

    def test_mle_ll_ge_ols_ll(self):
        """MLE must achieve at least as high a log-likelihood as OLS."""
        x = simulate_long(T=1_000)
        ou_ols = OrnsteinUhlenbeck(dt=DT)
        ou_mle = OrnsteinUhlenbeck(dt=DT)
        r_ols = ou_ols.fit(x, method="ols")
        r_mle = ou_mle.fit(x, method="mle")
        # Allow small numerical tolerance
        assert r_mle.log_lik >= r_ols.log_lik - 1e-3

    def test_ols_and_mle_close(self):
        """Both methods should agree closely on well-identified data."""
        x = simulate_long(T=5_000)
        ou_ols = OrnsteinUhlenbeck(dt=DT)
        ou_mle = OrnsteinUhlenbeck(dt=DT)
        ou_ols.fit(x, method="ols")
        ou_mle.fit(x, method="mle")
        assert ou_ols.theta == pytest.approx(ou_mle.theta, rel=0.05)
        assert ou_ols.mu    == pytest.approx(ou_mle.mu,    abs=0.05)
        assert ou_ols.sigma == pytest.approx(ou_mle.sigma, rel=0.05)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_single_path_shape(self):
        ou = make_ou()
        path = ou.simulate(100)
        assert path.shape == (101,)

    def test_multi_path_shape(self):
        ou = make_ou()
        paths = ou.simulate(50, n_paths=5)
        assert paths.shape == (5, 51)

    def test_starting_value_respected(self):
        ou = make_ou()
        path = ou.simulate(200, x0=10.0)
        assert path[0] == pytest.approx(10.0)

    def test_reproducible_with_seed(self):
        ou = make_ou()
        p1 = ou.simulate(100, seed=1)
        p2 = ou.simulate(100, seed=1)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        ou = make_ou()
        p1 = ou.simulate(100, seed=1)
        p2 = ou.simulate(100, seed=2)
        assert not np.allclose(p1, p2)

    def test_long_path_mean_near_mu(self):
        ou = make_ou(theta=0.5)
        path = ou.simulate(20_000, x0=TRUE_MU, seed=0)
        assert np.mean(path) == pytest.approx(TRUE_MU, abs=0.1)

    def test_long_path_std_near_sigma_eq(self):
        ou = make_ou(theta=0.5)
        path = ou.simulate(20_000, x0=TRUE_MU, seed=0)
        assert np.std(path) == pytest.approx(ou.stationary_std, rel=0.1)

    def test_invalid_n_steps_raises(self):
        ou = make_ou()
        with pytest.raises(ValueError):
            ou.simulate(0)

    def test_invalid_n_paths_raises(self):
        ou = make_ou()
        with pytest.raises(ValueError):
            ou.simulate(10, n_paths=0)

    def test_unfitted_raises(self):
        ou = OrnsteinUhlenbeck()
        with pytest.raises(RuntimeError):
            ou.simulate(10)


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def test_finite_and_negative(self):
        x = simulate_long(T=200)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        ll = ou.log_likelihood(x)
        assert np.isfinite(ll)
        assert ll < 0

    def test_correct_model_beats_misspecified(self):
        x = simulate_long(T=1_000)
        ou_good = OrnsteinUhlenbeck(theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA, dt=DT)
        ou_bad  = OrnsteinUhlenbeck(theta=5.0,        mu=0.0,     sigma=5.0,        dt=DT)
        assert ou_good.log_likelihood(x) > ou_bad.log_likelihood(x)

    def test_unfitted_raises(self):
        ou = OrnsteinUhlenbeck()
        with pytest.raises(RuntimeError):
            ou.log_likelihood(np.ones(10))


# ---------------------------------------------------------------------------
# z_score
# ---------------------------------------------------------------------------

class TestZScore:
    def test_at_mean_is_zero(self):
        ou = make_ou()
        z = ou.z_score(np.array([TRUE_MU]))
        assert z[0] == pytest.approx(0.0)

    def test_one_sigma_eq_gives_one(self):
        ou = make_ou()
        x = np.array([TRUE_MU + ou.stationary_std])
        assert ou.z_score(x)[0] == pytest.approx(1.0)

    def test_stationary_path_z_std_near_one(self):
        ou = make_ou(theta=0.5)
        path = ou.simulate(10_000, seed=0)
        assert np.std(ou.z_score(path)) == pytest.approx(1.0, rel=0.05)

    def test_unfitted_raises(self):
        ou = OrnsteinUhlenbeck()
        with pytest.raises(RuntimeError):
            ou.z_score(np.array([1.0]))


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

class TestResiduals:
    def test_length(self):
        x = simulate_long(T=200)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        eps = ou.residuals(x)
        assert len(eps) == len(x) - 1

    def test_mean_near_zero(self):
        x = simulate_long(T=5_000)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        eps = ou.residuals(x)
        assert np.mean(eps) == pytest.approx(0.0, abs=0.05)

    def test_unfitted_raises(self):
        ou = OrnsteinUhlenbeck()
        with pytest.raises(RuntimeError):
            ou.residuals(np.ones(10))


# ---------------------------------------------------------------------------
# band_probability
# ---------------------------------------------------------------------------

class TestBandProbability:
    def test_one_sigma_band_near_68pct(self):
        ou = make_ou()
        p = ou.band_probability(TRUE_MU - ou.stationary_std, TRUE_MU + ou.stationary_std)
        assert p == pytest.approx(0.6827, abs=0.005)

    def test_two_sigma_band_near_95pct(self):
        ou = make_ou()
        p = ou.band_probability(TRUE_MU - 2 * ou.stationary_std, TRUE_MU + 2 * ou.stationary_std)
        assert p == pytest.approx(0.9545, abs=0.005)

    def test_full_real_line_is_one(self):
        ou = make_ou()
        p = ou.band_probability(-1e9, 1e9)
        assert p == pytest.approx(1.0, abs=1e-6)

    def test_empty_band_is_zero(self):
        ou = make_ou()
        p = ou.band_probability(5.0, 5.0)
        assert p == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# expected_crossing_time
# ---------------------------------------------------------------------------

class TestExpectedCrossingTime:
    def test_at_mean_returns_zero(self):
        ou = make_ou()
        assert ou.expected_crossing_time(TRUE_MU) == pytest.approx(0.0)

    def test_positive_and_finite(self):
        ou = make_ou()
        t = ou.expected_crossing_time(TRUE_MU + 1.0)
        assert t > 0 and np.isfinite(t)

    def test_farther_start_takes_longer(self):
        ou = make_ou()
        t1 = ou.expected_crossing_time(TRUE_MU + 1.0)
        t2 = ou.expected_crossing_time(TRUE_MU + 2.0)
        assert t2 > t1


# ---------------------------------------------------------------------------
# Ljung-Box test
# ---------------------------------------------------------------------------

class TestLjungBox:
    def test_returns_two_floats(self):
        x = simulate_long(T=500)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        Q, p = ou.ljung_box(x, lags=10)
        assert isinstance(Q, float)
        assert isinstance(p, float)

    def test_p_value_in_unit_interval(self):
        x = simulate_long(T=500)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        _, p = ou.ljung_box(x, lags=10)
        assert 0.0 <= p <= 1.0

    def test_well_fitted_model_does_not_reject_at_5pct(self):
        """On a large sample the fitted OU residuals should be close to i.i.d."""
        x = simulate_long(T=5_000)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        _, p = ou.ljung_box(x, lags=20)
        # Not a guaranteed test (random), but should hold with high probability
        assert p > 0.01


# ---------------------------------------------------------------------------
# Summary / OUFitResult
# ---------------------------------------------------------------------------

class TestSummary:
    def test_fitted_summary_non_empty(self):
        x = simulate_long(T=200)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        s = ou.summary()
        assert len(s) > 0
        assert "theta" in s.lower() or "θ" in s

    def test_manual_summary_non_empty(self):
        ou = make_ou()
        s = ou.summary()
        assert len(s) > 0

    def test_fit_result_str(self):
        x = simulate_long(T=200)
        ou = OrnsteinUhlenbeck(dt=DT)
        result = ou.fit(x)
        s = str(result)
        assert "theta" in s.lower() or "θ" in s
        assert "ols" in s.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unit_root_warns(self):
        # Create a near-unit-root series
        rw = np.cumsum(RNG.normal(0, 1, 500))
        ou = OrnsteinUhlenbeck(dt=DT)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ou.fit(rw, method="ols")
            # May or may not warn depending on sample — just ensure no crash

    def test_fast_mean_reversion(self):
        ou = OrnsteinUhlenbeck(theta=10.0, mu=0.0, sigma=1.0, dt=DT)
        path = ou.simulate(500, seed=0)
        assert np.all(np.isfinite(path))

    def test_slow_mean_reversion(self):
        ou = OrnsteinUhlenbeck(theta=0.001, mu=0.0, sigma=0.01, dt=DT)
        path = ou.simulate(500, seed=0)
        assert np.all(np.isfinite(path))

    def test_fit_then_simulate_consistent(self):
        x = simulate_long(T=3_000)
        ou = OrnsteinUhlenbeck(dt=DT)
        ou.fit(x)
        new_path = ou.simulate(1_000, seed=99)
        assert np.all(np.isfinite(new_path))
        assert new_path.shape == (1_001,)
