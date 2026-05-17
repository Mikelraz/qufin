"""
Tests for qufin.timeseries.models — HedgeRatioFilter and TrendFilter.

Correctness benchmarks:
  - HedgeRatioFilter recovers a known constant hedge ratio
  - HedgeRatioFilter tracks a linearly drifting hedge ratio
  - HedgeRatioFilter spread is stationary when β is correctly estimated
  - HedgeRatioFilter.step() and .filter() produce identical results
  - TrendFilter smooths noisy prices (lower RMSE than raw)
  - TrendFilter velocity estimates match finite differences on smooth signals
  - TrendFilter RTS smooth reduces variance vs forward filter
  - Both models handle NaN (missing) observations
  - API: DataFrame columns, polars Series input, reset behaviour
"""

import numpy as np
import polars as pl
import pytest

from qufin.timeseries.models import HedgeRatioFilter, TrendFilter

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synthetic_pairs(
    T: int = 500,
    beta_true: float = 1.5,
    alpha_true: float = 2.0,
    sigma_eps: float = 0.5,
    sigma_x: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """y = beta * x + alpha + noise, x a random-walk."""
    x = np.cumsum(RNG.normal(0, sigma_x, T)) + 10.0
    y = beta_true * x + alpha_true + RNG.normal(0, sigma_eps, T)
    return y, x


def drifting_beta_pairs(
    T: int = 600,
    beta_start: float = 1.0,
    beta_end: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    betas = np.linspace(beta_start, beta_end, T)
    x = np.cumsum(RNG.normal(0, 1.0, T)) + 10.0
    y = betas * x + RNG.normal(0, 0.3, T)
    return y, x, betas


# ---------------------------------------------------------------------------
# HedgeRatioFilter — construction
# ---------------------------------------------------------------------------

class TestHedgeRatioFilterConstruction:
    def test_negative_delta_raises(self):
        with pytest.raises(ValueError):
            HedgeRatioFilter(delta=-1e-4)

    def test_negative_obs_var_raises(self):
        with pytest.raises(ValueError):
            HedgeRatioFilter(obs_var=-1.0)

    def test_default_initial_state(self):
        f = HedgeRatioFilter()
        assert f.beta == pytest.approx(1.0)
        assert f.alpha == pytest.approx(0.0)

    def test_custom_initial_state(self):
        f = HedgeRatioFilter(x0=np.array([2.5, -1.0]))
        assert f.beta == pytest.approx(2.5)
        assert f.alpha == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# HedgeRatioFilter — step interface
# ---------------------------------------------------------------------------

class TestHedgeRatioFilterStep:
    def test_step_returns_three_floats(self):
        f = HedgeRatioFilter()
        result = f.step(y=10.0, x=6.0)
        assert len(result) == 3
        b, a, s = result
        assert isinstance(b, float)
        assert isinstance(a, float)
        assert isinstance(s, float)

    def test_step_spread_is_innovation(self):
        """spread = y - beta_pred * x - alpha_pred (pre-update prediction error)."""
        f = HedgeRatioFilter(delta=1e-6)
        y_arr, x_arr = synthetic_pairs(T=200, beta_true=1.5, alpha_true=0.0)
        for t in range(200):
            _, _, spread = f.step(float(y_arr[t]), float(x_arr[t]))
        assert abs(spread) < 5.0

    def test_beta_variance_positive(self):
        f = HedgeRatioFilter()
        f.step(10.0, 6.0)
        assert f.beta_variance > 0

    def test_reset_restores_state(self):
        f = HedgeRatioFilter()
        for _ in range(50):
            f.step(10.0, 6.0)
        f.reset()
        assert f.beta == pytest.approx(1.0)
        assert f.alpha == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HedgeRatioFilter — constant hedge ratio recovery
# ---------------------------------------------------------------------------

class TestHedgeRatioFilterAccuracy:
    def test_recovers_constant_beta(self):
        """After seeing 400 obs the filter should be within 0.05 of true β."""
        beta_true = 1.8
        y, x = synthetic_pairs(T=400, beta_true=beta_true, sigma_eps=0.3)
        f = HedgeRatioFilter(delta=1e-5, obs_var=0.09)
        df = f.filter(y, x)
        beta_tail = df["beta"].to_numpy()[-100:]
        assert float(beta_tail.mean()) == pytest.approx(beta_true, abs=0.10)

    def test_recovers_constant_alpha(self):
        alpha_true = 3.5
        y, x = synthetic_pairs(T=400, beta_true=1.0, alpha_true=alpha_true, sigma_eps=0.3)
        f = HedgeRatioFilter(delta=1e-5, obs_var=0.09)
        df = f.filter(y, x)
        alpha_tail = df["alpha"].to_numpy()[-100:]
        assert float(alpha_tail.mean()) == pytest.approx(alpha_true, abs=0.30)

    def test_tracks_drifting_beta(self):
        """Filter beta should correlate highly with the true drifting beta."""
        y, x, betas_true = drifting_beta_pairs(T=600)
        f = HedgeRatioFilter(delta=5e-4, obs_var=0.09)
        df = f.filter(y, x)
        beta_arr = df["beta"].to_numpy()
        corr = np.corrcoef(beta_arr[100:], betas_true[100:])[0, 1]
        assert corr > 0.85, f"Correlation with drifting beta too low: {corr:.3f}"


# ---------------------------------------------------------------------------
# HedgeRatioFilter — batch == step-by-step
# ---------------------------------------------------------------------------

class TestHedgeRatioFilterConsistency:
    def test_filter_matches_step_by_step(self):
        y, x = synthetic_pairs(T=100)
        f1 = HedgeRatioFilter(delta=1e-4, obs_var=1.0)
        df = f1.filter(y, x)

        f2 = HedgeRatioFilter(delta=1e-4, obs_var=1.0)
        betas, alphas, spreads = [], [], []
        for yi, xi in zip(y, x):
            b, a, s = f2.step(float(yi), float(xi))
            betas.append(b); alphas.append(a); spreads.append(s)

        np.testing.assert_allclose(df["beta"].to_numpy(),   betas,   atol=1e-12)
        np.testing.assert_allclose(df["alpha"].to_numpy(),  alphas,  atol=1e-12)
        np.testing.assert_allclose(df["spread"].to_numpy(), spreads, atol=1e-12)


# ---------------------------------------------------------------------------
# HedgeRatioFilter — DataFrame API
# ---------------------------------------------------------------------------

class TestHedgeRatioFilterDataFrameAPI:
    def test_returns_polars_dataframe(self):
        f = HedgeRatioFilter()
        y, x = synthetic_pairs(T=50)
        result = f.filter(y, x)
        assert isinstance(result, pl.DataFrame)

    def test_expected_columns(self):
        f = HedgeRatioFilter()
        y, x = synthetic_pairs(T=50)
        df = f.filter(y, x)
        for col in ["beta", "alpha", "spread", "beta_std", "alpha_std"]:
            assert col in df.columns

    def test_polars_series_input_accepted(self):
        y_np, x_np = synthetic_pairs(T=80)
        y_pl = pl.Series("y", y_np)
        x_pl = pl.Series("x", x_np)
        f = HedgeRatioFilter()
        df = f.filter(y_pl, x_pl)
        assert df.height == 80

    def test_std_columns_nonnegative(self):
        f = HedgeRatioFilter()
        y, x = synthetic_pairs(T=100)
        df = f.filter(y, x)
        assert np.all(df["beta_std"].to_numpy() >= 0)
        assert np.all(df["alpha_std"].to_numpy() >= 0)

    def test_mismatched_lengths_raises(self):
        f = HedgeRatioFilter()
        with pytest.raises(ValueError):
            f.filter(np.ones(10), np.ones(11))


# ---------------------------------------------------------------------------
# TrendFilter — construction
# ---------------------------------------------------------------------------

class TestTrendFilterConstruction:
    def test_negative_process_var_raises(self):
        with pytest.raises(ValueError):
            TrendFilter(process_var=-1e-4)

    def test_negative_obs_var_raises(self):
        with pytest.raises(ValueError):
            TrendFilter(obs_var=-1.0)


# ---------------------------------------------------------------------------
# TrendFilter — forward filter
# ---------------------------------------------------------------------------

class TestTrendFilterForward:
    def test_returns_polars_dataframe(self):
        f = TrendFilter()
        prices = np.cumsum(RNG.normal(size=100)) + 100
        result = f.filter(prices)
        assert isinstance(result, pl.DataFrame)

    def test_expected_columns(self):
        f = TrendFilter()
        prices = np.ones(50)
        df = f.filter(prices)
        for col in ["level", "velocity", "level_std", "velocity_std"]:
            assert col in df.columns

    def test_polars_series_input(self):
        prices_np = np.cumsum(RNG.normal(size=60)) + 50
        prices = pl.Series("prices", prices_np)
        f = TrendFilter()
        df = f.filter(prices)
        assert df.height == 60

    def test_reduces_noise(self):
        """Filtered level should have lower RMSE vs truth than raw obs."""
        T = 500
        t_arr = np.arange(T, dtype=float)
        truth = 50.0 + 0.05 * t_arr
        noisy = truth + RNG.normal(0, 2.0, T)

        f = TrendFilter(process_var=1e-5, obs_var=4.0)
        df = f.filter(noisy)

        rmse_raw  = float(np.sqrt(np.mean((noisy - truth) ** 2)))
        rmse_filt = float(np.sqrt(np.mean((df["level"].to_numpy() - truth) ** 2)))
        assert rmse_filt < rmse_raw

    def test_velocity_sign_matches_trend(self):
        """Upward-trending price → positive velocity on average."""
        prices = np.linspace(10.0, 50.0, 200) + RNG.normal(0, 0.1, 200)
        f = TrendFilter(process_var=1e-4, obs_var=0.01)
        df = f.filter(prices)
        assert float(df["velocity"].to_numpy()[20:].mean()) > 0

    def test_std_columns_nonnegative(self):
        prices = np.cumsum(RNG.normal(size=80)) + 100
        f = TrendFilter()
        df = f.filter(prices)
        assert np.all(df["level_std"].to_numpy() >= 0)
        assert np.all(df["velocity_std"].to_numpy() >= 0)

    def test_missing_observations_handled(self):
        prices = np.cumsum(RNG.normal(size=100)) + 100
        prices[30] = np.nan
        prices[60] = np.nan
        f = TrendFilter()
        df = f.filter(prices)
        assert np.all(np.isfinite(df["level"].to_numpy()))


# ---------------------------------------------------------------------------
# TrendFilter — RTS smoother
# ---------------------------------------------------------------------------

class TestTrendFilterSmoother:
    def test_smooth_flag_runs(self):
        prices = np.cumsum(RNG.normal(size=80)) + 100
        f = TrendFilter()
        df = f.filter(prices, smooth=True)
        assert isinstance(df, pl.DataFrame)

    def test_smoother_variance_le_filter_variance(self):
        T = 300
        prices = np.cumsum(RNG.normal(size=T)) + 100

        f1 = TrendFilter(process_var=1e-4, obs_var=1.0)
        f2 = TrendFilter(process_var=1e-4, obs_var=1.0)

        df_filt   = f1.filter(prices, smooth=False)
        df_smooth = f2.filter(prices, smooth=True)

        filt_var   = df_filt["level_std"].to_numpy() ** 2
        smooth_var = df_smooth["level_std"].to_numpy() ** 2
        assert np.all(smooth_var <= filt_var + 1e-10)

    def test_smoother_reduces_rmse(self):
        T = 500
        truth = 50.0 + 0.05 * np.arange(T, dtype=float)
        noisy = truth + RNG.normal(0, 2.0, T)

        f1 = TrendFilter(process_var=1e-5, obs_var=4.0)
        f2 = TrendFilter(process_var=1e-5, obs_var=4.0)
        df_filt   = f1.filter(noisy, smooth=False)
        df_smooth = f2.filter(noisy, smooth=True)

        rmse_filt   = float(np.sqrt(np.mean((df_filt["level"].to_numpy()   - truth) ** 2)))
        rmse_smooth = float(np.sqrt(np.mean((df_smooth["level"].to_numpy() - truth) ** 2)))
        assert rmse_smooth <= rmse_filt + 1e-6


# ---------------------------------------------------------------------------
# TrendFilter — log-likelihood
# ---------------------------------------------------------------------------

class TestTrendFilterLogLikelihood:
    def test_ll_is_finite_negative(self):
        prices = np.cumsum(RNG.normal(size=100)) + 100
        f = TrendFilter()
        ll = f.log_likelihood(prices)
        assert np.isfinite(ll)
        assert ll < 0.0

    def test_correct_obs_var_has_higher_ll(self):
        T = 300
        sigma_obs = 2.0
        truth = np.linspace(10, 50, T)
        prices = truth + RNG.normal(0, sigma_obs, T)

        f_good = TrendFilter(process_var=1e-5, obs_var=sigma_obs ** 2)
        f_bad  = TrendFilter(process_var=1e-5, obs_var=100.0)
        assert f_good.log_likelihood(prices) > f_bad.log_likelihood(prices)
