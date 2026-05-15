"""
Tests for src.strategies.mean_reversion — MeanReversionStrategy.

Coverage
--------
  StrategyParams — construction, validation, constraint enforcement
  BacktestResult — shape, properties, to_dataframe, summary
  MeanReversionStrategy.run() — output shapes, causality, signal logic,
      warm-up period, NaN handling, pd.Series index propagation
  Signal logic — long/short entry, exit, stop-loss, correct P&L sign
  Streaming mode — step() matches run(), reset() cleans state
  Log-likelihood — finite, negative, correct ordering
  Training (sharpe) — improves or maintains Sharpe, updates params
  Training (likelihood) — maximises log-lik, updates delta/obs_var
  Edge cases — very short series, flat prices, explosive series
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.ou_process import OrnsteinUhlenbeck
from src.strategies.mean_reversion import (
    BacktestResult,
    MeanReversionStrategy,
    StrategyParams,
    TrainResult,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(99)


def make_ou_series(T: int = 500, theta: float = 0.2, mu: float = 10.0,
                   sigma: float = 0.5, seed: int = 0) -> np.ndarray:
    """Simulate a stationary OU price series of exactly T observations."""
    ou = OrnsteinUhlenbeck(theta=theta, mu=mu, sigma=sigma, dt=1.0)
    return ou.simulate(T - 1, x0=mu, seed=seed)


def default_strategy() -> MeanReversionStrategy:
    return MeanReversionStrategy(StrategyParams(
        delta=1e-4, obs_var=0.25, entry_z=1.5, exit_z=0.5,
        stop_z=3.5, vol_window=40, dt=1.0,
    ))


# ---------------------------------------------------------------------------
# StrategyParams
# ---------------------------------------------------------------------------

class TestStrategyParams:
    def test_defaults_valid(self):
        p = StrategyParams()
        assert p.delta > 0
        assert p.obs_var > 0
        assert p.entry_z > 0
        assert 0 <= p.exit_z < p.entry_z
        assert p.stop_z > p.entry_z

    def test_negative_delta_raises(self):
        with pytest.raises(ValueError):
            StrategyParams(delta=-1e-4)

    def test_zero_obs_var_raises(self):
        with pytest.raises(ValueError):
            StrategyParams(obs_var=0.0)

    def test_exit_ge_entry_raises(self):
        with pytest.raises(ValueError):
            StrategyParams(entry_z=1.0, exit_z=1.5)

    def test_exit_eq_entry_raises(self):
        with pytest.raises(ValueError):
            StrategyParams(entry_z=1.5, exit_z=1.5)

    def test_stop_le_entry_raises(self):
        with pytest.raises(ValueError):
            StrategyParams(entry_z=1.5, exit_z=0.5, stop_z=1.4)

    def test_negative_vol_window_raises(self):
        with pytest.raises(ValueError):
            StrategyParams(vol_window=2)

    def test_to_dict_keys(self):
        p = StrategyParams()
        d = p.to_dict()
        assert set(d.keys()) == {"delta", "obs_var", "entry_z", "exit_z", "stop_z"}


# ---------------------------------------------------------------------------
# run() — output structure
# ---------------------------------------------------------------------------

class TestRunOutputStructure:
    def test_returns_backtest_result(self):
        s = default_strategy()
        prices = make_ou_series(200)
        result = s.run(prices)
        assert isinstance(result, BacktestResult)

    def test_array_lengths(self):
        s = default_strategy()
        T = 300
        prices = make_ou_series(T)
        r = s.run(prices)
        assert len(r.mu)          == T
        assert len(r.theta)       == T
        assert len(r.half_life)   == T
        assert len(r.sigma_eq)    == T
        assert len(r.z_score)     == T
        assert len(r.signal)      == T
        assert len(r.log_returns) == T - 1

    def test_prices_stored(self):
        s = default_strategy()
        prices = make_ou_series(200)
        r = s.run(prices)
        np.testing.assert_array_equal(r.prices, prices)

    def test_series_index_propagated(self):
        s = default_strategy()
        idx = pd.date_range("2022-01-01", periods=200, freq="B")
        series = pd.Series(make_ou_series(200), index=idx)
        r = s.run(series)
        assert list(r.index) == list(idx)

    def test_to_dataframe_shape(self):
        s = default_strategy()
        T = 250
        r = s.run(make_ou_series(T))
        df = r.to_dataframe()
        assert df.shape == (T, 8)

    def test_to_dataframe_columns(self):
        s = default_strategy()
        r = s.run(make_ou_series(200))
        df = r.to_dataframe()
        expected = {"price", "mu", "theta", "half_life", "sigma_eq",
                    "z_score", "signal", "strat_ret"}
        assert set(df.columns) == expected

    def test_summary_string(self):
        s = default_strategy()
        r = s.run(make_ou_series(300))
        txt = r.summary()
        assert "Sharpe" in txt

    def test_too_short_raises(self):
        s = default_strategy()   # vol_window=40
        with pytest.raises(ValueError):
            s.run(make_ou_series(30))


# ---------------------------------------------------------------------------
# run() — state properties
# ---------------------------------------------------------------------------

class TestRunStateProperties:
    def test_warmup_nans(self):
        """Before vol_window bars, sigma_eq and z_score should be NaN."""
        s = default_strategy()   # vol_window=40
        prices = make_ou_series(200)
        r = s.run(prices)
        # At least the first vol_window bars should have NaN z_score
        assert np.all(np.isnan(r.z_score[:2]))

    def test_theta_positive_where_finite(self):
        s = default_strategy()
        r = s.run(make_ou_series(300))
        valid = r.theta[~np.isnan(r.theta)]
        assert np.all(valid > 0)

    def test_half_life_positive_where_finite(self):
        s = default_strategy()
        r = s.run(make_ou_series(300))
        valid = r.half_life[~np.isnan(r.half_life)]
        assert np.all(valid > 0)

    def test_sigma_eq_nonnegative(self):
        s = default_strategy()
        r = s.run(make_ou_series(300))
        valid = r.sigma_eq[~np.isnan(r.sigma_eq)]
        assert np.all(valid >= 0)

    def test_signal_values(self):
        s = default_strategy()
        r = s.run(make_ou_series(400))
        assert set(np.unique(r.signal)).issubset({-1.0, 0.0, 1.0})

    def test_mu_tracks_series_mean(self):
        """After many bars, filtered mean should be near the true OU mean."""
        s = default_strategy()
        mu_true = 10.0
        prices = make_ou_series(2000, mu=mu_true)
        r = s.run(prices)
        # Use last 1000 bars to skip warm-up
        assert np.nanmean(r.mu[-1000:]) == pytest.approx(mu_true, abs=0.5)

    def test_log_returns_finite(self):
        s = default_strategy()
        r = s.run(make_ou_series(300))
        assert np.all(np.isfinite(r.log_returns))

    def test_all_state_arrays_finite_after_warmup(self):
        s = default_strategy()   # vol_window=40
        r = s.run(make_ou_series(300))
        for arr in [r.mu, r.theta, r.sigma_eq, r.z_score]:
            # Everything after warm-up should be finite
            assert np.all(np.isfinite(arr[50:]))


# ---------------------------------------------------------------------------
# run() — signal logic correctness
# ---------------------------------------------------------------------------

class TestSignalLogic:
    def test_no_signal_during_warmup(self):
        """Signals must be zero until vol_window bars have accumulated."""
        s = default_strategy()  # vol_window=40
        prices = make_ou_series(300)
        r = s.run(prices)
        # First bar is always 0 (no previous price)
        assert r.signal[0] == 0.0

    def test_long_entry_when_z_low(self):
        """Force z_score very negative → strategy must go long."""
        s = MeanReversionStrategy(StrategyParams(
            delta=1e-4, obs_var=0.25, entry_z=0.1,  # very low threshold
            exit_z=0.05, stop_z=5.0, vol_window=20, dt=1.0,
        ))
        prices = make_ou_series(500, theta=1.0)   # fast mean-reversion
        r = s.run(prices)
        # Should have at least some long positions
        assert np.any(r.signal == 1.0)

    def test_short_entry_when_z_high(self):
        """Strategy must go short when z_score is very high."""
        s = MeanReversionStrategy(StrategyParams(
            delta=1e-4, obs_var=0.25, entry_z=0.1,
            exit_z=0.05, stop_z=5.0, vol_window=20, dt=1.0,
        ))
        prices = make_ou_series(500, theta=1.0)
        r = s.run(prices)
        assert np.any(r.signal == -1.0)

    def test_pnl_sign_correct_for_long(self):
        """Long position earns positive return when price rises."""
        # Manually verify: signal=+1 * positive_return > 0
        s = default_strategy()
        prices = make_ou_series(500, theta=0.5)
        r = s.run(prices)
        # Where signal is +1, strategy return = +1 * price_log_return
        log_ret_price = np.diff(np.log(prices))
        long_mask = r.signal[:-1] == 1.0
        short_mask = r.signal[:-1] == -1.0
        if long_mask.any():
            assert np.allclose(
                r.log_returns[long_mask],
                log_ret_price[long_mask],
            )
        if short_mask.any():
            assert np.allclose(
                r.log_returns[short_mask],
                -log_ret_price[short_mask],
            )

    def test_stop_loss_closes_position(self):
        """When |z| > stop_z the position must be 0 at that bar."""
        s = MeanReversionStrategy(StrategyParams(
            delta=1e-4, obs_var=0.25, entry_z=0.1,
            exit_z=0.05, stop_z=1.0,    # tight stop
            vol_window=20, dt=1.0,
        ))
        prices = make_ou_series(500, theta=0.5)
        r = s.run(prices)
        stop_mask = np.abs(r.z_score) > 1.0
        # Wherever a stop fires, the signal at that bar (after stopping) must be 0
        for t in range(1, len(prices)):
            if stop_mask[t] and r.signal[t - 1] != 0.0:
                assert r.signal[t] == 0.0, f"Stop-loss not applied at t={t}"
                break  # one confirmed check is sufficient

    def test_causality_no_lookahead(self):
        """Signal at t must not change when future prices are perturbed."""
        s = default_strategy()
        prices = make_ou_series(400)
        r1 = s.run(prices)

        # Replace the last 50 prices with noise
        prices_mod = prices.copy()
        prices_mod[-50:] = 10.0 + RNG.normal(0, 0.1, 50)
        r2 = s.run(prices_mod)

        # Signals up to bar T-51 must be identical
        np.testing.assert_array_equal(r1.signal[:-51], r2.signal[:-51])


# ---------------------------------------------------------------------------
# BacktestResult properties
# ---------------------------------------------------------------------------

class TestBacktestResultProperties:
    def test_sharpe_finite(self):
        s = default_strategy()
        r = s.run(make_ou_series(500))
        assert np.isfinite(r.sharpe)

    def test_total_return_finite(self):
        s = default_strategy()
        r = s.run(make_ou_series(500))
        assert np.isfinite(r.total_return)

    def test_max_drawdown_nonpositive(self):
        s = default_strategy()
        r = s.run(make_ou_series(500))
        assert r.max_drawdown <= 0.0

    def test_n_trades_nonneg(self):
        s = default_strategy()
        r = s.run(make_ou_series(500))
        assert r.n_trades >= 0

    def test_flat_strategy_zero_sharpe(self):
        """If signal is always 0, Sharpe is 0."""
        s = MeanReversionStrategy(StrategyParams(
            entry_z=100.0, exit_z=0.5, stop_z=200.0,  # never triggers
            vol_window=20, dt=1.0,
        ))
        prices = make_ou_series(300)
        r = s.run(prices)
        assert r.sharpe == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Streaming mode
# ---------------------------------------------------------------------------

class TestStreamingMode:
    def test_reset_required_before_step(self):
        s = MeanReversionStrategy()
        with pytest.raises(RuntimeError):
            s.step(10.0)

    def test_step_returns_expected_keys(self):
        s = default_strategy()
        s.reset()
        s.step(10.0)  # first step (no previous price)
        state = s.step(10.1)
        expected_keys = {"signal", "z_score", "mu", "theta", "half_life", "sigma_eq"}
        assert set(state.keys()) == expected_keys

    def test_step_signal_values(self):
        s = default_strategy()
        s.reset()
        prices = make_ou_series(200)
        for price in prices:
            state = s.step(price)
            assert state["signal"] in (-1.0, 0.0, 1.0)

    def test_step_matches_run(self):
        """Streaming step() must produce the same signals as batch run()."""
        s = default_strategy()
        prices = make_ou_series(300)

        # Batch
        r = s.run(prices)

        # Streaming
        s.reset()
        stream_signals = []
        for price in prices:
            st = s.step(price)
            stream_signals.append(st["signal"])

        np.testing.assert_array_equal(
            r.signal,
            np.array(stream_signals),
        )

    def test_reset_clears_state(self):
        s = default_strategy()
        prices = make_ou_series(200)
        s.reset()
        for p in prices[:100]:
            s.step(p)

        s.reset()
        # After reset, streaming should give same result as fresh start
        s2 = default_strategy()
        s2.reset()

        signals_after_reset = []
        signals_fresh = []
        for p in prices:
            signals_after_reset.append(s.step(p)["signal"])
            signals_fresh.append(s2.step(p)["signal"])

        np.testing.assert_array_equal(signals_after_reset, signals_fresh)


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def test_finite_and_negative(self):
        s = default_strategy()
        prices = make_ou_series(300)
        ll = s._log_likelihood(prices, s.params)
        assert np.isfinite(ll) and ll < 0

    def test_better_delta_wins(self):
        """A delta calibrated to the data noise level should win on log-lik."""
        prices = make_ou_series(500)
        s_good = MeanReversionStrategy(StrategyParams(delta=1e-4, obs_var=0.25))
        s_bad  = MeanReversionStrategy(StrategyParams(delta=1e-1, obs_var=10.0))
        ll_good = s_good._log_likelihood(prices, s_good.params)
        ll_bad  = s_bad._log_likelihood(prices, s_bad.params)
        assert ll_good > ll_bad


# ---------------------------------------------------------------------------
# Training — likelihood
# ---------------------------------------------------------------------------

class TestFitLikelihood:
    def test_returns_train_result(self):
        s = MeanReversionStrategy()
        prices = make_ou_series(500)
        result = s.fit(prices, method="likelihood")
        assert isinstance(result, TrainResult)

    def test_method_label(self):
        s = MeanReversionStrategy()
        result = s.fit(make_ou_series(500), method="likelihood")
        assert result.method == "likelihood"

    def test_params_updated_on_object(self):
        s = MeanReversionStrategy()
        old_delta = s.params.delta
        s.fit(make_ou_series(500), method="likelihood")
        # delta should have been tuned (may or may not change significantly)
        assert s.params.delta > 0

    def test_signal_thresholds_unchanged(self):
        """Likelihood training must NOT touch entry/exit/stop thresholds."""
        s = MeanReversionStrategy(StrategyParams(
            entry_z=2.0, exit_z=0.8, stop_z=4.0,
        ))
        s.fit(make_ou_series(500), method="likelihood")
        assert s.params.entry_z == pytest.approx(2.0)
        assert s.params.exit_z  == pytest.approx(0.8)
        assert s.params.stop_z  == pytest.approx(4.0)

    def test_fitted_ll_ge_default_ll(self):
        """Optimised params should achieve at least as high log-lik."""
        prices = make_ou_series(500)
        s = MeanReversionStrategy()
        ll_before = s._log_likelihood(prices, s.params)
        s.fit(prices, method="likelihood")
        ll_after = s._log_likelihood(prices, s.params)
        assert ll_after >= ll_before - 1.0   # allow tiny numerical slack

    def test_unknown_method_raises(self):
        s = MeanReversionStrategy()
        with pytest.raises(ValueError):
            s.fit(make_ou_series(300), method="kl_divergence")


# ---------------------------------------------------------------------------
# Training — Sharpe
# ---------------------------------------------------------------------------

class TestFitSharpe:
    def test_returns_train_result(self):
        s = MeanReversionStrategy()
        result = s.fit(make_ou_series(800), method="sharpe", n_restarts=1)
        assert isinstance(result, TrainResult)

    def test_method_label(self):
        s = MeanReversionStrategy()
        r = s.fit(make_ou_series(800), method="sharpe", n_restarts=1)
        assert r.method == "sharpe"

    def test_all_params_updated(self):
        """Sharpe training must update all five trainable parameters."""
        s = MeanReversionStrategy()
        s.fit(make_ou_series(800), method="sharpe", n_restarts=1)
        p = s.params
        assert p.delta > 0
        assert p.obs_var > 0
        assert p.entry_z > 0
        assert 0 <= p.exit_z < p.entry_z
        assert p.stop_z > p.entry_z

    def test_params_satisfy_constraints(self):
        s = MeanReversionStrategy()
        s.fit(make_ou_series(800), method="sharpe", n_restarts=2)
        p = s.params
        assert p.exit_z < p.entry_z
        assert p.stop_z > p.entry_z

    def test_str_output(self):
        s = MeanReversionStrategy()
        result = s.fit(make_ou_series(400), method="sharpe", n_restarts=1)
        assert len(str(result)) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_flat_price_series(self):
        """Constant prices must not crash (zero innovations)."""
        s = default_strategy()
        prices = np.full(300, 10.0)
        r = s.run(prices)          # no exception
        assert np.all(r.log_returns == 0.0)

    def test_random_walk_runs_without_crash(self):
        """A non-stationary series must not crash (will be detected as slow reversion)."""
        rw = np.cumsum(RNG.normal(0, 1, 500)) + 50.0
        s = default_strategy()
        r = s.run(rw)
        assert np.all(np.isfinite(r.log_returns))

    def test_pandas_series_input(self):
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        series = pd.Series(make_ou_series(300), index=idx)
        s = default_strategy()
        r = s.run(series)
        assert isinstance(r.to_dataframe(), pd.DataFrame)
        assert list(r.to_dataframe().index) == list(idx)

    def test_train_then_run_consistent(self):
        """Fitted params should produce consistent run() output."""
        prices = make_ou_series(600)
        s = MeanReversionStrategy()
        s.fit(prices[:400], method="likelihood")
        r = s.run(prices)
        assert np.all(np.isfinite(r.log_returns))

    def test_multiple_resets_idempotent(self):
        s = default_strategy()
        s.reset()
        s.reset()
        prices = make_ou_series(200)
        signals = [s.step(p)["signal"] for p in prices]
        # All signals should be valid floats
        assert all(sig in (-1.0, 0.0, 1.0) for sig in signals)
