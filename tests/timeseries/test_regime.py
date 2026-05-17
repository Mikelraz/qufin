"""
Tests for qufin.timeseries.regime — Markov-switching AR.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from qufin.timeseries.regime import MarkovSwitchingAR, MSARFitResult

RNG = np.random.default_rng(99)


def _two_regime_mean_shift(n_per: int = 400, seed: int = 1) -> np.ndarray:
    """Three blocks alternating between μ = ±1, σ = 0.4."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal(loc=-1.0, scale=0.4, size=n_per),
            rng.normal(loc=+1.0, scale=0.4, size=n_per),
            rng.normal(loc=-1.0, scale=0.4, size=n_per),
        ]
    )


class TestMSARConstruction:
    def test_defaults(self):
        m = MarkovSwitchingAR()
        assert m.p == 0 and m.k_regimes == 2

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            MarkovSwitchingAR(p=-1)

    def test_too_few_regimes(self):
        with pytest.raises(ValueError):
            MarkovSwitchingAR(k_regimes=1)

    def test_result_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            _ = MarkovSwitchingAR().result


class TestMSARFit:
    def setup_method(self) -> None:
        self.y = _two_regime_mean_shift(seed=2)
        self.model = MarkovSwitchingAR(p=0, k_regimes=2)
        self.res = self.model.fit(self.y, max_iter=200, tol=1e-7, seed=2)

    def test_returns_result(self):
        assert isinstance(self.res, MSARFitResult)

    def test_recovers_means(self):
        # Means should be roughly ±1 in some order
        sorted_mu = np.sort(self.res.mu)
        assert sorted_mu[0] == pytest.approx(-1.0, abs=0.2)
        assert sorted_mu[1] == pytest.approx(+1.0, abs=0.2)

    def test_recovers_variances(self):
        # σ² ≈ 0.16 for both regimes
        for s in self.res.sigma2:
            assert s == pytest.approx(0.16, abs=0.1)

    def test_transition_row_stochastic(self):
        row_sums = self.res.transition.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)
        assert np.all(self.res.transition >= 0)
        assert np.all(self.res.transition <= 1)

    def test_smoothed_probs_sum_to_one(self):
        sums = self.res.smoothed_probs.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_log_lik_finite(self):
        assert math.isfinite(self.res.log_lik)

    def test_most_likely_regime_shape(self):
        seq = self.res.most_likely_regime()
        assert seq.shape == (self.y.shape[0],)
        assert seq.dtype == np.int64
        assert seq.max() < self.res.k_regimes
        assert seq.min() >= 0

    def test_to_dataframe(self):
        df = self.res.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"t", "regime", "smoothed_prob", "filtered_prob"}


class TestMSARWithAR:
    def test_ar1_fits(self):
        rng = np.random.default_rng(33)
        # Two regimes, AR(1) with different φ
        y = np.zeros(800)
        regime = np.zeros(800, dtype=int)
        for t in range(1, 800):
            regime[t] = regime[t - 1]
            if rng.random() < 0.01:
                regime[t] = 1 - regime[t]
            phi = 0.7 if regime[t] == 0 else -0.3
            y[t] = phi * y[t - 1] + rng.standard_normal() * 0.4
        m = MarkovSwitchingAR(p=1, k_regimes=2)
        res = m.fit(y, max_iter=200, seed=4)
        assert res.phi.shape == (2, 1)
        # AR coefficients should span a meaningful range
        assert res.phi.max() - res.phi.min() > 0.2


class TestMSARSimulate:
    def test_simulate(self):
        y = _two_regime_mean_shift(n_per=200, seed=5)
        m = MarkovSwitchingAR(p=0, k_regimes=2)
        m.fit(y, max_iter=100, seed=5)
        sim_y, sim_states = m.simulate(500, seed=10)
        assert sim_y.shape == (500,)
        assert sim_states.shape == (500,)
        assert sim_states.max() < 2
        assert sim_states.min() >= 0

    def test_simulate_invalid_t_total(self):
        y = _two_regime_mean_shift(n_per=200, seed=6)
        m = MarkovSwitchingAR(p=0, k_regimes=2)
        m.fit(y, max_iter=50, seed=6)
        with pytest.raises(ValueError):
            m.simulate(0)


class TestMSARShortSeries:
    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            MarkovSwitchingAR(p=2, k_regimes=2).fit(np.zeros(3))
