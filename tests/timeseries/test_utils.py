"""Tests for src.timeseries.utils — differencing and info criteria."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.timeseries.utils import (
    difference,
    info_criteria,
    inverse_difference,
    seasonal_difference,
)

RNG = np.random.default_rng(42)


class TestDifference:
    def test_first_difference_matches_np_diff(self):
        x = RNG.standard_normal(20)
        np.testing.assert_allclose(difference(x, 1), np.diff(x, n=1))

    def test_second_difference_matches_np_diff(self):
        x = RNG.standard_normal(20)
        np.testing.assert_allclose(difference(x, 2), np.diff(x, n=2))

    def test_zero_returns_copy(self):
        x = RNG.standard_normal(5)
        out = difference(x, 0)
        np.testing.assert_array_equal(out, x)
        assert out is not x  # copy

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            difference(np.arange(5), -1)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="cannot apply"):
            difference(np.arange(2), 5)


class TestSeasonalDifference:
    def test_period_one_matches_diff(self):
        x = RNG.standard_normal(20)
        np.testing.assert_allclose(seasonal_difference(x, s=1, d_seasonal=1), np.diff(x))

    def test_period_four(self):
        x = np.arange(12, dtype=np.float64)
        out = seasonal_difference(x, s=4, d_seasonal=1)
        np.testing.assert_array_equal(out, np.full(8, 4.0))

    def test_zero_d_seasonal_returns_copy(self):
        x = RNG.standard_normal(8)
        out = seasonal_difference(x, s=2, d_seasonal=0)
        np.testing.assert_array_equal(out, x)


class TestInverseDifference:
    def test_first_order_round_trip(self):
        x = RNG.standard_normal(50)
        diffs = difference(x, 1)
        recon = inverse_difference(diffs, x[:1])
        np.testing.assert_allclose(recon, x, atol=1e-12)

    def test_second_order_round_trip(self):
        x = RNG.standard_normal(50)
        diffs = difference(x, 2)
        recon = inverse_difference(diffs, x[:2])
        np.testing.assert_allclose(recon, x, atol=1e-10)

    def test_third_order_round_trip(self):
        x = RNG.standard_normal(50)
        diffs = difference(x, 3)
        recon = inverse_difference(diffs, x[:3])
        np.testing.assert_allclose(recon, x, atol=1e-10)

    def test_zero_seeds_passes_through(self):
        diffs = np.arange(5, dtype=np.float64)
        out = inverse_difference(diffs, np.empty(0))
        np.testing.assert_array_equal(out, diffs)


class TestInfoCriteria:
    def test_values_match_formula(self):
        ll = -123.0
        n = 200
        k = 4
        aic, bic, hqic = info_criteria(ll, n, k)
        assert aic == pytest.approx(-2.0 * ll + 2.0 * k)
        assert bic == pytest.approx(-2.0 * ll + k * math.log(n))
        assert hqic == pytest.approx(-2.0 * ll + 2.0 * k * math.log(math.log(n)))

    def test_aic_bic_orderings(self):
        # For n > 8 (so log log n > 0 and log n > 2) the penalty ordering is:
        #   AIC penalty (= 2k) < HQIC penalty < BIC penalty.
        aic, bic, hqic = info_criteria(log_lik=0.0, n_obs=1000, n_params=5)
        assert aic < hqic < bic

    def test_invalid_n_obs(self):
        with pytest.raises(ValueError):
            info_criteria(0.0, 0, 1)

    def test_invalid_n_params(self):
        with pytest.raises(ValueError):
            info_criteria(0.0, 100, -1)
