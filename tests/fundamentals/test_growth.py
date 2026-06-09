"""Tests for qufin.fundamentals.growth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qufin.fundamentals import (
    FundamentalSnapshot,
    cagr,
    dupont_3factor,
    dupont_5factor,
    period_growth,
    sustainable_growth_rate,
    yoy_growth,
)


def test_cagr() -> None:
    assert cagr(100.0, 200.0, 2.0) == pytest.approx(math.sqrt(2.0) - 1.0)
    assert cagr(100.0, 100.0, 5.0) == pytest.approx(0.0)


def test_cagr_invalid_inputs_are_nan() -> None:
    assert math.isnan(cagr(0.0, 100.0, 2.0))
    assert math.isnan(cagr(-100.0, 100.0, 2.0))
    assert math.isnan(cagr(100.0, 200.0, 0.0))


def test_period_growth() -> None:
    values = np.array([100.0, 110.0, 121.0], dtype=np.float64)
    np.testing.assert_allclose(period_growth(values), [0.1, 0.1])


def test_period_growth_too_short_is_empty() -> None:
    assert period_growth(np.array([100.0])).shape == (0,)


def test_yoy_growth() -> None:
    prev = FundamentalSnapshot(revenue=1000.0)
    curr = FundamentalSnapshot(revenue=1300.0)
    assert yoy_growth(curr, prev, "revenue") == pytest.approx(0.3)


def test_yoy_growth_handles_negative_base() -> None:
    prev = FundamentalSnapshot(net_income=-50.0)
    curr = FundamentalSnapshot(net_income=50.0)
    # divides by |prev| so a swing from loss to profit is positive growth
    assert yoy_growth(curr, prev, "net_income") == pytest.approx(2.0)


def test_sustainable_growth_rate() -> None:
    assert sustainable_growth_rate(0.20, 0.40) == pytest.approx(0.12)


def test_dupont_3factor_reconstructs_roe(base_snapshot: FundamentalSnapshot) -> None:
    dp = dupont_3factor(base_snapshot)
    assert dp.n_factors == 3
    assert dp.roe == pytest.approx(0.18)
    product = math.prod(dp.factors.values())
    assert product == pytest.approx(dp.roe)
    assert dp.factors["net_margin"] == pytest.approx(0.144)
    assert dp.factors["equity_multiplier"] == pytest.approx(2.5)


def test_dupont_5factor_matches_3factor(base_snapshot: FundamentalSnapshot) -> None:
    dp = dupont_5factor(base_snapshot)
    assert dp.n_factors == 5
    assert dp.roe == pytest.approx(0.18)
    assert dp.factors["tax_burden"] == pytest.approx(0.8)
    assert dp.factors["interest_burden"] == pytest.approx(0.9)
