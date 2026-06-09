"""Tests for qufin.fundamentals.valuation."""

from __future__ import annotations

import pytest

from qufin.fundamentals import (
    gordon_growth_ddm,
    multi_stage_ddm,
    residual_income_value,
    two_stage_dcf,
    wacc,
)


def test_wacc() -> None:
    # E=600, D=400, V=1000; 0.6*0.10 + 0.4*0.05*(1-0.25) = 0.06 + 0.015 = 0.075
    assert wacc(600.0, 400.0, 0.10, 0.05, 0.25) == pytest.approx(0.075)


def test_gordon_growth() -> None:
    assert gordon_growth_ddm(2.0, 0.08, 0.03) == pytest.approx(40.0)


def test_gordon_growth_requires_r_above_g() -> None:
    with pytest.raises(ValueError, match="exceed growth"):
        gordon_growth_ddm(2.0, 0.03, 0.05)


def test_multi_stage_ddm_matches_manual_sum() -> None:
    dividends = [2.0, 2.2, 2.42]
    r, g = 0.10, 0.04
    expected = sum(d / (1.0 + r) ** t for t, d in enumerate(dividends, start=1))
    terminal = dividends[-1] * (1.0 + g) / (r - g)
    expected += terminal / (1.0 + r) ** len(dividends)
    assert multi_stage_ddm(dividends, g, r) == pytest.approx(expected)


def test_two_stage_dcf_matches_manual_discounting() -> None:
    fcf0, g_high, years, g_term, r = 100.0, 0.10, 5, 0.03, 0.09
    cash = [fcf0 * (1.0 + g_high) ** t for t in range(1, years + 1)]
    pv_explicit = sum(c / (1.0 + r) ** t for t, c in enumerate(cash, start=1))
    terminal = cash[-1] * (1.0 + g_term) / (r - g_term)
    pv_terminal = terminal / (1.0 + r) ** years
    expected_ev = pv_explicit + pv_terminal

    v = two_stage_dcf(fcf0, g_high, years, g_term, r, net_debt=200.0, shares_outstanding=50.0)
    assert v.enterprise_value == pytest.approx(expected_ev)
    assert v.equity_value == pytest.approx(expected_ev - 200.0)
    assert v.value_per_share == pytest.approx((expected_ev - 200.0) / 50.0)
    assert v.pv_terminal == pytest.approx(pv_terminal)
    assert v.pv_cash_flows.sum() == pytest.approx(pv_explicit)


def test_two_stage_dcf_validates_inputs() -> None:
    with pytest.raises(ValueError, match="high_years"):
        two_stage_dcf(100.0, 0.1, 0, 0.03, 0.09)
    with pytest.raises(ValueError, match="terminal_growth"):
        two_stage_dcf(100.0, 0.1, 5, 0.10, 0.09)


def test_residual_income_flat_case_reduces_to_book_times_roe_over_r() -> None:
    # With no growth and constant ROE, value -> B0 * ROE / r
    value = residual_income_value(1000.0, 0.12, 0.10, 0.0, 8, 0.0)
    assert value == pytest.approx(1000.0 * 0.12 / 0.10)


def test_residual_income_no_excess_return_equals_book() -> None:
    # ROE == cost of equity -> zero residual income -> value == book value
    value = residual_income_value(1000.0, 0.10, 0.10, 0.05, 5, 0.02)
    assert value == pytest.approx(1000.0)
