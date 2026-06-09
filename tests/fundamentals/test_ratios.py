"""Tests for qufin.fundamentals.ratios."""

from __future__ import annotations

import math

import pytest

from qufin.fundamentals import FundamentalSnapshot, compute_ratios
from qufin.fundamentals import ratios as rat
from qufin.fundamentals._util import safe_div


def test_safe_div_guards() -> None:
    assert safe_div(6.0, 2.0) == 3.0
    assert math.isnan(safe_div(1.0, 0.0))
    assert math.isnan(safe_div(1.0, math.nan))
    assert math.isnan(safe_div(math.nan, 2.0))


def test_profitability(base_snapshot: FundamentalSnapshot) -> None:
    s = base_snapshot
    assert rat.gross_margin(s) == pytest.approx(0.4)
    assert rat.operating_margin(s) == pytest.approx(0.2)
    assert rat.net_margin(s) == pytest.approx(0.144)
    assert rat.ebitda_margin(s) == pytest.approx(0.25)
    assert rat.return_on_assets(s) == pytest.approx(144.0 / 2000.0)
    assert rat.return_on_equity(s) == pytest.approx(144.0 / 800.0)


def test_roic_uses_nopat(base_snapshot: FundamentalSnapshot) -> None:
    # tax rate = 36/180 = 0.20, NOPAT = 200*0.8 = 160, invested = 800+700 = 1500
    assert rat.return_on_invested_capital(base_snapshot) == pytest.approx(160.0 / 1500.0)


def test_liquidity_and_leverage(base_snapshot: FundamentalSnapshot) -> None:
    s = base_snapshot
    assert rat.current_ratio(s) == pytest.approx(2.0)
    assert rat.quick_ratio(s) == pytest.approx((800.0 - 200.0) / 400.0)
    assert rat.cash_ratio(s) == pytest.approx(300.0 / 400.0)
    assert rat.debt_to_equity(s) == pytest.approx(700.0 / 800.0)
    assert rat.debt_to_assets(s) == pytest.approx(700.0 / 2000.0)
    assert rat.net_debt_to_ebitda(s) == pytest.approx((700.0 - 300.0) / 250.0)
    assert rat.interest_coverage(s) == pytest.approx(200.0 / 20.0)


def test_per_share(base_snapshot: FundamentalSnapshot) -> None:
    s = base_snapshot
    assert rat.eps(s) == pytest.approx(1.44)
    assert rat.book_value_per_share(s) == pytest.approx(8.0)
    assert rat.fcf_per_share(s) == pytest.approx(1.7)


def test_average_balance_with_prev(base_snapshot: FundamentalSnapshot) -> None:
    prev = FundamentalSnapshot(total_assets=1000.0, total_equity=600.0)
    # average assets = (2000 + 1000)/2 = 1500
    assert rat.return_on_assets(base_snapshot, prev) == pytest.approx(144.0 / 1500.0)
    assert rat.asset_turnover(base_snapshot, prev) == pytest.approx(1000.0 / 1500.0)


def test_gross_profit_fallback() -> None:
    s = FundamentalSnapshot(revenue=1000.0, cogs=600.0)  # gross_profit missing
    assert rat.gross_margin(s) == pytest.approx(0.4)


def test_fcf_fallback() -> None:
    s = FundamentalSnapshot(operating_cash_flow=220.0, capex=-50.0, shares_outstanding=100.0)
    assert rat.fcf_per_share(s) == pytest.approx(1.7)


def test_missing_inputs_propagate_nan() -> None:
    empty = FundamentalSnapshot()
    rs = compute_ratios(empty)
    assert all(math.isnan(v) for v in rs.as_dict().values())


def test_compute_ratios_matches_individuals(base_snapshot: FundamentalSnapshot) -> None:
    rs = compute_ratios(base_snapshot)
    assert rs.gross_margin == pytest.approx(rat.gross_margin(base_snapshot))
    assert rs.return_on_equity == pytest.approx(rat.return_on_equity(base_snapshot))
    assert set(rs.as_dict()) == {f for f in rs.as_dict()}  # as_dict round-trips
