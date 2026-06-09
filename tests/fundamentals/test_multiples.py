"""Tests for qufin.fundamentals.multiples."""

from __future__ import annotations

import math

import pytest

from qufin.fundamentals import FundamentalSnapshot
from qufin.fundamentals import multiples as mult


def test_equity_multiples(base_snapshot: FundamentalSnapshot) -> None:
    s = base_snapshot
    assert mult.pe_ratio(s) == pytest.approx(5000.0 / 144.0)
    assert mult.pb_ratio(s) == pytest.approx(5000.0 / 800.0)
    assert mult.ps_ratio(s) == pytest.approx(5000.0 / 1000.0)
    assert mult.earnings_yield(s) == pytest.approx(144.0 / 5000.0)
    assert mult.fcf_yield(s) == pytest.approx(170.0 / 5000.0)


def test_enterprise_multiples(base_snapshot: FundamentalSnapshot) -> None:
    s = base_snapshot
    # EV = market cap + net debt = 5000 + (700 - 300) = 5400
    assert mult.enterprise_value(s) == pytest.approx(5400.0)
    assert mult.ev_to_ebitda(s) == pytest.approx(5400.0 / 250.0)
    assert mult.ev_to_sales(s) == pytest.approx(5400.0 / 1000.0)


def test_dividend_yield_uses_absolute(base_snapshot: FundamentalSnapshot) -> None:
    # dividends_paid is reported negative (-40); yield should be positive
    assert mult.dividend_yield(base_snapshot) == pytest.approx(40.0 / 5000.0)


def test_peg_ratio(base_snapshot: FundamentalSnapshot) -> None:
    pe = mult.pe_ratio(base_snapshot)
    assert mult.peg_ratio(base_snapshot, 0.15) == pytest.approx(pe / 15.0)
    assert math.isnan(mult.peg_ratio(base_snapshot, 0.0))


def test_market_cap_fallback_from_price_and_shares() -> None:
    s = FundamentalSnapshot(net_income=100.0, price=50.0, shares_outstanding=100.0)
    # market_cap missing -> price * shares = 5000
    assert mult.pe_ratio(s) == pytest.approx(5000.0 / 100.0)


def test_missing_market_data_is_nan() -> None:
    s = FundamentalSnapshot(net_income=100.0)  # no price/shares/market_cap
    assert math.isnan(mult.pe_ratio(s))
