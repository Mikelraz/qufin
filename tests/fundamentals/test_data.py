"""Tests for qufin.fundamentals.data — statement coercion and assembly.

These exercise the pandas->polars coercion helper with a hand-built frame and
the polars-native containers directly; no network call to yfinance is made.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import polars as pl

from qufin.fundamentals import CANONICAL_INCOME, FinancialStatements
from qufin.fundamentals.data.yfinance import _align_periods, _coerce_statement


def test_coerce_statement_maps_labels_and_sorts() -> None:
    raw = pd.DataFrame(
        {
            pd.Timestamp("2023-12-31"): {
                "Total Revenue": 1000.0,
                "Net Income": 100.0,
                "Unmapped Row": 5.0,
            },
            pd.Timestamp("2022-12-31"): {
                "Total Revenue": 900.0,
                "Net Income": float("nan"),
                "Unmapped Row": 4.0,
            },
        }
    )
    out = _coerce_statement(raw, CANONICAL_INCOME)

    assert out.schema["period"] == pl.Date
    assert out.get_column("period").to_list() == [date(2022, 12, 31), date(2023, 12, 31)]
    assert out.get_column("revenue").to_list() == [900.0, 1000.0]
    # unmapped yfinance rows are dropped
    assert "Unmapped Row" not in out.columns
    # the NaN cell survives as a null
    assert out.get_column("net_income").to_list()[0] is None


def test_coerce_empty_statement_returns_period_only() -> None:
    out = _coerce_statement(pd.DataFrame(), CANONICAL_INCOME)
    assert out.height == 0
    assert out.columns == ["period"]


def test_align_periods_keeps_intersection() -> None:
    income = pl.DataFrame({"period": [date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)]})
    balance = pl.DataFrame({"period": [date(2022, 1, 1), date(2023, 1, 1), date(2024, 1, 1)]})
    cash_flow = pl.DataFrame({"period": [date(2022, 1, 1), date(2023, 1, 1)]})

    a, b, c = _align_periods(income, balance, cash_flow)
    common = [date(2022, 1, 1), date(2023, 1, 1)]
    assert a.get_column("period").to_list() == common
    assert b.get_column("period").to_list() == common
    assert c.get_column("period").to_list() == common


def test_financial_statements_snapshot() -> None:
    periods = [date(2022, 12, 31), date(2023, 12, 31)]
    income = pl.DataFrame(
        {"period": periods, "revenue": [900.0, 1000.0], "net_income": [90.0, 144.0]}
    )
    balance = pl.DataFrame(
        {"period": periods, "total_assets": [1800.0, 2000.0], "total_equity": [700.0, 800.0]}
    )
    cash_flow = pl.DataFrame(
        {"period": periods, "operating_cash_flow": [200.0, 220.0], "capex": [-40.0, -50.0]}
    )
    fs = FinancialStatements(income, balance, cash_flow, ticker="X", currency="USD")

    assert fs.periods() == periods

    curr = fs.snapshot(-1, price=50.0)
    assert curr.revenue == 1000.0
    assert curr.total_assets == 2000.0
    assert curr.operating_cash_flow == 220.0
    assert curr.price == 50.0
    assert curr.ticker == "X"
    assert curr.as_of == date(2023, 12, 31)
    assert curr.period == "2023-12-31"

    prev = fs.snapshot(-2)
    assert prev.revenue == 900.0
    assert math.isnan(prev.price)
