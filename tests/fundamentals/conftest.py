"""Shared fixtures for the fundamentals test suite.

Snapshots are hand-built with round numbers so expected ratios, scores, and
valuations can be checked against closed-form arithmetic.
"""

from __future__ import annotations

import pytest

from qufin.fundamentals import FundamentalSnapshot


@pytest.fixture
def base_snapshot() -> FundamentalSnapshot:
    """A healthy firm with clean, internally consistent figures."""
    return FundamentalSnapshot(
        revenue=1000.0,
        cogs=600.0,
        gross_profit=400.0,
        operating_income=200.0,
        ebit=200.0,
        ebitda=250.0,
        interest_expense=20.0,
        pretax_income=180.0,
        tax_expense=36.0,
        net_income=144.0,
        sga_expense=100.0,
        depreciation=50.0,
        total_assets=2000.0,
        current_assets=800.0,
        cash=300.0,
        inventory=200.0,
        receivables=150.0,
        total_liabilities=1200.0,
        current_liabilities=400.0,
        total_debt=700.0,
        long_term_debt=500.0,
        ppe_net=900.0,
        total_equity=800.0,
        retained_earnings=600.0,
        operating_cash_flow=220.0,
        capex=-50.0,
        free_cash_flow=170.0,
        dividends_paid=-40.0,
        price=50.0,
        shares_outstanding=100.0,
        market_cap=5000.0,
        ticker="TEST",
        period="FY",
    )


@pytest.fixture
def improving_pair() -> tuple[FundamentalSnapshot, FundamentalSnapshot]:
    """(current, previous) snapshots where the current year improves on all
    nine Piotroski dimensions, so the F-Score is 9."""
    prev = FundamentalSnapshot(
        revenue=1000.0,
        cogs=700.0,
        gross_profit=300.0,
        net_income=100.0,
        operating_cash_flow=120.0,
        total_assets=2000.0,
        current_assets=600.0,
        current_liabilities=400.0,
        long_term_debt=600.0,
        shares_outstanding=100.0,
    )
    curr = FundamentalSnapshot(
        revenue=1300.0,
        cogs=800.0,
        gross_profit=500.0,
        net_income=200.0,
        operating_cash_flow=250.0,
        total_assets=2100.0,
        current_assets=900.0,
        current_liabilities=400.0,
        long_term_debt=500.0,
        shares_outstanding=100.0,
    )
    return curr, prev
