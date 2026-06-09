"""
Financial ratios computed from a :class:`FundamentalSnapshot`.

Functions are grouped as profitability, liquidity, leverage/coverage,
efficiency, and per-share.  Every function returns a ``float`` and yields
``NaN`` (never raises) when a required line item is missing or a denominator is
zero, so a partial snapshot still produces a partial :class:`RatioSet`.

Ratios that conventionally use an *average* balance (turnover, ROA, ROE, ROIC)
accept an optional ``prev`` snapshot; when supplied, the relevant balance-sheet
figure is averaged across the two periods.  With no ``prev`` the ending balance
is used.
"""

from __future__ import annotations

import math

from ._types import FundamentalSnapshot, RatioSet
from ._util import safe_div


def _gross_profit(s: FundamentalSnapshot) -> float:
    if not math.isnan(s.gross_profit):
        return s.gross_profit
    return s.revenue - s.cogs


def _free_cash_flow(s: FundamentalSnapshot) -> float:
    if not math.isnan(s.free_cash_flow):
        return s.free_cash_flow
    return s.operating_cash_flow + s.capex


def _avg_balance(curr: float, prev: FundamentalSnapshot | None, attr: str) -> float:
    """Average ``curr`` with the matching ``prev`` balance when available."""
    if prev is None:
        return curr
    prior = getattr(prev, attr)
    if math.isnan(prior):
        return curr
    return 0.5 * (curr + prior)


# --- Profitability ---------------------------------------------------------


def gross_margin(s: FundamentalSnapshot) -> float:
    """Gross profit as a fraction of revenue."""
    return safe_div(_gross_profit(s), s.revenue)


def operating_margin(s: FundamentalSnapshot) -> float:
    """Operating income as a fraction of revenue."""
    return safe_div(s.operating_income, s.revenue)


def net_margin(s: FundamentalSnapshot) -> float:
    """Net income as a fraction of revenue."""
    return safe_div(s.net_income, s.revenue)


def ebitda_margin(s: FundamentalSnapshot) -> float:
    """EBITDA as a fraction of revenue."""
    return safe_div(s.ebitda, s.revenue)


def return_on_assets(s: FundamentalSnapshot, prev: FundamentalSnapshot | None = None) -> float:
    """Net income divided by (average) total assets."""
    return safe_div(s.net_income, _avg_balance(s.total_assets, prev, "total_assets"))


def return_on_equity(s: FundamentalSnapshot, prev: FundamentalSnapshot | None = None) -> float:
    """Net income divided by (average) common equity."""
    return safe_div(s.net_income, _avg_balance(s.total_equity, prev, "total_equity"))


def return_on_invested_capital(
    s: FundamentalSnapshot, prev: FundamentalSnapshot | None = None
) -> float:
    """NOPAT divided by invested capital (debt + equity).

    NOPAT = EBIT * (1 - effective tax rate), where the effective rate is
    ``tax_expense / pretax_income`` (falling back to 0 when unavailable).
    """
    tax_rate = safe_div(s.tax_expense, s.pretax_income)
    if math.isnan(tax_rate):
        tax_rate = 0.0
    nopat = s.ebit * (1.0 - tax_rate)
    invested = _avg_balance(s.total_equity, prev, "total_equity") + _avg_balance(
        s.total_debt, prev, "total_debt"
    )
    return safe_div(nopat, invested)


# --- Liquidity -------------------------------------------------------------


def current_ratio(s: FundamentalSnapshot) -> float:
    """Current assets over current liabilities."""
    return safe_div(s.current_assets, s.current_liabilities)


def quick_ratio(s: FundamentalSnapshot) -> float:
    """(Current assets minus inventory) over current liabilities (acid test)."""
    return safe_div(s.current_assets - s.inventory, s.current_liabilities)


def cash_ratio(s: FundamentalSnapshot) -> float:
    """Cash and equivalents over current liabilities."""
    return safe_div(s.cash, s.current_liabilities)


# --- Leverage & coverage ---------------------------------------------------


def debt_to_equity(s: FundamentalSnapshot) -> float:
    """Total debt over common equity."""
    return safe_div(s.total_debt, s.total_equity)


def debt_to_assets(s: FundamentalSnapshot) -> float:
    """Total debt over total assets."""
    return safe_div(s.total_debt, s.total_assets)


def net_debt_to_ebitda(s: FundamentalSnapshot) -> float:
    """Net debt (debt minus cash) over EBITDA."""
    return safe_div(s.net_debt(), s.ebitda)


def interest_coverage(s: FundamentalSnapshot) -> float:
    """EBIT over interest expense (times interest earned)."""
    return safe_div(s.ebit, s.interest_expense)


# --- Efficiency / activity -------------------------------------------------


def asset_turnover(s: FundamentalSnapshot, prev: FundamentalSnapshot | None = None) -> float:
    """Revenue divided by (average) total assets."""
    return safe_div(s.revenue, _avg_balance(s.total_assets, prev, "total_assets"))


def inventory_turnover(s: FundamentalSnapshot, prev: FundamentalSnapshot | None = None) -> float:
    """Cost of goods sold divided by (average) inventory."""
    return safe_div(s.cogs, _avg_balance(s.inventory, prev, "inventory"))


def receivables_turnover(s: FundamentalSnapshot, prev: FundamentalSnapshot | None = None) -> float:
    """Revenue divided by (average) accounts receivable."""
    return safe_div(s.revenue, _avg_balance(s.receivables, prev, "receivables"))


# --- Per share -------------------------------------------------------------


def eps(s: FundamentalSnapshot) -> float:
    """Earnings per (diluted) share."""
    return safe_div(s.net_income, s.shares_outstanding)


def book_value_per_share(s: FundamentalSnapshot) -> float:
    """Common equity per (diluted) share."""
    return safe_div(s.total_equity, s.shares_outstanding)


def fcf_per_share(s: FundamentalSnapshot) -> float:
    """Free cash flow per (diluted) share."""
    return safe_div(_free_cash_flow(s), s.shares_outstanding)


# --- Aggregator ------------------------------------------------------------


def compute_ratios(s: FundamentalSnapshot, prev: FundamentalSnapshot | None = None) -> RatioSet:
    """Compute the full :class:`RatioSet` for ``s``.

    Args:
        s: The period snapshot to analyse.
        prev: Optional prior-period snapshot; when given, balance-based ratios
            (ROA, ROE, ROIC, turnover) use the two-period average balance.

    Returns:
        A :class:`RatioSet` with every ratio (missing inputs -> ``NaN``).
    """
    return RatioSet(
        gross_margin=gross_margin(s),
        operating_margin=operating_margin(s),
        net_margin=net_margin(s),
        ebitda_margin=ebitda_margin(s),
        return_on_assets=return_on_assets(s, prev),
        return_on_equity=return_on_equity(s, prev),
        return_on_invested_capital=return_on_invested_capital(s, prev),
        current_ratio=current_ratio(s),
        quick_ratio=quick_ratio(s),
        cash_ratio=cash_ratio(s),
        debt_to_equity=debt_to_equity(s),
        debt_to_assets=debt_to_assets(s),
        net_debt_to_ebitda=net_debt_to_ebitda(s),
        interest_coverage=interest_coverage(s),
        asset_turnover=asset_turnover(s, prev),
        inventory_turnover=inventory_turnover(s, prev),
        receivables_turnover=receivables_turnover(s, prev),
        eps=eps(s),
        book_value_per_share=book_value_per_share(s),
        fcf_per_share=fcf_per_share(s),
    )
