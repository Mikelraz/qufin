"""
Market-based valuation multiples computed from a :class:`FundamentalSnapshot`.

These require the snapshot's market fields (``price``/``shares_outstanding``/
``market_cap``) to be populated; the data loader fills them from quote data.
Equity multiples use market capitalisation; enterprise multiples use
:meth:`FundamentalSnapshot.ev`.  As with ratios, a missing or zero denominator
yields ``NaN``.
"""

from __future__ import annotations

import math

from ._types import FundamentalSnapshot
from ._util import safe_div


def _free_cash_flow(s: FundamentalSnapshot) -> float:
    if not math.isnan(s.free_cash_flow):
        return s.free_cash_flow
    return s.operating_cash_flow + s.capex


def enterprise_value(s: FundamentalSnapshot) -> float:
    """Enterprise value: market cap plus net debt (debt minus cash)."""
    return s.ev()


def pe_ratio(s: FundamentalSnapshot) -> float:
    """Price-to-earnings: market cap over net income."""
    return safe_div(s.market_capitalisation(), s.net_income)


def pb_ratio(s: FundamentalSnapshot) -> float:
    """Price-to-book: market cap over common equity."""
    return safe_div(s.market_capitalisation(), s.total_equity)


def ps_ratio(s: FundamentalSnapshot) -> float:
    """Price-to-sales: market cap over revenue."""
    return safe_div(s.market_capitalisation(), s.revenue)


def ev_to_ebitda(s: FundamentalSnapshot) -> float:
    """Enterprise value over EBITDA."""
    return safe_div(s.ev(), s.ebitda)


def ev_to_sales(s: FundamentalSnapshot) -> float:
    """Enterprise value over revenue."""
    return safe_div(s.ev(), s.revenue)


def earnings_yield(s: FundamentalSnapshot) -> float:
    """Net income over market cap (the inverse of the P/E ratio)."""
    return safe_div(s.net_income, s.market_capitalisation())


def fcf_yield(s: FundamentalSnapshot) -> float:
    """Free cash flow over market cap."""
    return safe_div(_free_cash_flow(s), s.market_capitalisation())


def dividend_yield(s: FundamentalSnapshot) -> float:
    """Cash dividends paid over market cap.

    yfinance reports ``dividends_paid`` as a negative cash outflow; the yield is
    reported here as a positive fraction.
    """
    return safe_div(abs(s.dividends_paid), s.market_capitalisation())


def peg_ratio(s: FundamentalSnapshot, eps_growth: float) -> float:
    """Price/earnings-to-growth ratio.

    Args:
        s: Snapshot (market fields populated).
        eps_growth: Expected annual EPS growth as a *decimal* (0.15 = 15 %).

    Returns:
        ``P/E`` divided by the growth rate in percentage points
        (``eps_growth * 100``).  ``NaN`` if P/E or growth is unavailable/zero.
    """
    return safe_div(pe_ratio(s), eps_growth * 100.0)
