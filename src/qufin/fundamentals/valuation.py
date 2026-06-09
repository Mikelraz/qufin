"""
Intrinsic valuation models: discounted cash flow, dividend discount, and
residual income.

Unlike the ratio/multiple helpers these take explicit forward-looking
assumptions (growth rates, discount rates) rather than reading them from a
snapshot — forecasting inputs is out of scope for this package.  All models
assume end-of-period cash flows and use a constant discount rate.

A perpetual-growth terminal value is undefined when the discount rate does not
exceed the terminal growth rate, so those functions raise :class:`ValueError`
rather than returning a misleading number.
"""

from __future__ import annotations

import math

import numpy as np

from ._types import NAN, Valuation
from ._util import safe_div


def wacc(
    equity_value: float,
    debt_value: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
) -> float:
    """Weighted average cost of capital.

    Args:
        equity_value: Market value of equity.
        debt_value: Market value of debt.
        cost_of_equity: Required return on equity (decimal).
        cost_of_debt: Pre-tax cost of debt (decimal).
        tax_rate: Marginal tax rate (decimal); debt is tax-shielded.

    Returns:
        Blended cost of capital, or ``NaN`` if total capital is zero.
    """
    total = equity_value + debt_value
    we = safe_div(equity_value, total)
    wd = safe_div(debt_value, total)
    if math.isnan(we) or math.isnan(wd):
        return math.nan
    return we * cost_of_equity + wd * cost_of_debt * (1.0 - tax_rate)


def gordon_growth_ddm(d1: float, discount_rate: float, growth: float) -> float:
    """Gordon constant-growth dividend discount model.

    Args:
        d1: Expected dividend one period from now (``D_0 * (1 + g)``).
        discount_rate: Required return on equity (decimal).
        growth: Perpetual dividend growth rate (decimal).

    Returns:
        Present value of the growing perpetuity, ``d1 / (r - g)``.

    Raises:
        ValueError: If ``discount_rate <= growth``.
    """
    if discount_rate <= growth:
        raise ValueError("discount_rate must exceed growth for a finite Gordon value")
    return d1 / (discount_rate - growth)


def multi_stage_ddm(dividends: list[float], terminal_growth: float, discount_rate: float) -> float:
    """Multi-stage DDM: explicit dividend forecast plus a Gordon terminal value.

    Args:
        dividends: Forecast dividends for years ``1..N`` (``dividends[t-1]`` is
            the year-``t`` dividend).
        terminal_growth: Perpetual growth applied after year ``N``.
        discount_rate: Required return on equity (decimal).

    Returns:
        Present value of the explicit dividends plus the discounted terminal
        value computed from the final forecast dividend.

    Raises:
        ValueError: If ``dividends`` is empty or ``discount_rate <= terminal_growth``.
    """
    if not dividends:
        raise ValueError("dividends must contain at least one forecast period")
    if discount_rate <= terminal_growth:
        raise ValueError("discount_rate must exceed terminal_growth")

    n = len(dividends)
    t = np.arange(1, n + 1, dtype=np.float64)
    cash = np.asarray(dividends, dtype=np.float64)
    disc = (1.0 + discount_rate) ** t
    pv_explicit = float(np.sum(cash / disc))

    terminal_value = cash[-1] * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1.0 + discount_rate) ** n
    return pv_explicit + pv_terminal


def two_stage_dcf(
    fcf0: float,
    high_growth: float,
    high_years: int,
    terminal_growth: float,
    discount_rate: float,
    *,
    net_debt: float = 0.0,
    shares_outstanding: float = NAN,
) -> Valuation:
    """Two-stage free-cash-flow-to-firm DCF valuation.

    Free cash flow grows at ``high_growth`` for ``high_years`` years, then at
    ``terminal_growth`` in perpetuity.  Enterprise value is the present value of
    the explicit cash flows plus the discounted terminal value; equity value
    subtracts ``net_debt``.

    Args:
        fcf0: Most recent (year-0) free cash flow to the firm.
        high_growth: Growth rate during the explicit horizon (decimal).
        high_years: Length of the explicit horizon in years (>= 1).
        terminal_growth: Perpetual growth after the horizon (decimal).
        discount_rate: WACC (decimal).
        net_debt: Debt minus cash, subtracted to reach equity value.
        shares_outstanding: Diluted shares; if NaN, ``value_per_share`` is NaN.

    Returns:
        A :class:`Valuation` with enterprise/equity/per-share values and the
        present-value decomposition.

    Raises:
        ValueError: If ``high_years < 1`` or ``discount_rate <= terminal_growth``.
    """
    if high_years < 1:
        raise ValueError("high_years must be >= 1")
    if discount_rate <= terminal_growth:
        raise ValueError("discount_rate must exceed terminal_growth")

    t = np.arange(1, high_years + 1, dtype=np.float64)
    cash = fcf0 * (1.0 + high_growth) ** t
    disc = (1.0 + discount_rate) ** t
    pv_cash_flows = cash / disc

    terminal_value = cash[-1] * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1.0 + discount_rate) ** high_years

    enterprise_value = float(np.sum(pv_cash_flows)) + pv_terminal
    equity_value = enterprise_value - net_debt
    value_per_share = safe_div(equity_value, shares_outstanding)

    return Valuation(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        terminal_value=terminal_value,
        pv_terminal=pv_terminal,
        pv_cash_flows=pv_cash_flows,
        discount_rate=discount_rate,
    )


def residual_income_value(
    book_value0: float,
    roe: float,
    cost_of_equity: float,
    growth: float,
    years: int,
    terminal_growth: float,
) -> float:
    """Residual-income (Edwards-Bell-Ohlson) equity value.

    Residual income in year ``t`` is ``(ROE - r) * B_{t-1}`` where book value
    compounds at ``growth``.  Value equals starting book value plus the present
    value of explicit residual income plus a discounted terminal residual-income
    perpetuity.

    With ``growth = terminal_growth = 0`` and constant ``roe`` this reduces to
    the textbook ``B_0 * roe / r`` (a firm earning a constant ROE on book
    equity in perpetuity).

    Args:
        book_value0: Current book value of equity.
        roe: Forecast return on equity (decimal).
        cost_of_equity: Required return on equity (decimal).
        growth: Annual book-value growth over the explicit horizon (decimal).
        years: Explicit horizon length in years (>= 1).
        terminal_growth: Perpetual residual-income growth after the horizon.

    Returns:
        Estimated intrinsic equity value.

    Raises:
        ValueError: If ``years < 1`` or ``cost_of_equity <= terminal_growth``.
    """
    if years < 1:
        raise ValueError("years must be >= 1")
    if cost_of_equity <= terminal_growth:
        raise ValueError("cost_of_equity must exceed terminal_growth")

    spread = roe - cost_of_equity
    t = np.arange(1, years + 1, dtype=np.float64)
    book_prev = book_value0 * (1.0 + growth) ** (t - 1.0)
    residual_income = spread * book_prev
    disc = (1.0 + cost_of_equity) ** t
    pv_explicit = float(np.sum(residual_income / disc))

    terminal_ri = residual_income[-1] * (1.0 + terminal_growth)
    terminal_value = terminal_ri / (cost_of_equity - terminal_growth)
    pv_terminal = terminal_value / (1.0 + cost_of_equity) ** years

    return book_value0 + pv_explicit + pv_terminal
