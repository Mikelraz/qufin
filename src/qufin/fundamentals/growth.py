"""
Growth metrics and DuPont return-on-equity decomposition.

Growth helpers operate either on two snapshots (year-over-year) or on a raw
series of period values (e.g. a revenue history).  The DuPont decompositions
break ROE into its multiplicative drivers so that two firms with the same ROE
can be distinguished by *how* they earn it (margins vs turnover vs leverage).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ._types import DuPont, FundamentalSnapshot
from ._util import safe_div


def cagr(begin: float, end: float, periods: float) -> float:
    """Compound annual growth rate between ``begin`` and ``end``.

    Args:
        begin: Starting value (must be positive).
        end: Ending value (must be positive).
        periods: Number of compounding periods between the two values.

    Returns:
        ``(end / begin) ** (1 / periods) - 1``, or ``NaN`` when inputs are
        non-positive or ``periods <= 0`` (a real CAGR is undefined there).
    """
    if begin <= 0.0 or end <= 0.0 or periods <= 0.0 or math.isnan(begin) or math.isnan(end):
        return math.nan
    return (end / begin) ** (1.0 / periods) - 1.0


def period_growth(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Period-over-period growth rates of a value series.

    Args:
        values: 1-D array of period values (ascending in time), length >= 2.

    Returns:
        Array of length ``len(values) - 1`` where element *t* is
        ``values[t+1] / values[t] - 1``.
    """
    if values.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    return values[1:] / values[:-1] - 1.0


def yoy_growth(curr: FundamentalSnapshot, prev: FundamentalSnapshot, field: str) -> float:
    """Growth in a single snapshot field from ``prev`` to ``curr``.

    Divides by ``|prev_value|`` so the sign of the growth reflects the direction
    of change even when the base is negative (e.g. a swing in earnings).

    Args:
        curr: Current-period snapshot.
        prev: Prior-period snapshot.
        field: Name of the :class:`FundamentalSnapshot` field to compare.

    Returns:
        ``(curr - prev) / |prev|``, or ``NaN`` if either value is missing/zero.
    """
    cur_v = float(getattr(curr, field))
    prev_v = float(getattr(prev, field))
    return safe_div(cur_v - prev_v, abs(prev_v))


def sustainable_growth_rate(roe: float, payout_ratio: float) -> float:
    """Sustainable growth rate: ``ROE * (1 - payout_ratio)``.

    The fastest a firm can grow equity without issuing shares or raising
    leverage, funded entirely by retained earnings.

    Args:
        roe: Return on equity (decimal).
        payout_ratio: Fraction of earnings paid as dividends (decimal).

    Returns:
        Retention-funded growth rate.
    """
    return roe * (1.0 - payout_ratio)


def dupont_3factor(s: FundamentalSnapshot) -> DuPont:
    """Three-factor DuPont decomposition of ROE.

    ``ROE = net margin × asset turnover × equity multiplier``.
    """
    net_margin = safe_div(s.net_income, s.revenue)
    asset_turn = safe_div(s.revenue, s.total_assets)
    equity_mult = safe_div(s.total_assets, s.total_equity)
    factors = {
        "net_margin": net_margin,
        "asset_turnover": asset_turn,
        "equity_multiplier": equity_mult,
    }
    return DuPont(roe=net_margin * asset_turn * equity_mult, factors=factors, n_factors=3)


def dupont_5factor(s: FundamentalSnapshot) -> DuPont:
    """Five-factor DuPont decomposition of ROE.

    ``ROE = tax burden × interest burden × operating margin × asset turnover ×
    equity multiplier``, isolating the tax and interest drags from operating
    performance.
    """
    tax_burden = safe_div(s.net_income, s.pretax_income)
    interest_burden = safe_div(s.pretax_income, s.ebit)
    operating_margin = safe_div(s.ebit, s.revenue)
    asset_turn = safe_div(s.revenue, s.total_assets)
    equity_mult = safe_div(s.total_assets, s.total_equity)
    factors = {
        "tax_burden": tax_burden,
        "interest_burden": interest_burden,
        "operating_margin": operating_margin,
        "asset_turnover": asset_turn,
        "equity_multiplier": equity_mult,
    }
    roe = tax_burden * interest_burden * operating_margin * asset_turn * equity_mult
    return DuPont(roe=roe, factors=factors, n_factors=5)
