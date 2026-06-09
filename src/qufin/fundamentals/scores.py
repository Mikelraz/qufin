"""
Composite fundamental scores: Piotroski F-Score (quality/strength), Altman
Z-Score (bankruptcy risk), and Beneish M-Score (earnings-manipulation risk).

The F-Score and M-Score are *change* models and require both the current and
prior-period snapshots.  The Z-Score is a point-in-time model.  Missing inputs
cause individual signals to fail (F-Score) or push the score to ``NaN`` (Z/M),
in which case the binary flag defaults to the non-alarming outcome.
"""

from __future__ import annotations

import math

from ._types import FScore, FundamentalSnapshot, MScore, ZScore, ZZone
from ._util import safe_div
from .ratios import (
    asset_turnover,
    current_ratio,
    gross_margin,
    return_on_assets,
)


def piotroski_f_score(curr: FundamentalSnapshot, prev: FundamentalSnapshot) -> FScore:
    """Compute the 9-point Piotroski F-Score.

    Each signal scores 1 when the firm improved on a dimension of
    profitability, leverage/liquidity, or operating efficiency relative to the
    prior period.  Missing data fails a signal (conservative).

    Args:
        curr: Current-period snapshot.
        prev: Prior-period snapshot.

    Returns:
        An :class:`FScore` whose ``score`` property sums the nine signals.
    """
    roa_curr = return_on_assets(curr)
    roa_prev = return_on_assets(prev)
    lev_curr = safe_div(curr.long_term_debt, curr.total_assets)
    lev_prev = safe_div(prev.long_term_debt, prev.total_assets)

    return FScore(
        positive_net_income=roa_curr > 0.0,
        positive_operating_cash_flow=curr.operating_cash_flow > 0.0,
        rising_roa=roa_curr > roa_prev,
        cash_flow_exceeds_net_income=curr.operating_cash_flow > curr.net_income,
        falling_leverage=lev_curr < lev_prev,
        rising_current_ratio=current_ratio(curr) > current_ratio(prev),
        no_new_shares=curr.shares_outstanding <= prev.shares_outstanding,
        rising_gross_margin=gross_margin(curr) > gross_margin(prev),
        rising_asset_turnover=asset_turnover(curr) > asset_turnover(prev),
    )


def _z_zone(score: float, safe_cut: float, distress_cut: float) -> ZZone:
    if math.isnan(score):
        return "distress"
    if score > safe_cut:
        return "safe"
    if score < distress_cut:
        return "distress"
    return "grey"


def altman_z_score(s: FundamentalSnapshot) -> ZScore:
    """Original 5-factor Altman Z-Score for public manufacturing firms.

    ``Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5`` where X1 is working
    capital / assets, X2 retained earnings / assets, X3 EBIT / assets, X4
    market value of equity / total liabilities, and X5 sales / assets.

    Zones: ``Z > 2.99`` safe, ``Z < 1.81`` distress, otherwise grey.
    """
    working_capital = s.current_assets - s.current_liabilities
    x1 = safe_div(working_capital, s.total_assets)
    x2 = safe_div(s.retained_earnings, s.total_assets)
    x3 = safe_div(s.ebit, s.total_assets)
    x4 = safe_div(s.market_capitalisation(), s.total_liabilities)
    x5 = safe_div(s.revenue, s.total_assets)
    score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    return ZScore(score=score, zone=_z_zone(score, 2.99, 1.81), variant="z")


def altman_z_double_prime(s: FundamentalSnapshot) -> ZScore:
    """4-factor Altman Z''-Score for non-manufacturers / emerging markets.

    ``Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4`` using *book* value of equity
    in X4 (book equity / total liabilities) rather than market value, so it does
    not depend on a quote.

    Zones: ``Z'' > 2.6`` safe, ``Z'' < 1.1`` distress, otherwise grey.
    """
    working_capital = s.current_assets - s.current_liabilities
    x1 = safe_div(working_capital, s.total_assets)
    x2 = safe_div(s.retained_earnings, s.total_assets)
    x3 = safe_div(s.ebit, s.total_assets)
    x4 = safe_div(s.total_equity, s.total_liabilities)
    score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    return ZScore(score=score, zone=_z_zone(score, 2.6, 1.1), variant="z_double_prime")


def _gross_margin_raw(s: FundamentalSnapshot) -> float:
    return safe_div(s.revenue - s.cogs, s.revenue)


def _asset_quality(s: FundamentalSnapshot) -> float:
    """Fraction of assets that are neither current nor net PP&E."""
    return 1.0 - safe_div(s.current_assets + s.ppe_net, s.total_assets)


def _depreciation_rate(s: FundamentalSnapshot) -> float:
    return safe_div(s.depreciation, s.depreciation + s.ppe_net)


def _leverage(s: FundamentalSnapshot) -> float:
    return safe_div(s.current_liabilities + s.long_term_debt, s.total_assets)


def beneish_m_score(curr: FundamentalSnapshot, prev: FundamentalSnapshot) -> MScore:
    """Compute the Beneish 8-variable M-Score of earnings-manipulation risk.

    ``M = -4.84 + 0.92·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI + 0.115·DEPI
    - 0.172·SGAI + 4.679·TATA - 0.327·LVGI``.  A score above ``-1.78`` flags a
    likely manipulator.

    The asset-quality index uses ``1 - (current_assets + net PP&E) / assets`` as
    the non-current-asset proxy, matching the canonical Beneish definition.

    Args:
        curr: Current-period snapshot.
        prev: Prior-period snapshot.

    Returns:
        An :class:`MScore` carrying the eight indices, the score, and the flag.
    """
    dsri = safe_div(
        safe_div(curr.receivables, curr.revenue),
        safe_div(prev.receivables, prev.revenue),
    )
    gmi = safe_div(_gross_margin_raw(prev), _gross_margin_raw(curr))
    aqi = safe_div(_asset_quality(curr), _asset_quality(prev))
    sgi = safe_div(curr.revenue, prev.revenue)
    depi = safe_div(_depreciation_rate(prev), _depreciation_rate(curr))
    sgai = safe_div(
        safe_div(curr.sga_expense, curr.revenue),
        safe_div(prev.sga_expense, prev.revenue),
    )
    lvgi = safe_div(_leverage(curr), _leverage(prev))
    tata = safe_div(curr.net_income - curr.operating_cash_flow, curr.total_assets)

    score = (
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )
    return MScore(
        dsri=dsri,
        gmi=gmi,
        aqi=aqi,
        sgi=sgi,
        depi=depi,
        sgai=sgai,
        lvgi=lvgi,
        tata=tata,
        score=score,
        manipulator=score > -1.78,
    )
