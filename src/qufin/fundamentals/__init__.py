"""Quantitative fundamental analysis tools.

Submodules
----------
data
    Statement loaders (yfinance) producing :class:`FinancialStatements` and
    :class:`FundamentalSnapshot` containers.
ratios
    Profitability, liquidity, leverage, efficiency and per-share ratios plus
    the :func:`compute_ratios` aggregator.
multiples
    Market-based valuation multiples (P/E, P/B, EV/EBITDA, FCF yield, PEG, ...).
valuation
    Intrinsic models: two-stage DCF, Gordon and multi-stage DDM, residual income.
scores
    Composite scores: Piotroski F, Altman Z / Z'', Beneish M.
growth
    CAGR, period/year-over-year growth, sustainable growth, DuPont decomposition.
screen
    Cross-sectional standardisation and weighted-composite ranking of a universe.

All analytics operate on a :class:`FundamentalSnapshot` (or a polars universe
frame) and never touch the network, so they are deterministic and testable.

Typical workflow::

    from qufin.fundamentals import (
        load_snapshot, load_financial_statements,
        compute_ratios, dupont_5factor,
        two_stage_dcf, altman_z_score, piotroski_f_score, beneish_m_score,
        rank_universe,
    )

    stmts = load_financial_statements("MSFT")
    curr = stmts.snapshot(-1, price=430.0)
    prev = stmts.snapshot(-2)

    ratios = compute_ratios(curr, prev)
    z      = altman_z_score(curr)
    f      = piotroski_f_score(curr, prev)
    dcf    = two_stage_dcf(curr.free_cash_flow, 0.10, 5, 0.03, 0.09,
                           net_debt=curr.net_debt(),
                           shares_outstanding=curr.shares_outstanding)
"""

from qufin.fundamentals._types import (
    CANONICAL_BALANCE,
    CANONICAL_CASHFLOW,
    CANONICAL_INCOME,
    DuPont,
    FinancialStatements,
    FScore,
    FundamentalSnapshot,
    MScore,
    RatioSet,
    Valuation,
    ZScore,
)
from qufin.fundamentals.data import load_financial_statements, load_snapshot
from qufin.fundamentals.growth import (
    cagr,
    dupont_3factor,
    dupont_5factor,
    period_growth,
    sustainable_growth_rate,
    yoy_growth,
)
from qufin.fundamentals.multiples import (
    dividend_yield,
    earnings_yield,
    enterprise_value,
    ev_to_ebitda,
    ev_to_sales,
    fcf_yield,
    pb_ratio,
    pe_ratio,
    peg_ratio,
    ps_ratio,
)
from qufin.fundamentals.ratios import (
    asset_turnover,
    book_value_per_share,
    cash_ratio,
    compute_ratios,
    current_ratio,
    debt_to_assets,
    debt_to_equity,
    ebitda_margin,
    eps,
    fcf_per_share,
    gross_margin,
    interest_coverage,
    inventory_turnover,
    net_debt_to_ebitda,
    net_margin,
    operating_margin,
    quick_ratio,
    receivables_turnover,
    return_on_assets,
    return_on_equity,
    return_on_invested_capital,
)
from qufin.fundamentals.scores import (
    altman_z_double_prime,
    altman_z_score,
    beneish_m_score,
    piotroski_f_score,
)
from qufin.fundamentals.screen import (
    composite_score,
    percentile_rank,
    rank_universe,
    zscore,
)
from qufin.fundamentals.valuation import (
    gordon_growth_ddm,
    multi_stage_ddm,
    residual_income_value,
    two_stage_dcf,
    wacc,
)

__all__ = [
    # types
    "FundamentalSnapshot",
    "FinancialStatements",
    "RatioSet",
    "Valuation",
    "FScore",
    "ZScore",
    "MScore",
    "DuPont",
    "CANONICAL_INCOME",
    "CANONICAL_BALANCE",
    "CANONICAL_CASHFLOW",
    # data
    "load_financial_statements",
    "load_snapshot",
    # ratios
    "gross_margin",
    "operating_margin",
    "net_margin",
    "ebitda_margin",
    "return_on_assets",
    "return_on_equity",
    "return_on_invested_capital",
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "net_debt_to_ebitda",
    "interest_coverage",
    "asset_turnover",
    "inventory_turnover",
    "receivables_turnover",
    "eps",
    "book_value_per_share",
    "fcf_per_share",
    "compute_ratios",
    # multiples
    "enterprise_value",
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "ev_to_ebitda",
    "ev_to_sales",
    "earnings_yield",
    "fcf_yield",
    "dividend_yield",
    "peg_ratio",
    # valuation
    "wacc",
    "gordon_growth_ddm",
    "multi_stage_ddm",
    "two_stage_dcf",
    "residual_income_value",
    # scores
    "piotroski_f_score",
    "altman_z_score",
    "altman_z_double_prime",
    "beneish_m_score",
    # growth
    "cagr",
    "period_growth",
    "yoy_growth",
    "sustainable_growth_rate",
    "dupont_3factor",
    "dupont_5factor",
    # screen
    "zscore",
    "percentile_rank",
    "composite_score",
    "rank_universe",
]
