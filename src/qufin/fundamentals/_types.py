"""
Shared types and result containers for the fundamentals subpackage.

A :class:`FundamentalSnapshot` is the central abstraction: a flat bundle of
canonical, point-in-time accounting figures for a single entity and reporting
period.  Every analytic in this package (ratios, multiples, valuation models,
scores) consumes a snapshot rather than a live ticker, so the analytics are
pure and unit-testable against textbook values.

Conventions
-----------
* All monetary figures are in the issuer's reporting currency and the *same*
  unit (yfinance reports absolute currency units, not thousands/millions).
* Missing line items are represented by ``float('nan')`` — never ``0.0`` — so
  that a genuinely zero balance is distinguishable from an unavailable one.
* Flow figures (income statement, cash flow) are for the period; stock figures
  (balance sheet) are end-of-period balances.
* ``shares_outstanding`` is the diluted share count where available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import date
from typing import Literal, Self

import numpy as np
import polars as pl
from numpy.typing import NDArray

Frequency = Literal["annual", "quarterly"]
ZVariant = Literal["z", "z_double_prime"]
ZZone = Literal["safe", "grey", "distress"]

NAN: float = float("nan")

# yfinance statement labels -> canonical FundamentalSnapshot field names. yfinance
# occasionally renames rows between releases; unmapped rows are simply ignored and
# the corresponding snapshot field stays NaN, so analytics degrade gracefully.
CANONICAL_INCOME: dict[str, str] = {
    "Total Revenue": "revenue",
    "Operating Revenue": "revenue",
    "Cost Of Revenue": "cogs",
    "Gross Profit": "gross_profit",
    "Operating Income": "operating_income",
    "EBIT": "ebit",
    "EBITDA": "ebitda",
    "Normalized EBITDA": "ebitda",
    "Interest Expense": "interest_expense",
    "Pretax Income": "pretax_income",
    "Tax Provision": "tax_expense",
    "Net Income": "net_income",
    "Net Income Common Stockholders": "net_income",
    "Selling General And Administration": "sga_expense",
    "Diluted Average Shares": "shares_outstanding",
}

CANONICAL_BALANCE: dict[str, str] = {
    "Total Assets": "total_assets",
    "Current Assets": "current_assets",
    "Cash And Cash Equivalents": "cash",
    "Cash Cash Equivalents And Short Term Investments": "cash",
    "Inventory": "inventory",
    "Accounts Receivable": "receivables",
    "Receivables": "receivables",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Current Liabilities": "current_liabilities",
    "Total Debt": "total_debt",
    "Long Term Debt": "long_term_debt",
    "Net PPE": "ppe_net",
    "Stockholders Equity": "total_equity",
    "Common Stock Equity": "total_equity",
    "Retained Earnings": "retained_earnings",
}

CANONICAL_CASHFLOW: dict[str, str] = {
    "Operating Cash Flow": "operating_cash_flow",
    "Cash Flow From Continuing Operating Activities": "operating_cash_flow",
    "Capital Expenditure": "capex",
    "Free Cash Flow": "free_cash_flow",
    "Depreciation And Amortization": "depreciation",
    "Depreciation Amortization Depletion": "depreciation",
    "Cash Dividends Paid": "dividends_paid",
    "Common Stock Dividend Paid": "dividends_paid",
}


@dataclass(slots=True, frozen=True)
class FundamentalSnapshot:
    """Canonical accounting figures for one entity at one reporting period.

    Every field defaults to ``NaN`` (unavailable).  Construct directly for
    tests, or via :meth:`from_mapping` from a ``{canonical_field: value}`` dict.

    Attributes:
        revenue: Total/operating revenue for the period.
        cogs: Cost of revenue (cost of goods sold).
        gross_profit: Revenue minus COGS.
        operating_income: Income from operations.
        ebit: Earnings before interest and taxes.
        ebitda: EBIT plus depreciation and amortisation.
        interest_expense: Interest expense for the period (reported as positive).
        pretax_income: Income before tax.
        tax_expense: Income tax provision.
        net_income: Net income attributable to common stockholders.
        sga_expense: Selling, general & administrative expense (for Beneish SGAI).
        depreciation: Depreciation & amortisation for the period (Beneish DEPI).
        total_assets: End-of-period total assets.
        current_assets: End-of-period current assets.
        cash: Cash and cash equivalents (and short-term investments where merged).
        inventory: End-of-period inventory.
        receivables: Accounts receivable.
        total_liabilities: Total liabilities (net of minority interest).
        current_liabilities: End-of-period current liabilities.
        total_debt: Total interest-bearing debt.
        long_term_debt: Long-term portion of debt.
        ppe_net: Net property, plant & equipment (Beneish AQI/DEPI).
        total_equity: Common stockholders' equity (book value of equity).
        retained_earnings: Accumulated retained earnings.
        operating_cash_flow: Cash from operating activities.
        capex: Capital expenditure (reported by yfinance as a negative number).
        free_cash_flow: Operating cash flow plus capex.
        dividends_paid: Cash dividends paid (reported as a negative number).
        price: Share price as of ``as_of`` (market data, not from statements).
        shares_outstanding: Diluted shares outstanding.
        market_cap: ``price * shares_outstanding`` (or vendor-reported).
        ticker: Issuer symbol.
        period: Human-readable period label (e.g. ``"FY2025"`` or ``"TTM"``).
        as_of: Calendar date the figures pertain to / were observed.
    """

    revenue: float = NAN
    cogs: float = NAN
    gross_profit: float = NAN
    operating_income: float = NAN
    ebit: float = NAN
    ebitda: float = NAN
    interest_expense: float = NAN
    pretax_income: float = NAN
    tax_expense: float = NAN
    net_income: float = NAN
    sga_expense: float = NAN
    depreciation: float = NAN

    total_assets: float = NAN
    current_assets: float = NAN
    cash: float = NAN
    inventory: float = NAN
    receivables: float = NAN
    total_liabilities: float = NAN
    current_liabilities: float = NAN
    total_debt: float = NAN
    long_term_debt: float = NAN
    ppe_net: float = NAN
    total_equity: float = NAN
    retained_earnings: float = NAN

    operating_cash_flow: float = NAN
    capex: float = NAN
    free_cash_flow: float = NAN
    dividends_paid: float = NAN

    price: float = NAN
    shares_outstanding: float = NAN
    market_cap: float = NAN

    ticker: str = ""
    period: str = ""
    as_of: date | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, float], **meta: object) -> Self:
        """Build a snapshot from a ``{field_name: value}`` mapping.

        Keys not matching a snapshot field are ignored; missing fields stay
        ``NaN``.  ``meta`` may carry ``ticker``/``period``/``as_of``.
        """
        known = {f.name for f in fields(cls)}
        numeric = {k: float(v) for k, v in values.items() if k in known}
        extra = {k: v for k, v in meta.items() if k in known}
        return cls(**numeric, **extra)  # type: ignore[arg-type]

    def market_capitalisation(self) -> float:
        """Market cap, falling back to ``price * shares_outstanding``."""
        if not math.isnan(self.market_cap):
            return self.market_cap
        return self.price * self.shares_outstanding

    def net_debt(self) -> float:
        """Total debt net of cash and equivalents."""
        return self.total_debt - self.cash

    def ev(self) -> float:
        """Enterprise value: market cap plus net debt."""
        return self.market_capitalisation() + self.net_debt()


@dataclass(slots=True)
class FinancialStatements:
    """Period-indexed financial statements for a single issuer.

    Each frame has a ``period`` column of dtype :class:`polars.Date` (one row
    per fiscal period, ascending) plus one column per canonical line item.  The
    three frames are aligned on a common, sorted set of periods so that row *i*
    refers to the same period across all of them.

    Attributes:
        income: Income-statement frame.
        balance: Balance-sheet frame.
        cash_flow: Cash-flow-statement frame.
        ticker: Issuer symbol.
        currency: Reporting currency (ISO 4217), if known.
        frequency: ``"annual"`` or ``"quarterly"``.
    """

    income: pl.DataFrame
    balance: pl.DataFrame
    cash_flow: pl.DataFrame
    ticker: str = ""
    currency: str = ""
    frequency: Frequency = "annual"

    def periods(self) -> list[date]:
        """Ascending list of fiscal-period dates common to all statements."""
        if "period" not in self.income.columns:
            return []
        return self.income.get_column("period").to_list()

    def snapshot(
        self,
        i: int = -1,
        *,
        price: float = NAN,
        shares_outstanding: float = NAN,
        market_cap: float = NAN,
    ) -> FundamentalSnapshot:
        """Assemble a :class:`FundamentalSnapshot` from period row ``i``.

        Args:
            i: Period row index (default ``-1`` = most recent).
            price: Optional share price to attach (market data).
            shares_outstanding: Optional override for diluted share count.
            market_cap: Optional market capitalisation.

        Returns:
            A snapshot merging the matching row of every statement frame.
        """
        values: dict[str, float] = {}
        for frame in (self.income, self.balance, self.cash_flow):
            if frame.height == 0:
                continue
            row = frame.row(i, named=True)
            for k, v in row.items():
                if k == "period" or v is None:
                    continue
                values[k] = float(v)

        if not math.isnan(shares_outstanding):
            values["shares_outstanding"] = shares_outstanding

        periods = self.periods()
        as_of = periods[i] if periods else None
        label = as_of.isoformat() if as_of is not None else ""
        return FundamentalSnapshot.from_mapping(
            values,
            ticker=self.ticker,
            period=label,
            as_of=as_of,
            price=price,
            market_cap=market_cap,
        )


@dataclass(slots=True)
class RatioSet:
    """Aggregated financial ratios for a single snapshot.

    All values are dimensionless decimals or turnover counts; any ratio whose
    denominator was zero or unavailable is ``NaN``.
    """

    gross_margin: float
    operating_margin: float
    net_margin: float
    ebitda_margin: float
    return_on_assets: float
    return_on_equity: float
    return_on_invested_capital: float
    current_ratio: float
    quick_ratio: float
    cash_ratio: float
    debt_to_equity: float
    debt_to_assets: float
    net_debt_to_ebitda: float
    interest_coverage: float
    asset_turnover: float
    inventory_turnover: float
    receivables_turnover: float
    eps: float
    book_value_per_share: float
    fcf_per_share: float

    def as_dict(self) -> dict[str, float]:
        """Return ``{ratio_name: value}`` for quick inspection."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(slots=True)
class Valuation:
    """Result of an intrinsic discounted-cash-flow valuation.

    Attributes:
        enterprise_value: PV of explicit free cash flows plus terminal value.
        equity_value: Enterprise value minus net debt.
        value_per_share: Equity value divided by shares outstanding (``NaN`` if
            no share count was supplied).
        terminal_value: Undiscounted Gordon-growth terminal value at the end of
            the explicit forecast horizon.
        pv_terminal: Present value of ``terminal_value``.
        pv_cash_flows: Present value of each explicit-horizon cash flow.
        discount_rate: Discount rate used (WACC for FCFF, cost of equity for FCFE).
    """

    enterprise_value: float
    equity_value: float
    value_per_share: float
    terminal_value: float
    pv_terminal: float
    pv_cash_flows: NDArray[np.float64]
    discount_rate: float


@dataclass(slots=True, frozen=True)
class FScore:
    """Piotroski F-Score: nine binary signals of fundamental strength.

    A high score (8–9) flags improving, financially sound firms; a low score
    (0–2) flags deteriorating ones.  Each attribute is the boolean outcome of
    the corresponding signal.
    """

    positive_net_income: bool
    positive_operating_cash_flow: bool
    rising_roa: bool
    cash_flow_exceeds_net_income: bool
    falling_leverage: bool
    rising_current_ratio: bool
    no_new_shares: bool
    rising_gross_margin: bool
    rising_asset_turnover: bool

    @property
    def score(self) -> int:
        """Sum of the nine signals (0–9)."""
        return sum(getattr(self, f.name) for f in fields(self))

    def as_dict(self) -> dict[str, bool]:
        """Return ``{signal_name: passed}`` for the nine components."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(slots=True, frozen=True)
class ZScore:
    """Altman bankruptcy-risk Z-Score with zone classification.

    Attributes:
        score: The computed Z (or Z'') value.
        zone: ``"safe"``, ``"grey"`` or ``"distress"`` per the variant's cut-offs.
        variant: Which model produced the score.
    """

    score: float
    zone: ZZone
    variant: ZVariant


@dataclass(slots=True, frozen=True)
class MScore:
    """Beneish M-Score: likelihood that earnings have been manipulated.

    The eight indices feed the canonical 8-variable model.  ``manipulator`` is
    ``True`` when ``score > -1.78`` (the standard threshold).
    """

    dsri: float
    gmi: float
    aqi: float
    sgi: float
    depi: float
    sgai: float
    lvgi: float
    tata: float
    score: float
    manipulator: bool


@dataclass(slots=True, frozen=True)
class DuPont:
    """DuPont decomposition of return on equity.

    Attributes:
        roe: Reconstructed ROE (product of the factors).
        factors: Ordered ``{factor_name: value}`` mapping (3 or 5 entries).
        n_factors: 3 or 5.
    """

    roe: float
    factors: dict[str, float]
    n_factors: int = 3
