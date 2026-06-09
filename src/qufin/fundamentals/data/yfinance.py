"""
yfinance financial-statement loader.

yfinance returns each statement as a pandas DataFrame indexed by line-item
label with one column per reporting period (most recent first).  We transpose
to a period-indexed frame, convert to polars immediately, and keep only the
columns that map to a canonical :class:`FundamentalSnapshot` field.

Limitations
-----------
* Statement coverage and exact row labels vary by issuer and yfinance release;
  unmapped rows are dropped and the corresponding fields stay ``NaN``.
* The three statements are aligned on their common set of periods so that a
  given row index refers to the same fiscal period across all of them.
* Market fields (price, shares, market cap) come from ``fast_info`` and are
  only attached by :func:`load_snapshot`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import polars as pl

from .._types import (
    CANONICAL_BALANCE,
    CANONICAL_CASHFLOW,
    CANONICAL_INCOME,
    NAN,
    FinancialStatements,
    Frequency,
    FundamentalSnapshot,
)

if TYPE_CHECKING:
    import pandas as pd


_EMPTY = pl.DataFrame(schema={"period": pl.Date})


def _coerce_statement(statement: pd.DataFrame | None, label_map: dict[str, str]) -> pl.DataFrame:
    """Transpose a yfinance statement to a canonical period-indexed polars frame.

    Args:
        statement: yfinance pandas frame (index = line items, columns = periods).
        label_map: yfinance label -> canonical field name.

    Returns:
        A polars frame with an ascending ``period`` (Date) column plus one
        Float64 column per mapped, present line item.
    """
    if statement is None or statement.empty:
        return _EMPTY.clone()

    deduped = statement[~statement.index.duplicated(keep="first")]
    transposed = deduped.transpose()
    transposed.index.name = "period"
    transposed = transposed.reset_index()

    wide = pl.from_pandas(transposed)

    selected: list[pl.Expr] = [pl.col("period").cast(pl.Date).alias("period")]
    seen: set[str] = set()
    for label, field in label_map.items():
        if label in wide.columns and field not in seen:
            selected.append(pl.col(label).cast(pl.Float64, strict=False).alias(field))
            seen.add(field)

    return wide.select(selected).sort("period")


def _align_periods(*frames: pl.DataFrame) -> tuple[pl.DataFrame, ...]:
    """Restrict every frame to the periods common to all of them, sorted."""
    period_sets = [set(f.get_column("period").to_list()) for f in frames if f.height > 0]
    if not period_sets:
        return frames
    common = set.intersection(*period_sets)
    return tuple(f.filter(pl.col("period").is_in(list(common))).sort("period") for f in frames)


def load_financial_statements(
    ticker: str, *, frequency: Frequency = "annual"
) -> FinancialStatements:
    """Fetch income, balance-sheet, and cash-flow statements via yfinance.

    Args:
        ticker: Issuer symbol (e.g. ``"MSFT"``).
        frequency: ``"annual"`` or ``"quarterly"``.

    Returns:
        A :class:`FinancialStatements` with the three statements aligned on
        their common periods.

    Raises:
        ImportError: If yfinance is not installed.
        ValueError: If ``frequency`` is not recognised.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance is required: `uv add yfinance`") from e

    tk = yf.Ticker(ticker)
    match frequency:
        case "annual":
            inc, bal, cf = tk.income_stmt, tk.balance_sheet, tk.cashflow
        case "quarterly":
            inc, bal, cf = (
                tk.quarterly_income_stmt,
                tk.quarterly_balance_sheet,
                tk.quarterly_cashflow,
            )
        case _:
            raise ValueError(f"frequency must be 'annual' or 'quarterly', got {frequency!r}")

    income, balance, cash_flow = _align_periods(
        _coerce_statement(inc, CANONICAL_INCOME),
        _coerce_statement(bal, CANONICAL_BALANCE),
        _coerce_statement(cf, CANONICAL_CASHFLOW),
    )

    currency = ""
    try:
        currency = str(tk.fast_info["currency"])
    except (KeyError, TypeError, ValueError):
        pass

    return FinancialStatements(
        income=income,
        balance=balance,
        cash_flow=cash_flow,
        ticker=ticker,
        currency=currency,
        frequency=frequency,
    )


def _fast_value(fast_info: Any, key: str) -> float:
    try:
        value = fast_info[key]
    except (KeyError, TypeError, ValueError):
        return NAN
    if value is None:
        return NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def load_snapshot(ticker: str, *, frequency: Frequency = "annual") -> FundamentalSnapshot:
    """Build a :class:`FundamentalSnapshot` for the latest period of ``ticker``.

    Loads the statements, takes the most recent period, and enriches it with
    price, shares outstanding, and market cap from ``fast_info``.

    Args:
        ticker: Issuer symbol.
        frequency: ``"annual"`` or ``"quarterly"``.

    Returns:
        The latest-period snapshot with market fields attached.

    Raises:
        ImportError: If yfinance is not installed.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance is required: `uv add yfinance`") from e

    statements = load_financial_statements(ticker, frequency=frequency)
    fast_info = yf.Ticker(ticker).fast_info
    price = _fast_value(fast_info, "last_price")
    shares = _fast_value(fast_info, "shares")
    market_cap = _fast_value(fast_info, "market_cap")

    return statements.snapshot(
        -1,
        price=price,
        shares_outstanding=shares if not math.isnan(shares) else NAN,
        market_cap=market_cap,
    )
