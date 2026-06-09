"""Data loaders for fundamental statements (currently yfinance)."""

from qufin.fundamentals.data.yfinance import (
    load_financial_statements,
    load_snapshot,
)

__all__ = [
    "load_financial_statements",
    "load_snapshot",
]
