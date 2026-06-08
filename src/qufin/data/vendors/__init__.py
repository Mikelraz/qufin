"""Vendor adapters: OHLC, actions, and (for some vendors) option chains."""

from __future__ import annotations

from ._ib_common import (
    IBKRError,
    IBKRErrorCategory,
    IBKRErrorListener,
    MarketDataType,
    classify_error,
)
from .base import ActionsSource, OHLCSource
from .csv import CsvOHLC
from .yfinance import Interval as YFInterval
from .yfinance import YFinanceOHLC, load_ohlc, load_ohlc_many

__all__ = [
    "ActionsSource",
    "CsvOHLC",
    "IBKRError",
    "IBKRErrorCategory",
    "IBKRErrorListener",
    "IBKRHistoricalOHLC",
    "MarketDataType",
    "OHLCSource",
    "YFInterval",
    "YFinanceOHLC",
    "classify_error",
    "load_alpaca_ohlc",
    "load_alpaca_option_chain",
    "load_ibkr_ohlc",
    "load_ohlc",
    "load_ohlc_many",
]


def __getattr__(name: str):
    if name in ("load_alpaca_ohlc", "load_alpaca_option_chain", "AlpacaOHLC"):
        from . import alpaca

        return getattr(alpaca, name)
    if name in ("load_ibkr_ohlc", "IBKRHistoricalOHLC"):
        from . import ibkr

        return getattr(ibkr, name)
    raise AttributeError(name)
