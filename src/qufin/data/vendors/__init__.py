"""Vendor adapters: OHLC, actions, and (for some vendors) option chains."""

from __future__ import annotations

from .base import ActionsSource, OHLCSource
from .csv import CsvOHLC
from .yfinance import Interval as YFInterval
from .yfinance import YFinanceOHLC, load_ohlc, load_ohlc_many

__all__ = [
    "ActionsSource",
    "CsvOHLC",
    "OHLCSource",
    "YFInterval",
    "YFinanceOHLC",
    "load_alpaca_ohlc",
    "load_alpaca_option_chain",
    "load_ohlc",
    "load_ohlc_many",
]


def __getattr__(name: str):
    if name in ("load_alpaca_ohlc", "load_alpaca_option_chain", "AlpacaOHLC"):
        from . import alpaca

        return getattr(alpaca, name)
    raise AttributeError(name)
