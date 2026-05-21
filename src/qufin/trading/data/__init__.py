"""Market data loaders and parquet-backed cache."""

from __future__ import annotations

from .cache import ParquetCache
from .yfinance_ohlc import load_ohlc, load_ohlc_many

__all__ = [
    "ParquetCache",
    "load_alpaca_ohlc",
    "load_alpaca_option_chain",
    "load_ohlc",
    "load_ohlc_many",
]


def __getattr__(name: str):
    if name in ("load_alpaca_ohlc", "load_alpaca_option_chain"):
        from . import alpaca_data
        return getattr(alpaca_data, name)
    raise AttributeError(name)
