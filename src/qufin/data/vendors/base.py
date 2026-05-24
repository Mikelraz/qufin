"""
Vendor-source protocols.

Concrete adapters (yfinance, Alpaca, CSV, custom) implement these so the
data pipeline can plug them in interchangeably.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

import polars as pl

from .._types import OHLCV


@runtime_checkable
class OHLCSource(Protocol):
    """A source of historical OHLCV bars for one or more symbols."""

    def fetch(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> OHLCV: ...

    def fetch_many(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> dict[str, OHLCV]: ...


@runtime_checkable
class ActionsSource(Protocol):
    """A source of corporate actions (splits, dividends, ...) for one or more symbols."""

    def fetch(self, symbols: Sequence[str]) -> pl.DataFrame: ...
