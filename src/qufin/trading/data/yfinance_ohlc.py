"""
yfinance OHLC bar loader.

yfinance returns a pandas DataFrame; we convert to polars immediately and
coerce to ``qufin.wyckoff._types.BAR_SCHEMA`` so the result is a drop-in
input for both the wyckoff toolkit and the trading engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

import polars as pl

from ...wyckoff._types import BAR_SCHEMA, OHLCV

if TYPE_CHECKING:
    import pandas as pd

Interval = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
]


def load_ohlc(
    symbol: str,
    *,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    period: str | None = None,
    interval: Interval = "1d",
    auto_adjust: bool = True,
) -> OHLCV:
    """Download OHLC bars for one ticker from yfinance.

    Either ``period`` (e.g. ``'5y'``) or an explicit ``start``/``end`` pair
    must be supplied. Returns a polars-backed ``OHLCV`` whose schema matches
    ``BAR_SCHEMA``. Timestamps are coerced to tz-aware UTC.
    """
    import yfinance as yf

    pdf: pd.DataFrame = yf.download(
        symbol,
        start=start,
        end=end,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if pdf.empty:
        raise ValueError(f"yfinance returned no data for {symbol!r}")
    # yfinance may return MultiIndex columns when a single symbol is requested
    # alongside others; flatten if so.
    if hasattr(pdf.columns, "nlevels") and pdf.columns.nlevels > 1:
        pdf.columns = [c[0] for c in pdf.columns]
    pdf = pdf.reset_index().rename(
        columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )[["timestamp", "open", "high", "low", "close", "volume"]]
    df = pl.from_pandas(pdf)
    # Coerce to BAR_SCHEMA.
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", time_zone="UTC")),
        *(pl.col(c).cast(pl.Float64()) for c in ("open", "high", "low", "close", "volume")),
    )
    # Drop nulls produced by partial bars.
    df = df.drop_nulls(subset=["open", "high", "low", "close"])
    return OHLCV.from_records(df, symbol=symbol)


def load_ohlc_many(
    symbols: list[str],
    *,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    period: str | None = None,
    interval: Interval = "1d",
    auto_adjust: bool = True,
) -> dict[str, OHLCV]:
    """Download OHLC bars for multiple tickers and return a symbol → OHLCV dict."""
    return {
        sym: load_ohlc(
            sym,
            start=start,
            end=end,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
        )
        for sym in symbols
    }


# Schema re-export for callers building frames directly.
__all__ = ["BAR_SCHEMA", "Interval", "load_ohlc", "load_ohlc_many"]
