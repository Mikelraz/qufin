"""
Alpaca historical OHLC + option-chain loader.

Wraps ``alpaca-py``'s historical data clients and emits frames that match
``BAR_SCHEMA`` and ``CHAIN_SCHEMA``. ``alpaca`` is imported lazily; the rest
of the data subpackage works without it installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import polars as pl

from ...options._types import CHAIN_SCHEMA, OptionChain
from .._types import OHLCV

TimeFrameUnit = Literal["Min", "Hour", "Day"]


def load_alpaca_ohlc(
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    amount: int = 1,
    unit: TimeFrameUnit = "Day",
    feed: str = "iex",
    api_key: str | None = None,
    secret_key: str | None = None,
) -> OHLCV:
    """Download historical bars from Alpaca and return an ``OHLCV`` frame."""
    import os

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.timeframe import TimeFrameUnit as TFUnit

    api_key = api_key or os.environ.get("ALPACA_API_KEY")
    secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(api_key, secret_key)
    unit_map: dict[str, TFUnit] = {"Min": TFUnit.Minute, "Hour": TFUnit.Hour, "Day": TFUnit.Day}
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        timeframe=TimeFrame(amount=amount, unit=unit_map[unit]),
        feed=feed,
    )
    bars = client.get_stock_bars(req).df.reset_index()
    bars = bars[bars["symbol"] == symbol] if "symbol" in bars.columns else bars
    df = pl.from_pandas(bars).rename({"timestamp": "timestamp"})
    df = df.select(
        pl.col("timestamp").cast(pl.Datetime("ns", time_zone="UTC")),
        pl.col("open").cast(pl.Float64()),
        pl.col("high").cast(pl.Float64()),
        pl.col("low").cast(pl.Float64()),
        pl.col("close").cast(pl.Float64()),
        pl.col("volume").cast(pl.Float64()),
    )
    return OHLCV.from_records(df, symbol=symbol)


def load_alpaca_option_chain(
    underlying: str,
    *,
    as_of: date,
    expiry: date,
    spot: float,
    api_key: str | None = None,
    secret_key: str | None = None,
) -> OptionChain:
    """Snapshot the option chain for ``underlying`` at one expiry from Alpaca."""
    import os

    from alpaca.data.historical import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest

    api_key = api_key or os.environ.get("ALPACA_API_KEY")
    secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
    client = OptionHistoricalDataClient(api_key, secret_key)
    req = OptionChainRequest(underlying_symbol=underlying, expiration_date=expiry)
    chain = client.get_option_chain(req)

    rows: list[dict[str, object]] = []
    for sym, snap in chain.items():
        root_len = len(underlying)
        body = sym[root_len:].lstrip("0").ljust(15)
        try:
            yymmdd = body[:6]
            right = body[6]
            strike_raw = int(body[7:15])
        except (ValueError, IndexError):
            continue
        rows.append(
            {
                "expiry": expiry,
                "strike": float(strike_raw) / 1000.0,
                "option_type": "C" if right == "C" else "P",
                "bid": float(snap.latest_quote.bid_price if snap.latest_quote else 0.0),
                "ask": float(snap.latest_quote.ask_price if snap.latest_quote else 0.0),
                "last": float(snap.latest_trade.price if snap.latest_trade else 0.0),
                "volume": int(snap.latest_trade.size if snap.latest_trade else 0),
                "open_interest": 0,
                "iv": 0.0,
            }
        )
        del yymmdd
    df = pl.DataFrame(rows, schema=CHAIN_SCHEMA)
    return OptionChain.from_records(df, spot=spot, as_of=as_of, underlying=underlying)


@dataclass(slots=True)
class AlpacaOHLC:
    """``OHLCSource``-conforming wrapper around the Alpaca bars endpoint."""

    feed: str = "iex"
    api_key: str | None = None
    secret_key: str | None = None

    def fetch(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> OHLCV:
        amount, unit = _parse_interval(interval)
        return load_alpaca_ohlc(
            symbol,
            start=start,
            end=end,
            amount=amount,
            unit=unit,
            feed=self.feed,
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

    def fetch_many(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> dict[str, OHLCV]:
        return {
            sym: self.fetch(sym, start=start, end=end, interval=interval) for sym in symbols
        }


_UNIT_BY_SUFFIX: dict[str, TimeFrameUnit] = {
    "m": "Min",
    "min": "Min",
    "h": "Hour",
    "hour": "Hour",
    "d": "Day",
    "day": "Day",
}


def _parse_interval(interval: str) -> tuple[int, TimeFrameUnit]:
    """Parse e.g. ``'1d'``, ``'5m'``, ``'1h'`` into (amount, alpaca unit)."""
    s = interval.strip().lower()
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i == 0:
        amount = 1
    else:
        amount = int(s[:i])
    suffix = s[i:] or "d"
    if suffix not in _UNIT_BY_SUFFIX:
        raise ValueError(f"unsupported Alpaca interval: {interval!r}")
    return amount, _UNIT_BY_SUFFIX[suffix]
