"""
Interactive Brokers historical OHLC loader.

Wraps ``ib_async``'s ``reqHistoricalDataAsync`` and emits frames that
match ``BAR_SCHEMA``. A local TWS or IB Gateway must be running and have
the API enabled; see ``docs/ibkr_setup.md`` for the one-time setup.

``ib_async`` is imported lazily so the rest of the data subpackage stays
usable without the optional ``trading-live`` dependency group installed.

Notes
-----
* IBKR's historical-data API requires a market-data subscription for
  real-time bars. Paper accounts can use the free *delayed* feed by
  passing ``use_delayed=True`` (the default).
* IBKR throttles historical requests (~60 reqs per 10 min for sub-30s
  bars). ``fetch_many`` issues calls serially with a small inter-request
  sleep to stay under the cap; for large universes prefer cached
  Parquet over hot fetches.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import polars as pl

from .._types import BAR_SCHEMA, OHLCV
from ._ib_common import IBKRErrorListener, MarketDataType, connect_ib, safe_float

BarSize = Literal[
    "1 secs",
    "5 secs",
    "10 secs",
    "15 secs",
    "30 secs",
    "1 min",
    "2 mins",
    "3 mins",
    "5 mins",
    "10 mins",
    "15 mins",
    "20 mins",
    "30 mins",
    "1 hour",
    "2 hours",
    "3 hours",
    "4 hours",
    "8 hours",
    "1 day",
    "1 week",
    "1 month",
]
WhatToShow = Literal["TRADES", "MIDPOINT", "BID", "ASK", "BID_ASK"]


def load_ibkr_ohlc(
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    bar_size: BarSize = "1 day",
    what_to_show: WhatToShow = "TRADES",
    use_rth: bool = True,
    use_delayed: bool = True,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 11,
    exchange: str = "SMART",
    currency: str = "USD",
) -> OHLCV:
    """Download historical bars from IBKR and return an ``OHLCV`` frame.

    Parameters
    ----------
    symbol         Equity ticker (e.g. ``"SPY"``).
    start, end     UTC-aware datetimes bounding the window.
    bar_size       IBKR bar-size string; see ``BarSize`` literal.
    what_to_show   Data type. ``"TRADES"`` matches the rest of qufin's pipeline.
    use_rth        Regular trading hours only.
    use_delayed    Switch the TWS data source to delayed (free on paper).
    host, port     TWS/Gateway endpoint. ``7497`` = paper TWS, ``4002`` = paper Gateway.
    client_id      Unique connection id; pick something other than what your live
                   broker connection uses (``IBKRBroker`` defaults to ``1``).
    """
    return asyncio.run(
        _fetch_async(
            symbol,
            start=start,
            end=end,
            bar_size=bar_size,
            what_to_show=what_to_show,
            use_rth=use_rth,
            use_delayed=use_delayed,
            host=host,
            port=port,
            client_id=client_id,
            exchange=exchange,
            currency=currency,
        )
    )


async def _fetch_async(
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    bar_size: BarSize,
    what_to_show: WhatToShow,
    use_rth: bool,
    use_delayed: bool,
    host: str,
    port: int,
    client_id: int,
    exchange: str,
    currency: str,
) -> OHLCV:
    from ib_async import Stock

    # DELAYED is sufficient for backtesting off paper accounts without a
    # real-time data subscription; the listener surfaces any subscription gaps.
    listener = IBKRErrorListener()
    ib = await connect_ib(
        host,
        port,
        client_id,
        market_data_type=MarketDataType.DELAYED if use_delayed else None,
        listener=listener,
    )
    try:
        contract = Stock(symbol, exchange, currency)
        await ib.qualifyContractsAsync(contract)

        duration_str = _duration_string(start, end)
        # IBKR's endDateTime must be in UTC with the explicit suffix; empty string = now.
        end_str = end.astimezone(UTC).strftime("%Y%m%d-%H:%M:%S")

        raw = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_str,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,  # 2 = epoch seconds in UTC
        )
    finally:
        listener.detach()
        ib.disconnect()

    return _bars_to_ohlcv(raw, symbol=symbol)


def _bars_to_ohlcv(bars: Sequence[Any], *, symbol: str) -> OHLCV:
    """Convert ``ib_async`` BarData rows into a ``BAR_SCHEMA`` polars frame."""
    rows: list[dict[str, Any]] = []
    for b in bars:
        ts = b.date
        match ts:
            case datetime():
                ts_utc = ts.astimezone(UTC) if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
            case _:
                # ``date`` for daily bars when formatDate=1; coerce to UTC midnight.
                ts_utc = datetime(ts.year, ts.month, ts.day, tzinfo=UTC)
        rows.append(
            {
                "timestamp": ts_utc,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": safe_float(b.volume) or 0.0,
            }
        )
    df = pl.DataFrame(rows, schema=BAR_SCHEMA) if rows else pl.DataFrame(schema=BAR_SCHEMA)
    return OHLCV.from_records(df, symbol=symbol)


def _duration_string(start: datetime, end: datetime) -> str:
    """Build an IBKR ``durationStr`` (e.g. ``"30 D"``, ``"6 M"``, ``"2 Y"``).

    IBKR accepts S / D / W / M / Y. We pick the smallest unit that fits the
    window — that's what keeps single-request payloads inside the 1000-bar
    soft limit for the most common bar sizes.
    """
    if end <= start:
        raise ValueError("end must be after start")
    span_secs = (end - start).total_seconds()
    span_days = span_secs / 86_400.0
    if span_secs <= 86_400:
        return f"{int(math.ceil(span_secs))} S"
    if span_days <= 30:
        return f"{int(math.ceil(span_days))} D"
    # IBKR per-unit caps: W ≤ 52, M ≤ 12. Anything that would round above
    # those (≈ a full year) jumps straight to Y, otherwise the server replies
    # with "must be made in years".
    if span_days <= 52 * 7:
        return f"{int(math.ceil(span_days / 7.0))} W"
    return f"{int(math.ceil(span_days / 365.0))} Y"


@dataclass(slots=True)
class IBKRHistoricalOHLC:
    """``OHLCSource``-conforming wrapper around the IBKR historical bars endpoint.

    Parameters
    ----------
    host, port     TWS/Gateway endpoint. Defaults match ``IBKRBroker``.
    client_id      Connection id; must not collide with other live clients.
    use_delayed    Use delayed market data (free on paper accounts).
    use_rth        Regular trading hours only.
    what_to_show   IBKR data type.
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 11
    use_delayed: bool = True
    use_rth: bool = True
    what_to_show: WhatToShow = "TRADES"
    pacing_seconds: float = 1.0

    def fetch(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> OHLCV:
        bar_size = _interval_to_bar_size(interval)
        return load_ibkr_ohlc(
            symbol,
            start=start,
            end=end,
            bar_size=bar_size,
            what_to_show=self.what_to_show,
            use_rth=self.use_rth,
            use_delayed=self.use_delayed,
            host=self.host,
            port=self.port,
            client_id=self.client_id,
        )

    def fetch_many(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> dict[str, OHLCV]:
        return asyncio.run(self._fetch_many_async(symbols, start=start, end=end, interval=interval))

    async def _fetch_many_async(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> dict[str, OHLCV]:
        from ib_async import Stock

        bar_size = _interval_to_bar_size(interval)
        duration_str = _duration_string(start, end)
        end_str = end.astimezone(UTC).strftime("%Y%m%d-%H:%M:%S")

        listener = IBKRErrorListener()
        ib = await connect_ib(
            self.host,
            self.port,
            self.client_id,
            market_data_type=MarketDataType.DELAYED if self.use_delayed else None,
            listener=listener,
        )
        out: dict[str, OHLCV] = {}
        try:
            for sym in symbols:
                contract = Stock(sym, "SMART", "USD")
                await ib.qualifyContractsAsync(contract)
                raw = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_str,
                    durationStr=duration_str,
                    barSizeSetting=bar_size,
                    whatToShow=self.what_to_show,
                    useRTH=self.use_rth,
                    formatDate=2,
                )
                out[sym] = _bars_to_ohlcv(raw, symbol=sym)
                await asyncio.sleep(self.pacing_seconds)
        finally:
            listener.detach()
            ib.disconnect()
        return out


# qufin-style short intervals → IBKR bar-size strings.
_INTERVAL_TO_BAR_SIZE: dict[str, BarSize] = {
    "1s": "1 secs",
    "5s": "5 secs",
    "15s": "15 secs",
    "30s": "30 secs",
    "1m": "1 min",
    "2m": "2 mins",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "2h": "2 hours",
    "4h": "4 hours",
    "1d": "1 day",
    "1w": "1 week",
    "1mo": "1 month",
}


def _interval_to_bar_size(interval: str) -> BarSize:
    key = interval.strip().lower()
    if key not in _INTERVAL_TO_BAR_SIZE:
        raise ValueError(
            f"unsupported IBKR interval: {interval!r}; "
            f"valid values: {sorted(_INTERVAL_TO_BAR_SIZE)}"
        )
    return _INTERVAL_TO_BAR_SIZE[key]
