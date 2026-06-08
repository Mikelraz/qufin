"""
One-shot market-data quoting over a read-only IBKR client.

``IBKRBroker`` owns order placement and account state; quoting is a separate
concern that needs its own ``clientId`` (IBKR rejects duplicate ids) and its
own market-data-type setting. The free functions here snapshot a stock or
option contract on *any* connected ``ib_async.IB`` instance, and
``QuoteSession`` is a convenience wrapper that manages a dedicated read-only
connection so a chain explorer, a price alert, or a notebook can pull quotes
without opening an order-capable broker connection.

The snapshot pattern (subscribe delayed market data, wait for the feed to land
ticks, read bid/ask/Greeks, unsubscribe) is shared by every IBKR tool in this
project; keeping it here means the scripts stay thin and the behaviour stays
consistent.

``ib_async`` is imported lazily so importing this module never requires the
optional ``trading-live`` dependency group.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ...data.vendors._ib_common import (
    IBKRErrorListener,
    MarketDataType,
    connect_ib,
    safe_float,
    snapshot,
    snapshot_many,
)
from ...options._types import CALL, OptionContract

__all__ = [
    "MarketDataType",
    "Quote",
    "QuoteSession",
    "quote_option",
    "quote_options",
    "quote_stock",
]

# Generic ticks that make delayed option feeds populate OI / IV / volume.
# 100=OptionVolume, 101=OptionOpenInterest, 106=ImpliedVol, 165=MiscStats.
_OPTION_GENERIC_TICKS = "100,101,106,165"


@dataclass(slots=True, frozen=True)
class Quote:
    """A single market-data snapshot of a stock or option contract."""

    bid: float | None = None
    ask: float | None = None
    last: float | None = None  # last trade, falling back to prior close
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    open_interest: float | None = None
    volume: float | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None or self.bid <= 0.0 or self.ask <= 0.0:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float | None:
        m = self.mid
        if m is None or m <= 0.0 or self.bid is None or self.ask is None:
            return None
        return (self.ask - self.bid) / m * 100.0

    @property
    def price(self) -> float | None:
        """Best available reference price: mid when quotable, else last."""
        m = self.mid
        return m if m is not None else self.last


def _ib_option(contract: OptionContract) -> Any:
    from ib_async import Option

    return Option(
        contract.underlying,
        contract.expiry.strftime("%Y%m%d"),
        float(contract.strike),
        "C" if contract.option_type == CALL else "P",
        "SMART",
    )


def _ticker_to_quote(ticker: Any, *, is_option: bool, is_call: bool) -> Quote:
    greeks = (
        ticker.modelGreeks
        or getattr(ticker, "lastGreeks", None)
        or getattr(ticker, "bidGreeks", None)
        or getattr(ticker, "askGreeks", None)
    )
    oi: float | None = None
    if is_option:
        tick = "callOpenInterest" if is_call else "putOpenInterest"
        oi = safe_float(getattr(ticker, tick, None))
    return Quote(
        bid=safe_float(ticker.bid),
        ask=safe_float(ticker.ask),
        last=safe_float(ticker.last) or safe_float(ticker.close),
        iv=safe_float(greeks.impliedVol) if greeks is not None else None,
        delta=safe_float(greeks.delta) if greeks is not None else None,
        gamma=safe_float(greeks.gamma) if greeks is not None else None,
        theta=safe_float(greeks.theta) if greeks is not None else None,
        vega=safe_float(greeks.vega) if greeks is not None else None,
        open_interest=oi,
        volume=safe_float(ticker.volume),
    )


async def quote_stock(ib: Any, symbol: str, *, settle_seconds: float = 4.0) -> Quote:
    """Snapshot a single equity on an already-connected ``IB`` instance."""
    from ib_async import Stock

    contract = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(contract)
    ticker = await snapshot(ib, contract, settle_seconds=settle_seconds, await_quotable=True)
    return _ticker_to_quote(ticker, is_option=False, is_call=False)


async def quote_option(ib: Any, contract: OptionContract, *, settle_seconds: float = 4.0) -> Quote:
    """Snapshot a single option contract on an already-connected ``IB``.

    Returns an empty :class:`Quote` if the contract does not qualify (not
    listed, or the account lacks options permissions).
    """
    opt = _ib_option(contract)
    await ib.qualifyContractsAsync(opt)
    if not getattr(opt, "conId", 0):
        return Quote()
    ticker = await snapshot(
        ib,
        opt,
        generic_ticks=_OPTION_GENERIC_TICKS,
        settle_seconds=settle_seconds,
        await_quotable=True,
    )
    return _ticker_to_quote(ticker, is_option=True, is_call=opt.right == "C")


async def quote_options(
    ib: Any,
    contracts: Sequence[OptionContract],
    *,
    settle_seconds: float = 4.0,
    batch_size: int = 60,
    pacing_seconds: float = 1.0,
) -> dict[OptionContract, Quote]:
    """Snapshot many option contracts concurrently, in batches.

    Subscribes up to ``batch_size`` contracts at once (paper Gateway allows
    ~100 concurrent market-data lines), waits ``settle_seconds`` for the
    delayed feed to land, reads, then unsubscribes before the next batch.
    Unqualified contracts map to an empty :class:`Quote`.
    """
    result: dict[OptionContract, Quote] = {c: Quote() for c in contracts}
    opts = [_ib_option(c) for c in contracts]
    await ib.qualifyContractsAsync(*opts)
    pairs = [(c, o) for c, o in zip(contracts, opts, strict=True) if getattr(o, "conId", 0)]
    if not pairs:
        return result

    tickers = await snapshot_many(
        ib,
        [o for _, o in pairs],
        generic_ticks=_OPTION_GENERIC_TICKS,
        settle_seconds=settle_seconds,
        batch_size=batch_size,
        pacing_seconds=pacing_seconds,
    )
    for (c, o), ticker in zip(pairs, tickers, strict=True):
        result[c] = _ticker_to_quote(ticker, is_option=True, is_call=o.right == "C")
    return result


@dataclass(slots=True)
class QuoteSession:
    """A dedicated read-only IBKR connection for market-data snapshots.

    Use as an async context manager::

        async with QuoteSession(port=4002, client_id=91) as qs:
            spot = await qs.spot("CRWV")
            q = await qs.option_quote(contract)

    Parameters
    ----------
    host, port         TWS/Gateway endpoint. 7497 = paper TWS, 4002 = paper Gateway.
    client_id          Unique client id; must differ from other connected clients.
    market_data_type   ``reqMarketDataType`` to request after connecting.
    """

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 90
    market_data_type: MarketDataType = MarketDataType.DELAYED_FROZEN
    _ib: Any = field(default=None, init=False, repr=False)
    _errors: IBKRErrorListener = field(default_factory=IBKRErrorListener, init=False)

    async def connect(self) -> None:
        self._ib = await connect_ib(
            self.host,
            self.port,
            self.client_id,
            market_data_type=self.market_data_type,
            listener=self._errors,
        )

    async def disconnect(self) -> None:
        if self._ib is not None:
            self._errors.detach()
            await asyncio.to_thread(self._ib.disconnect)
            self._ib = None

    @property
    def errors(self) -> IBKRErrorListener:
        """Classified IBKR messages seen on this read-only session."""
        return self._errors

    async def __aenter__(self) -> QuoteSession:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()

    @property
    def ib(self) -> Any:
        if self._ib is None:
            raise RuntimeError("QuoteSession is not connected; await session.connect() first")
        return self._ib

    async def stock_quote(self, symbol: str, *, settle_seconds: float = 4.0) -> Quote:
        return await quote_stock(self.ib, symbol, settle_seconds=settle_seconds)

    async def spot(self, symbol: str, *, settle_seconds: float = 4.0) -> float | None:
        """Best available underlying price: last, else mid, else bid."""
        q = await self.stock_quote(symbol, settle_seconds=settle_seconds)
        return q.last or q.mid or q.bid

    async def option_quote(self, contract: OptionContract, *, settle_seconds: float = 4.0) -> Quote:
        return await quote_option(self.ib, contract, settle_seconds=settle_seconds)

    async def option_quotes(
        self,
        contracts: Sequence[OptionContract],
        *,
        settle_seconds: float = 4.0,
        batch_size: int = 60,
        pacing_seconds: float = 1.0,
    ) -> dict[OptionContract, Quote]:
        return await quote_options(
            self.ib,
            contracts,
            settle_seconds=settle_seconds,
            batch_size=batch_size,
            pacing_seconds=pacing_seconds,
        )
