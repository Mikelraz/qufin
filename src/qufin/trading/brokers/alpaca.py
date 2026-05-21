"""
Alpaca broker adapter.

Wraps the ``alpaca-py`` SDK behind the ``Broker`` protocol. Stocks and
US-listed equity options are both supported by Alpaca's API; the adapter
routes ``Order`` (with a ``str`` symbol) and ``Order`` (with an
``OptionContract`` asset) through the same code path.

The import of ``alpaca`` is lazy so the rest of the trading subpackage
remains usable without the optional dependency installed.

Credentials
-----------
Reads ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` from the process env
unless explicitly supplied. ``paper=True`` (default) routes to
``https://paper-api.alpaca.markets``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ...options._types import CALL, OptionContract
from .._types import (
    AccountSnapshot,
    BarEvent,
    Fill,
    Order,
    OrderId,
    OrderType,
    Position,
    SymbolOrContract,
    TimeInForce,
)


def _format_option_symbol(contract: OptionContract) -> str:
    """Build the OCC-format symbol Alpaca uses for an option leg.

    Format: ``ROOT[YYMMDD][C|P][strike*1000 padded to 8 digits]``.
    e.g. ``AAPL  240119C00185000``. The 6-char root padding is intentional —
    Alpaca normalises whitespace internally.
    """
    root = contract.underlying.upper().ljust(6)
    expiry = contract.expiry.strftime("%y%m%d")
    cp = "C" if contract.option_type == CALL else "P"
    strike = f"{int(round(contract.strike * 1000)):08d}"
    return f"{root}{expiry}{cp}{strike}".replace(" ", "")


def _alpaca_symbol(asset: SymbolOrContract) -> str:
    if isinstance(asset, OptionContract):
        return _format_option_symbol(asset)
    return str(asset)


def _to_alpaca_tif(tif: TimeInForce) -> Any:
    from alpaca.trading.enums import TimeInForce as AlpacaTIF
    match tif:
        case TimeInForce.DAY:
            return AlpacaTIF.DAY
        case TimeInForce.GTC:
            return AlpacaTIF.GTC
        case TimeInForce.IOC:
            return AlpacaTIF.IOC
        case TimeInForce.FOK:
            return AlpacaTIF.FOK


@dataclass(slots=True)
class AlpacaBroker:
    """Alpaca paper / live broker.

    Parameters
    ----------
    api_key, secret_key  Defaults to ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``.
    paper                Route to the paper endpoint when True (default).
    feed                 ``"iex"`` (free) or ``"sip"`` (paid) data feed.
    """

    api_key: str | None = None
    secret_key: str | None = None
    paper: bool = True
    feed: str = "iex"
    _client: Any = field(default=None, init=False, repr=False)
    _stream: Any = field(default=None, init=False, repr=False)
    _fill_queue: asyncio.Queue[Fill] = field(default_factory=asyncio.Queue)
    _connected: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = self.secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are required. Pass them "
                "explicitly or set the env vars."
            )

    async def connect(self) -> None:
        from alpaca.trading.client import TradingClient
        self._client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        if self._stream is not None:
            await asyncio.to_thread(self._stream.stop)
            self._stream = None

    async def account(self) -> AccountSnapshot:
        self._require_connected()
        acct = await asyncio.to_thread(self._client.get_account)
        now = datetime.now(tz=UTC)
        return AccountSnapshot(
            timestamp=now,
            cash=float(acct.cash),
            equity=float(acct.equity),
            buying_power=float(acct.buying_power),
            margin_used=float(acct.initial_margin) if hasattr(acct, "initial_margin") else 0.0,
            # Alpaca exposes only a day-trade counter here, not realised day PnL.
            day_pnl=float(getattr(acct, "daytrade_count", 0.0)),
            total_pnl=float(acct.equity) - float(acct.last_equity),
        )

    async def positions(self) -> list[Position]:
        self._require_connected()
        raw = await asyncio.to_thread(self._client.get_all_positions)
        out: list[Position] = []
        for p in raw:
            out.append(
                Position(
                    asset=p.symbol,
                    qty=float(p.qty),
                    avg_price=float(p.avg_entry_price),
                    last_mark=float(p.current_price),
                    unrealised_pnl=float(p.unrealized_pl),
                    multiplier=100.0 if getattr(p, "asset_class", "") == "us_option" else 1.0,
                )
            )
        return out

    async def place_order(self, order: Order) -> OrderId:
        self._require_connected()
        from alpaca.trading.enums import OrderSide
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        symbol = _alpaca_symbol(order.asset)
        side = OrderSide.BUY if order.qty > 0 else OrderSide.SELL
        qty = abs(order.qty)
        tif = _to_alpaca_tif(order.tif)

        match order.order_type:
            case OrderType.MARKET:
                req: Any = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=tif)
            case OrderType.LIMIT:
                req = LimitOrderRequest(
                    symbol=symbol, qty=qty, side=side, time_in_force=tif,
                    limit_price=order.limit_price,
                )
            case OrderType.STOP:
                req = StopOrderRequest(
                    symbol=symbol, qty=qty, side=side, time_in_force=tif,
                    stop_price=order.stop_price,
                )
            case OrderType.STOP_LIMIT:
                req = StopLimitOrderRequest(
                    symbol=symbol, qty=qty, side=side, time_in_force=tif,
                    stop_price=order.stop_price, limit_price=order.limit_price,
                )
        placed = await asyncio.to_thread(self._client.submit_order, req)
        return str(placed.id)

    async def cancel_order(self, order_id: OrderId) -> None:
        self._require_connected()
        await asyncio.to_thread(self._client.cancel_order_by_id, order_id)

    def stream_bars(self, symbols: Sequence[str]) -> AsyncIterator[BarEvent]:
        return self._bar_stream(symbols)

    def stream_fills(self) -> AsyncIterator[Fill]:
        return self._fill_stream()

    # ------------------------------------------------------------------

    async def _bar_stream(self, symbols: Sequence[str]) -> AsyncIterator[BarEvent]:
        from alpaca.data.live import StockDataStream

        self._stream = StockDataStream(self.api_key, self.secret_key, feed=self.feed)
        queue: asyncio.Queue[BarEvent] = asyncio.Queue()

        async def handler(bar: Any) -> None:
            await queue.put(
                BarEvent(
                    symbol=str(bar.symbol),
                    timestamp=bar.timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                    index=-1,
                )
            )

        for sym in symbols:
            self._stream.subscribe_bars(handler, sym)
        # alpaca-py SDK runs the stream synchronously; run it in a thread.
        asyncio.create_task(asyncio.to_thread(self._stream.run))
        while self._connected:
            yield await queue.get()

    async def _fill_stream(self) -> AsyncIterator[Fill]:
        while True:
            yield await self._fill_queue.get()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("AlpacaBroker is not connected; await broker.connect() first")
