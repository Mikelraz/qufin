"""
Interactive Brokers adapter via ``ib_async``.

Connects to a locally running TWS or IB Gateway. Default ports are
``7497`` for paper and ``7496`` for live. Stocks and option contracts are
both routed through the same ``placeOrder`` call; option-contract
qualification (``qualifyContracts``) is performed lazily on first use.

``ib_async`` is imported lazily so the rest of the trading subpackage
remains usable without it.
"""

from __future__ import annotations

import asyncio
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
)


def _ib_contract(asset: SymbolOrContract) -> Any:
    from ib_async import Option, Stock
    if isinstance(asset, OptionContract):
        return Option(
            asset.underlying,
            asset.expiry.strftime("%Y%m%d"),
            asset.strike,
            "C" if asset.option_type == CALL else "P",
            "SMART",
        )
    return Stock(str(asset), "SMART", "USD")


def _ib_order(order: Order) -> Any:
    from ib_async import LimitOrder, MarketOrder, StopLimitOrder, StopOrder
    action = "BUY" if order.qty > 0 else "SELL"
    qty = abs(order.qty)
    match order.order_type:
        case OrderType.MARKET:
            return MarketOrder(action, qty)
        case OrderType.LIMIT:
            return LimitOrder(action, qty, order.limit_price)
        case OrderType.STOP:
            return StopOrder(action, qty, order.stop_price)
        case OrderType.STOP_LIMIT:
            return StopLimitOrder(action, qty, order.limit_price, order.stop_price)


@dataclass(slots=True)
class IBKRBroker:
    """Interactive Brokers paper or live adapter.

    Parameters
    ----------
    host       TWS/Gateway host (default ``"127.0.0.1"``).
    port       7497 = paper TWS, 7496 = live TWS, 4002 = paper Gateway, 4001 = live Gateway.
    client_id  Unique client id; must differ from other connected clients.
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    _ib: Any = field(default=None, init=False, repr=False)
    _fill_queue: asyncio.Queue[Fill] = field(default_factory=asyncio.Queue)
    _connected: bool = field(default=False, init=False)

    async def connect(self) -> None:
        from ib_async import IB
        self._ib = IB()
        await self._ib.connectAsync(self.host, self.port, clientId=self.client_id)
        self._ib.execDetailsEvent += self._on_exec  # type: ignore[attr-defined]
        self._connected = True

    async def disconnect(self) -> None:
        if self._ib is not None:
            await asyncio.to_thread(self._ib.disconnect)
        self._connected = False

    async def account(self) -> AccountSnapshot:
        self._require_connected()
        summary = await self._ib.accountSummaryAsync()
        values: dict[str, float] = {}
        for row in summary:
            try:
                values[row.tag] = float(row.value)
            except ValueError:
                continue
        now = datetime.now(tz=UTC)
        return AccountSnapshot(
            timestamp=now,
            cash=values.get("TotalCashValue", 0.0),
            equity=values.get("NetLiquidation", 0.0),
            buying_power=values.get("BuyingPower", 0.0),
            margin_used=values.get("MaintMarginReq", 0.0),
            day_pnl=values.get("UnrealizedPnL", 0.0),
            total_pnl=values.get("RealizedPnL", 0.0) + values.get("UnrealizedPnL", 0.0),
        )

    async def positions(self) -> list[Position]:
        self._require_connected()
        raw = await self._ib.reqPositionsAsync()
        out: list[Position] = []
        for p in raw:
            out.append(
                Position(
                    asset=p.contract.symbol,
                    qty=float(p.position),
                    avg_price=float(p.avgCost),
                    multiplier=100.0 if p.contract.secType == "OPT" else 1.0,
                )
            )
        return out

    async def place_order(self, order: Order) -> OrderId:
        self._require_connected()
        contract = _ib_contract(order.asset)
        await self._ib.qualifyContractsAsync(contract)
        ib_order = _ib_order(order)
        trade = self._ib.placeOrder(contract, ib_order)
        return str(trade.order.orderId)

    async def cancel_order(self, order_id: OrderId) -> None:
        self._require_connected()
        for trade in self._ib.trades():
            if str(trade.order.orderId) == str(order_id):
                self._ib.cancelOrder(trade.order)
                return

    def stream_bars(self, symbols: Sequence[str]) -> AsyncIterator[BarEvent]:
        return self._bar_stream(symbols)

    def stream_fills(self) -> AsyncIterator[Fill]:
        return self._fill_stream()

    # ------------------------------------------------------------------

    async def _bar_stream(self, symbols: Sequence[str]) -> AsyncIterator[BarEvent]:
        from ib_async import Stock
        queue: asyncio.Queue[BarEvent] = asyncio.Queue()
        for sym in symbols:
            contract = Stock(sym, "SMART", "USD")
            bars = self._ib.reqRealTimeBars(contract, 5, "TRADES", useRTH=False)
            bars.updateEvent += lambda _bars, has_new, sym=sym: (
                queue.put_nowait(
                    BarEvent(
                        symbol=sym,
                        timestamp=_bars[-1].time,
                        open=float(_bars[-1].open_),
                        high=float(_bars[-1].high),
                        low=float(_bars[-1].low),
                        close=float(_bars[-1].close),
                        volume=float(_bars[-1].volume),
                        index=-1,
                    )
                )
                if has_new
                else None
            )
        while self._connected:
            yield await queue.get()

    async def _fill_stream(self) -> AsyncIterator[Fill]:
        while True:
            yield await self._fill_queue.get()

    def _on_exec(self, trade: Any, fill: Any) -> None:
        # IB exec event → canonical Fill; pushed onto the queue.
        order = trade.order
        contract = trade.contract
        asset: SymbolOrContract
        if contract.secType == "OPT":
            from datetime import date as _date
            asset = OptionContract(
                strike=float(contract.strike),
                expiry=_date.fromisoformat(
                    f"{contract.lastTradeDateOrContractMonth[:4]}-"
                    f"{contract.lastTradeDateOrContractMonth[4:6]}-"
                    f"{contract.lastTradeDateOrContractMonth[6:8]}"
                ),
                option_type=contract.right,
                underlying=contract.symbol,
            )
        else:
            asset = contract.symbol
        sign = 1.0 if order.action == "BUY" else -1.0
        self._fill_queue.put_nowait(
            Fill(
                order_id=str(order.orderId),
                asset=asset,
                timestamp=fill.time,
                qty=sign * float(fill.shares),
                price=float(fill.price),
                commission=float(getattr(fill.commissionReport, "commission", 0.0)),
            )
        )

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("IBKRBroker is not connected; await broker.connect() first")
