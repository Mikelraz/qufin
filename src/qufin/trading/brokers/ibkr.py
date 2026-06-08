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

from ...data.vendors._ib_common import (
    IBKRErrorCategory,
    IBKRErrorListener,
    MarketDataType,
    connect_ib,
    safe_float,
)
from ...options._types import CALL, OptionContract
from .._types import (
    AccountSnapshot,
    BarEvent,
    Fill,
    Order,
    OrderId,
    OrderRejectedError,
    OrderStatus,
    OrderType,
    Position,
    SymbolOrContract,
    TimeInForce,
)

_TIF_TO_IB: dict[TimeInForce, str] = {
    TimeInForce.DAY: "DAY",
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
}


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


def _unknown_status(order_id: OrderId) -> OrderStatus:
    return OrderStatus(order_id=str(order_id), status="Unknown")


def _ib_order(order: Order) -> Any:
    from ib_async import LimitOrder, MarketOrder, StopLimitOrder, StopOrder

    action = "BUY" if order.qty > 0 else "SELL"
    qty = abs(order.qty)
    tif = _TIF_TO_IB[order.tif]
    # outsideRth=True lets orders submitted outside Regular Trading Hours
    # survive Gateway's default "cancel-outside-RTH" preset instead of being
    # silently cancelled with error 10349. Paper accounts typically want this.
    # ``Order.__post_init__`` already guarantees the price fields below are set
    # for the relevant types; the explicit guards keep this translation total
    # (and satisfy the type checker) at the IB boundary.
    match order.order_type:
        case OrderType.MARKET:
            return MarketOrder(action, qty, tif=tif, outsideRth=True)
        case OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("LIMIT order requires limit_price")
            return LimitOrder(action, qty, order.limit_price, tif=tif, outsideRth=True)
        case OrderType.STOP:
            if order.stop_price is None:
                raise ValueError("STOP order requires stop_price")
            return StopOrder(action, qty, order.stop_price, tif=tif, outsideRth=True)
        case OrderType.STOP_LIMIT:
            if order.limit_price is None or order.stop_price is None:
                raise ValueError("STOP_LIMIT order requires limit_price and stop_price")
            return StopLimitOrder(
                action, qty, order.limit_price, order.stop_price, tif=tif, outsideRth=True
            )
        case _:
            raise ValueError(f"unsupported order type: {order.order_type!r}")


@dataclass(slots=True)
class IBKRBroker:
    """Interactive Brokers paper or live adapter.

    Parameters
    ----------
    host       TWS/Gateway host (default ``"127.0.0.1"``).
    port       7497 = paper TWS, 7496 = live TWS, 4002 = paper Gateway, 4001 = live Gateway.
    client_id  Unique client id; must differ from other connected clients.
    market_data_type   Optional ``reqMarketDataType`` applied on connect (e.g.
                       ``MarketDataType.DELAYED_FROZEN`` for paper accounts).
    connect_timeout    Seconds to wait for the connection before failing.
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    market_data_type: MarketDataType | None = None
    connect_timeout: float = 8.0
    _ib: Any = field(default=None, init=False, repr=False)
    _fill_queue: asyncio.Queue[Fill] = field(default_factory=asyncio.Queue)
    _connected: bool = field(default=False, init=False)
    _errors: IBKRErrorListener = field(default_factory=IBKRErrorListener, init=False)

    async def connect(self) -> None:
        self._ib = await connect_ib(
            self.host,
            self.port,
            self.client_id,
            timeout=self.connect_timeout,
            market_data_type=self.market_data_type,
            listener=self._errors,
        )
        self._ib.execDetailsEvent += self._on_exec
        self._connected = True

    async def disconnect(self) -> None:
        if self._ib is not None:
            self._errors.detach()
            await asyncio.to_thread(self._ib.disconnect)
            self._ib = None
        self._connected = False

    @property
    def errors(self) -> IBKRErrorListener:
        """Classified IBKR messages seen since connect (data / order / connection)."""
        return self._errors

    @property
    def has_subscription_issue(self) -> bool:
        """True once a market-data subscription / delayed-feed warning has arrived."""
        return self._errors.has_subscription_issue

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
            asset: SymbolOrContract
            if p.contract.secType == "OPT":
                # Build a proper OptionContract so callers can distinguish
                # multiple option positions on the same underlying.
                from datetime import date as _date

                ymd = p.contract.lastTradeDateOrContractMonth
                asset = OptionContract(
                    strike=float(p.contract.strike),
                    expiry=_date.fromisoformat(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"),
                    option_type=p.contract.right,
                    underlying=p.contract.symbol,
                )
            else:
                asset = p.contract.symbol
            out.append(
                Position(
                    asset=asset,
                    qty=float(p.position),
                    avg_price=float(p.avgCost),
                    multiplier=100.0 if p.contract.secType == "OPT" else 1.0,
                )
            )
        return out

    async def place_order(
        self, order: Order, *, wait: bool = False, timeout: float = 5.0
    ) -> OrderId:
        """Submit ``order`` and return its broker order id.

        By default returns as soon as the order is transmitted (unchanged
        behaviour). With ``wait=True`` it blocks until the order reaches a
        working or terminal state and raises :class:`OrderRejectedError` if the
        broker rejects or marks it inactive.
        """
        self._require_connected()
        contract = _ib_contract(order.asset)
        await self._ib.qualifyContractsAsync(contract)
        ib_order = _ib_order(order)
        trade = self._ib.placeOrder(contract, ib_order)
        order_id = str(trade.order.orderId)
        if wait:
            await self.wait_for_status(order_id, timeout=timeout)
        return order_id

    async def order_status(self, order_id: OrderId) -> OrderStatus:
        """Current status of a live order; ``status="Unknown"`` if not found."""
        self._require_connected()
        status = self._status_from_trades(order_id)
        return status if status is not None else _unknown_status(order_id)

    async def wait_for_status(self, order_id: OrderId, *, timeout: float = 5.0) -> OrderStatus:
        """Poll until the order is working/terminal (or ``timeout``); raise on reject.

        Transient states (``PendingSubmit`` and ``ib_async``'s ``ValidationError``,
        which appears when a benign warning arrives mid-submit) are *not* treated
        as final — this is what stops a successfully-resting order from looking
        like a failure.
        """
        self._require_connected()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        latest: OrderStatus | None = None
        while loop.time() < deadline:
            status = self._status_from_trades(order_id)
            if status is not None:
                latest = status
                if status.is_working or status.is_done:
                    break
            await asyncio.sleep(0.1)
        result = latest if latest is not None else _unknown_status(order_id)
        if result.is_rejected:
            raise OrderRejectedError(result)
        return result

    def _status_from_trades(self, order_id: OrderId) -> OrderStatus | None:
        for trade in self._ib.trades():
            if str(trade.order.orderId) == str(order_id):
                return self._to_order_status(trade)
        return None

    def _to_order_status(self, trade: Any) -> OrderStatus:
        raw = trade.orderStatus
        status = str(raw.status)
        reason = (
            self._reject_reason(trade)
            if status in ("Inactive", "Cancelled", "ApiCancelled")
            else None
        )
        return OrderStatus(
            order_id=str(trade.order.orderId),
            status=status,
            filled=float(raw.filled),
            remaining=float(raw.remaining),
            avg_fill_price=(safe_float(raw.avgFillPrice) or None),
            reject_reason=reason,
        )

    def _reject_reason(self, trade: Any) -> str | None:
        last = self._errors.last(IBKRErrorCategory.ORDER_REJECT)
        if last is not None:
            return last.message
        for entry in reversed(getattr(trade, "log", [])):
            message = getattr(entry, "message", "")
            if message:
                return str(message)
        return None

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
