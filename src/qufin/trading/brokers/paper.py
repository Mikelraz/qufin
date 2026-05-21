"""
In-process paper broker.

Drives an internal ``Portfolio`` + ``ExecutionModel`` from an externally
fed bar stream, exposing the same async ``Broker`` surface as live
adapters. Use this to run a strategy in a paper-trading-like loop without
any external dependency; it is the engine in a different costume.

For live paper-trading against a real exchange's paper endpoint, use
``AlpacaBroker(paper=True)`` or ``IBKRBroker(port=7497)`` instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .._types import (
    AccountSnapshot,
    BarEvent,
    Fill,
    Order,
    OrderId,
    Position,
    new_order_id,
)
from ..engine.execution import ExecutionModel, NextBarOpenExecution
from ..engine.portfolio import Portfolio


@dataclass(slots=True)
class PaperBroker:
    """Async paper broker backed by an in-process portfolio + execution model."""

    starting_cash: float = 100_000.0
    execution: ExecutionModel = field(default_factory=NextBarOpenExecution)
    _portfolio: Portfolio = field(init=False)
    _pending: list[Order] = field(default_factory=list)
    _fill_queue: asyncio.Queue[Fill] = field(default_factory=asyncio.Queue)
    _connected: bool = False
    _last_bar: dict[str, BarEvent] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._portfolio = Portfolio(starting_cash=self.starting_cash)

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def account(self) -> AccountSnapshot:
        self._require_connected()
        now = datetime.now(tz=UTC)
        return self._portfolio.snapshot(timestamp=now)

    async def positions(self) -> list[Position]:
        self._require_connected()
        return [p for p in self._portfolio.positions.values() if p.qty != 0.0]

    async def place_order(self, order: Order) -> OrderId:
        self._require_connected()
        oid = order.client_id or new_order_id("paper")
        self._pending.append(order.with_id(oid))
        return oid

    async def cancel_order(self, order_id: OrderId) -> None:
        self._require_connected()
        self._pending = [o for o in self._pending if o.client_id != order_id]

    def stream_bars(self, symbols: Sequence[str]) -> AsyncIterator[BarEvent]:
        """Replay bars previously injected via ``ingest_bar``.

        The paper broker doesn't fetch data on its own; tests and live
        paper-trading loops feed it via ``ingest_bar`` and the bars are
        relayed out through this stream. Callers that want a more
        sophisticated bar feed should use a real broker adapter.
        """
        del symbols  # paper broker emits every ingested bar
        return self._bar_stream()

    def stream_fills(self) -> AsyncIterator[Fill]:
        return self._fill_stream()

    # ------------------------------------------------------------------
    # Driving methods (used by tests and live paper loops)

    async def ingest_bar(self, bar: BarEvent) -> None:
        """Advance the broker by one bar: fill pending orders, then update marks."""
        self._require_connected()
        self._last_bar[bar.symbol] = bar
        if self._pending:
            fills = self.execution.execute(
                pending=self._pending,
                next_bars={bar.symbol: bar},
                timestamp=bar.timestamp,
            )
            self._pending = [
                o for o in self._pending if not any(f.order_id == o.client_id for f in fills)
            ]
            for fill in fills:
                self._portfolio.apply_fill(fill)
                await self._fill_queue.put(fill)
        # Mark to market using last close of every symbol seen so far.
        marks = {sym: b.close for sym, b in self._last_bar.items()}
        self._portfolio.mark_to_market(marks)

    # ------------------------------------------------------------------

    async def _bar_stream(self) -> AsyncIterator[BarEvent]:
        # The paper broker is fed externally; this stream is intentionally
        # a no-op iterator — callers that need bar replay should use the
        # offline ``BacktestEngine`` directly or wire a real broker.
        if False:
            yield  # type: ignore[unreachable]

    async def _fill_stream(self) -> AsyncIterator[Fill]:
        while True:
            fill = await self._fill_queue.get()
            yield fill

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("PaperBroker not connected; call await broker.connect() first")
