"""
Broker protocol — what every concrete broker (paper or live) must implement.

The surface is intentionally minimal. Each call returns the canonical
``Order`` / ``Fill`` / ``Position`` / ``AccountSnapshot`` types from
``trading._types`` so strategy code is identical across brokers. Streaming
is exposed as ``AsyncIterator``; backends without a native push channel
implement it as a polling generator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from .._types import (
    AccountSnapshot,
    BarEvent,
    Fill,
    Order,
    OrderId,
    Position,
)


@runtime_checkable
class Broker(Protocol):
    """Single, asyncio-based interface for paper and live brokers.

    Lifecycle: ``connect`` once, then any combination of ``account``,
    ``positions``, ``place_order``, ``cancel_order``, ``stream_bars``,
    ``stream_fills``; ``disconnect`` at the end.
    """

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def account(self) -> AccountSnapshot: ...

    async def positions(self) -> list[Position]: ...

    async def place_order(self, order: Order) -> OrderId: ...

    async def cancel_order(self, order_id: OrderId) -> None: ...

    def stream_bars(self, symbols: Sequence[str]) -> AsyncIterator[BarEvent]: ...

    def stream_fills(self) -> AsyncIterator[Fill]: ...
