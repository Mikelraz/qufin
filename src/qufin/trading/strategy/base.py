"""
Strategy protocol — what the engine expects from any signal source.

A ``Strategy`` is a stateful callable that the engine drives bar-by-bar. It
sees only events at or before the current bar (look-ahead is impossible by
construction — the engine enforces it). The strategy returns ``Order`` or
``Signal`` lists; the engine handles routing, sizing (for ``Signal``), and
fill simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import polars as pl

from .._types import (
    AccountSnapshot,
    BarEvent,
    Fill,
    Order,
    Position,
    Signal,
    SymbolOrContract,
)


@dataclass(slots=True)
class StrategyContext:
    """Read-only view onto engine state passed to every strategy callback.

    The engine constructs and updates this once per bar. Strategies must
    treat it as read-only — mutating it does not affect the engine.

    Attributes
    ----------
    account     Latest account snapshot at the current bar's close.
    positions   Dict of open positions keyed by their asset.
    history     Per-symbol polars frame of bars up to and including the
                current bar (slice of the source). Use this for indicator
                / feature computation.
    bar         The current bar event (last row of ``history``).
    """

    account: AccountSnapshot
    positions: dict[SymbolOrContract, Position]
    history: dict[str, pl.DataFrame]
    bar: BarEvent
    fills_since_last_bar: list[Fill] = field(default_factory=list)


@runtime_checkable
class Strategy(Protocol):
    """Anything driving the engine implements this protocol.

    The four callbacks form the strategy's lifecycle:

    * ``on_start`` — once at the start of a run; useful to allocate buffers
      or fit a parameter from warm-up data.
    * ``on_bar`` — called for every bar; returns ``Order`` or ``Signal`` to
      execute on the next bar. An empty list means "do nothing".
    * ``on_fill`` — notification when one of the strategy's orders fills;
      typically used to update internal state (e.g. an entry price). The
      engine has already updated the portfolio by the time this is called.
    * ``on_end`` — once after the final bar. Useful for cleanup or logging.

    All callbacks are synchronous by design. Live trading lifts this onto
    asyncio inside the broker layer, not here.
    """

    def on_start(self, ctx: StrategyContext) -> None: ...

    def on_bar(self, ctx: StrategyContext) -> list[Order | Signal]: ...

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None: ...

    def on_end(self, ctx: StrategyContext) -> None: ...


class StrategyBase:
    """Convenience base class with no-op default callbacks.

    Subclass this when you only want to override ``on_bar``. Does not use
    ``__slots__`` so subclasses are free to declare arbitrary attributes.
    """

    def on_start(self, ctx: StrategyContext) -> None:  # noqa: B027 — intentional no-op
        return None

    def on_bar(self, ctx: StrategyContext) -> list[Order | Signal]:
        return []

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:  # noqa: B027
        del fill, ctx

    def on_end(self, ctx: StrategyContext) -> None:  # noqa: B027
        del ctx
