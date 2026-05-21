"""
Event-driven backtest engine.

Drives a ``Strategy`` against a ``Clock`` of historical bars, routes the
resulting orders through an ``ExecutionModel``, and accumulates fills into
a ``Portfolio``. Outputs a ``BacktestReport`` that downstream evaluation
consumes.

Look-ahead invariant
--------------------
At every step ``t`` the strategy sees ``history[:t+1]`` (the current bar
inclusive). Orders emitted at step ``t`` are queued and only filled at
step ``t+1`` by ``ExecutionModel.execute(next_bars=bars[t+1], …)``. Thus a
strategy cannot peek at price data beyond what it has already seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .._types import (
    AccountSnapshot,
    BacktestReport,
    BarEvent,
    Fill,
    Order,
    Signal,
    SignalKind,
    SymbolOrContract,
    new_order_id,
)
from ..strategy.base import Strategy, StrategyContext
from .clock import Clock
from .execution import ExecutionModel, NextBarOpenExecution
from .options_engine import OptionsEngine
from .portfolio import Portfolio


@dataclass(slots=True)
class EngineConfig:
    starting_cash: float = 100_000.0
    history_window: int = 0  # 0 = unlimited; otherwise rolling window length
    record_positions_eod: bool = False


@dataclass(slots=True)
class BacktestEngine:
    """Bar-driven backtest engine."""

    strategy: Strategy
    clock: Clock
    execution: ExecutionModel = field(default_factory=NextBarOpenExecution)
    options_engine: OptionsEngine | None = None
    config: EngineConfig = field(default_factory=EngineConfig)

    def run(self) -> BacktestReport:
        portfolio = Portfolio(starting_cash=self.config.starting_cash)
        pending_orders: list[Order] = []
        history: dict[str, list[dict[str, object]]] = {s: [] for s in self.clock.bars}
        trades: list[dict[str, object]] = []
        prev_equity: float | None = None
        last_marks: dict[SymbolOrContract, float] = {}

        steps = list(self.clock.iter_steps())
        if not steps:
            return BacktestReport(
                equity_curve=pl.DataFrame(schema=_EQUITY_SCHEMA),
                trades=pl.DataFrame(schema=_TRADE_SCHEMA),
            )

        # on_start with the very first bar as context
        first_ts, first_bars = steps[0]
        any_bar = next(iter(first_bars.values())) if first_bars else None
        if any_bar is None:
            raise RuntimeError("Clock yielded an empty first step")
        bootstrap_ctx = StrategyContext(
            account=portfolio.snapshot(timestamp=first_ts),
            positions=portfolio.positions,
            history={s: pl.DataFrame(schema=_BAR_RECORD_SCHEMA) for s in self.clock.bars},
            bar=any_bar,
        )
        # Drop the bootstrap snapshot — the first real step records its own.
        portfolio.history.pop()
        self.strategy.on_start(bootstrap_ctx)

        for ts, step_bars in steps:
            # 1. Extend history with this step's bars.
            for sym, bar in step_bars.items():
                history[sym].append(_bar_to_row(bar))
                last_marks[sym] = bar.close

            # 2. Fill any orders from the previous step using *this* step's bars.
            new_fills: list[Fill] = []
            if pending_orders:
                new_fills = self.execution.execute(
                    pending=pending_orders, next_bars=step_bars, timestamp=ts
                )
                pending_orders = [
                    o for o in pending_orders if _still_open(o, new_fills)
                ]
            for fill in new_fills:
                portfolio.apply_fill(fill)
                trades.append(_fill_to_row(fill))

            # 3. Mark to market.
            marks = dict(last_marks)
            if self.options_engine is not None:
                opt_marks = self.options_engine.mark_options(
                    timestamp=ts,
                    positions=portfolio.positions,
                    equity_marks=last_marks,
                )
                marks.update(opt_marks)
            portfolio.mark_to_market(marks)

            # 4. Snapshot.
            snap = portfolio.snapshot(timestamp=ts, prev_equity=prev_equity)
            prev_equity = snap.equity

            # 5. Notify strategy of fills.
            ctx = self._make_context(snap, portfolio, history, step_bars, new_fills)
            for fill in new_fills:
                self.strategy.on_fill(fill, ctx)

            # 6. Ask for new orders / signals.
            intents = self.strategy.on_bar(ctx) if step_bars else []
            for intent in intents:
                order = self._intent_to_order(intent, snap, portfolio, last_marks)
                if order is not None:
                    pending_orders.append(order)

        # Finalise.
        end_ctx = self._make_context(
            portfolio.history[-1] if portfolio.history else bootstrap_ctx.account,
            portfolio,
            history,
            {},
            [],
        )
        self.strategy.on_end(end_ctx)

        equity_curve = pl.DataFrame(
            [s.to_row() for s in portfolio.history], schema=_EQUITY_SCHEMA
        )
        trades_df = pl.DataFrame(trades, schema=_TRADE_SCHEMA) if trades else pl.DataFrame(
            schema=_TRADE_SCHEMA
        )
        return BacktestReport(equity_curve=equity_curve, trades=trades_df)

    # ------------------------------------------------------------------

    def _make_context(
        self,
        snap: AccountSnapshot,
        portfolio: Portfolio,
        history: dict[str, list[dict[str, object]]],
        step_bars: dict[str, BarEvent],
        fills: list[Fill],
    ) -> StrategyContext:
        any_bar = next(iter(step_bars.values()), None) if step_bars else None
        if any_bar is None and portfolio.history:
            # Synthesise a placeholder bar so on_end has something to read.
            placeholder_sym = next(iter(self.clock.bars))
            any_bar = BarEvent(
                symbol=placeholder_sym,
                timestamp=snap.timestamp,
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0.0,
                index=-1,
            )
        assert any_bar is not None
        window = self.config.history_window
        history_frames: dict[str, pl.DataFrame] = {}
        for sym, rows in history.items():
            if not rows:
                history_frames[sym] = pl.DataFrame(schema=_BAR_RECORD_SCHEMA)
                continue
            sliced = rows[-window:] if window > 0 else rows
            history_frames[sym] = pl.DataFrame(sliced, schema=_BAR_RECORD_SCHEMA)
        return StrategyContext(
            account=snap,
            positions=portfolio.positions,
            history=history_frames,
            bar=any_bar,
            fills_since_last_bar=fills,
        )

    def _intent_to_order(
        self,
        intent: Order | Signal,
        snap: AccountSnapshot,
        portfolio: Portfolio,
        last_marks: dict[SymbolOrContract, float],
    ) -> Order | None:
        if isinstance(intent, Order):
            if intent.client_id == "":
                return intent.with_id(new_order_id())
            return intent
        # Signal → Order
        match intent.kind:
            case SignalKind.ORDER:
                assert intent.order is not None
                if intent.order.client_id == "":
                    return intent.order.with_id(new_order_id())
                return intent.order
            case SignalKind.TARGET_QTY:
                current = portfolio.positions.get(intent.asset)
                cur_qty = current.qty if current is not None else 0.0
                delta = intent.value - cur_qty
                if delta == 0.0:
                    return None
                return Order(asset=intent.asset, qty=delta, tag=intent.tag).with_id(
                    new_order_id()
                )
            case SignalKind.TARGET_WEIGHT:
                mark = last_marks.get(intent.asset)
                if mark is None or mark <= 0.0:
                    return None
                multiplier = 100.0 if not isinstance(intent.asset, str) else 1.0
                target_notional = intent.value * snap.equity
                target_qty = target_notional / (mark * multiplier)
                current = portfolio.positions.get(intent.asset)
                cur_qty = current.qty if current is not None else 0.0
                delta = target_qty - cur_qty
                if abs(delta) < 1e-9:
                    return None
                return Order(asset=intent.asset, qty=delta, tag=intent.tag).with_id(
                    new_order_id()
                )


# ---------------------------------------------------------------------------
# Schemas

_BAR_RECORD_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ns", time_zone="UTC"),
    "symbol": pl.Utf8(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "index": pl.Int64(),
}

_EQUITY_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ns", time_zone="UTC"),
    "cash": pl.Float64(),
    "equity": pl.Float64(),
    "buying_power": pl.Float64(),
    "margin_used": pl.Float64(),
    "day_pnl": pl.Float64(),
    "total_pnl": pl.Float64(),
}

_TRADE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ns", time_zone="UTC"),
    "asset": pl.Utf8(),
    "qty": pl.Float64(),
    "price": pl.Float64(),
    "commission": pl.Float64(),
    "slippage": pl.Float64(),
    "tag": pl.Utf8(),
    "order_id": pl.Utf8(),
}


def _bar_to_row(bar: BarEvent) -> dict[str, object]:
    return {
        "timestamp": bar.timestamp,
        "symbol": bar.symbol,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "index": bar.index,
    }


def _fill_to_row(fill: Fill) -> dict[str, object]:
    from ...options._types import OptionContract
    if isinstance(fill.asset, OptionContract):
        c = fill.asset
        label = f"{c.underlying}-{c.expiry}-{c.strike}{c.option_type}"
    else:
        label = str(fill.asset)
    return {
        "timestamp": fill.timestamp,
        "asset": label,
        "qty": fill.qty,
        "price": fill.price,
        "commission": fill.commission,
        "slippage": fill.slippage,
        "tag": "",
        "order_id": fill.order_id,
    }


def _still_open(order: Order, fills: list[Fill]) -> bool:
    """True if no fill has consumed this order's full quantity yet.

    Today the execution models either fill the full quantity or leave the
    order pending — so this collapses to "no matching fill found". The
    function exists so partial fills can be added without restructuring.
    """
    return not any(f.order_id == order.client_id for f in fills)
