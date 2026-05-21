"""
Portfolio state: cash, positions, equity curve.

Mark-to-market is driven by the engine — the portfolio receives marks via
``mark_to_market`` rather than computing them itself, so equities and
options can use different pricing paths (last close vs. Black-Scholes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ...options._types import OptionContract
from .._types import (
    AccountSnapshot,
    Fill,
    Position,
    SymbolOrContract,
)


@dataclass(slots=True)
class Portfolio:
    """Mutable portfolio state.

    Attributes
    ----------
    starting_cash   Initial cash balance.
    cash            Current cash balance.
    positions       Open positions keyed by asset.
    history         Per-bar snapshots (appended via ``snapshot``).
    """

    starting_cash: float
    cash: float = 0.0
    positions: dict[SymbolOrContract, Position] = field(default_factory=dict)
    history: list[AccountSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = self.starting_cash

    # ------------------------------------------------------------------
    # Fill processing

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and the relevant position by a single fill.

        Realised PnL crystallises on closing or flipping trades. Average
        cost is reset when the position flips sign or returns to zero.
        """
        multiplier = 100.0 if isinstance(fill.asset, OptionContract) else 1.0
        notional = fill.qty * fill.price * multiplier
        self.cash -= notional
        self.cash -= fill.commission

        pos = self.positions.get(fill.asset)
        if pos is None:
            self.positions[fill.asset] = Position(
                asset=fill.asset,
                qty=fill.qty,
                avg_price=fill.price,
                last_mark=fill.price,
                multiplier=multiplier,
            )
            return

        new_qty = pos.qty + fill.qty
        same_sign = (pos.qty > 0 and fill.qty > 0) or (pos.qty < 0 and fill.qty < 0)
        if same_sign:
            # Adding to existing position — weighted-average cost.
            pos.avg_price = (pos.avg_price * pos.qty + fill.price * fill.qty) / new_qty
            pos.qty = new_qty
        else:
            # Reducing or flipping.
            closed_qty = min(abs(fill.qty), abs(pos.qty))
            sign_pos = 1.0 if pos.qty > 0 else -1.0
            realised = sign_pos * closed_qty * (fill.price - pos.avg_price) * pos.multiplier
            pos.realised_pnl += realised
            pos.qty = new_qty
            if new_qty == 0.0:
                pos.avg_price = 0.0
            elif (new_qty > 0) != (pos.qty - fill.qty > 0):
                # Flipped through zero: cost basis = the fill price.
                pos.avg_price = fill.price

        if pos.qty == 0.0:
            # Keep the position around so callers can inspect realised PnL,
            # but mark it flat. Engines should drop flats from sizing.
            pos.avg_price = 0.0
            pos.last_mark = fill.price

    # ------------------------------------------------------------------
    # Mark-to-market

    def mark_to_market(self, marks: dict[SymbolOrContract, float]) -> None:
        """Update each position's unrealised PnL and last mark."""
        for asset, mark in marks.items():
            pos = self.positions.get(asset)
            if pos is None or pos.qty == 0.0:
                if pos is not None:
                    pos.last_mark = mark
                continue
            pos.last_mark = mark
            pos.unrealised_pnl = (mark - pos.avg_price) * pos.qty * pos.multiplier

    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values() if p.qty != 0.0)

    def total_pnl(self) -> float:
        return self.equity() - self.starting_cash

    # ------------------------------------------------------------------
    # Snapshots

    def snapshot(self, *, timestamp: datetime, prev_equity: float | None = None) -> AccountSnapshot:
        eq = self.equity()
        day_pnl = 0.0 if prev_equity is None else eq - prev_equity
        snap = AccountSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            equity=eq,
            buying_power=max(self.cash, 0.0),
            margin_used=0.0,
            day_pnl=day_pnl,
            total_pnl=self.total_pnl(),
        )
        self.history.append(snap)
        return snap
