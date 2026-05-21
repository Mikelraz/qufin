"""
Execution models: how an order becomes a fill.

The default model fills market orders at the next bar's open with the
configured slippage; limit/stop orders are evaluated against the next
bar's high/low range. Same-bar-close execution is also available for
strategies that emit orders meant to fill at the current bar.

All models expose the same ``execute`` signature so they are swappable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from .._types import (
    AssetKind,
    BarEvent,
    CommissionModel,
    Fill,
    FixedCommission,
    NoSlippage,
    Order,
    OrderType,
    PercentSlippage,
    SlippageModel,
)


class ExecutionModel(Protocol):
    """Convert pending orders into fills using the bar(s) immediately ahead."""

    slippage: SlippageModel
    commissions: CommissionModel

    def execute(
        self,
        *,
        pending: list[Order],
        next_bars: dict[str, BarEvent],
        timestamp: datetime,
    ) -> list[Fill]: ...


def _option_underlying(asset: object) -> str:
    from ...options._types import OptionContract  # local import to avoid cycle at type-check
    if isinstance(asset, OptionContract):
        return asset.underlying
    return str(asset)


def _kind(asset: object) -> AssetKind:
    from ...options._types import OptionContract
    return AssetKind.OPTION if isinstance(asset, OptionContract) else AssetKind.EQUITY


@dataclass(slots=True)
class NextBarOpenExecution:
    """Market orders fill at the next bar's open; limit/stop checked against the bar's range.

    Limit buys fill if ``low <= limit_price``; limit sells if ``high >= limit_price``.
    Stop buys trigger when ``high >= stop_price``; stop sells when ``low <= stop_price``.
    Stop-limit orders use the same trigger and then the limit rule above.
    Unfilled limit/stop orders stay pending (the engine carries them forward).
    """

    slippage: SlippageModel = field(default_factory=NoSlippage)
    commissions: CommissionModel = field(default_factory=FixedCommission)

    def execute(
        self,
        *,
        pending: list[Order],
        next_bars: dict[str, BarEvent],
        timestamp: datetime,
    ) -> list[Fill]:
        fills: list[Fill] = []
        for order in pending:
            sym = _option_underlying(order.asset)
            bar = next_bars.get(sym)
            if bar is None:
                continue
            kind = _kind(order.asset)
            price = self._fillable_price(order, bar)
            if price is None:
                continue
            fill_price = self.slippage.adjust(ref_price=price, qty=order.qty, asset_kind=kind)
            commission = self.commissions.charge(qty=order.qty, price=fill_price, asset_kind=kind)
            fills.append(
                Fill(
                    order_id=order.client_id,
                    asset=order.asset,
                    timestamp=timestamp,
                    qty=order.qty,
                    price=fill_price,
                    commission=commission,
                    slippage=(
                        self.slippage.bps if isinstance(self.slippage, PercentSlippage) else 0.0
                    ),
                )
            )
        return fills

    @staticmethod
    def _fillable_price(order: Order, bar: BarEvent) -> float | None:
        match order.order_type:
            case OrderType.MARKET:
                return bar.open
            case OrderType.LIMIT:
                assert order.limit_price is not None
                if order.qty > 0 and bar.low <= order.limit_price:
                    return min(bar.open, order.limit_price)
                if order.qty < 0 and bar.high >= order.limit_price:
                    return max(bar.open, order.limit_price)
                return None
            case OrderType.STOP:
                assert order.stop_price is not None
                if order.qty > 0 and bar.high >= order.stop_price:
                    return max(bar.open, order.stop_price)
                if order.qty < 0 and bar.low <= order.stop_price:
                    return min(bar.open, order.stop_price)
                return None
            case OrderType.STOP_LIMIT:
                assert order.stop_price is not None
                assert order.limit_price is not None
                triggered = (
                    (order.qty > 0 and bar.high >= order.stop_price)
                    or (order.qty < 0 and bar.low <= order.stop_price)
                )
                if not triggered:
                    return None
                if order.qty > 0 and bar.low <= order.limit_price:
                    return min(max(bar.open, order.stop_price), order.limit_price)
                if order.qty < 0 and bar.high >= order.limit_price:
                    return max(min(bar.open, order.stop_price), order.limit_price)
                return None


@dataclass(slots=True)
class SameBarCloseExecution:
    """Fill market orders at the current bar's close; for parity/intraday testing only."""

    slippage: SlippageModel = field(default_factory=NoSlippage)
    commissions: CommissionModel = field(default_factory=FixedCommission)

    def execute(
        self,
        *,
        pending: list[Order],
        next_bars: dict[str, BarEvent],
        timestamp: datetime,
    ) -> list[Fill]:
        fills: list[Fill] = []
        for order in pending:
            sym = _option_underlying(order.asset)
            bar = next_bars.get(sym)
            if bar is None or order.order_type != OrderType.MARKET:
                continue
            kind = _kind(order.asset)
            fill_price = self.slippage.adjust(
                ref_price=bar.close, qty=order.qty, asset_kind=kind
            )
            commission = self.commissions.charge(qty=order.qty, price=fill_price, asset_kind=kind)
            fills.append(
                Fill(
                    order_id=order.client_id,
                    asset=order.asset,
                    timestamp=timestamp,
                    qty=order.qty,
                    price=fill_price,
                    commission=commission,
                )
            )
        return fills
