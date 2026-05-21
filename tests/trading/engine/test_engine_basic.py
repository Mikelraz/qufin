"""End-to-end engine smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

from qufin.trading import (
    BacktestEngine,
    Order,
    Signal,
    SignalKind,
    Strategy,
)
from qufin.trading._types import OrderType
from qufin.trading.engine.clock import Clock
from qufin.trading.strategy.base import StrategyBase, StrategyContext


@dataclass
class BuyAndHold(StrategyBase):
    """Take a 100% target weight on the first bar and hold."""

    symbol: str
    triggered: bool = False

    def on_bar(self, ctx: StrategyContext) -> list[Order | Signal]:
        if self.triggered:
            return []
        self.triggered = True
        return [Signal(asset=self.symbol, kind=SignalKind.TARGET_WEIGHT, value=1.0)]


def test_engine_runs_buy_and_hold(synthetic_bars):
    strategy = BuyAndHold(symbol="AAA")
    engine = BacktestEngine(strategy=strategy, clock=Clock(bars=synthetic_bars))
    report = engine.run()
    assert report.equity_curve.height == synthetic_bars["AAA"].height
    # After the first bar fires the signal, exactly one fill should appear.
    assert report.trades.height == 1
    # Equity must end higher than starting cash since the underlying ramps up.
    final_equity = float(report.equity_curve["equity"][-1])
    assert final_equity > 100_000.0


def test_strategy_protocol_is_runtime_checkable():
    assert isinstance(BuyAndHold(symbol="X"), Strategy)


def test_order_validation_rejects_zero_qty():
    import pytest

    with pytest.raises(ValueError):
        Order(asset="AAA", qty=0.0, order_type=OrderType.MARKET)
