"""Lookahead invariant: strategies can only see bars up to the current one."""

from __future__ import annotations

from dataclasses import dataclass, field

from qufin.trading import BacktestEngine
from qufin.trading.engine.clock import Clock
from qufin.trading.strategy.base import StrategyBase, StrategyContext


@dataclass
class LookaheadProbe(StrategyBase):
    """Records the index of the latest bar visible on each call to ``on_bar``."""

    seen_indices: list[int] = field(default_factory=list)

    def on_bar(self, ctx: StrategyContext) -> list:
        frame = ctx.history["AAA"]
        if frame.height == 0:
            return []
        self.seen_indices.append(int(frame["index"][-1]))
        return []


def test_strategy_never_sees_future_bars(synthetic_bars):
    """At step t the latest visible bar's index must be exactly t.

    Strict equality — if the engine ever leaked the next bar into ``history``
    before ``on_bar``, the recorded index would be t+1 for some t.
    """
    probe = LookaheadProbe()
    engine = BacktestEngine(strategy=probe, clock=Clock(bars=synthetic_bars))
    engine.run()
    n = synthetic_bars["AAA"].height
    assert probe.seen_indices == list(range(n))
