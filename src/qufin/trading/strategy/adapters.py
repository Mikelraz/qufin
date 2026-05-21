"""
Adapters that wrap existing ``qufin.strategies`` models in the
``Strategy`` protocol so they run on the new engine.

For this first iteration we ship a single adapter for
``MeanReversionStrategy`` as a worked example — its online ``step(price)``
API makes the adapter trivial. The other strategies
(``GARCHVolTargetStrategy``, ``CointegrationPairsStrategy``,
``RegimeSwitchingStrategy``) follow the same template: instantiate, reset
on ``on_start``, call the model's per-bar update inside ``on_bar``, and
translate its output into a ``Signal`` with ``SignalKind.TARGET_WEIGHT``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...strategies.mean_reversion import MeanReversionStrategy, StrategyParams
from .._types import Fill, Order, Signal, SignalKind
from .base import StrategyBase, StrategyContext


@dataclass
class MeanReversionAdapter(StrategyBase):
    """Run ``qufin.strategies.MeanReversionStrategy`` on the new engine.

    The wrapped model's online ``step(price)`` is fed the latest close at
    every bar and its ``signal`` field becomes a target weight in
    {-1, 0, +1}.
    """

    symbol: str
    params: StrategyParams
    _model: MeanReversionStrategy | None = field(default=None, init=False, repr=False)

    def on_start(self, ctx: StrategyContext) -> None:
        del ctx
        self._model = MeanReversionStrategy(params=self.params)
        self._model.reset()

    def on_bar(self, ctx: StrategyContext) -> list[Order | Signal]:
        frame = ctx.history.get(self.symbol)
        if frame is None or frame.height == 0 or self._model is None:
            return []
        latest_close = float(frame["close"][-1])
        state = self._model.step(latest_close)
        signal = float(state["signal"])
        return [Signal(asset=self.symbol, kind=SignalKind.TARGET_WEIGHT, value=signal)]

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        del fill, ctx
