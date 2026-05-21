"""
Strategy adapter for a fitted sklearn estimator.

``MLSignalStrategy`` keeps a feature set, a fitted estimator, and a
``thresholds`` policy that translates predictions into target weights. The
strategy is engine-side: it expects the estimator to already be fit (e.g.
via ``walk_forward_splits``) before the engine runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._types import Fill, Order, Signal, SignalKind
from ...strategy.base import StrategyBase, StrategyContext
from .features import FeatureSet


@dataclass(slots=True)
class ThresholdPolicy:
    """Maps a scalar prediction to a target weight in [-1, 1].

    A classifier with class ``+1 / 0 / -1`` is mapped 1:1. For a regressor
    (return forecast), positions are taken when ``|pred| > entry`` and
    closed when ``|pred| < exit``.
    """

    entry: float = 0.0005
    exit: float = 0.0
    max_weight: float = 1.0

    def to_weight(self, pred: float, prev: float) -> float:
        if abs(pred) < self.exit:
            return 0.0
        if abs(pred) >= self.entry:
            return self.max_weight if pred > 0 else -self.max_weight
        return prev


@dataclass
class MLSignalStrategy(StrategyBase):
    """Wrap a fitted sklearn estimator into the ``Strategy`` protocol.

    Parameters
    ----------
    symbol      Underlying symbol to trade.
    estimator   Fitted sklearn estimator with ``predict`` (regressor or
                ``predict_proba``-less classifier emitting -1/0/+1).
    features    ``FeatureSet`` matching what the estimator was trained on.
    policy      Threshold policy turning predictions into target weights.
    warmup      Minimum number of bars before any prediction is emitted.
    """

    symbol: str
    estimator: Any
    features: FeatureSet
    policy: ThresholdPolicy = field(default_factory=ThresholdPolicy)
    warmup: int = 50
    _prev_weight: float = field(default=0.0, init=False, repr=False)

    def on_bar(self, ctx: StrategyContext) -> list[Order | Signal]:
        frame = ctx.history.get(self.symbol)
        if frame is None or frame.height < self.warmup:
            return []
        X = self.features.transform(frame)
        if X.shape[0] == 0 or np.isnan(X[-1]).any():
            return []
        pred = float(np.asarray(self.estimator.predict(X[-1:].reshape(1, -1)))[0])
        weight = self.policy.to_weight(pred, self._prev_weight)
        self._prev_weight = weight
        return [Signal(asset=self.symbol, kind=SignalKind.TARGET_WEIGHT, value=weight)]

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        del fill, ctx
