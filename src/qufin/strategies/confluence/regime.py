"""Regime classification combining HMM posterior with rule-based phases.

The cash-defense layer reads only ``RegimeResult.cash_defense_active`` —
which fires when the rolling probability of *distribution* or *markdown*
exceeds the configured threshold for ``regime_persistence_bars``
consecutive bars. The HMM is refitted on a rolling window inside the
classifier so it remains causal in a walk-forward backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from qufin.wyckoff import (
    OHLCV,
    WyckoffHMMClassifier,
)

MacroPhase = Literal["Accumulation", "Markup", "Distribution", "Markdown"]
_BEAR_PHASES: frozenset[MacroPhase] = frozenset({"Distribution", "Markdown"})


@dataclass(slots=True)
class RegimeResult:
    phases: list[MacroPhase]
    p_bear: np.ndarray
    cash_defense: np.ndarray  # bool array per bar


@dataclass(slots=True)
class RegimeClassifier:
    """Rolling HMM macro-phase classifier with a cash-defense flag.

    The HMM is treated as a feature, not as truth. ``cash_defense`` is the
    operational signal the strategy consumes: True means close everything
    and stay flat until it clears.
    """

    warmup_bars: int = 250
    refit_period: int = 21
    bear_threshold: float = 0.60
    persistence_bars: int = 2
    seed: int | None = 0

    def fit_predict(self, bars: OHLCV) -> RegimeResult:
        n = bars.n_bars
        if n < self.warmup_bars:
            return RegimeResult(
                phases=["Accumulation"] * n,
                p_bear=np.zeros(n, dtype=np.float64),
                cash_defense=np.zeros(n, dtype=bool),
            )

        phases: list[MacroPhase] = ["Accumulation"] * n
        p_bear = np.zeros(n, dtype=np.float64)

        cls = WyckoffHMMClassifier(seed=self.seed)
        for end in range(self.warmup_bars, n + 1, self.refit_period):
            window = bars.data.slice(0, end)
            sub = OHLCV.from_records(window, symbol=bars.symbol)
            res = cls.fit_predict(sub)
            start = end - self.refit_period if end > self.warmup_bars else 0
            for i in range(start, end):
                label = res.labels[i] if i < len(res.labels) else "Accumulation"
                phases[i] = label  # type: ignore[assignment]
                p_bear[i] = 1.0 if label in _BEAR_PHASES else 0.0

        cash_defense = self._persistent_above(p_bear, self.bear_threshold, self.persistence_bars)
        return RegimeResult(phases=phases, p_bear=p_bear, cash_defense=cash_defense)

    @staticmethod
    def _persistent_above(values: np.ndarray, threshold: float, k: int) -> np.ndarray:
        flag = values >= threshold
        out = np.zeros_like(flag, dtype=bool)
        run = 0
        for i in range(flag.shape[0]):
            run = run + 1 if flag[i] else 0
            out[i] = run >= k
        return out


def regime_frame(result: RegimeResult, timestamps: pl.Series) -> pl.DataFrame:
    """Convenience: pack the regime result into a polars frame for joins."""
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "phase": result.phases,
            "p_bear": result.p_bear,
            "cash_defense": result.cash_defense,
        }
    )
