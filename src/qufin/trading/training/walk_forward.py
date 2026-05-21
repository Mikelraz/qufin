"""
Walk-forward cross-validation.

Splits the bar history into a sequence of rolling (or expanding)
train/test windows. Each window first runs hyperparameter search on the
train slice, then evaluates the best params on the test slice. An
optional ``embargo`` of bars between train and test prevents look-ahead
through autocorrelated returns (López de Prado, 2018).

Reuses ``timeseries.RollingBacktest`` only conceptually — the splits are
defined directly here so they can carry strategy parameters rather than
forecast residuals.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .._types import BacktestReport
from ..engine.clock import Clock
from ..engine.engine import BacktestEngine, EngineConfig
from ..strategy.base import Strategy
from .search import GridSearch, Objective

StrategyFactory = Callable[[Mapping[str, Any]], Strategy]


@dataclass(slots=True)
class WalkForwardResult:
    """Per-fold train/test report.

    Attributes
    ----------
    folds          polars frame with columns: fold_id, train_start, train_end,
                   test_start, test_end, best_params (string-encoded), train_score,
                   test_score.
    test_reports   one ``BacktestReport`` per fold, in fold order.
    """

    folds: pl.DataFrame
    test_reports: list[BacktestReport]


@dataclass(slots=True)
class WalkForwardCV:
    """Rolling or expanding walk-forward search + evaluation.

    Parameters
    ----------
    strategy_factory  Callable that returns a ``Strategy`` from a parameter dict.
    grid              Hyperparameter grid; passed to inner ``GridSearch``.
    bars              Symbol → bar frame (BAR_SCHEMA). Splits are applied
                      identically to every symbol's frame using row position.
    objective         Scalar metric (higher = better).
    train_size        Bars per train window.
    test_size         Bars per test window.
    step              Step between consecutive folds (defaults to ``test_size``,
                      yielding non-overlapping test windows).
    embargo           Bars to drop between train and test for leakage control.
    expanding         If True, the train window grows by ``step`` each fold; if
                      False, it slides.
    """

    strategy_factory: StrategyFactory
    grid: Mapping[str, Iterable[Any]]
    bars: dict[str, pl.DataFrame]
    objective: Objective
    train_size: int
    test_size: int
    step: int | None = None
    embargo: int = 0
    expanding: bool = False
    engine_config: EngineConfig = field(default_factory=EngineConfig)
    n_jobs: int = 1

    def run(self) -> WalkForwardResult:
        step = self.step if self.step is not None else self.test_size
        n = min(f.height for f in self.bars.values())
        if self.train_size + self.embargo + self.test_size > n:
            raise ValueError(
                f"train+embargo+test ({self.train_size + self.embargo + self.test_size}) "
                f"exceeds available bars ({n})"
            )

        fold_rows: list[dict[str, Any]] = []
        test_reports: list[BacktestReport] = []
        fold_id = 0
        train_start = 0
        cur_train_size = self.train_size
        while True:
            train_end = train_start + cur_train_size
            test_start = train_end + self.embargo
            test_end = test_start + self.test_size
            if test_end > n:
                break

            train_bars = {s: f.slice(train_start, cur_train_size) for s, f in self.bars.items()}
            test_bars = {
                s: f.slice(test_start, self.test_size) for s, f in self.bars.items()
            }

            inner = GridSearch(
                strategy_factory=self.strategy_factory,
                grid=self.grid,
                bars_factory=lambda b=train_bars: dict(b),
                objective=self.objective,
                engine_config=self.engine_config,
                n_jobs=self.n_jobs,
            )
            train_result = inner.run()

            strategy = self.strategy_factory(train_result.best_params)
            engine = BacktestEngine(
                strategy=strategy, clock=Clock(bars=test_bars), config=self.engine_config
            )
            test_report = engine.run()
            test_score = self.objective(test_report)

            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "best_params": str(train_result.best_params),
                    "train_score": train_result.best_score,
                    "test_score": float(test_score),
                }
            )
            test_reports.append(test_report)

            fold_id += 1
            if self.expanding:
                cur_train_size += step
            else:
                train_start += step

        return WalkForwardResult(folds=pl.DataFrame(fold_rows), test_reports=test_reports)
