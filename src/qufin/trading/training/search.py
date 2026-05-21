"""
Hyperparameter search engines.

``GridSearch`` runs every Cartesian combination of the supplied parameter
ranges; trivially parallelised by ``concurrent.futures.ProcessPoolExecutor``
when ``n_jobs > 1``.

Bayesian search is left for a follow-up — the scipy-based scaffold lives
here but is intentionally lightweight; users wanting full Gaussian-process
surrogates should reach for a dedicated package (``optuna``, ``skopt``).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .._types import BacktestReport
from ..engine.clock import Clock
from ..engine.engine import BacktestEngine, EngineConfig
from ..strategy.base import Strategy

StrategyFactory = Callable[[Mapping[str, Any]], Strategy]
Objective = Callable[[BacktestReport], float]


@dataclass(slots=True)
class GridSearchResult:
    """Output of a grid search.

    Attributes
    ----------
    table       polars frame with one row per param combination, containing
                each parameter plus the ``objective`` column.
    best_params Best-scoring parameter combination.
    best_score  Score at ``best_params``.
    """

    table: pl.DataFrame
    best_params: dict[str, Any]
    best_score: float


@dataclass(slots=True)
class GridSearch:
    """Brute-force Cartesian search over a parameter grid.

    Parameters
    ----------
    strategy_factory  Callable that returns a ``Strategy`` given a parameter dict.
    grid              Mapping from parameter name to the iterable of values to try.
    bars_factory      Callable that returns the dict of symbol → bar frames
                      consumed by the engine's ``Clock``. Re-invoked per point
                      so strategies can't accidentally mutate shared state.
    objective         Scalar function over a ``BacktestReport`` (higher = better).
    engine_config     Configuration passed to every engine instantiation.
    n_jobs            1 = serial; >1 = process pool.
    """

    strategy_factory: StrategyFactory
    grid: Mapping[str, Iterable[Any]]
    bars_factory: Callable[[], dict[str, pl.DataFrame]]
    objective: Objective
    engine_config: EngineConfig = field(default_factory=EngineConfig)
    n_jobs: int = 1

    def run(self) -> GridSearchResult:
        names = list(self.grid.keys())
        value_lists = [list(v) for v in self.grid.values()]
        combos = [dict(zip(names, point, strict=True)) for point in itertools.product(*value_lists)]
        if not combos:
            raise ValueError("grid is empty")
        if self.n_jobs <= 1:
            scores = [self._score(c) for c in combos]
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                scores = list(pool.map(self._score, combos))

        rows: list[dict[str, Any]] = []
        for params, score in zip(combos, scores, strict=True):
            rows.append({**params, "objective": float(score)})
        table = pl.DataFrame(rows)
        best_idx = int(scores.index(max(scores)))
        return GridSearchResult(
            table=table, best_params=combos[best_idx], best_score=float(scores[best_idx])
        )

    def _score(self, params: Mapping[str, Any]) -> float:
        strategy = self.strategy_factory(params)
        clock = Clock(bars=self.bars_factory())
        engine = BacktestEngine(strategy=strategy, clock=clock, config=self.engine_config)
        report = engine.run()
        return self.objective(report)
