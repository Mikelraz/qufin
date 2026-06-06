"""
Cointegration pair screening.

Scan a universe of price series for cointegrated pairs — the candidate
selection step that precedes any pairs / statistical-arbitrage strategy.  Wraps
the formal tests in :mod:`qufin.timeseries.cointegration` and ranks the
surviving pairs by significance, reporting the hedge ratio and mean-reversion
half-life of each spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
import polars as pl

from ..models.spread import half_life, spread
from ..timeseries.cointegration import engle_granger, johansen

ScreenMethod = Literal["engle_granger", "johansen"]


@dataclass(slots=True, frozen=True)
class PairScreenResult:
    """One screened, cointegrated pair (``y`` regressed on / hedged with ``x``)."""

    y: str
    x: str
    beta: float
    alpha: float
    p_value: float
    statistic: float
    half_life: float
    method: str

    def __str__(self) -> str:
        return (
            f"{self.y}~{self.x}: β={self.beta:.4f}, p={self.p_value:.4f}, "
            f"half-life={self.half_life:.1f}"
        )


def _as_panel(prices: Any, names: list[str] | None) -> tuple[np.ndarray, list[str]]:
    """Return a ``(T, N)`` float64 panel and matching column names."""
    if isinstance(prices, pl.DataFrame):
        return prices.to_numpy().astype(np.float64), list(prices.columns)
    arr = np.ascontiguousarray(np.asarray(prices, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) panel, got shape {arr.shape}.")
    cols = names if names is not None else [f"asset_{i}" for i in range(arr.shape[1])]
    if len(cols) != arr.shape[1]:
        raise ValueError(f"names has {len(cols)} entries but panel has {arr.shape[1]} columns.")
    return arr, cols


def _screen_eg(
    yv: np.ndarray, xv: np.ndarray, ny: str, nx: str, alpha: float
) -> PairScreenResult | None:
    """Engle-Granger in both directions; keep the more significant orientation."""
    best: PairScreenResult | None = None
    for dep, indep, dn, indn in ((yv, xv, ny, nx), (xv, yv, nx, ny)):
        res = engle_granger(dep, indep, trend="c")
        if best is None or res.p_value < best.p_value:
            beta = float(res.beta[0])
            sp = spread(dep, indep, beta, res.alpha)
            best = PairScreenResult(
                y=dn,
                x=indn,
                beta=beta,
                alpha=float(res.alpha),
                p_value=float(res.p_value),
                statistic=float(res.adf_stat),
                half_life=half_life(sp),
                method="engle_granger",
            )
    # The loop always runs, so ``best`` is assigned; gate on significance.
    return best if best.p_value <= alpha else None


def _screen_johansen(
    yv: np.ndarray, xv: np.ndarray, ny: str, nx: str, alpha: float
) -> PairScreenResult | None:
    res = johansen(np.column_stack([yv, xv]), alpha=alpha)
    if res.rank_trace < 1:
        return None
    vec = res.eigenvectors[:, 0]
    if abs(vec[0]) < 1e-12:
        return None
    beta = float(-vec[1] / vec[0])
    sp = spread(yv, xv, beta, 0.0)
    return PairScreenResult(
        y=ny,
        x=nx,
        beta=beta,
        alpha=0.0,
        p_value=float("nan"),  # Johansen reports a statistic vs critical value, not a p-value
        statistic=float(res.trace_stats[0]),
        half_life=half_life(sp),
        method="johansen",
    )


def screen_pairs(
    prices: Any,
    *,
    names: list[str] | None = None,
    method: ScreenMethod = "engle_granger",
    alpha: float = 0.05,
    use_log: bool = True,
    max_half_life: float | None = None,
) -> list[PairScreenResult]:
    """
    Screen all asset pairs in a price panel for cointegration.

    Parameters
    ----------
    prices         Price panel: a ``pl.DataFrame`` (columns are asset names) or
                   a ``(T, N)`` array (pass ``names`` for labels).
    names          Column labels when ``prices`` is an array.
    method         ``"engle_granger"`` (tests both orientations, p-value ranked)
                   or ``"johansen"`` (trace-statistic ranked).
    alpha          Significance level for the cointegration gate.
    use_log        Run the tests on log prices (recommended for equities/FX).
    max_half_life  If set, drop pairs whose spread half-life exceeds this (too
                   slow to trade) or is non-positive / infinite.

    Returns
    -------
    list[PairScreenResult]
        Cointegrated pairs, most significant first.
    """
    panel, cols = _as_panel(prices, names)
    if panel.shape[1] < 2:
        raise ValueError("need at least 2 assets to screen.")
    if use_log:
        if np.any(panel <= 0.0):
            raise ValueError("all prices must be > 0 when use_log is True.")
        panel = np.log(panel)

    out: list[PairScreenResult] = []
    for i, j in combinations(range(panel.shape[1]), 2):
        yv, xv = panel[:, i], panel[:, j]
        if method == "engle_granger":
            res = _screen_eg(yv, xv, cols[i], cols[j], alpha)
        else:
            res = _screen_johansen(yv, xv, cols[i], cols[j], alpha)
        if res is None:
            continue
        if max_half_life is not None and not (0.0 < res.half_life <= max_half_life):
            continue
        out.append(res)

    # Engle-Granger: ascending p-value.  Johansen: descending trace statistic.
    if method == "engle_granger":
        out.sort(key=lambda r: r.p_value)
    else:
        out.sort(key=lambda r: -r.statistic)
    return out
