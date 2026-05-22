"""Kelly + Ledoit-Wolf covariance sizing.

Per-symbol edge is estimated from a rolling realised hit-rate ``p`` and
win/loss ratio ``b`` of the strategy's own historical signals on that
symbol. The Kelly fraction is then ``f* = (p·b − (1−p)) / b``, floored at
zero (negative edge → no position) and capped at ``kelly_fraction`` (the
half-Kelly safety multiplier).

Portfolio assembly is a small projection-style step: take the per-symbol
target weights, scale them down to satisfy the variance budget (using the
Ledoit-Wolf shrunk annualised covariance), then apply single-name and
correlation-cluster caps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from qufin.portfolio import (
    annualize_cov,
    cov_to_corr,
    ledoit_wolf_cov,
    simple_returns,
)

from .params import ConfluenceParams


@dataclass(slots=True, frozen=True)
class SymbolEdge:
    symbol: str
    p_win: float
    b_ratio: float
    n_trades: int

    def kelly(self, fraction: float) -> float:
        p, b = self.p_win, self.b_ratio
        if b <= 0.0:
            return 0.0
        f_star = (p * b - (1.0 - p)) / b
        return max(0.0, min(f_star, 1.0)) * fraction


@dataclass(slots=True)
class KellyCovarianceSizer:
    params: ConfluenceParams

    def edge(self, trade_returns: np.ndarray, symbol: str) -> SymbolEdge:
        """Compute per-symbol (p, b) from a tail of realised trade returns."""
        p = self.params
        if trade_returns.size < p.edge_min_trades:
            return SymbolEdge(symbol, p.default_edge_p, p.default_edge_b, trade_returns.size)
        wins = trade_returns[trade_returns > 0.0]
        losses = trade_returns[trade_returns < 0.0]
        if wins.size == 0 or losses.size == 0:
            return SymbolEdge(symbol, p.default_edge_p, p.default_edge_b, trade_returns.size)
        p_win = float(wins.size / trade_returns.size)
        b_ratio = float(wins.mean() / abs(losses.mean()))
        return SymbolEdge(symbol, p_win, b_ratio, trade_returns.size)

    def allocate(
        self,
        *,
        candidates: list[str],
        edges: dict[str, SymbolEdge],
        returns_panel: pl.DataFrame,
        gex_size_multiplier: float = 1.0,
    ) -> dict[str, float]:
        """Return target weights for each candidate symbol.

        ``returns_panel`` is a long-format polars frame with columns
        ``{timestamp, symbol, ret}`` covering at least
        ``cov_lookback_days`` rows per symbol.
        """
        p = self.params
        if not candidates:
            return {}

        target = {
            sym: edges.get(sym, SymbolEdge(sym, p.default_edge_p, p.default_edge_b, 0)).kelly(
                p.kelly_fraction
            )
            * gex_size_multiplier
            for sym in candidates
        }

        cov_ann = self._covariance(candidates, returns_panel)
        if cov_ann is None:
            target = self._apply_single_name_cap(target, p.max_weight_per_name)
            target = self._apply_total_cap(target, p.max_total_exposure)
            return target

        w = np.array([target[s] for s in candidates], dtype=np.float64)
        port_var = float(w @ cov_ann @ w)
        budget = p.sigma_target_annual**2
        if port_var > budget and port_var > 0.0:
            w *= np.sqrt(budget / port_var)
        target = {s: float(w[i]) for i, s in enumerate(candidates)}

        target = self._apply_single_name_cap(target, p.max_weight_per_name)
        target = self._apply_cluster_cap(target, candidates, cov_ann, p)
        target = self._apply_total_cap(target, p.max_total_exposure)
        return target

    def _covariance(
        self,
        symbols: list[str],
        returns_panel: pl.DataFrame,
    ) -> np.ndarray | None:
        p = self.params
        if returns_panel.is_empty():
            return None
        wide = (
            returns_panel.filter(pl.col("symbol").is_in(symbols))
            .pivot(values="ret", index="timestamp", on="symbol")
            .drop("timestamp")
        )
        present = [s for s in symbols if s in wide.columns]
        if len(present) < 2:
            return None
        wide = wide.select(present).drop_nulls()
        if wide.height < max(30, p.cov_lookback_days // 4):
            return None
        mat = wide.tail(p.cov_lookback_days).to_numpy().astype(np.float64, copy=False)
        cov = ledoit_wolf_cov(mat)
        cov_ann = annualize_cov(cov, p.periods_per_year)
        idx = {s: i for i, s in enumerate(present)}
        full = np.zeros((len(symbols), len(symbols)), dtype=np.float64)
        diag_fill = np.diag(cov_ann).mean() if cov_ann.size else 0.04
        for i, si in enumerate(symbols):
            for j, sj in enumerate(symbols):
                if si in idx and sj in idx:
                    full[i, j] = cov_ann[idx[si], idx[sj]]
                elif i == j:
                    full[i, j] = diag_fill
        return full

    @staticmethod
    def _apply_single_name_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
        return {s: min(w, cap) for s, w in weights.items()}

    @staticmethod
    def _apply_total_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
        total = sum(weights.values())
        if total <= cap or total <= 0.0:
            return weights
        scale = cap / total
        return {s: w * scale for s, w in weights.items()}

    @staticmethod
    def _apply_cluster_cap(
        weights: dict[str, float],
        symbols: list[str],
        cov_ann: np.ndarray,
        params: ConfluenceParams,
    ) -> dict[str, float]:
        corr = cov_to_corr(cov_ann)
        clusters = _greedy_clusters(symbols, corr, params.cluster_corr_threshold)
        for cluster in clusters:
            total = sum(weights[s] for s in cluster)
            if total > params.max_cluster_weight and total > 0.0:
                scale = params.max_cluster_weight / total
                for s in cluster:
                    weights[s] *= scale
        return weights


def _greedy_clusters(
    symbols: list[str],
    corr: np.ndarray,
    threshold: float,
) -> list[list[str]]:
    """Single-linkage clustering on the |corr| ≥ threshold graph."""
    n = len(symbols)
    visited = [False] * n
    clusters: list[list[str]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        members: list[int] = []
        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            members.append(j)
            for k in range(n):
                if not visited[k] and abs(corr[j, k]) >= threshold:
                    stack.append(k)
        clusters.append([symbols[m] for m in members])
    return clusters


def panel_returns(price_panel: pl.DataFrame) -> pl.DataFrame:
    """Long-format ``{timestamp, symbol, ret}`` from a wide price frame.

    ``price_panel`` columns: ``timestamp`` + one column per symbol of close
    prices.
    """
    symbols = [c for c in price_panel.columns if c != "timestamp"]
    if not symbols:
        return pl.DataFrame(schema={"timestamp": pl.Datetime, "symbol": pl.Utf8, "ret": pl.Float64})
    rets = simple_returns(price_panel.select(symbols))
    rets = rets.with_columns(price_panel["timestamp"])
    return rets.unpivot(
        index="timestamp",
        on=symbols,
        variable_name="symbol",
        value_name="ret",
    ).drop_nulls()
