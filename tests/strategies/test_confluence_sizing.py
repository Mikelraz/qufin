"""Unit tests for KellyCovarianceSizer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from qufin.strategies.confluence import (
    ConfluenceParams,
    KellyCovarianceSizer,
    SymbolEdge,
)


def _synthetic_returns_panel(
    symbols: list[str],
    *,
    n_days: int = 400,
    block_corr: float = 0.85,
) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    factor = rng.normal(scale=0.01, size=n_days)
    rows = []
    start = datetime(2020, 1, 2, tzinfo=UTC)
    ts = [start + timedelta(days=i) for i in range(n_days)]
    for s in symbols:
        idio = rng.normal(scale=0.005, size=n_days)
        r = block_corr * factor + np.sqrt(1 - block_corr**2) * idio
        rows.append(pl.DataFrame({"timestamp": ts, "symbol": [s] * n_days, "ret": r}))
    return pl.concat(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", time_zone="UTC"))
    )


def test_kelly_edge_returns_default_when_few_trades() -> None:
    p = ConfluenceParams()
    sizer = KellyCovarianceSizer(p)
    edge = sizer.edge(np.array([], dtype=np.float64), "SPY")
    assert edge.p_win == p.default_edge_p
    assert edge.b_ratio == p.default_edge_b


def test_kelly_edge_computes_from_returns() -> None:
    p = ConfluenceParams(edge_min_trades=5)
    sizer = KellyCovarianceSizer(p)
    rets = np.array([0.02, 0.03, -0.01, 0.04, -0.015, 0.02], dtype=np.float64)
    edge = sizer.edge(rets, "SPY")
    assert 0.0 < edge.p_win < 1.0
    assert edge.b_ratio > 0.0


def test_sizer_respects_single_name_cap() -> None:
    p = ConfluenceParams(max_weight_per_name=0.04)
    sizer = KellyCovarianceSizer(p)
    symbols = ["A", "B", "C"]
    panel = _synthetic_returns_panel(symbols)
    edges = {s: SymbolEdge(s, 0.7, 2.0, 50) for s in symbols}
    weights = sizer.allocate(candidates=symbols, edges=edges, returns_panel=panel)
    for w in weights.values():
        assert w <= p.max_weight_per_name + 1e-9


def test_sizer_respects_total_exposure_cap() -> None:
    p = ConfluenceParams(max_total_exposure=0.10, max_weight_per_name=1.0)
    sizer = KellyCovarianceSizer(p)
    symbols = ["A", "B", "C", "D"]
    panel = _synthetic_returns_panel(symbols)
    edges = {s: SymbolEdge(s, 0.7, 2.0, 50) for s in symbols}
    weights = sizer.allocate(candidates=symbols, edges=edges, returns_panel=panel)
    assert sum(weights.values()) <= p.max_total_exposure + 1e-9


def test_sizer_respects_cluster_cap() -> None:
    p = ConfluenceParams(
        max_weight_per_name=1.0,
        max_cluster_weight=0.10,
        cluster_corr_threshold=0.5,
        max_total_exposure=1.0,
    )
    sizer = KellyCovarianceSizer(p)
    symbols = ["A", "B", "C"]
    panel = _synthetic_returns_panel(symbols, block_corr=0.95)
    edges = {s: SymbolEdge(s, 0.7, 2.0, 50) for s in symbols}
    weights = sizer.allocate(candidates=symbols, edges=edges, returns_panel=panel)
    assert sum(weights.values()) <= p.max_cluster_weight + 1e-9


def test_sizer_gex_multiplier_halves_exposure() -> None:
    p = ConfluenceParams(
        max_weight_per_name=1.0,
        max_total_exposure=1.0,
        sigma_target_annual=10.0,
        max_cluster_weight=1.0,
    )
    sizer = KellyCovarianceSizer(p)
    symbols = ["A", "B"]
    panel = _synthetic_returns_panel(symbols)
    edges = {s: SymbolEdge(s, 0.7, 2.0, 50) for s in symbols}
    full = sizer.allocate(
        candidates=symbols, edges=edges, returns_panel=panel, gex_size_multiplier=1.0
    )
    halved = sizer.allocate(
        candidates=symbols, edges=edges, returns_panel=panel, gex_size_multiplier=0.5
    )
    assert sum(halved.values()) <= sum(full.values()) + 1e-9
