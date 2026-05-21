"""Tearsheet plumbing: metrics populate and align with portfolio.metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from qufin.portfolio.metrics import sharpe_ratio
from qufin.trading import BacktestReport
from qufin.trading.evaluation import tearsheet


def test_tearsheet_summary_matches_portfolio_metrics():
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0008, scale=0.01, size=252)
    equity = 100_000.0 * np.cumprod(1.0 + returns)
    equity = np.insert(equity, 0, 100_000.0)
    start = datetime(2024, 1, 2, tzinfo=UTC)
    ts = [start + timedelta(days=i) for i in range(len(equity))]
    eq_curve = pl.DataFrame({
        "timestamp": ts,
        "cash": [0.0] * len(equity),
        "equity": equity,
        "buying_power": [0.0] * len(equity),
        "margin_used": [0.0] * len(equity),
        "day_pnl": [0.0] * len(equity),
        "total_pnl": [0.0] * len(equity),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", time_zone="UTC")))

    report = BacktestReport(equity_curve=eq_curve, trades=pl.DataFrame())
    ts_report = tearsheet(report, periods_per_year=252)

    # Sharpe match within numerical noise.
    expected = sharpe_ratio(report.returns(), periods_per_year=252)
    assert abs(ts_report.summary["sharpe"] - expected) < 1e-9
    # Summary side effect: report's own summary mirrors the tearsheet's.
    assert report.summary["sharpe"] == ts_report.summary["sharpe"]
