"""
Performance tearsheet — aggregate metrics, monthly tables, rolling Sharpe.

All metric computations reuse ``qufin.portfolio.metrics``. The tearsheet
attaches its summary dict to the source ``BacktestReport.summary`` as a
side effect so downstream callers can serialise the report directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from ...portfolio.metrics import (
    annualized_volatility,
    calmar_ratio,
    conditional_var,
    historical_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from .._types import BacktestReport


@dataclass(slots=True)
class TearSheet:
    """Structured performance report."""

    summary: dict[str, float]
    monthly_returns: pl.DataFrame
    rolling_sharpe: pl.DataFrame


def tearsheet(
    report: BacktestReport,
    *,
    periods_per_year: int = 252,
    rolling_window: int = 63,
    risk_free_rate: float = 0.0,
) -> TearSheet:
    """Compute the standard performance pack from a backtest report."""
    returns = report.returns()
    n = len(returns)
    if n == 0:
        empty = pl.DataFrame()
        return TearSheet(summary={}, monthly_returns=empty, rolling_sharpe=empty)

    total_return = float(np.prod(1.0 + returns) - 1.0)
    cagr = (
        float((1.0 + total_return) ** (periods_per_year / n) - 1.0) if n > 0 else 0.0
    )
    summary: dict[str, float] = {
        "total_return": total_return,
        "cagr": cagr,
        "annualised_vol": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar": calmar_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "var_95": historical_var(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "hit_rate": float(np.mean(returns > 0)),
        "n_bars": float(n),
    }
    if report.trades.height > 0:
        summary["n_trades"] = float(report.trades.height)
        # Profit factor: gross-win / gross-loss using returns sign as proxy.
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        summary["profit_factor"] = float(gains / losses) if losses > 0 else math.inf

    # Monthly returns table.
    equity = report.equity_curve
    monthly: pl.DataFrame = (
        equity.with_columns(
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
        )
        .group_by(["year", "month"], maintain_order=True)
        .agg(pl.col("equity").last().alias("eq_end"), pl.col("equity").first().alias("eq_start"))
        .with_columns(((pl.col("eq_end") / pl.col("eq_start")) - 1.0).alias("monthly_return"))
        .select("year", "month", "monthly_return")
    )

    # Rolling Sharpe.
    if n >= rolling_window + 1:
        rs = np.full(n, np.nan, dtype=np.float64)
        for i in range(rolling_window, n + 1):
            window = returns[i - rolling_window : i]
            rs[i - 1] = sharpe_ratio(window, risk_free_rate, periods_per_year)
        rolling = pl.DataFrame(
            {"timestamp": equity["timestamp"][1:], "rolling_sharpe": rs}
        )
    else:
        rolling = pl.DataFrame({"timestamp": [], "rolling_sharpe": []})

    report.summary = summary
    return TearSheet(summary=summary, monthly_returns=monthly, rolling_sharpe=rolling)
