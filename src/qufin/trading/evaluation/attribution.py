"""
Per-symbol / per-trade PnL attribution.

Operates on a ``BacktestReport.trades`` frame. The report's ``asset``
column is a stringified key (``SYMBOL`` for equities, an option label for
contracts); attribution groups on it directly.
"""

from __future__ import annotations

import polars as pl

from .._types import BacktestReport


def per_symbol_pnl(report: BacktestReport) -> pl.DataFrame:
    """Aggregate trade-level PnL by asset.

    PnL per trade is approximated as ``-qty * price - commission`` (cash flow).
    Aggregating over an asset's full history yields realised PnL plus the
    closing notional of any open position.
    """
    if report.trades.height == 0:
        return pl.DataFrame(
            schema={
                "asset": pl.Utf8(),
                "n_trades": pl.Int64(),
                "gross_pnl": pl.Float64(),
                "commission": pl.Float64(),
                "net_pnl": pl.Float64(),
            }
        )
    return (
        report.trades.with_columns(
            (-(pl.col("qty") * pl.col("price"))).alias("gross_pnl"),
        )
        .group_by("asset", maintain_order=True)
        .agg(
            pl.len().alias("n_trades"),
            pl.col("gross_pnl").sum().alias("gross_pnl"),
            pl.col("commission").sum().alias("commission"),
        )
        .with_columns((pl.col("gross_pnl") - pl.col("commission")).alias("net_pnl"))
    )
