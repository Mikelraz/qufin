"""
Multi-strategy comparison.

Side-by-side metric table for any number of reports and a Diebold-Mariano
test for strategy pairs (one-sided alternative: A's returns dominate B's).
DM is delegated to ``qufin.timeseries.forecast_eval``.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from ...timeseries.forecast_eval import DMResult, diebold_mariano
from .._types import BacktestReport
from .tearsheet import tearsheet


def compare(
    reports: Mapping[str, BacktestReport], *, periods_per_year: int = 252
) -> pl.DataFrame:
    """Run ``tearsheet`` over each report and stack the summaries into a frame.

    The output frame has one row per strategy and columns for every metric
    in ``TearSheet.summary``.
    """
    rows: list[dict[str, float | str]] = []
    for name, report in reports.items():
        ts = tearsheet(report, periods_per_year=periods_per_year)
        rows.append({"strategy": name, **ts.summary})
    return pl.DataFrame(rows)


def diebold_mariano_pair(
    a: BacktestReport, b: BacktestReport, *, h: int = 1, loss: str = "squared"
) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy between two strategies.

    Uses each strategy's negative per-period return as a forecast error
    against a zero target; a strategy with consistently larger gains gets
    a lower loss. ``h`` is the forecast horizon (1 for bar-to-bar) and
    ``loss`` is forwarded to ``timeseries.forecast_eval.diebold_mariano``.
    """
    ra = a.returns()
    rb = b.returns()
    n = min(len(ra), len(rb))
    if n == 0:
        raise ValueError("both reports must contain at least one period of returns")
    return diebold_mariano(-ra[:n], -rb[:n], h=h, loss=loss)
