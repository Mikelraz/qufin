"""
Objective functions used by hyperparameter search.

Every objective takes a ``BacktestReport`` and returns a scalar. By
convention higher is better — searches maximise. Wrappers around
``qufin.portfolio.metrics`` so we don't duplicate the maths.
"""

from __future__ import annotations

from ...portfolio.metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from .._types import BacktestReport


def sharpe_objective(report: BacktestReport, *, periods_per_year: int = 252) -> float:
    """Annualised Sharpe of the report's equity returns."""
    returns = report.returns()
    if len(returns) < 2:
        return 0.0
    return sharpe_ratio(returns, periods_per_year=periods_per_year)


def sortino_objective(report: BacktestReport, *, periods_per_year: int = 252) -> float:
    """Annualised Sortino."""
    returns = report.returns()
    if len(returns) < 2:
        return 0.0
    return sortino_ratio(returns, periods_per_year=periods_per_year)


def calmar_objective(report: BacktestReport, *, periods_per_year: int = 252) -> float:
    """Calmar ratio (annualised return / max drawdown)."""
    returns = report.returns()
    if len(returns) < 2:
        return 0.0
    return calmar_ratio(returns, periods_per_year=periods_per_year)


def penalised_drawdown_objective(
    report: BacktestReport, *, lambda_dd: float = 5.0, periods_per_year: int = 252
) -> float:
    """Sharpe minus ``lambda_dd × max_drawdown``.

    The penalty term lets searches trade off Sharpe against drawdown
    tolerance with a single dial — increase ``lambda_dd`` to favour
    shallower drawdowns at the cost of raw Sharpe.
    """
    returns = report.returns()
    if len(returns) < 2:
        return 0.0
    return sharpe_ratio(returns, periods_per_year=periods_per_year) - lambda_dd * max_drawdown(
        returns
    )
