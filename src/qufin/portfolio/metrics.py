from __future__ import annotations

import math

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True)
def _drawdown_series(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Peak-to-trough drawdown at each step, expressed as a fraction of peak wealth."""
    n = len(returns)
    dd = np.empty(n)
    peak = 1.0
    value = 1.0
    for i in range(n):
        value *= 1.0 + returns[i]
        if value > peak:
            peak = value
        dd[i] = (peak - value) / peak
    return dd


def max_drawdown(returns: NDArray[np.float64]) -> float:
    """Maximum peak-to-trough drawdown (positive value)."""
    return float(np.max(_drawdown_series(returns)))


def annualized_volatility(returns: NDArray[np.float64], periods_per_year: int = 252) -> float:
    """Annualized sample standard deviation of returns."""
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio. risk_free_rate is an annual rate."""
    excess = returns - risk_free_rate / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (downside-deviation denominator)."""
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return math.inf
    downside_std = math.sqrt(float(np.mean(downside**2)))
    if downside_std < 1e-12:
        return math.inf
    return float(np.mean(excess) / downside_std * math.sqrt(periods_per_year))


def calmar_ratio(
    returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio: annualized return divided by maximum drawdown."""
    mdd = max_drawdown(returns)
    if mdd < 1e-12:
        return math.inf
    compound = float(np.prod(1.0 + returns))
    ann_ret = (compound ** (periods_per_year / len(returns))) - 1.0
    return ann_ret / mdd


def historical_var(returns: NDArray[np.float64], confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at confidence level (returned as a positive loss)."""
    return float(-np.percentile(returns, (1.0 - confidence) * 100.0))


def conditional_var(returns: NDArray[np.float64], confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR) at confidence level (positive loss)."""
    threshold = np.percentile(returns, (1.0 - confidence) * 100.0)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return historical_var(returns, confidence)
    return float(-np.mean(tail))


def portfolio_metrics(
    weights: NDArray[np.float64],
    returns_matrix: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Aggregate metrics for a weighted portfolio from a (T × n) returns matrix."""
    port_returns = returns_matrix @ weights
    return {
        "annualized_return": float(
            (np.prod(1.0 + port_returns) ** (periods_per_year / len(port_returns))) - 1.0
        ),
        "annualized_volatility": annualized_volatility(port_returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(port_returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(port_returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(port_returns),
        "calmar_ratio": calmar_ratio(port_returns, periods_per_year),
        "var_95": historical_var(port_returns),
        "cvar_95": conditional_var(port_returns),
    }
