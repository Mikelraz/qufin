"""Risk and performance metrics computed on numpy return arrays.

All functions expect a 1-D array of *period* returns (e.g. daily simple
returns) unless stated otherwise.  Annualisation is applied internally using
the ``periods_per_year`` parameter.

Unit conventions
----------------
- Returns are dimensionless decimals, not percent (0.01 = 1 %).
- ``risk_free_rate`` is always an *annual* rate; functions scale it to
  per-period before arithmetic.
- VaR and CVaR are returned as *positive* loss magnitudes.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True)
def _drawdown_series(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute peak-to-trough drawdown at each time step.

    Internally tracks the running peak wealth level so only a single pass
    over the data is needed — O(T) time and O(T) space.

    Args:
        returns: 1-D array of period returns.

    Returns:
        Array of the same length where element *t* is the fraction of peak
        wealth lost as of period *t*: ``(peak_t - value_t) / peak_t``.
    """
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
    """Maximum peak-to-trough drawdown over the full history.

    Computed as the largest fraction of peak wealth ever lost before a new
    high is reached::

        MDD = max_t  (peak_t - value_t) / peak_t

    where ``peak_t = max_{s <= t} wealth_s`` and ``wealth_t = prod(1 + r_s)``.

    Args:
        returns: 1-D array of period returns.

    Returns:
        Maximum drawdown as a positive decimal (0.20 = 20 % drawdown).
    """
    return float(np.max(_drawdown_series(returns)))


def annualized_volatility(returns: NDArray[np.float64], periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns (sample, ddof=1).

    Args:
        returns: 1-D array of period returns.
        periods_per_year: 252 for daily, 52 for weekly, 12 for monthly.

    Returns:
        Annualized volatility as a decimal.
    """
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio using the sample standard deviation.

    Defined as::

        Sharpe = (E[r - r_f/T]) / std(r - r_f/T)  *  sqrt(T)

    where ``T = periods_per_year``.  Returns 0.0 when the standard deviation
    of excess returns is effectively zero (constant return stream).

    Args:
        returns: 1-D array of period returns.
        risk_free_rate: Annual risk-free rate (e.g. 0.04 for 4 %).
        periods_per_year: Calendar periods per year.

    Returns:
        Annualized Sharpe ratio (dimensionless).
    """
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
    """Annualized Sortino ratio using only the downside deviation.

    Unlike the Sharpe ratio, only negative excess returns contribute to the
    denominator::

        Sortino = E[excess] / downside_std  *  sqrt(T)
        downside_std = sqrt(E[min(excess, 0)^2])

    Returns ``math.inf`` when there are no negative excess return periods.

    Args:
        returns: 1-D array of period returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Calendar periods per year.

    Returns:
        Annualized Sortino ratio (dimensionless).
    """
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
    """Calmar ratio: geometric annualized return divided by maximum drawdown.

    A higher Calmar ratio indicates better return per unit of drawdown risk.
    Returns ``math.inf`` when maximum drawdown is effectively zero.

    Args:
        returns: 1-D array of period returns.
        periods_per_year: Calendar periods per year.

    Returns:
        Calmar ratio (dimensionless).
    """
    mdd = max_drawdown(returns)
    if mdd < 1e-12:
        return math.inf
    compound = float(np.prod(1.0 + returns))
    ann_ret = (compound ** (periods_per_year / len(returns))) - 1.0
    return ann_ret / mdd


def historical_var(returns: NDArray[np.float64], confidence: float = 0.95) -> float:
    """Historical (non-parametric) Value-at-Risk at the given confidence level.

    VaR is the loss not exceeded with probability ``confidence``::

        VaR_alpha = -quantile_{1-alpha}(returns)

    Reported as a positive number (e.g. 0.02 = a 2 % loss threshold).

    Args:
        returns: 1-D array of period returns.
        confidence: Confidence level, e.g. 0.95 for 95 % VaR.

    Returns:
        VaR as a positive decimal loss.
    """
    return float(-np.percentile(returns, (1.0 - confidence) * 100.0))


def conditional_var(returns: NDArray[np.float64], confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR / ES): mean loss beyond the VaR threshold.

    CVaR is the average of all returns that fall below the ``-VaR`` threshold,
    making it a coherent risk measure unlike VaR::

        CVaR_alpha = -E[r | r <= -VaR_alpha]

    Always >= VaR at the same confidence level.

    Args:
        returns: 1-D array of period returns.
        confidence: Confidence level, e.g. 0.95 for 95 % CVaR.

    Returns:
        CVaR as a positive decimal loss.
    """
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
    """Compute all standard risk/return metrics for a weighted portfolio.

    Constructs the portfolio return series ``r_p = returns_matrix @ weights``
    and then evaluates every metric on that series.

    Args:
        weights: 1-D weight vector of shape (n,), should sum to 1.
        returns_matrix: Period returns of shape (T, n): rows = time, columns = assets.
            Must contain *period* returns (not annualized).
        risk_free_rate: Annual risk-free rate used for Sharpe/Sortino/Calmar.
        periods_per_year: Calendar periods per year (252 daily, 52 weekly, 12 monthly).

    Returns:
        Dictionary with keys:
            - ``annualized_return``: geometric annualized return.
            - ``annualized_volatility``: annualized std of portfolio returns.
            - ``sharpe_ratio``: annualized Sharpe.
            - ``sortino_ratio``: annualized Sortino.
            - ``max_drawdown``: maximum peak-to-trough drawdown (positive).
            - ``calmar_ratio``: annualized return / max drawdown.
            - ``var_95``: 95 % historical VaR (positive loss).
            - ``cvar_95``: 95 % expected shortfall (positive loss).
    """
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
