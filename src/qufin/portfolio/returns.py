"""Price-to-return transformations operating on polars DataFrames.

Convention
----------
- Input DataFrames have one optional date column (default ``"date"``) plus one
  float column per asset.  Any column whose name equals ``date_col`` is treated
  as the time index and is carried through unchanged.
- Simple and log returns both drop the first row, which becomes NaN after
  the one-period lag.
- All return figures are dimensionless (not in percent).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from numpy.typing import NDArray


def _asset_cols(df: pl.DataFrame, date_col: str) -> list[str]:
    return [c for c in df.columns if c != date_col]


def simple_returns(prices: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Compute period-over-period simple returns: ``r_t = P_t / P_{t-1} - 1``.

    Args:
        prices: DataFrame of asset prices.  Must contain at least one numeric
            column.  A date column named ``date_col`` is preserved as-is.
        date_col: Name of the date/time index column.  Excluded from the
            return calculation.

    Returns:
        DataFrame of the same shape minus one row (the lagged first row is
        dropped).  Date and asset columns are preserved in their original order.
    """
    cols = _asset_cols(prices, date_col)
    return prices.with_columns([(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in cols]).slice(
        1
    )


def log_returns(prices: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Compute period-over-period log returns: ``r_t = ln(P_t / P_{t-1})``.

    Log returns are additive over time (unlike simple returns) and are the
    natural choice for modelling with normal distributions.  For small returns
    ``log(1 + r_simple) ≈ r_log``.

    Args:
        prices: DataFrame of asset prices.
        date_col: Name of the date/time index column.

    Returns:
        DataFrame of log returns with one fewer row than ``prices``.
    """
    cols = _asset_cols(prices, date_col)
    return prices.with_columns(
        [(pl.col(c) / pl.col(c).shift(1)).log(math.e).alias(c) for c in cols]
    ).slice(1)


def cumulative_returns(returns: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Convert period returns to cumulative wealth relative to the start.

    The output at time *t* is ``(1+r_1)(1+r_2)...(1+r_t) - 1``, i.e. total
    return since the first period in the input.

    Args:
        returns: DataFrame of period returns (simple, not log).
        date_col: Name of the date/time index column.

    Returns:
        DataFrame of the same shape with each asset column replaced by its
        running cumulative return.
    """
    cols = _asset_cols(returns, date_col)
    return returns.with_columns([((1 + pl.col(c)).cum_prod() - 1).alias(c) for c in cols])


def annualize_return(total_return: float, n_periods: int, periods_per_year: int) -> float:
    """Annualize a total compounded return observed over ``n_periods``.

    Uses the geometric (compound) formula::

        annualized = (1 + total_return) ^ (periods_per_year / n_periods) - 1

    Args:
        total_return: Total return over the observation window, e.g. 0.25 for 25 %.
        n_periods: Number of periods in the observation window.
        periods_per_year: Calendar periods per year (252 daily, 52 weekly, 12 monthly).

    Returns:
        Annualized return as a decimal.
    """
    return (1.0 + total_return) ** (periods_per_year / n_periods) - 1.0


def annualized_returns(
    returns: pl.DataFrame,
    periods_per_year: int,
    date_col: str = "date",
) -> dict[str, float]:
    """Compute the geometric annualized return for each asset column.

    NaN values are dropped before compounding so that assets with incomplete
    histories are handled gracefully.

    Args:
        returns: DataFrame of simple period returns.
        periods_per_year: Calendar periods per year (252 daily, 52 weekly, 12 monthly).
        date_col: Name of the date/time index column.

    Returns:
        Mapping ``{asset_name: annualized_return}`` where returns are decimals.
    """
    cols = _asset_cols(returns, date_col)
    result: dict[str, float] = {}
    for col in cols:
        arr = returns[col].drop_nulls().to_numpy()
        compound = float(np.prod(1.0 + arr))
        result[col] = annualize_return(compound - 1.0, len(arr), periods_per_year)
    return result


def to_returns_matrix(
    returns: pl.DataFrame,
    date_col: str = "date",
) -> tuple[NDArray[np.float64], list[str]]:
    """Extract a (T × n) float64 numpy array and ordered asset name list.

    Downstream numpy/scipy code (covariance estimation, optimization) works
    with the raw matrix.  This function is the bridge between the polars
    DataFrame representation and the numpy world.

    Args:
        returns: DataFrame of period returns.
        date_col: Name of the date/time index column to exclude.

    Returns:
        Tuple of:
            - ``matrix``: shape (T, n), row = time step, column = asset.
            - ``asset_names``: list of column names in column order.
    """
    cols = _asset_cols(returns, date_col)
    mat = returns.select(cols).to_numpy().astype(np.float64)
    return mat, cols
