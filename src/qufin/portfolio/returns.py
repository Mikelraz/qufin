from __future__ import annotations

import math

import numpy as np
import polars as pl
from numpy.typing import NDArray


def _asset_cols(df: pl.DataFrame, date_col: str) -> list[str]:
    return [c for c in df.columns if c != date_col]


def simple_returns(prices: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Period-over-period simple returns: r_t = P_t/P_{t-1} - 1."""
    cols = _asset_cols(prices, date_col)
    return prices.with_columns([(pl.col(c) / pl.col(c).shift(1) - 1).alias(c) for c in cols]).slice(
        1
    )


def log_returns(prices: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Period-over-period log returns: r_t = ln(P_t / P_{t-1})."""
    cols = _asset_cols(prices, date_col)
    return prices.with_columns(
        [(pl.col(c) / pl.col(c).shift(1)).log(math.e).alias(c) for c in cols]
    ).slice(1)


def cumulative_returns(returns: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Transform period returns into cumulative returns: (1+r_1)·…·(1+r_t) - 1."""
    cols = _asset_cols(returns, date_col)
    return returns.with_columns([((1 + pl.col(c)).cum_prod() - 1).alias(c) for c in cols])


def annualize_return(total_return: float, n_periods: int, periods_per_year: int) -> float:
    """Annualize a total compounded return over n_periods."""
    return (1.0 + total_return) ** (periods_per_year / n_periods) - 1.0


def annualized_returns(
    returns: pl.DataFrame,
    periods_per_year: int,
    date_col: str = "date",
) -> dict[str, float]:
    """Geometric annualized return for each asset column."""
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
    """Extract (T × n) float64 matrix and asset name list from a returns DataFrame."""
    cols = _asset_cols(returns, date_col)
    mat = returns.select(cols).to_numpy().astype(np.float64)
    return mat, cols
