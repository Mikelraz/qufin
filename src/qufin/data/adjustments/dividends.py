"""
Total-return series from price bars + cash dividends.

For each bar with a cash dividend on its ex-date, the log return is::

    log_ret[t] = log((close[t] + div[t]) / close[t-1])

When no dividend lands on bar ``t``, the standard ``log(close[t]/close[t-1])``
is used. The first bar's return is 0 by convention.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .._types import OHLCV
from .actions import ACTIONS_SCHEMA


def total_return_series(bars: OHLCV, actions: pl.DataFrame) -> pl.Series:
    """Return a per-bar total-return log-return Series aligned to ``bars``."""
    if actions.is_empty():
        return _plain_log_returns(bars.close())
    _validate_schema(actions)
    divs = actions.filter(
        (pl.col("symbol") == bars.symbol) & (pl.col("kind") == "cash_div")
    ).sort("timestamp")
    closes = bars.close()
    if divs.is_empty():
        return _plain_log_returns(closes)

    bar_ts_ns = bars.data["timestamp"].cast(pl.Int64).to_numpy()
    div_ts_ns = divs["timestamp"].cast(pl.Int64).to_numpy()
    div_cash = divs["cash"].to_numpy()

    div_per_bar = np.zeros_like(closes)
    idx = np.searchsorted(bar_ts_ns, div_ts_ns)
    for k, i in enumerate(idx):
        if 0 <= i < len(bar_ts_ns) and bar_ts_ns[i] == div_ts_ns[k]:
            div_per_bar[i] += div_cash[k]

    log_ret = np.zeros_like(closes)
    log_ret[1:] = np.log((closes[1:] + div_per_bar[1:]) / closes[:-1])
    return pl.Series("total_return", log_ret)


def total_return_index(
    bars: OHLCV, actions: pl.DataFrame, *, base: float = 1.0
) -> pl.Series:
    """Cumulative total-return index starting at ``base``."""
    log_ret = total_return_series(bars, actions).to_numpy()
    cum = np.exp(np.cumsum(log_ret)) * base
    return pl.Series("total_return_index", cum)


def _plain_log_returns(closes: np.ndarray) -> pl.Series:
    out = np.zeros_like(closes)
    out[1:] = np.log(closes[1:] / closes[:-1])
    return pl.Series("total_return", out)


def _validate_schema(actions: pl.DataFrame) -> None:
    missing = set(ACTIONS_SCHEMA) - set(actions.columns)
    if missing:
        raise ValueError(f"actions frame is missing columns: {sorted(missing)}")
