"""
Backward-adjust OHLCV for splits and (optionally) cash dividends.

For each bar at ``ts``, the cumulative split factor is the product of
``1/ratio`` over every split with ex-date strictly greater than ``ts`` —
i.e. splits that have not yet happened in the as-of frame of ``ts``. Prices
are multiplied by this factor; volume is divided.

When ``include_dividends`` is True, cash dividends are folded into the same
backward factor using the textbook ``(prev_close - div) / prev_close``
factor on the ex-date — yielding total-return-adjusted prices.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .._types import OHLCV
from .actions import ACTIONS_SCHEMA


def apply_splits(bars: OHLCV, actions: pl.DataFrame) -> OHLCV:
    """Return a new ``OHLCV`` with prices and volume back-adjusted for splits.

    ``actions`` must match ``ACTIONS_SCHEMA``. Rows with ``kind != 'split'``
    or whose ``symbol`` differs from ``bars.symbol`` are ignored.
    """
    return _back_adjust(bars, actions, include_dividends=False)


def back_adjust(bars: OHLCV, actions: pl.DataFrame) -> OHLCV:
    """Back-adjust prices for both splits and cash dividends (total-return basis)."""
    return _back_adjust(bars, actions, include_dividends=True)


def _back_adjust(bars: OHLCV, actions: pl.DataFrame, *, include_dividends: bool) -> OHLCV:
    if actions.is_empty():
        return bars
    _validate_schema(actions)
    relevant = actions.filter(
        (pl.col("symbol") == bars.symbol)
        & pl.col("kind").is_in(["split", "cash_div"] if include_dividends else ["split"])
    ).sort("timestamp")
    if relevant.is_empty():
        return bars

    bar_ts_ns = bars.data["timestamp"].cast(pl.Int64).to_numpy()
    closes = bars.data["close"].to_numpy().astype(np.float64, copy=True)

    price_mult_at_bar = np.ones(bar_ts_ns.shape, dtype=np.float64)
    vol_mult_at_bar = np.ones(bar_ts_ns.shape, dtype=np.float64)

    action_ts_ns = relevant["timestamp"].cast(pl.Int64).to_numpy()
    kinds = relevant["kind"].to_list()
    ratios = relevant["ratio"].to_numpy()
    cashes = relevant["cash"].to_numpy()

    for i in range(len(action_ts_ns) - 1, -1, -1):
        ex_ns = action_ts_ns[i]
        # bars strictly before the ex-date get this action folded in
        mask = bar_ts_ns < ex_ns
        match kinds[i]:
            case "split":
                price_mult_at_bar[mask] *= 1.0 / ratios[i]
                vol_mult_at_bar[mask] *= ratios[i]
            case "cash_div":
                prev_idx = int(np.searchsorted(bar_ts_ns, ex_ns, side="left")) - 1
                if prev_idx < 0:
                    continue
                prev_close_adj = closes[prev_idx] * price_mult_at_bar[prev_idx]
                if prev_close_adj <= 0:
                    continue
                factor = (prev_close_adj - cashes[i]) / prev_close_adj
                if factor <= 0:
                    continue
                price_mult_at_bar[mask] *= factor

    adjusted = bars.data.with_columns(
        (pl.col("open") * pl.Series(price_mult_at_bar)).alias("open"),
        (pl.col("high") * pl.Series(price_mult_at_bar)).alias("high"),
        (pl.col("low") * pl.Series(price_mult_at_bar)).alias("low"),
        (pl.col("close") * pl.Series(price_mult_at_bar)).alias("close"),
        (pl.col("volume") * pl.Series(vol_mult_at_bar)).alias("volume"),
    )
    return OHLCV.from_records(adjusted, symbol=bars.symbol)


def _validate_schema(actions: pl.DataFrame) -> None:
    missing = set(ACTIONS_SCHEMA) - set(actions.columns)
    if missing:
        raise ValueError(f"actions frame is missing columns: {sorted(missing)}")
