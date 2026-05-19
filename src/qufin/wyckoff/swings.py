"""
Swing-point detection for OHLCV bar sequences.

Two algorithms are provided:

* ``find_swings`` — Bill-Williams fractal pivots over a fixed ``(left, right)``
  bar window. Deterministic, no parameter for percent moves.
* ``zigzag`` — Percent-reversal ZigZag that emits a pivot only after a
  counter-move of at least ``pct``. Suited to noisy intraday data.

Both return ``list[SwingPoint]`` in chronological order with the bar index,
timestamp, price, kind (``'H'`` or ``'L'``), and a discrete "strength"
qualifier.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from ._kernels import fractal_swings, zigzag_swings
from ._types import OHLCV, SwingPoint


def find_swings(bars: OHLCV, left: int = 3, right: int = 3) -> list[SwingPoint]:
    """
    Detect Bill-Williams fractal swing points.

    A bar is a swing high iff its ``high`` strictly exceeds the highs of the
    ``left`` bars before and the ``right`` bars after. Swing lows are the dual.

    Parameters
    ----------
    bars   The OHLCV sequence.
    left   Number of preceding bars the pivot must dominate (``>= 1``).
    right  Number of following bars the pivot must dominate (``>= 1``).

    Returns
    -------
    list of SwingPoint in chronological order. ``strength`` equals
    ``min(left, right)`` for every emitted pivot.
    """
    if left < 1 or right < 1:
        raise ValueError(f"left and right must be >= 1, got ({left}, {right})")
    high = bars.high()
    low = bars.low()
    indices, kinds = fractal_swings(high, low, left, right)
    timestamps = bars.timestamps()
    strength = min(left, right)
    out: list[SwingPoint] = []
    for i in range(indices.shape[0]):
        idx = int(indices[i])
        kind: str = "H" if kinds[i] == 1 else "L"
        price = float(high[idx]) if kind == "H" else float(low[idx])
        ts = timestamps[idx]
        out.append(
            SwingPoint(
                idx=idx,
                timestamp=_to_datetime(ts),
                price=price,
                kind=kind,  # type: ignore[arg-type]
                strength=strength,
            )
        )
    return out


def zigzag(bars: OHLCV, pct: float = 0.03) -> list[SwingPoint]:
    """
    Detect percent-reversal ZigZag pivots.

    Emits a pivot whenever the counter-direction high/low has reversed by at
    least ``pct`` (e.g. ``0.05`` for 5%) from the running extreme of the
    current leg.

    Strength is set to ``1`` for ZigZag pivots (no analytic equivalent to the
    fractal width).
    """
    if not 0.0 < pct < 1.0:
        raise ValueError(f"pct must be in (0, 1), got {pct}")
    high = bars.high()
    low = bars.low()
    indices, kinds = zigzag_swings(high, low, pct)
    timestamps = bars.timestamps()
    out: list[SwingPoint] = []
    for i in range(indices.shape[0]):
        idx = int(indices[i])
        kind: str = "H" if kinds[i] == 1 else "L"
        price = float(high[idx]) if kind == "H" else float(low[idx])
        ts = timestamps[idx]
        out.append(
            SwingPoint(
                idx=idx,
                timestamp=_to_datetime(ts),
                price=price,
                kind=kind,  # type: ignore[arg-type]
                strength=1,
            )
        )
    return out


def swing_extremes(swings: list[SwingPoint]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(high_prices, low_prices)`` from a list of swing points.

    Convenience for downstream consumers that work in numpy and don't care
    about timestamps or strengths.
    """
    highs = np.array([s.price for s in swings if s.kind == "H"], dtype=np.float64)
    lows = np.array([s.price for s in swings if s.kind == "L"], dtype=np.float64)
    return highs, lows


def _to_datetime(ts: object) -> datetime:
    """Coerce a numpy / polars timestamp to a python ``datetime``."""
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, np.datetime64):
        # Convert to UTC-aware python datetime via int64 nanoseconds.
        ns = ts.astype("datetime64[ns]").astype(np.int64)

        return datetime.fromtimestamp(ns / 1e9, tz=UTC)
    raise TypeError(f"cannot coerce {type(ts).__name__} to datetime")
