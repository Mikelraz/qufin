"""
Numba-jitted scan kernels for the Wyckoff subpackage.

These operate exclusively on primitive numpy arrays of float64 / int64 so the
JIT-compiled loops can run with the GIL released. Polars and dataclasses are
kept on the Python side.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def fractal_swings(
    high: np.ndarray, low: np.ndarray, left: int, right: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bill-Williams style fractal pivots.

    A bar ``i`` is a swing high iff ``high[i] > high[j]`` for the ``left`` bars
    before and the ``right`` bars after; a swing low is the dual condition on
    ``low``. Strict inequalities ensure uniqueness on flat plateaus.

    Returns
    -------
    indices : int64 array of pivot bar positions, in chronological order.
    kinds   : int8 array, ``+1`` for swing high, ``-1`` for swing low.
    """
    n = high.shape[0]
    idx_buf = np.empty(n, dtype=np.int64)
    kind_buf = np.empty(n, dtype=np.int8)
    count = 0
    for i in range(left, n - right):
        is_high = True
        for j in range(1, left + 1):
            if high[i - j] >= high[i]:
                is_high = False
                break
        if is_high:
            for j in range(1, right + 1):
                if high[i + j] >= high[i]:
                    is_high = False
                    break
        if is_high:
            idx_buf[count] = i
            kind_buf[count] = 1
            count += 1
            continue
        is_low = True
        for j in range(1, left + 1):
            if low[i - j] <= low[i]:
                is_low = False
                break
        if is_low:
            for j in range(1, right + 1):
                if low[i + j] <= low[i]:
                    is_low = False
                    break
        if is_low:
            idx_buf[count] = i
            kind_buf[count] = -1
            count += 1
    return idx_buf[:count].copy(), kind_buf[:count].copy()


@njit(cache=True)
def zigzag_swings(high: np.ndarray, low: np.ndarray, pct: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Percent-reversal ZigZag.

    Tracks the current extreme in each direction; emits a pivot whenever the
    counter-direction high/low reverses by at least ``pct`` (e.g. ``0.03`` for
    3%). The first pivot is the starting bar (direction inferred from the
    first qualifying reversal).
    """
    n = high.shape[0]
    idx_buf = np.empty(n, dtype=np.int64)
    kind_buf = np.empty(n, dtype=np.int8)
    count = 0
    if n == 0:
        return idx_buf[:0].copy(), kind_buf[:0].copy()

    # 0 = unknown, +1 = up-leg (looking for highs), -1 = down-leg (looking for lows)
    direction = 0
    ext_idx = 0
    ext_high = high[0]
    ext_low = low[0]

    for i in range(1, n):
        if direction == 0:
            if low[i] < ext_low:
                ext_low = low[i]
                ext_idx = i
            if high[i] > ext_high:
                ext_high = high[i]
                ext_idx = i
            # Direction decided once a counter-move exceeds pct from the extreme.
            up_move = (high[i] - ext_low) / ext_low if ext_low > 0 else 0.0
            dn_move = (ext_high - low[i]) / ext_high if ext_high > 0 else 0.0
            if up_move >= pct and up_move >= dn_move:
                # ext_low was a valid low; switch to up-leg.
                idx_buf[count] = ext_idx
                kind_buf[count] = -1
                count += 1
                direction = 1
                ext_idx = i
                ext_high = high[i]
            elif dn_move >= pct:
                idx_buf[count] = ext_idx
                kind_buf[count] = 1
                count += 1
                direction = -1
                ext_idx = i
                ext_low = low[i]
        elif direction == 1:
            if high[i] > ext_high:
                ext_high = high[i]
                ext_idx = i
            dn_move = (ext_high - low[i]) / ext_high if ext_high > 0 else 0.0
            if dn_move >= pct:
                idx_buf[count] = ext_idx
                kind_buf[count] = 1
                count += 1
                direction = -1
                ext_idx = i
                ext_low = low[i]
        else:  # direction == -1
            if low[i] < ext_low:
                ext_low = low[i]
                ext_idx = i
            up_move = (high[i] - ext_low) / ext_low if ext_low > 0 else 0.0
            if up_move >= pct:
                idx_buf[count] = ext_idx
                kind_buf[count] = -1
                count += 1
                direction = 1
                ext_idx = i
                ext_high = high[i]

    # Final pending extreme — emit it so the sequence ends at the last leg's tip.
    if direction == 1:
        idx_buf[count] = ext_idx
        kind_buf[count] = 1
        count += 1
    elif direction == -1:
        idx_buf[count] = ext_idx
        kind_buf[count] = -1
        count += 1

    return idx_buf[:count].copy(), kind_buf[:count].copy()


@njit(cache=True)
def pnf_columns_from_closes(
    closes: np.ndarray, box_size: float, reversal: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build P&F columns from a close-only series.

    Returns four parallel int/float arrays describing each column:
    ``directions`` (+1 for X / up column, -1 for O / down column),
    ``start_indices`` (bar where the column began),
    ``lows`` and ``highs`` (column extreme prices in price units).
    """
    n = closes.shape[0]
    dir_buf = np.empty(n, dtype=np.int8)
    start_buf = np.empty(n, dtype=np.int64)
    low_buf = np.empty(n, dtype=np.float64)
    high_buf = np.empty(n, dtype=np.float64)
    cols = 0
    if n == 0 or box_size <= 0.0:
        return (
            dir_buf[:0].copy(),
            start_buf[:0].copy(),
            low_buf[:0].copy(),
            high_buf[:0].copy(),
        )

    # Anchor first column at the first close, direction undecided until the
    # price moves by a full box from the anchor.
    anchor = closes[0]
    cur_dir = 0
    cur_low = anchor
    cur_high = anchor
    cur_start = 0

    for i in range(1, n):
        c = closes[i]
        if cur_dir == 0:
            up_boxes = int((c - cur_high) / box_size)
            dn_boxes = int((cur_low - c) / box_size)
            if up_boxes >= 1:
                cur_dir = 1
                cur_high = cur_low + up_boxes * box_size
            elif dn_boxes >= 1:
                cur_dir = -1
                cur_low = cur_high - dn_boxes * box_size
        elif cur_dir == 1:
            up_boxes = int((c - cur_high) / box_size)
            if up_boxes >= 1:
                cur_high = cur_high + up_boxes * box_size
            else:
                dn_boxes = int((cur_high - c) / box_size)
                if dn_boxes >= reversal:
                    dir_buf[cols] = 1
                    start_buf[cols] = cur_start
                    low_buf[cols] = cur_low
                    high_buf[cols] = cur_high
                    cols += 1
                    prior_top = cur_high
                    cur_dir = -1
                    cur_start = i
                    cur_high = prior_top - box_size
                    cur_low = prior_top - dn_boxes * box_size
        else:  # cur_dir == -1
            dn_boxes = int((cur_low - c) / box_size)
            if dn_boxes >= 1:
                cur_low = cur_low - dn_boxes * box_size
            else:
                up_boxes = int((c - cur_low) / box_size)
                if up_boxes >= reversal:
                    dir_buf[cols] = -1
                    start_buf[cols] = cur_start
                    low_buf[cols] = cur_low
                    high_buf[cols] = cur_high
                    cols += 1
                    prior_bottom = cur_low
                    cur_dir = 1
                    cur_start = i
                    cur_low = prior_bottom + box_size
                    cur_high = prior_bottom + up_boxes * box_size

    if cur_dir != 0:
        dir_buf[cols] = cur_dir
        start_buf[cols] = cur_start
        low_buf[cols] = cur_low
        high_buf[cols] = cur_high
        cols += 1

    return (
        dir_buf[:cols].copy(),
        start_buf[:cols].copy(),
        low_buf[:cols].copy(),
        high_buf[:cols].copy(),
    )
