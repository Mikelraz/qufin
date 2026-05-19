"""
Point-and-Figure charting and cause-and-effect counts.

P&F construction follows the classic close-only convention with a fixed box
size and a 3-box reversal by default. Two counting methods project price
targets from a congestion column:

* **Horizontal count** — column width (number of columns in a base) × box ×
  reversal, projected from the breakout level.
* **Vertical count** — column length (number of boxes in the thrust column)
  × box × reversal, projected from the column origin.

These quantify the *Law of Cause and Effect*: the longer or wider the
preparation, the larger the subsequent move.
"""

from __future__ import annotations

import numpy as np

from ._kernels import pnf_columns_from_closes
from ._types import (
    OHLCV,
    CauseEffectTarget,
    PnFChart,
    PnFColumn,
)
from .bars import atr


def default_box_size(bars: OHLCV, *, atr_window: int = 14, frac: float = 0.5) -> float:
    """
    Suggest a box size proportional to median ATR.

    ``frac × median(ATR)`` is a robust starting point: it scales with the
    instrument's typical bar range and degrades gracefully on illiquid or
    noisy series.
    """
    if frac <= 0.0:
        raise ValueError(f"frac must be > 0, got {frac}")
    a = atr(bars, window=atr_window)
    med = float(np.nanmedian(a))
    if not np.isfinite(med) or med <= 0.0:
        raise ValueError("could not compute a valid ATR-based box size")
    return frac * med


def pnf_from_bars(
    bars: OHLCV,
    *,
    box_size: float | None = None,
    reversal: int = 3,
    atr_window: int = 14,
) -> PnFChart:
    """
    Construct a P&F chart from the bar close sequence.

    Parameters
    ----------
    bars        OHLCV input.
    box_size    Price increment per box; if None, derived from ATR.
    reversal    Number of boxes required for a counter-column.
    atr_window  ATR window used when ``box_size`` is None.
    """
    if reversal < 1:
        raise ValueError(f"reversal must be >= 1, got {reversal}")
    if box_size is None:
        box_size = default_box_size(bars, atr_window=atr_window)
    if box_size <= 0.0:
        raise ValueError(f"box_size must be > 0, got {box_size}")
    closes = bars.close()
    dirs, starts, lows, highs = pnf_columns_from_closes(closes, box_size, reversal)
    columns: list[PnFColumn] = []
    for k in range(dirs.shape[0]):
        n_boxes = int(round((highs[k] - lows[k]) / box_size)) + 1
        columns.append(
            PnFColumn(
                direction="X" if dirs[k] == 1 else "O",
                start_idx=int(starts[k]),
                boxes_low=float(lows[k]),
                boxes_high=float(highs[k]),
                n_boxes=n_boxes,
            )
        )
    return PnFChart(box_size=float(box_size), reversal=reversal, columns=columns)


def vertical_count(chart: PnFChart, column_idx: int) -> CauseEffectTarget:
    """
    Compute a vertical-count price target for a single thrust column.

    The target equals ``n_boxes × box_size × reversal`` projected in the
    column's direction from the column's origin (its low for X columns; its
    high for O columns).
    """
    if not 0 <= column_idx < chart.n_columns:
        raise ValueError(f"column_idx out of range: {column_idx}")
    col = chart.columns[column_idx]
    count = col.n_boxes
    move = count * chart.box_size * chart.reversal
    if col.direction == "X":
        anchor = col.boxes_low
        projected = anchor + move
        direction: str = "up"
        breakout = col.boxes_high
    else:
        anchor = col.boxes_high
        projected = anchor - move
        direction = "down"
        breakout = col.boxes_low
    return CauseEffectTarget(
        anchor_col=int(column_idx),
        count_boxes=int(count),
        box_size=float(chart.box_size),
        reversal=int(chart.reversal),
        breakout_price=float(breakout),
        projected_price=float(projected),
        direction=direction,  # type: ignore[arg-type]
        method="vertical",
    )


def horizontal_count(chart: PnFChart, column_idx: int) -> CauseEffectTarget:
    """
    Compute a horizontal-count target for a congestion base ending at ``column_idx``.

    The "base" is the longest contiguous run of columns (ending at the column
    immediately before ``column_idx``) whose combined low-high band lies within
    one box of the breakout level. Target = ``base_width × box × reversal``.
    """
    if not 0 <= column_idx < chart.n_columns:
        raise ValueError(f"column_idx out of range: {column_idx}")
    if column_idx == 0:
        raise ValueError("horizontal_count requires at least one base column")
    breakout_col = chart.columns[column_idx]
    direction = breakout_col.direction
    box = chart.box_size

    if direction == "X":
        breakout_level = breakout_col.boxes_low
    else:
        breakout_level = breakout_col.boxes_high

    width = 0
    for k in range(column_idx - 1, -1, -1):
        col = chart.columns[k]
        if col.boxes_low <= breakout_level + box and col.boxes_high >= breakout_level - box:
            width += 1
        else:
            break
    if width == 0:
        width = 1

    move = width * box * chart.reversal
    if direction == "X":
        projected = breakout_level + move
        dir_label: str = "up"
    else:
        projected = breakout_level - move
        dir_label = "down"
    return CauseEffectTarget(
        anchor_col=int(column_idx),
        count_boxes=int(width),
        box_size=float(chart.box_size),
        reversal=int(chart.reversal),
        breakout_price=float(breakout_level),
        projected_price=float(projected),
        direction=dir_label,  # type: ignore[arg-type]
        method="horizontal",
    )
