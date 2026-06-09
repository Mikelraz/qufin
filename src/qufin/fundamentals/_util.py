"""Internal numeric helpers shared across the fundamentals analytics."""

from __future__ import annotations

import math


def safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning ``NaN`` for a zero, NaN, or otherwise invalid divisor.

    Fundamental data is riddled with gaps; propagating ``NaN`` rather than
    raising lets a caller compute a whole :class:`RatioSet` even when a few
    line items are missing.
    """
    if denominator == 0.0 or math.isnan(denominator) or math.isnan(numerator):
        return math.nan
    return numerator / denominator
