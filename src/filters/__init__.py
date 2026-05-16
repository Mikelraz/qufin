"""
Backward-compatibility shim — Kalman filter code now lives in src.timeseries.

The original ``src.filters`` package was migrated into the new ``src.timeseries``
subpackage during Phase 2 of the time-series build-out.  This module re-exports
the moved symbols so existing callers (notebooks, ``scripts/validate_kalman_ou.py``,
and ``tests/filters/*``) keep working unchanged.

Use the canonical paths in new code:

    from src.timeseries import KalmanFilter, FilterResult, SmootherResult
    from src.timeseries import HedgeRatioFilter, TrendFilter
"""

from __future__ import annotations

from ..timeseries.kalman import FilterResult, KalmanFilter, SmootherResult
from ..timeseries.models import HedgeRatioFilter, TrendFilter

__all__ = [
    "KalmanFilter",
    "FilterResult",
    "SmootherResult",
    "HedgeRatioFilter",
    "TrendFilter",
]
