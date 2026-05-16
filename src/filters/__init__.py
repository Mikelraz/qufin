"""
Linear Kalman Filter subpackage — backward-compatible re-exports.

The implementation has moved to ``src.timeseries``.  This package re-exports
everything so that existing code importing from ``src.filters`` continues to work.

Preferred import going forward::

    from src.timeseries.kalman import KalmanFilter, FilterResult, SmootherResult
    from src.timeseries.models import HedgeRatioFilter, TrendFilter
"""

from ..timeseries.kalman import FilterResult, KalmanFilter, SmootherResult
from ..timeseries.models import HedgeRatioFilter, TrendFilter

__all__ = [
    "KalmanFilter",
    "FilterResult",
    "SmootherResult",
    "HedgeRatioFilter",
    "TrendFilter",
]
