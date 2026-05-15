"""
Linear Kalman Filter subpackage for financial time series.

Exports:
    KalmanFilter    - General-purpose linear Kalman filter with RTS smoother
    FilterResult    - Dataclass holding forward-pass outputs
    SmootherResult  - Dataclass holding backward-smoother outputs
    HedgeRatioFilter - Tracks a time-varying hedge ratio (beta + intercept)
    TrendFilter     - Constant-velocity smoother for price/return series
"""

from .kalman import FilterResult, KalmanFilter, SmootherResult
from .models import HedgeRatioFilter, TrendFilter

__all__ = [
    "KalmanFilter",
    "FilterResult",
    "SmootherResult",
    "HedgeRatioFilter",
    "TrendFilter",
]
