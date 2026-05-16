"""
Backward-compatible shim.

KalmanFilter, FilterResult, and SmootherResult have moved to
``src.timeseries.kalman``.  This module re-exports them so that any
existing code importing from ``src.filters.kalman`` continues to work.
"""

from ..timeseries.kalman import FilterResult, KalmanFilter, SmootherResult

__all__ = ["FilterResult", "KalmanFilter", "SmootherResult"]
