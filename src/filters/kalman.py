"""Backward-compatibility shim.  See ``src/filters/__init__.py``."""

from __future__ import annotations

from ..timeseries.kalman import FilterResult, KalmanFilter, SmootherResult

__all__ = ["FilterResult", "KalmanFilter", "SmootherResult"]
