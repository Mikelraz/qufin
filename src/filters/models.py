"""
Backward-compatible shim.

HedgeRatioFilter and TrendFilter have moved to ``src.timeseries.models``.
This module re-exports them so that any existing code importing from
``src.filters.models`` continues to work.
"""

from ..timeseries.models import HedgeRatioFilter, TrendFilter

__all__ = ["HedgeRatioFilter", "TrendFilter"]
