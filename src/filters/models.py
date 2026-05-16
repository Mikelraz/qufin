"""Backward-compatibility shim.  See ``src/filters/__init__.py``."""

from __future__ import annotations

from ..timeseries.models import HedgeRatioFilter, TrendFilter

__all__ = ["HedgeRatioFilter", "TrendFilter"]
