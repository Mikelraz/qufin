"""Performance reporting, attribution, and multi-strategy comparison."""

from __future__ import annotations

from .attribution import per_symbol_pnl
from .compare import compare
from .tearsheet import TearSheet, tearsheet

__all__ = ["TearSheet", "compare", "per_symbol_pnl", "tearsheet"]
