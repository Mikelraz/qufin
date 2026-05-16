"""
GARCH-family volatility models.

Single-asset models
-------------------
    GARCH    Bollerslev (1986)
    EGARCH   Nelson (1991)
    GJR      Glosten-Jagannathan-Runkle (1993)
    EWMA     RiskMetrics exponentially-weighted moving average (closed form)

Multivariate DCC-GARCH is deferred to Phase 5 (``garch/dcc.py``).
"""

from __future__ import annotations

from .egarch import EGARCH, EGARCHFitResult
from .ewma import EWMA, EWMAResult
from .garch import GARCH, GARCHFitResult
from .gjr import GJR, GJRFitResult

__all__ = [
    "GARCH",
    "GARCHFitResult",
    "EGARCH",
    "EGARCHFitResult",
    "GJR",
    "GJRFitResult",
    "EWMA",
    "EWMAResult",
]
