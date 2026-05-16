"""
GARCH-family volatility models.

Single-asset models
-------------------
    GARCH    Bollerslev (1986)
    EGARCH   Nelson (1991)
    GJR      Glosten-Jagannathan-Runkle (1993)
    EWMA     RiskMetrics exponentially-weighted moving average (closed form)

Multivariate
------------
    DCC      Engle (2002) Dynamic Conditional Correlation GARCH
"""

from __future__ import annotations

from .dcc import DCC, DCCFitResult
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
    "DCC",
    "DCCFitResult",
]
