"""
Cointegration tests and models.

* ``engle_granger`` — Engle-Granger two-step residual ADF test.
* ``johansen``      — Johansen (1991) trace and maximum-eigenvalue tests.
* ``vecm``          — Reduced-rank VECM estimation given a known cointegration rank.
"""

from __future__ import annotations

from .engle_granger import EngleGrangerResult, engle_granger
from .johansen import JohansenResult, johansen
from .vecm import VECMResult, vecm

__all__ = [
    "EngleGrangerResult",
    "engle_granger",
    "JohansenResult",
    "johansen",
    "VECMResult",
    "vecm",
]
