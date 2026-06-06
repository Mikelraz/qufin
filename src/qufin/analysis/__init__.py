"""
Quantitative analysis tools.

Factors
-------
    momentum             time-series momentum (Moskowitz-Ooi-Pedersen) and
                         cross-sectional momentum (Jegadeesh-Titman)

Screening
---------
    cointegration_screen rank a universe's pairs by cointegration significance
"""

from .cointegration_screen import PairScreenResult, screen_pairs
from .momentum import (
    MomentumFactorResult,
    cross_sectional_momentum,
    time_series_momentum,
    trailing_return,
    volatility_scaled_signal,
)

__all__ = [
    "MomentumFactorResult",
    "cross_sectional_momentum",
    "time_series_momentum",
    "trailing_return",
    "volatility_scaled_signal",
    "PairScreenResult",
    "screen_pairs",
]
