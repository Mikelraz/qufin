"""
Time-series modelling subpackage.

Phase 1 — foundations
---------------------
    _types        Protocols and shared result containers (ForecastResult, BacktestEvalResult)
    _io           numpy / polars conversions and input validation
    _kernels      numba-jitted numerical kernels (sample_acf, durbin_levinson, lag_matrix)
    utils         differencing, info criteria (AIC / BIC / HQIC)
    stationarity  ADF, KPSS, Phillips-Perron, variance-ratio tests
    diagnostics   Ljung-Box, Jarque-Bera, ARCH-LM, ACF / PACF with confidence bands

Future phases will add ARMA / ARIMA / SARIMA (Phase 2), VAR + cointegration
(Phase 3), GARCH family (Phase 4), DCC / regime / forecast evaluation (Phase 5).
"""

from __future__ import annotations

from ._types import (
    BacktestEvalResult,
    ForecastResult,
    HasInfoCriteria,
    HasResiduals,
)
from .diagnostics import (
    ACFResult,
    acf,
    arch_lm,
    jarque_bera,
    ljung_box,
    pacf,
)
from .stationarity import (
    ADFResult,
    KPSSResult,
    PPResult,
    VRResult,
    adf,
    kpss,
    phillips_perron,
    variance_ratio,
)
from .utils import (
    difference,
    info_criteria,
    inverse_difference,
    seasonal_difference,
)

__all__ = [
    # Protocols and shared types
    "BacktestEvalResult",
    "ForecastResult",
    "HasInfoCriteria",
    "HasResiduals",
    # Diagnostics
    "ACFResult",
    "acf",
    "arch_lm",
    "jarque_bera",
    "ljung_box",
    "pacf",
    # Stationarity
    "ADFResult",
    "KPSSResult",
    "PPResult",
    "VRResult",
    "adf",
    "kpss",
    "phillips_perron",
    "variance_ratio",
    # Utilities
    "difference",
    "info_criteria",
    "inverse_difference",
    "seasonal_difference",
]
