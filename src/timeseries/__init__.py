"""
Time-series modelling subpackage.

Kalman filter
-------------
    kalman        KalmanFilter, FilterResult, SmootherResult
    models        HedgeRatioFilter, TrendFilter

Phase 1 — foundations
---------------------
    _types        Protocols and shared result containers (ForecastResult, BacktestEvalResult)
    _io           numpy / polars conversions and input validation
    _kernels      numba-jitted numerical kernels (sample_acf, durbin_levinson, lag_matrix)
    utils         differencing, info criteria (AIC / BIC / HQIC)
    stationarity  ADF, KPSS, Phillips-Perron, variance-ratio tests
    diagnostics   Ljung-Box, Jarque-Bera, ARCH-LM, ACF / PACF with confidence bands

Phase 2 — ARMA family + state-space wrapper
--------------------------------------------
    arima         AR, MA, ARMA, ARIMA, SARIMA (estimation, forecasting, simulation)
    statespace    ARMAStateSpace (Kalman filter + RTS smoother over ARMA observations)

Phase 3 — VAR + cointegration
-----------------------------
    var           VAR(p), Granger causality, impulse response functions
    cointegration Engle-Granger, Johansen, VECM

Phase 4 — GARCH family
----------------------
    garch         GARCH, EGARCH, GJR, EWMA (single-asset volatility models)

Future phases will add DCC / regime / forecast evaluation (Phase 5).
"""

from __future__ import annotations

from ._types import (
    BacktestEvalResult,
    ForecastResult,
    HasInfoCriteria,
    HasResiduals,
)
from .arima import (
    AR,
    ARIMA,
    ARMA,
    MA,
    SARIMA,
    ARFitResult,
    ARIMAFitResult,
    ARMAFitResult,
    MAFitResult,
    SARIMAFitResult,
)
from .cointegration import (
    EngleGrangerResult,
    JohansenResult,
    VECMResult,
    engle_granger,
    johansen,
    vecm,
)
from .diagnostics import (
    ACFResult,
    acf,
    arch_lm,
    jarque_bera,
    ljung_box,
    pacf,
)
from .garch import (
    EGARCH,
    EWMA,
    GARCH,
    GJR,
    EGARCHFitResult,
    EWMAResult,
    GARCHFitResult,
    GJRFitResult,
)
from .kalman import FilterResult, KalmanFilter, SmootherResult
from .models import HedgeRatioFilter, TrendFilter
from .statespace import ARMAStateSpace, StateSpaceResult
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
from .var import (
    VAR,
    GrangerResult,
    VARFitResult,
    granger_causality,
    impulse_response,
)

__all__ = [
    # Protocols and shared types
    "BacktestEvalResult",
    "ForecastResult",
    "HasInfoCriteria",
    "HasResiduals",
    # Kalman filter
    "FilterResult",
    "KalmanFilter",
    "SmootherResult",
    # Pre-built models
    "HedgeRatioFilter",
    "TrendFilter",
    # ARMA family
    "AR",
    "ARIMA",
    "ARMA",
    "MA",
    "SARIMA",
    "ARFitResult",
    "ARIMAFitResult",
    "ARMAFitResult",
    "MAFitResult",
    "SARIMAFitResult",
    # State-space
    "ARMAStateSpace",
    "StateSpaceResult",
    # VAR
    "VAR",
    "VARFitResult",
    "GrangerResult",
    "granger_causality",
    "impulse_response",
    # Cointegration
    "EngleGrangerResult",
    "engle_granger",
    "JohansenResult",
    "johansen",
    "VECMResult",
    "vecm",
    # GARCH family
    "GARCH",
    "GARCHFitResult",
    "EGARCH",
    "EGARCHFitResult",
    "GJR",
    "GJRFitResult",
    "EWMA",
    "EWMAResult",
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
