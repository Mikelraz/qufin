"""
Price-indicators subpackage — classic technical-analysis primitives.

Layout
------
    _types               Result containers and shared helpers
    _kernels             numba-jitted hot loops (EMA, RSI, ADX, SAR, Supertrend…)
    moving_averages      SMA, EMA, WMA, DEMA, TEMA, HMA, KAMA
    momentum             RSI, MACD, Stochastic, ROC, CCI, Williams %R
    volatility           true_range, ATR, Bollinger, Keltner, Donchian
    trend                ADX (+DI / -DI), Aroon, Parabolic SAR, Supertrend, Ichimoku
    volume               OBV, VWAP (cum + rolling), MFI, CMF, A/D line
    support_resistance   Pivot points (classic + Fibonacci), swing clustering

Quick start
-----------
    import numpy as np
    from qufin.indicators import ema, rsi, macd, bollinger_bands, atr

    close = np.array([...])
    high  = np.array([...])
    low   = np.array([...])

    ma   = ema(close, window=20)
    rs   = rsi(close, window=14)
    m    = macd(close)                       # MACDResult(macd, signal, hist)
    bb   = bollinger_bands(close, 20, 2.0)   # BollingerBands(middle, upper, …)
    rng  = atr(high, low, close, window=14)

OHLCV interop with the Wyckoff subpackage:

    from qufin.wyckoff import OHLCV
    from qufin.indicators import rsi, supertrend

    bars: OHLCV = ...
    rs = rsi(bars.close(), window=14)
    st = supertrend(bars.high(), bars.low(), bars.close())
"""

from __future__ import annotations

from ._types import (
    ADXResult,
    AroonResult,
    BollingerBands,
    DonchianChannels,
    IchimokuResult,
    KeltnerChannels,
    MACDResult,
    PivotPoints,
    StochasticResult,
    SupertrendResult,
    SupportResistanceLevel,
)
from .momentum import cci, macd, roc, rsi, stochastic, williams_r
from .moving_averages import dema, ema, hma, kama, sma, tema, wma
from .support_resistance import (
    cluster_levels,
    fibonacci_pivot_points,
    pivot_points,
    pivot_points_series,
    support_resistance_from_swings,
)
from .trend import adx, aroon, ichimoku, parabolic_sar, supertrend
from .volatility import (
    atr,
    bollinger_bands,
    donchian_channels,
    keltner_channels,
    true_range,
)
from .volume import (
    accumulation_distribution,
    cmf,
    mfi,
    obv,
    rolling_vwap,
    vwap,
)

__all__ = [
    # Result types
    "MACDResult",
    "BollingerBands",
    "KeltnerChannels",
    "DonchianChannels",
    "StochasticResult",
    "ADXResult",
    "AroonResult",
    "SupertrendResult",
    "IchimokuResult",
    "PivotPoints",
    "SupportResistanceLevel",
    # Moving averages
    "sma",
    "ema",
    "wma",
    "dema",
    "tema",
    "hma",
    "kama",
    # Momentum
    "rsi",
    "macd",
    "stochastic",
    "roc",
    "cci",
    "williams_r",
    # Volatility
    "true_range",
    "atr",
    "bollinger_bands",
    "keltner_channels",
    "donchian_channels",
    # Trend
    "adx",
    "aroon",
    "parabolic_sar",
    "supertrend",
    "ichimoku",
    # Volume
    "obv",
    "vwap",
    "rolling_vwap",
    "mfi",
    "cmf",
    "accumulation_distribution",
    # Support / resistance
    "pivot_points",
    "fibonacci_pivot_points",
    "pivot_points_series",
    "cluster_levels",
    "support_resistance_from_swings",
]
