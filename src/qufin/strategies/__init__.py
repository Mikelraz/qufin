"""Trading strategy implementations"""

from .cointegration_pairs import (
    CointegrationPairsStrategy,
    PairsBacktestResult,
    PairsParams,
)
from .garch_vol_target import (
    GARCHVolTargetParams,
    GARCHVolTargetResult,
    GARCHVolTargetStrategy,
)
from .hull_backtest import HullBacktestResult, backtest_hull
from .hull_strategy import (
    generate_signals,
    momentum_filter,
    multi_timeframe_filter,
    vwap_filter,
)
from .hull_suite import (
    HullBand,
    HullRibbon,
    ehma,
    hma,
    hull_ribbon,
    hull_slope,
    price_vs_ribbon,
    thma,
    wma,
)
from .mean_reversion import (
    BacktestResult,
    MeanReversionStrategy,
    StrategyParams,
    TrainResult,
)
from .regime_switching import (
    RegimeSwitchingParams,
    RegimeSwitchingResult,
    RegimeSwitchingStrategy,
)
from .time_series_momentum import (
    TimeSeriesMomentumStrategy,
    TSMOMParams,
    TSMOMResult,
    TSMOMTrainResult,
)

__all__ = [
    # Mean reversion
    "MeanReversionStrategy",
    "StrategyParams",
    "BacktestResult",
    "TrainResult",
    # GARCH vol target
    "GARCHVolTargetStrategy",
    "GARCHVolTargetParams",
    "GARCHVolTargetResult",
    # Cointegration pairs
    "CointegrationPairsStrategy",
    "PairsParams",
    "PairsBacktestResult",
    # Regime switching
    "RegimeSwitchingStrategy",
    "RegimeSwitchingParams",
    "RegimeSwitchingResult",
    # Time-series momentum
    "TimeSeriesMomentumStrategy",
    "TSMOMParams",
    "TSMOMResult",
    "TSMOMTrainResult",
    # Hull Suite — indicator
    "wma",
    "hma",
    "thma",
    "ehma",
    "hull_ribbon",
    "hull_slope",
    "price_vs_ribbon",
    "HullBand",
    "HullRibbon",
    # Hull Suite — strategy + filters
    "generate_signals",
    "multi_timeframe_filter",
    "vwap_filter",
    "momentum_filter",
    # Hull Suite — backtest
    "backtest_hull",
    "HullBacktestResult",
]
