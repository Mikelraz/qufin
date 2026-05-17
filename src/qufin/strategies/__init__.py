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
]
