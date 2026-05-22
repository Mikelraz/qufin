"""Parameter container for the Wyckoff-Hull confluence strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HullVariant = Literal["hma", "thma", "ehma"]


@dataclass(slots=True, frozen=True)
class ConfluenceParams:
    """All knobs for the strategy in a single immutable container.

    Defaults match the values cited in the WHCG specification. The dataclass
    is frozen so a fitted parameter set can be hashed for walk-forward
    bookkeeping.
    """

    # Wyckoff structure
    swing_window: int = 5
    range_min_bars: int = 20
    event_lookback: int = 5

    # Hull ribbon
    hull_fast_length: int = 50
    hull_slow_length: int = 60
    hull_fast_type: HullVariant = "hma"
    hull_slow_type: HullVariant = "ehma"

    # Momentum
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Volume confirmation
    obv_slope_window: int = 20
    cmf_window: int = 20

    # ATR + chandelier trail
    atr_window: int = 22
    chandelier_window: int = 22
    chandelier_atr_mult: float = 3.0

    # Confluence rule
    min_confluences: int = 3

    # Regime cash-defense
    regime_distribution_threshold: float = 0.6
    regime_persistence_bars: int = 2
    regime_warmup_bars: int = 250
    regime_refit_period: int = 21

    # GEX defense overlay
    use_gex_overlay: bool = True
    gex_macro_symbol: str = "SPY"
    gex_put_wall_buffer_pct: float = 0.01

    # Sizing — half-Kelly
    kelly_fraction: float = 0.5
    edge_lookback_days: int = 252
    edge_min_trades: int = 10
    default_edge_p: float = 0.50
    default_edge_b: float = 1.0

    # Portfolio constraints
    sigma_target_annual: float = 0.12
    max_weight_per_name: float = 0.05
    max_total_exposure: float = 1.0
    cluster_corr_threshold: float = 0.70
    max_cluster_weight: float = 0.30
    cov_lookback_days: int = 252
    periods_per_year: int = 252

    # Execution
    starting_cash: float = 100_000.0
    slippage_bps: float = 1.0
