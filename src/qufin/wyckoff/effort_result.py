"""
Effort vs Result — Wyckoff's third law.

Effort is volume; result is the resulting price movement. A *bullish*
divergence is heavy effort with little downward result (supply absorbed);
a *bearish* divergence is heavy effort with little upward result (demand
absorbed). Either configuration signals that the visible side is being
absorbed by the opposite side — a hallmark of composite-operator activity.

We quantify the relationship as the rolling z-score of

    eff_t = log(volume_t + 1)            (effort)
    res_t = |close_t - open_t| / TR_t    (result, in [0, 1])

and flag bars where ``z_eff >= effort_z`` and ``z_res <= -result_z``
(``divergence_strength = z_eff - z_res``).

Absorption flags are necessary but not sufficient: composite-operator
activity is only revealed when the suppressed side wins out *after* the
bar. The ``subsequent_shift_divergence`` routine confirms each absorption
bar by measuring the close-to-close shift over a lookahead window,
normalising by ATR, and classifying bullish / bearish divergences.

The complementary ``price_movement_harmony`` routine inspects every bar
against its rolling trend: a healthy trend prints rising volume on bars
that move *with* the trend. When price obeys the trend on dry volume the
move is disharmonious — a bullish divergence inside a downtrend (supply
drying up) or a bearish divergence inside an uptrend (demand drying up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numba import njit

from ._types import OHLCV
from .bars import atr, rolling_slope, rolling_zscore, true_range

DivergenceKind = Literal["bullish", "bearish"]
HarmonyKind = Literal[
    "bullish_divergence",
    "bearish_divergence",
    "bullish_harmony",
    "bearish_harmony",
]


@dataclass(slots=True)
class EffortResult:
    """
    Rolling effort-vs-result statistics for a bar sequence.

    Attributes
    ----------
    z_effort        Rolling z-score of log volume.
    z_result        Rolling z-score of |close-open| / true_range.
    divergence      ``z_effort - z_result`` (high effort + low result).
    flag_absorption Bool array — True where effort is high but result is low.
    """

    z_effort: np.ndarray
    z_result: np.ndarray
    divergence: np.ndarray
    flag_absorption: np.ndarray


def effort_vs_result(
    bars: OHLCV,
    *,
    window: int = 50,
    effort_z: float = 1.5,
    result_z: float = -0.5,
) -> EffortResult:
    """
    Compute rolling effort vs result statistics and absorption flags.

    Parameters
    ----------
    bars       OHLCV sequence.
    window     Rolling window for the z-scores; default 50 bars.
    effort_z   Effort z-score threshold for absorption flag; default 1.5.
    result_z   Result z-score threshold (upper bound) for absorption flag;
               default -0.5 (i.e. result is at least half a sigma BELOW
               its rolling mean).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if effort_z <= 0.0:
        raise ValueError(f"effort_z must be > 0, got {effort_z}")

    vol = bars.volume()
    log_vol = np.log(vol + 1.0)
    tr = true_range(bars)
    body = np.abs(bars.close() - bars.open())
    result = np.where(tr > 0.0, body / tr, 0.0)

    z_eff = rolling_zscore(log_vol, window)
    z_res = rolling_zscore(result, window)
    diverg = z_eff - z_res
    flag = (z_eff >= effort_z) & (z_res <= result_z)
    return EffortResult(
        z_effort=z_eff,
        z_result=z_res,
        divergence=diverg,
        flag_absorption=flag,
    )


@dataclass(slots=True, frozen=True)
class DivergenceEvent:
    """
    A confirmed effort-vs-result divergence.

    A *bullish* event marks an absorption bar followed by a meaningful
    upward shift in price; a *bearish* event marks an absorption bar
    followed by a meaningful downward shift.

    Attributes
    ----------
    idx                 Bar index of the originating absorption bar.
    kind                ``"bullish"`` or ``"bearish"``.
    bar_sign            Sign of ``close - open`` at ``idx`` (-1, 0, +1).
    shift               Signed close-to-close shift over the lookahead.
    shift_atr           ``shift`` normalised by the bar's ATR.
    divergence_strength ``z_effort - z_result`` at ``idx``.
    lookahead           Number of bars used to measure the shift.
    """

    idx: int
    kind: DivergenceKind
    bar_sign: int
    shift: float
    shift_atr: float
    divergence_strength: float
    lookahead: int


@dataclass(slots=True)
class ShiftDivergence:
    """
    Per-bar subsequent-shift statistics and confirmed divergence events.

    Attributes
    ----------
    shift           Signed close-to-close shift ``close[i+L] - close[i]`` in
                    price units; NaN where the lookahead runs past the end.
    shift_atr       ``shift`` divided by ATR at the bar; NaN during ATR or
                    lookahead warmup.
    kind            Int8 array — ``+1`` bullish, ``-1`` bearish, ``0`` none.
    flag_confirmed  Bool array — True at absorption bars whose subsequent
                    shift opposes the visible body and exceeds the threshold.
    events          List of confirmed ``DivergenceEvent`` ordered by index.
    """

    shift: np.ndarray
    shift_atr: np.ndarray
    kind: np.ndarray
    flag_confirmed: np.ndarray
    events: list[DivergenceEvent] = field(default_factory=list)


@njit(cache=True)
def _subsequent_shift_kernel(close: np.ndarray, lookahead: int) -> np.ndarray:
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = i + lookahead
        if j < n:
            out[i] = close[j] - close[i]
        else:
            out[i] = np.nan
    return out


def subsequent_shift_divergence(
    bars: OHLCV,
    effort_result: EffortResult,
    *,
    lookahead: int = 5,
    shift_atr_threshold: float = 0.5,
    atr_window: int = 14,
    require_opposing_body: bool = False,
) -> ShiftDivergence:
    """
    Confirm absorption bars by their subsequent close-to-close shift.

    For every bar flagged in ``effort_result.flag_absorption`` we measure
    ``close[i + lookahead] - close[i]`` and normalise by ``ATR(i)``. The bar
    is tagged *bullish* when the normalised shift is ``>= +threshold`` and
    *bearish* when it is ``<= -threshold``. Tagging is gated by the lookahead
    and ATR warmups: the originating bar must have a finite ATR and the
    lookahead bar must exist.

    Parameters
    ----------
    bars                   OHLCV sequence (same instance used for
                           ``effort_vs_result``).
    effort_result          Output of :func:`effort_vs_result` for ``bars``.
    lookahead              Forward window for the shift; default 5 bars.
    shift_atr_threshold    Minimum ``|shift| / ATR`` for confirmation;
                           default 0.5.
    atr_window             Window for the ATR normaliser; default 14.
    require_opposing_body  When True, only confirm a divergence whose
                           subsequent shift opposes the visible body of the
                           absorption bar (i.e. bullish requires
                           ``close <= open`` at the bar and vice versa).
                           Skipping this check is the default since absorption
                           bars have near-zero bodies by construction.
    """
    if lookahead < 1:
        raise ValueError(f"lookahead must be >= 1, got {lookahead}")
    if shift_atr_threshold <= 0.0:
        raise ValueError(f"shift_atr_threshold must be > 0, got {shift_atr_threshold}")

    n = bars.n_bars
    flag = effort_result.flag_absorption
    if flag.shape[0] != n:
        raise ValueError(f"effort_result length {flag.shape[0]} does not match bars length {n}")

    close = bars.close()
    open_ = bars.open()
    a = atr(bars, window=atr_window)
    shift = _subsequent_shift_kernel(close, lookahead)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift_atr_arr = np.where((a > 0.0) & np.isfinite(a), shift / a, np.nan)

    body_sign = np.sign(close - open_).astype(np.int8)
    bullish = flag & (shift_atr_arr >= shift_atr_threshold)
    bearish = flag & (shift_atr_arr <= -shift_atr_threshold)
    if require_opposing_body:
        bullish &= body_sign <= 0
        bearish &= body_sign >= 0

    kind = np.zeros(n, dtype=np.int8)
    kind[bullish] = 1
    kind[bearish] = -1
    confirmed = bullish | bearish

    diverg = effort_result.divergence
    events: list[DivergenceEvent] = [
        DivergenceEvent(
            idx=int(i),
            kind="bullish" if kind[i] == 1 else "bearish",
            bar_sign=int(body_sign[i]),
            shift=float(shift[i]),
            shift_atr=float(shift_atr_arr[i]),
            divergence_strength=float(diverg[i]),
            lookahead=lookahead,
        )
        for i in np.flatnonzero(confirmed)
    ]
    return ShiftDivergence(
        shift=shift,
        shift_atr=shift_atr_arr,
        kind=kind,
        flag_confirmed=confirmed,
        events=events,
    )


@dataclass(slots=True, frozen=True)
class HarmonyEvent:
    """
    A trend / volume harmony or divergence at a single bar.

    Attributes
    ----------
    idx         Bar index.
    kind        One of ``bullish_harmony``, ``bearish_harmony``,
                ``bullish_divergence``, ``bearish_divergence``.
    trend_sign  ``+1`` uptrend, ``-1`` downtrend, ``0`` flat.
    move_sign   Sign of ``close - open`` at the bar (-1, 0, +1).
    z_volume    Rolling z-score of log volume at the bar.
    slope_atr   Trend slope (price per bar) at the bar, normalised by ATR.
    """

    idx: int
    kind: HarmonyKind
    trend_sign: int
    move_sign: int
    z_volume: float
    slope_atr: float


@dataclass(slots=True)
class PriceMovementHarmony:
    """
    Trend / volume harmony arrays and divergence events.

    Attributes
    ----------
    trend_sign       Int8 array — sign of the rolling slope, gated by
                     ``trend_atr_threshold``. ``0`` outside a clear trend.
    move_sign        Int8 array — sign of ``close - open`` per bar.
    z_volume         Rolling z-score of log volume.
    slope_atr        Rolling slope of close divided by ATR.
    harmony          Int8 array — ``+1`` confirming harmony (in-trend move
                     on heavy volume), ``-1`` divergence (in-trend move on
                     light volume), ``0`` otherwise.
    flag_divergence  Bool array — True where ``harmony == -1``.
    flag_harmony     Bool array — True where ``harmony == +1``.
    events           List of divergence events (in-trend moves on light
                     volume) in bar order. Harmony bars are not emitted as
                     events; query ``flag_harmony`` for those.
    """

    trend_sign: np.ndarray
    move_sign: np.ndarray
    z_volume: np.ndarray
    slope_atr: np.ndarray
    harmony: np.ndarray
    flag_divergence: np.ndarray
    flag_harmony: np.ndarray
    events: list[HarmonyEvent] = field(default_factory=list)


def price_movement_harmony(
    bars: OHLCV,
    *,
    trend_window: int = 20,
    vol_window: int = 50,
    trend_atr_threshold: float = 0.05,
    high_vol_z: float = 0.5,
    low_vol_z: float = 0.0,
    atr_window: int = 14,
) -> PriceMovementHarmony:
    """
    Detect trend / volume harmony and divergence per bar.

    For every bar we classify three things: the prevailing trend direction
    (sign of the rolling slope, gated so micro-noise is treated as flat),
    the bar's move direction (sign of ``close - open``), and the bar's
    volume regime (rolling z-score of log volume). When the move aligns
    with the trend, heavy volume confirms harmony and light volume signals
    divergence.

    Mapping
    -------
    Uptrend + up bar + high volume    → ``bullish_harmony``     (+1)
    Downtrend + down bar + high volume → ``bearish_harmony``     (+1)
    Uptrend + up bar + low volume     → ``bearish_divergence``  (-1)
    Downtrend + down bar + low volume → ``bullish_divergence``  (-1)
    Anything else                      → 0

    Parameters
    ----------
    bars                  OHLCV sequence.
    trend_window          Window for the rolling slope of close. Default 20.
    vol_window            Window for the log-volume z-score. Default 50.
    trend_atr_threshold   Minimum ``|slope| / ATR`` for a bar to count as
                          trending; below this the trend is treated as flat
                          and no harmony / divergence is emitted. Default 0.05
                          ATR per bar.
    high_vol_z            Volume z-score floor for a *harmony* tag.
                          Default 0.5.
    low_vol_z             Volume z-score ceiling for a *divergence* tag.
                          Default 0.0 (any below-average volume).
    atr_window            Window for the ATR normaliser. Default 14.
    """
    if trend_window < 2:
        raise ValueError(f"trend_window must be >= 2, got {trend_window}")
    if vol_window < 2:
        raise ValueError(f"vol_window must be >= 2, got {vol_window}")
    if trend_atr_threshold < 0.0:
        raise ValueError(f"trend_atr_threshold must be >= 0, got {trend_atr_threshold}")
    if high_vol_z < low_vol_z:
        raise ValueError(f"high_vol_z ({high_vol_z}) must be >= low_vol_z ({low_vol_z})")

    close = bars.close()
    open_ = bars.open()
    vol = bars.volume()

    slope = rolling_slope(close, trend_window)
    a = atr(bars, window=atr_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_atr = np.where((a > 0.0) & np.isfinite(a), slope / a, np.nan)

    z_vol = rolling_zscore(np.log(vol + 1.0), vol_window)

    trend_sign = np.zeros(close.shape[0], dtype=np.int8)
    valid_slope = np.isfinite(slope_atr)
    trend_sign[valid_slope & (slope_atr >= trend_atr_threshold)] = 1
    trend_sign[valid_slope & (slope_atr <= -trend_atr_threshold)] = -1

    move_sign = np.sign(close - open_).astype(np.int8)

    aligned = (trend_sign != 0) & (move_sign == trend_sign)
    valid_vol = np.isfinite(z_vol)

    harmony_mask = aligned & valid_vol & (z_vol >= high_vol_z)
    divergence_mask = aligned & valid_vol & (z_vol <= low_vol_z)

    harmony = np.zeros(close.shape[0], dtype=np.int8)
    harmony[harmony_mask] = 1
    harmony[divergence_mask] = -1

    events: list[HarmonyEvent] = []
    for i in np.flatnonzero(divergence_mask):
        ts = int(trend_sign[i])
        kind: HarmonyKind = "bearish_divergence" if ts == 1 else "bullish_divergence"
        events.append(
            HarmonyEvent(
                idx=int(i),
                kind=kind,
                trend_sign=ts,
                move_sign=int(move_sign[i]),
                z_volume=float(z_vol[i]),
                slope_atr=float(slope_atr[i]),
            )
        )

    return PriceMovementHarmony(
        trend_sign=trend_sign,
        move_sign=move_sign,
        z_volume=z_vol,
        slope_atr=slope_atr,
        harmony=harmony,
        flag_divergence=divergence_mask,
        flag_harmony=harmony_mask,
        events=events,
    )
