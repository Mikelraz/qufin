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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._types import OHLCV
from .bars import rolling_zscore, true_range


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
