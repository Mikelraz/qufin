"""
Wyckoff schematic event detectors.

Each detector inspects an ``OHLCV`` sequence (optionally constrained to a
``TradingRange``) and returns the events it found. All detectors are
deterministic and threshold-based; defaults follow conventional Wyckoff
heuristics but every threshold is exposed as a keyword argument.

Accumulation schematic events
-----------------------------
* **PS** — Preliminary Support: first high-volume support bar after a decline.
* **SC** — Selling Climax: outsize volume + range, close in upper portion.
* **AR** — Automatic Rally: first material rally off the SC low.
* **ST** — Secondary Test: low retest within tolerance of SC on lighter volume.
* **Spring** — False break of range support, recovered within ``recovery_bars``.
* **SOS** — Sign of Strength: wide-range, high-volume bar closing above range.
* **LPS** — Last Point of Support: pullback after SOS holding above support.

Distribution schematic events
-----------------------------
* **BC** — Buying Climax (dual of SC).
* **UT** — Upthrust (dual of Spring within range).
* **UTAD** — Upthrust After Distribution (UT *after* an SOW has formed).
* **SOW** — Sign of Weakness: wide-range, high-volume bar closing below range.
* **LPSY** — Last Point of Supply: rally after SOW failing below resistance.
"""

from __future__ import annotations

import numpy as np

from ._types import (
    OHLCV,
    ClimaxEvent,
    SpringEvent,
    StructuralEvent,
    TradingRange,
)
from .bars import atr, bar_range_zscore, normalize_volume, rolling_slope


def _nanmedian_or_nan(x: np.ndarray) -> float:
    """``np.nanmedian`` that returns NaN (silently) on an all-NaN slice."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def detect_climax(
    bars: OHLCV,
    *,
    z_volume: float = 2.5,
    z_range: float = 2.0,
    trend_window: int = 20,
    trend_slope_min: float = 0.0,
    close_position: float = 1.0 / 3.0,
    vol_window: int = 50,
) -> list[ClimaxEvent]:
    """
    Detect Selling-Climax (SC) and Buying-Climax (BC) bars.

    A *Selling Climax* requires:
        * negative trend leading in (slope of close over ``trend_window``
          bars < ``-trend_slope_min``);
        * volume z-score ≥ ``z_volume`` (over ``vol_window``);
        * (high-low) range z-score ≥ ``z_range``;
        * close in the upper ``close_position`` fraction of the bar's range.

    A *Buying Climax* is the dual on a rising trend with the close in the
    lower ``close_position`` fraction of the bar.
    """
    if not 0.0 < close_position <= 0.5:
        raise ValueError(f"close_position must be in (0, 0.5], got {close_position}")
    n = bars.n_bars
    if n < max(trend_window, vol_window):
        return []
    z_v = normalize_volume(bars, window=vol_window)
    z_r = bar_range_zscore(bars, window=vol_window)
    slope = rolling_slope(bars.close(), window=trend_window)
    h = bars.high()
    l = bars.low()  # noqa: E741
    c = bars.close()

    out: list[ClimaxEvent] = []
    for i in range(n):
        if not (np.isfinite(z_v[i]) and np.isfinite(z_r[i]) and np.isfinite(slope[i])):
            continue
        if z_v[i] < z_volume or z_r[i] < z_range:
            continue
        rng = h[i] - l[i]
        if rng <= 0.0:
            continue
        close_pos_from_low = (c[i] - l[i]) / rng
        if slope[i] < -trend_slope_min and close_pos_from_low >= 1.0 - close_position:
            out.append(
                ClimaxEvent(
                    idx=i,
                    kind="SC",
                    z_volume=float(z_v[i]),
                    z_range=float(z_r[i]),
                    price=float(l[i]),
                )
            )
        elif slope[i] > trend_slope_min and close_pos_from_low <= close_position:
            out.append(
                ClimaxEvent(
                    idx=i,
                    kind="BC",
                    z_volume=float(z_v[i]),
                    z_range=float(z_r[i]),
                    price=float(h[i]),
                )
            )
    return out


def detect_automatic_rally(
    bars: OHLCV,
    climax: ClimaxEvent,
    *,
    max_bars: int = 30,
) -> StructuralEvent | None:
    """
    Locate the Automatic Rally (after SC) or Automatic Reaction (after BC).

    Returns the first swing extreme within ``max_bars`` of the climax that
    moves in the recovery direction without a new climax-side extreme being
    set in between.
    """
    n = bars.n_bars
    end = min(n, climax.idx + 1 + max_bars)
    if end <= climax.idx + 1:
        return None
    h = bars.high()
    l = bars.low()  # noqa: E741
    c = bars.close()
    z_v = normalize_volume(bars, window=50)
    if climax.kind == "SC":
        running_low = l[climax.idx]
        best_idx = -1
        best_high = -np.inf
        for i in range(climax.idx + 1, end):
            if l[i] < running_low:
                break
            if h[i] > best_high:
                best_high = float(h[i])
                best_idx = i
        if best_idx > climax.idx and best_high > c[climax.idx]:
            return StructuralEvent(
                idx=best_idx,
                kind="AR",
                price=best_high,
                z_volume=float(z_v[best_idx]) if np.isfinite(z_v[best_idx]) else 0.0,
            )
        return None
    # BC -> Automatic Reaction (still labelled AR)
    running_high = h[climax.idx]
    best_idx = -1
    best_low = np.inf
    for i in range(climax.idx + 1, end):
        if h[i] > running_high:
            break
        if l[i] < best_low:
            best_low = float(l[i])
            best_idx = i
    if best_idx > climax.idx and best_low < c[climax.idx]:
        return StructuralEvent(
            idx=best_idx,
            kind="AR",
            price=best_low,
            z_volume=float(z_v[best_idx]) if np.isfinite(z_v[best_idx]) else 0.0,
        )
    return None


def detect_secondary_test(
    bars: OHLCV,
    climax: ClimaxEvent,
    automatic_rally: StructuralEvent,
    *,
    tolerance_atr: float = 1.0,
    max_bars: int = 60,
    atr_window: int = 14,
) -> StructuralEvent | None:
    """
    Detect Secondary Test (ST) — a return to the climax extreme on lighter volume.
    """
    if automatic_rally.kind != "AR":
        raise ValueError("automatic_rally.kind must be 'AR'")
    n = bars.n_bars
    start = automatic_rally.idx + 1
    end = min(n, start + max_bars)
    if end <= start:
        return None
    h = bars.high()
    l = bars.low()  # noqa: E741
    v = bars.volume()
    a = atr(bars, window=atr_window)
    z_v = normalize_volume(bars, window=50)

    climax_extreme = climax.price
    climax_volume = float(v[climax.idx])
    atr_ref = float(a[climax.idx]) if np.isfinite(a[climax.idx]) else float(np.nanmedian(a))
    if not np.isfinite(atr_ref) or atr_ref <= 0.0:
        return None
    tol = tolerance_atr * atr_ref

    best_idx = -1
    best_price = climax_extreme
    if climax.kind == "SC":
        for i in range(start, end):
            if l[i] <= climax_extreme + tol and l[i] >= climax_extreme - tol:
                if v[i] < climax_volume and l[i] < best_price + tol:
                    best_idx = i
                    best_price = float(l[i])
                    break
    else:  # BC
        for i in range(start, end):
            if h[i] >= climax_extreme - tol and h[i] <= climax_extreme + tol:
                if v[i] < climax_volume and h[i] > best_price - tol:
                    best_idx = i
                    best_price = float(h[i])
                    break
    if best_idx < 0:
        return None
    return StructuralEvent(
        idx=best_idx,
        kind="ST",
        price=best_price,
        z_volume=float(z_v[best_idx]) if np.isfinite(z_v[best_idx]) else 0.0,
    )


def detect_spring(
    bars: OHLCV,
    trading_range: TradingRange,
    *,
    max_penetration_atr: float = 1.5,
    recovery_bars: int = 5,
    atr_window: int = 14,
) -> list[SpringEvent]:
    """
    Detect Springs — false breakdowns of range support followed by recovery.

    A bar is a spring candidate when:
        * its low penetrates support by ≤ ``max_penetration_atr × ATR``;
        * within the next ``recovery_bars``, the close re-enters the range
          (i.e. ``close > support``);
        * the penetration bar's volume z-score is not abnormally high
          (springs are typically *not* climactic).
    """
    n = bars.n_bars
    l = bars.low()  # noqa: E741
    c = bars.close()
    a = atr(bars, window=atr_window)
    z_v = normalize_volume(bars, window=50)
    support = trading_range.support
    end = min(n, trading_range.end_idx)

    out: list[SpringEvent] = []
    for i in range(trading_range.start_idx, end):
        atr_ref = float(a[i]) if np.isfinite(a[i]) else _nanmedian_or_nan(a[: i + 1])
        if not np.isfinite(atr_ref) or atr_ref <= 0.0:
            continue
        penetration = support - float(l[i])
        if penetration <= 0.0 or penetration > max_penetration_atr * atr_ref:
            continue
        recovered_in: int = -1
        for k in range(1, recovery_bars + 1):
            j = i + k
            if j >= n:
                break
            if c[j] > support:
                recovered_in = k
                break
        if recovered_in < 0:
            continue
        out.append(
            SpringEvent(
                idx=i,
                kind="Spring",
                penetration=float(penetration),
                recovery_bars=int(recovered_in),
                z_volume=float(z_v[i]) if np.isfinite(z_v[i]) else 0.0,
            )
        )
    return out


def detect_upthrust(
    bars: OHLCV,
    trading_range: TradingRange,
    *,
    max_penetration_atr: float = 1.5,
    recovery_bars: int = 5,
    atr_window: int = 14,
    is_utad: bool = False,
) -> list[SpringEvent]:
    """
    Detect Upthrusts (UT) — the dual of Spring above resistance.

    Set ``is_utad=True`` to label the events as ``"UTAD"`` (upthrust after
    distribution) for use in distribution schematics where the UT occurs
    after an SOW.
    """
    n = bars.n_bars
    h = bars.high()
    c = bars.close()
    a = atr(bars, window=atr_window)
    z_v = normalize_volume(bars, window=50)
    resistance = trading_range.resistance
    end = min(n, trading_range.end_idx)
    label: str = "UTAD" if is_utad else "UT"

    out: list[SpringEvent] = []
    for i in range(trading_range.start_idx, end):
        atr_ref = float(a[i]) if np.isfinite(a[i]) else _nanmedian_or_nan(a[: i + 1])
        if not np.isfinite(atr_ref) or atr_ref <= 0.0:
            continue
        penetration = float(h[i]) - resistance
        if penetration <= 0.0 or penetration > max_penetration_atr * atr_ref:
            continue
        recovered_in: int = -1
        for k in range(1, recovery_bars + 1):
            j = i + k
            if j >= n:
                break
            if c[j] < resistance:
                recovered_in = k
                break
        if recovered_in < 0:
            continue
        out.append(
            SpringEvent(
                idx=i,
                kind=label,  # type: ignore[arg-type]
                penetration=float(penetration),
                recovery_bars=int(recovered_in),
                z_volume=float(z_v[i]) if np.isfinite(z_v[i]) else 0.0,
            )
        )
    return out


def detect_sos_lps(
    bars: OHLCV,
    trading_range: TradingRange,
    *,
    breakout_z_volume: float = 1.0,
    breakout_z_range: float = 1.0,
    lookahead: int = 30,
    pullback_atr: float = 1.5,
    atr_window: int = 14,
) -> tuple[list[StructuralEvent], list[StructuralEvent]]:
    """
    Detect Sign of Strength (SOS) and Last Point of Support (LPS).

    An SOS is a wide-range, high-volume bar after the range that closes above
    ``trading_range.resistance``. An LPS is the first subsequent pullback that
    holds within ``pullback_atr × ATR`` above the resistance line, on declining
    relative volume.

    Returns ``(sos_events, lps_events)``.
    """
    n = bars.n_bars
    start = trading_range.end_idx
    end = min(n, start + lookahead)
    if end <= start:
        return [], []
    l = bars.low()  # noqa: E741
    c = bars.close()
    z_v = normalize_volume(bars, window=50)
    z_r = bar_range_zscore(bars, window=50)
    a = atr(bars, window=atr_window)
    sos: list[StructuralEvent] = []
    for i in range(start, end):
        if not (np.isfinite(z_v[i]) and np.isfinite(z_r[i])):
            continue
        if z_v[i] >= breakout_z_volume and z_r[i] >= breakout_z_range:
            if c[i] > trading_range.resistance:
                sos.append(
                    StructuralEvent(
                        idx=i,
                        kind="SOS",
                        price=float(c[i]),
                        z_volume=float(z_v[i]),
                    )
                )
    lps: list[StructuralEvent] = []
    for s in sos:
        for j in range(s.idx + 1, min(n, s.idx + 1 + lookahead)):
            atr_ref = float(a[j]) if np.isfinite(a[j]) else float(np.nanmedian(a[: j + 1]))
            if not np.isfinite(atr_ref) or atr_ref <= 0.0:
                continue
            if l[j] >= trading_range.resistance - pullback_atr * atr_ref:
                if l[j] <= trading_range.resistance + pullback_atr * atr_ref:
                    if z_v[j] < s.z_volume:
                        lps.append(
                            StructuralEvent(
                                idx=j,
                                kind="LPS",
                                price=float(l[j]),
                                z_volume=float(z_v[j]) if np.isfinite(z_v[j]) else 0.0,
                            )
                        )
                        break
            elif c[j] < trading_range.resistance - pullback_atr * atr_ref:
                break
    return sos, lps


def detect_sow_lpsy(
    bars: OHLCV,
    trading_range: TradingRange,
    *,
    breakout_z_volume: float = 1.0,
    breakout_z_range: float = 1.0,
    lookahead: int = 30,
    pullback_atr: float = 1.5,
    atr_window: int = 14,
) -> tuple[list[StructuralEvent], list[StructuralEvent]]:
    """
    Detect Sign of Weakness (SOW) and Last Point of Supply (LPSY).

    Dual of ``detect_sos_lps`` on the downside.
    """
    n = bars.n_bars
    start = trading_range.end_idx
    end = min(n, start + lookahead)
    if end <= start:
        return [], []
    h = bars.high()
    c = bars.close()
    z_v = normalize_volume(bars, window=50)
    z_r = bar_range_zscore(bars, window=50)
    a = atr(bars, window=atr_window)
    sow: list[StructuralEvent] = []
    for i in range(start, end):
        if not (np.isfinite(z_v[i]) and np.isfinite(z_r[i])):
            continue
        if z_v[i] >= breakout_z_volume and z_r[i] >= breakout_z_range:
            if c[i] < trading_range.support:
                sow.append(
                    StructuralEvent(
                        idx=i,
                        kind="SOW",
                        price=float(c[i]),
                        z_volume=float(z_v[i]),
                    )
                )
    lpsy: list[StructuralEvent] = []
    for s in sow:
        for j in range(s.idx + 1, min(n, s.idx + 1 + lookahead)):
            atr_ref = float(a[j]) if np.isfinite(a[j]) else float(np.nanmedian(a[: j + 1]))
            if not np.isfinite(atr_ref) or atr_ref <= 0.0:
                continue
            if h[j] <= trading_range.support + pullback_atr * atr_ref:
                if h[j] >= trading_range.support - pullback_atr * atr_ref:
                    if z_v[j] < s.z_volume:
                        lpsy.append(
                            StructuralEvent(
                                idx=j,
                                kind="LPSY",
                                price=float(h[j]),
                                z_volume=float(z_v[j]) if np.isfinite(z_v[j]) else 0.0,
                            )
                        )
                        break
            elif c[j] > trading_range.support + pullback_atr * atr_ref:
                break
    return sow, lpsy


def detect_preliminary_support(
    bars: OHLCV,
    *,
    trend_window: int = 30,
    z_volume_min: float = 1.5,
    vol_window: int = 50,
) -> list[StructuralEvent]:
    """
    Detect Preliminary Support (PS) — high-volume support bars during a decline.

    A bar qualifies as PS if:
        * the rolling slope of close over ``trend_window`` is negative;
        * the bar's volume z-score ≥ ``z_volume_min``;
        * the close is materially above the bar's low (lower wick).
    """
    n = bars.n_bars
    if n < max(trend_window, vol_window):
        return []
    z_v = normalize_volume(bars, window=vol_window)
    slope = rolling_slope(bars.close(), window=trend_window)
    h = bars.high()
    l = bars.low()  # noqa: E741
    c = bars.close()
    out: list[StructuralEvent] = []
    for i in range(n):
        if not (np.isfinite(z_v[i]) and np.isfinite(slope[i])):
            continue
        if slope[i] >= 0.0 or z_v[i] < z_volume_min:
            continue
        rng = h[i] - l[i]
        if rng <= 0.0:
            continue
        if (c[i] - l[i]) / rng < 0.5:
            continue
        out.append(
            StructuralEvent(
                idx=i,
                kind="PS",
                price=float(l[i]),
                z_volume=float(z_v[i]),
            )
        )
    return out
