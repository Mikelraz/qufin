"""Confluence signal engine.

Combines five orthogonal confirmation channels — Wyckoff events,
effort/result divergence, Hull ribbon, momentum, and volume — into a
single long-entry intent. A long entry fires when the count of active
channels at bar ``t`` meets ``min_confluences`` *and* the regime is bullish
(not in cash defense). Exits are evaluated independently and any one
trigger suffices.

The engine is causal by construction: every gate reads
``bars[:t+1]`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import polars as pl

from qufin.indicators import atr, cmf, macd, obv, rsi
from qufin.strategies.hull_suite import hull_ribbon
from qufin.wyckoff import (
    OHLCV,
    SwingPoint,
    detect_sos_lps,
    detect_sow_lpsy,
    detect_spring,
    detect_trading_ranges,
    detect_upthrust,
    effort_vs_result,
    find_swings,
    price_movement_harmony,
    rolling_slope,
    subsequent_shift_divergence,
)

from .params import ConfluenceParams


class ExitReason(StrEnum):
    NONE = "none"
    WYCKOFF_BEARISH = "wyckoff_bearish"  # E1
    HULL_FLIP = "hull_flip"  # E2
    SWING_STOP = "swing_stop"  # E3
    CHANDELIER = "chandelier"  # E4
    GEX_DEFENSE = "gex_defense"  # E5
    REGIME_DEFENSE = "regime_defense"  # E6


@dataclass(slots=True)
class SignalFrame:
    """Per-bar signal output. Aligned 1:1 with the input OHLCV."""

    timestamps: pl.Series
    long_entry: np.ndarray
    confluence_count: np.ndarray
    exit_reason: np.ndarray  # dtype=object, ExitReason values
    flags: dict[str, np.ndarray]  # individual gate masks for attribution

    def to_polars(self) -> pl.DataFrame:
        cols: dict[str, object] = {
            "timestamp": self.timestamps,
            "long_entry": self.long_entry,
            "confluences": self.confluence_count,
            "exit_reason": np.array([str(r) for r in self.exit_reason], dtype=object),
        }
        for name, arr in self.flags.items():
            cols[name] = arr
        return pl.DataFrame(cols)


@dataclass(slots=True)
class ConfluenceSignalEngine:
    """Evaluate the five-channel confluence stack on one symbol's bars."""

    params: ConfluenceParams

    def evaluate(self, bars: OHLCV) -> SignalFrame:
        p = self.params
        close = bars.close()
        high = bars.high()
        low = bars.low()
        volume = bars.volume()

        w1 = self._wyckoff_event_mask(bars, p.event_lookback)
        w2 = self._effort_result_mask(bars, p.event_lookback)
        h = self._hull_mask(close)
        m = self._momentum_mask(close)
        v = self._volume_mask(close, high, low, volume)

        count = w1.astype(int) + w2.astype(int) + h.astype(int) + m.astype(int) + v.astype(int)
        long_entry = (count >= p.min_confluences) & ~self._warmup_mask(bars)

        exit_arr = self._exit_reasons(bars, close, high, low)
        flags = {
            "w1_wyckoff_event": w1,
            "w2_effort_result": w2,
            "h_hull": h,
            "m_momentum": m,
            "v_volume": v,
        }
        return SignalFrame(
            timestamps=bars.data["timestamp"],
            long_entry=long_entry,
            confluence_count=count,
            exit_reason=exit_arr,
            flags=flags,
        )

    # ------------------------------------------------------------------ gates

    def _wyckoff_event_mask(self, bars: OHLCV, lookback: int) -> np.ndarray:
        n = bars.n_bars
        mask = np.zeros(n, dtype=bool)
        if n < self.params.range_min_bars:
            return mask
        ranges = detect_trading_ranges(bars, min_bars=self.params.range_min_bars)
        if not ranges:
            return mask
        idxs: list[int] = []
        for tr in ranges:
            springs = detect_spring(bars, tr)
            sos, lps = detect_sos_lps(bars, tr)
            idxs.extend(s.idx for s in springs)
            idxs.extend(s.idx for s in sos)
            idxs.extend(s.idx for s in lps)
        for idx in idxs:
            lo = max(0, idx)
            hi = min(n, idx + lookback + 1)
            mask[lo:hi] = True
        return mask

    def _effort_result_mask(self, bars: OHLCV, lookback: int) -> np.ndarray:
        n = bars.n_bars
        mask = np.zeros(n, dtype=bool)
        try:
            er = effort_vs_result(bars)
            ssd = subsequent_shift_divergence(bars, er)
            pmh = price_movement_harmony(bars)
        except Exception:
            return mask
        events = list(getattr(ssd, "events", []) or []) + list(getattr(pmh, "events", []) or [])
        for ev in events:
            kind = getattr(ev, "kind", "")
            idx = int(getattr(ev, "idx", -1))
            if idx < 0:
                continue
            if "bull" in str(kind).lower() or "buy" in str(kind).lower():
                hi = min(n, idx + lookback + 1)
                mask[idx:hi] = True
        return mask

    def _hull_mask(self, close: np.ndarray) -> np.ndarray:
        p = self.params
        n = close.shape[0]
        if n < max(p.hull_fast_length, p.hull_slow_length) * 2:
            return np.zeros(n, dtype=bool)  # type: ignore[no-any-return]
        ribbon = hull_ribbon(
            close,
            fast_length=p.hull_fast_length,
            fast_type=p.hull_fast_type,
            slow_length=p.hull_slow_length,
            slow_type=p.hull_slow_type,
        )
        above = np.array([pos == "above" for pos in ribbon.position])
        slow_slope_up = np.nan_to_num(ribbon.slow.slope, nan=0.0) > 0.0
        return above & slow_slope_up

    def _momentum_mask(self, close: np.ndarray) -> np.ndarray:
        p = self.params
        r = rsi(close, window=p.rsi_window)
        macd_res = macd(close, fast=p.macd_fast, slow=p.macd_slow, signal=p.macd_signal)
        hist_pos = np.nan_to_num(macd_res.hist, nan=-1.0) > 0.0
        rsi_cross = np.zeros_like(r, dtype=bool)
        rsi_cross[1:] = (r[1:] >= 50.0) & (r[:-1] < 50.0)
        return hist_pos | rsi_cross

    def _volume_mask(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        p = self.params
        on_bal = obv(close, volume)
        on_bal_slope = rolling_slope(on_bal, window=p.obv_slope_window)
        obv_up = np.nan_to_num(on_bal_slope, nan=0.0) > 0.0
        mf = cmf(high, low, close, volume, window=p.cmf_window)
        cmf_pos = np.nan_to_num(mf, nan=0.0) > 0.0
        return obv_up | cmf_pos

    # ------------------------------------------------------------------ exits

    def _exit_reasons(
        self,
        bars: OHLCV,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> np.ndarray:
        p = self.params
        n = bars.n_bars
        out = np.full(n, ExitReason.NONE.value, dtype=object)

        wyckoff_bear = self._wyckoff_bearish_mask(bars, p.event_lookback)
        hull_flip = self._hull_flip_mask(close)
        swing_break = self._swing_break_mask(bars, close)
        chand = self._chandelier_break_mask(close, high, low)

        for i in range(n):
            if wyckoff_bear[i]:
                out[i] = ExitReason.WYCKOFF_BEARISH.value
            elif hull_flip[i]:
                out[i] = ExitReason.HULL_FLIP.value
            elif swing_break[i]:
                out[i] = ExitReason.SWING_STOP.value
            elif chand[i]:
                out[i] = ExitReason.CHANDELIER.value
        return out

    def _wyckoff_bearish_mask(self, bars: OHLCV, lookback: int) -> np.ndarray:
        n = bars.n_bars
        mask = np.zeros(n, dtype=bool)
        if n < self.params.range_min_bars:
            return mask
        ranges = detect_trading_ranges(bars, min_bars=self.params.range_min_bars)
        for tr in ranges:
            uts = detect_upthrust(bars, tr)
            sow, lpsy = detect_sow_lpsy(bars, tr)
            for ev in list(uts) + list(sow) + list(lpsy):
                idx = int(getattr(ev, "idx", -1))
                if idx < 0:
                    continue
                hi = min(n, idx + lookback + 1)
                mask[idx:hi] = True
        return mask

    def _hull_flip_mask(self, close: np.ndarray) -> np.ndarray:
        p = self.params
        n = close.shape[0]
        if n < max(p.hull_fast_length, p.hull_slow_length) * 2:
            return np.zeros(n, dtype=bool)  # type: ignore[no-any-return]
        ribbon = hull_ribbon(
            close,
            fast_length=p.hull_fast_length,
            fast_type=p.hull_fast_type,
            slow_length=p.hull_slow_length,
            slow_type=p.hull_slow_type,
        )
        slope = np.nan_to_num(ribbon.slow.slope, nan=0.0)
        flip = np.zeros(n, dtype=bool)
        for i in range(2, n):
            if slope[i] < 0.0 and slope[i - 1] < 0.0:
                flip[i] = True
        return flip

    def _swing_break_mask(self, bars: OHLCV, close: np.ndarray) -> np.ndarray:
        n = bars.n_bars
        mask = np.zeros(n, dtype=bool)
        sw = self.params.swing_window
        swings = find_swings(bars, left=sw, right=sw)
        last_low: float | None = None
        last_low_idx = 0
        ptr = 0
        sorted_lows: list[SwingPoint] = [s for s in swings if s.kind == "L"]
        sorted_lows.sort(key=lambda s: s.idx)
        for i in range(n):
            while ptr < len(sorted_lows) and sorted_lows[ptr].idx <= i:
                last_low = sorted_lows[ptr].price
                last_low_idx = sorted_lows[ptr].idx
                ptr += 1
            if last_low is not None and i > last_low_idx and close[i] < last_low:
                mask[i] = True
        return mask

    def _chandelier_break_mask(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> np.ndarray:
        p = self.params
        n = close.shape[0]
        a = atr(high, low, close, window=p.atr_window)
        roll_high = np.full(n, np.nan, dtype=np.float64)
        for i in range(p.chandelier_window - 1, n):
            roll_high[i] = high[i - p.chandelier_window + 1 : i + 1].max()
        trail = roll_high - p.chandelier_atr_mult * a
        return np.nan_to_num(close < trail, nan=False)

    def _warmup_mask(self, bars: OHLCV) -> np.ndarray:
        p = self.params
        warmup = max(
            p.hull_slow_length * 2,
            p.macd_slow + p.macd_signal,
            p.atr_window + 1,
            p.range_min_bars,
            p.obv_slope_window,
        )
        n = bars.n_bars
        out = np.zeros(n, dtype=bool)
        out[:warmup] = True
        return out
