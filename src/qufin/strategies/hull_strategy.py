"""
Hull Suite signal logic and confluence filters.

This module turns the dual-band ribbon from :mod:`qufin.strategies.hull_suite`
into +1 / −1 / 0 trading signals, and provides the confluence filters needed
to reduce false entries:

* :func:`generate_signals` — core ribbon-break + slope rule.
* :func:`multi_timeframe_filter` — gate a lower-timeframe signal by the
  higher-timeframe ribbon direction (e.g. 15-min entry only when 4-H ribbon
  is green for longs).
* :func:`vwap_filter` — keep intraday longs only above VWAP, shorts only
  below.
* :func:`momentum_filter` — optional RSI / MACD timing gate.

Signal logic (no classic MA crossovers)
---------------------------------------
* **Long entry** when price *breaks above the ribbon from below or from
  inside* AND both bands are rising (ribbon turns green).
* **Short entry** when price breaks *below the ribbon from above or from
  inside* AND both bands are falling (ribbon turns red).
* **Long exit** when close < ribbon-bottom OR the fast band turns red.
* **Short exit** when close > ribbon-top OR the fast band turns green.
* While price is *inside* the ribbon with no slope flip, the existing
  position is held — interpreted as a consolidation/retest, not a reversal.

Usage
-----
    >>> import polars as pl
    >>> from qufin.strategies.hull_strategy import generate_signals
    >>> bars: pl.DataFrame = ...  # columns: open/high/low/close/volume
    >>> sig = generate_signals(bars, fast_length=50, slow_length=60)
    >>> # sig is an int8 polars Series of {-1, 0, +1}
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..indicators._types import to_numpy_1d
from .hull_suite import HullRibbon, HullVariant, hull_ribbon

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _close_array(ohlcv: pl.DataFrame | np.ndarray | pl.Series) -> np.ndarray:
    """Extract the close column from an OHLCV DataFrame, or pass through arrays."""
    match ohlcv:
        case pl.DataFrame():
            if "close" not in ohlcv.columns:
                raise ValueError("OHLCV DataFrame must contain a 'close' column")
            return to_numpy_1d(ohlcv["close"])
        case pl.Series() | np.ndarray():
            return to_numpy_1d(ohlcv)
        case _:
            return to_numpy_1d(np.asarray(ohlcv))


def _as_int_signals(arr: np.ndarray) -> pl.Series:
    """Wrap a {-1, 0, +1} float array as an Int8 polars Series."""
    return pl.Series("signal", arr.astype(np.int8), dtype=pl.Int8)


# ---------------------------------------------------------------------------
# Core signal generation
# ---------------------------------------------------------------------------


def _signals_from_ribbon(close: np.ndarray, ribbon: HullRibbon) -> np.ndarray:
    """
    Stateful position evolution from a price series + pre-computed ribbon.

    The state machine is fully causal: ``position[t]`` depends only on
    information available through bar ``t`` (close, ribbon values up to t).
    """
    n = close.shape[0]
    pos = np.zeros(n, dtype=np.int8)

    fast_v = ribbon.fast.values
    slow_v = ribbon.slow.values
    fast_sl = ribbon.fast.slope
    slow_sl = ribbon.slow.slope
    where = ribbon.position

    cur = 0  # current position {-1, 0, +1}
    for t in range(n):
        if np.isnan(fast_v[t]) or np.isnan(slow_v[t]) or np.isnan(close[t]):
            pos[t] = cur
            continue

        top = max(fast_v[t], slow_v[t])
        bot = min(fast_v[t], slow_v[t])
        where_t = where[t]
        where_prev = where[t - 1] if t > 0 else ""
        f_up = fast_sl[t] > 0.0
        f_dn = fast_sl[t] < 0.0
        s_up = slow_sl[t] > 0.0
        s_dn = slow_sl[t] < 0.0

        # Exits first — a fast-band colour flip closes immediately so we never
        # hold counter-trend during what becomes the next entry's confirmation.
        if cur > 0 and (close[t] < bot or f_dn):
            cur = 0
        elif cur < 0 and (close[t] > top or f_up):
            cur = 0

        # Entries — require a fresh break (prev bar not yet on the new side)
        # AND both bands sloping with the break.
        if cur == 0:
            if where_t == "above" and where_prev != "above" and f_up and s_up:
                cur = 1
            elif where_t == "below" and where_prev != "below" and f_dn and s_dn:
                cur = -1

        pos[t] = cur

    return pos


def generate_signals(
    ohlcv_df: pl.DataFrame | np.ndarray | pl.Series,
    fast_length: int = 50,
    slow_length: int = 60,
    fast_type: HullVariant = "hma",
    slow_type: HullVariant = "ehma",
    length_multiplier: float = 1.0,
) -> pl.Series:
    """
    Generate a {-1, 0, +1} trading signal from price + the Hull ribbon.

    Parameters
    ----------
    ohlcv_df : polars.DataFrame, polars.Series, or 1-D array
        Either an OHLCV frame with a ``'close'`` column or a bare price
        series.
    fast_length : int, default 50
        Lookback for the fast band.  55 is a common swing-trading tweak.
    slow_length : int, default 60
        Lookback for the slow band.
    fast_type, slow_type : {'hma', 'thma', 'ehma'}
        Hull variant per band.  Defaults: fast=``hma``, slow=``ehma``.
    length_multiplier : float, default 1.0
        Scales both lengths uniformly — useful to preview a higher-timeframe
        ribbon on a single chart.

    Returns
    -------
    polars.Series
        Int8 series of length T with values in {-1, 0, +1}:

        * +1 — long while price holds above the ribbon and the fast band is
          green.
        * −1 — short while price holds below the ribbon and the fast band is
          red.
        * 0 — flat (no entry has confirmed, or an exit rule fired).
    """
    close = _close_array(ohlcv_df)
    ribbon = hull_ribbon(
        close,
        fast_length=fast_length,
        fast_type=fast_type,
        slow_length=slow_length,
        slow_type=slow_type,
        length_multiplier=length_multiplier,
    )
    pos = _signals_from_ribbon(close, ribbon)
    return _as_int_signals(pos)


# ---------------------------------------------------------------------------
# Confluence filters
# ---------------------------------------------------------------------------


def multi_timeframe_filter(
    signals_htf: pl.Series | np.ndarray,
    signals_ltf: pl.Series | np.ndarray,
) -> pl.Series:
    """
    Gate a lower-timeframe signal by the higher-timeframe trend direction.

    The two series must already be aligned bar-for-bar — the caller is
    responsible for forward-filling the HTF signal onto the LTF time grid
    (typical pattern: ``ltf.join_asof(htf, on='timestamp', strategy='backward')``).

    Rule: an LTF long is only kept if the HTF signal is ``+1`` on the same
    bar; an LTF short only if HTF is ``-1``.  Anything else is forced to 0.
    This implements the spec's "reject counter-trend signals on lower
    timeframes" rule.
    """
    htf = to_numpy_1d(signals_htf)
    ltf = to_numpy_1d(signals_ltf)
    if htf.shape[0] != ltf.shape[0]:
        raise ValueError(
            f"signals_htf and signals_ltf must be aligned and same length "
            f"({htf.shape[0]} vs {ltf.shape[0]})"
        )
    aligned = ((ltf > 0) & (htf > 0)).astype(np.int8) - ((ltf < 0) & (htf < 0)).astype(np.int8)
    return _as_int_signals(aligned)


def vwap_filter(
    signals: pl.Series | np.ndarray,
    price: pl.Series | np.ndarray,
    vwap: pl.Series | np.ndarray,
) -> pl.Series:
    """
    Drop signals whose direction disagrees with price-vs-VWAP.

    For intraday Hull entries the spec says: only longs when price > VWAP,
    only shorts when price < VWAP.  All other bars are forced to flat.

    Works with daily-anchored, session-anchored, or rolling VWAPs — the
    function only consumes the VWAP value itself, so any anchoring choice
    upstream is fine.
    """
    sig = to_numpy_1d(signals)
    p = to_numpy_1d(price)
    v = to_numpy_1d(vwap)
    if not (sig.shape[0] == p.shape[0] == v.shape[0]):
        raise ValueError("signals, price, vwap must share length")
    out = np.zeros_like(sig, dtype=np.int8)
    above = p > v
    below = p < v
    out[(sig > 0) & above] = 1
    out[(sig < 0) & below] = -1
    return _as_int_signals(out)


def momentum_filter(
    signals: pl.Series | np.ndarray,
    rsi_series: pl.Series | np.ndarray | None = None,
    macd_series: pl.Series | np.ndarray | None = None,
    rsi_long_floor: float = 50.0,
    rsi_short_ceiling: float = 50.0,
) -> pl.Series:
    """
    Optional RSI / MACD timing gate for Hull signals.

    The Hull-Suite rule of thumb is: take a long only when the Hull is
    *re-accelerating*, i.e. RSI is turning up from oversold while the ribbon
    flips green; take a short only when RSI is rolling over from
    overbought.  This filter enforces that with a simple threshold check:

    * Long allowed when ``rsi > rsi_long_floor`` (default 50 — bullish half)
      AND ``macd > 0`` if MACD is supplied.
    * Short allowed when ``rsi < rsi_short_ceiling`` (default 50) AND
      ``macd < 0`` if MACD is supplied.

    Either of ``rsi_series`` or ``macd_series`` may be ``None`` to skip that
    leg of the gate.  If both are ``None`` the signals are passed through
    unchanged.
    """
    sig = to_numpy_1d(signals).astype(np.int8)
    if rsi_series is None and macd_series is None:
        return _as_int_signals(sig)

    n = sig.shape[0]
    long_ok = np.ones(n, dtype=bool)
    short_ok = np.ones(n, dtype=bool)

    if rsi_series is not None:
        rsi = to_numpy_1d(rsi_series)
        if rsi.shape[0] != n:
            raise ValueError("rsi_series length mismatch")
        long_ok &= rsi > rsi_long_floor
        short_ok &= rsi < rsi_short_ceiling

    if macd_series is not None:
        macd = to_numpy_1d(macd_series)
        if macd.shape[0] != n:
            raise ValueError("macd_series length mismatch")
        long_ok &= macd > 0.0
        short_ok &= macd < 0.0

    out = np.zeros(n, dtype=np.int8)
    out[(sig > 0) & long_ok] = 1
    out[(sig < 0) & short_ok] = -1
    return _as_int_signals(out)
