"""Wyckoff event detectors: SC, AR, ST, Spring, UT, SOS, LPS."""

from __future__ import annotations

import numpy as np

from qufin.wyckoff import (
    TradingRange,
    detect_automatic_rally,
    detect_climax,
    detect_secondary_test,
    detect_sos_lps,
    detect_sow_lpsy,
    detect_spring,
    detect_upthrust,
)
from tests.wyckoff.conftest import make_ohlcv


def _baseline_drift(n: int, slope: float, sigma: float, seed: int, base: float = 100.0):
    rng = np.random.default_rng(seed)
    log_steps = rng.normal(slope, sigma, n)
    closes = base * np.exp(np.cumsum(log_steps))
    opens = np.concatenate(([base], closes[:-1]))
    noise = rng.uniform(0.0, 0.5 * sigma, n) * closes
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    vols = rng.lognormal(10.0, 0.2, n)
    return opens, highs, lows, closes, vols


def test_selling_climax_at_planted_bar() -> None:
    # 80 bars of downtrend, then a wide-range climax bar with massive volume
    # and a close in the upper third — followed by 30 calmer bars.
    n_down = 80
    n_after = 30
    n = n_down + 1 + n_after
    opens, highs, lows, closes, vols = _baseline_drift(n, slope=-0.01, sigma=0.005, seed=0)
    k = n_down
    rng_size = 6.0
    lows[k] = closes[k - 1] - rng_size
    highs[k] = closes[k - 1] + rng_size * 0.5
    closes[k] = highs[k] - rng_size * 0.3  # upper third
    opens[k] = closes[k - 1]
    vols[k] = vols.mean() * 30.0

    bars = make_ohlcv(opens, highs, lows, closes, vols)
    climaxes = detect_climax(bars, vol_window=40, trend_window=20)
    sc_indices = [c.idx for c in climaxes if c.kind == "SC"]
    assert k in sc_indices


def test_automatic_rally_after_sc() -> None:
    n_down = 80
    n_after = 30
    n = n_down + 1 + n_after
    opens, highs, lows, closes, vols = _baseline_drift(n, slope=-0.01, sigma=0.005, seed=2)
    k = n_down
    rng_size = 6.0
    lows[k] = closes[k - 1] - rng_size
    highs[k] = closes[k - 1] + rng_size * 0.5
    closes[k] = highs[k] - rng_size * 0.3
    opens[k] = closes[k - 1]
    vols[k] = vols.mean() * 30.0
    # Force a clean rally for the AR.
    for j in range(k + 1, k + 15):
        closes[j] = closes[j - 1] + 0.5
        opens[j] = closes[j - 1]
        highs[j] = closes[j] + 0.1
        lows[j] = opens[j] - 0.1

    bars = make_ohlcv(opens, highs, lows, closes, vols)
    climaxes = detect_climax(bars, vol_window=40, trend_window=20)
    sc = next(c for c in climaxes if c.kind == "SC" and c.idx == k)
    ar = detect_automatic_rally(bars, sc, max_bars=20)
    assert ar is not None
    assert ar.kind == "AR"
    assert ar.idx > sc.idx
    assert ar.price > sc.price


def test_spring_detection_recovers_into_range() -> None:
    # Build a lateral range [98, 102] for 60 bars, then a single bar dipping
    # to 97.5 and recovering above support within 2 bars.
    n = 80
    rng = np.random.default_rng(3)
    closes = 100.0 + 0.5 * rng.normal(0, 1, n)
    closes = np.clip(closes, 98.5, 101.5)
    opens = closes
    highs = closes + 0.2
    lows = closes - 0.2
    vols = np.full(n, 1.0)
    k = 60
    lows[k] = 97.5
    closes[k] = 98.5  # close back inside range edge
    closes[k + 1] = 99.5
    closes[k + 2] = 100.5
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    tr = TradingRange(start_idx=5, end_idx=75, support=98.0, resistance=102.0)
    springs = detect_spring(bars, tr, max_penetration_atr=10.0)
    spring_indices = [s.idx for s in springs]
    assert k in spring_indices


def test_upthrust_detection_mirrors_spring() -> None:
    n = 80
    rng = np.random.default_rng(4)
    closes = 100.0 + 0.5 * rng.normal(0, 1, n)
    closes = np.clip(closes, 98.5, 101.5)
    opens = closes
    highs = closes + 0.2
    lows = closes - 0.2
    vols = np.full(n, 1.0)
    k = 60
    highs[k] = 102.5
    closes[k] = 101.5
    closes[k + 1] = 100.5
    closes[k + 2] = 99.5
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    tr = TradingRange(start_idx=5, end_idx=75, support=98.0, resistance=102.0)
    upthrusts = detect_upthrust(bars, tr, max_penetration_atr=10.0)
    assert any(u.idx == k for u in upthrusts)


def test_sos_lps_after_range() -> None:
    # 80 bars range [98, 102], then a powerful breakout bar above resistance
    # on high volume, then a pullback that holds above resistance.
    n_range = 80
    n_after = 30
    n = n_range + n_after
    rng = np.random.default_rng(5)
    closes = np.empty(n)
    closes[:n_range] = np.clip(100.0 + 0.5 * rng.normal(0, 1, n_range), 98.5, 101.5)
    opens = closes.copy()
    highs = closes + 0.2
    lows = closes - 0.2
    vols = rng.lognormal(10.0, 0.1, n)
    # SOS bar: wide-range, high-volume close > resistance.
    k_sos = n_range
    closes[k_sos] = 105.0
    opens[k_sos] = 102.0
    highs[k_sos] = 106.0
    lows[k_sos] = 101.9
    vols[k_sos] *= 20.0
    # Pullback: bar drifting back to ~103, low volume.
    for j in range(k_sos + 1, k_sos + 6):
        closes[j] = 103.0
        opens[j] = 103.5
        highs[j] = 103.7
        lows[j] = 102.5
        vols[j] = vols[k_sos] * 0.1

    bars = make_ohlcv(opens, highs, lows, closes, vols)
    tr = TradingRange(start_idx=5, end_idx=n_range, support=98.0, resistance=102.0)
    sos, lps = detect_sos_lps(bars, tr)
    assert any(s.idx == k_sos for s in sos)
    assert len(lps) >= 1
    assert lps[0].idx > k_sos


def test_secondary_test_lower_volume_required() -> None:
    n = 150
    opens, highs, lows, closes, vols = _baseline_drift(n, slope=-0.01, sigma=0.005, seed=6)
    k = 80
    # SC bar
    rng_size = 6.0
    lows[k] = closes[k - 1] - rng_size
    highs[k] = closes[k - 1] + rng_size * 0.5
    closes[k] = highs[k] - rng_size * 0.3
    opens[k] = closes[k - 1]
    vols[k] = vols.mean() * 30.0
    # Rally up
    for j in range(k + 1, k + 15):
        closes[j] = closes[j - 1] + 0.5
        opens[j] = closes[j - 1]
        highs[j] = closes[j] + 0.1
        lows[j] = opens[j] - 0.1
    # Secondary test back near SC low on light volume
    ks = k + 25
    closes[ks] = lows[k] + 0.5
    opens[ks] = closes[ks - 1]
    lows[ks] = lows[k] + 0.1
    highs[ks] = closes[ks - 1] + 0.1
    vols[ks] = vols[k] * 0.05

    bars = make_ohlcv(opens, highs, lows, closes, vols)
    climaxes = detect_climax(bars, vol_window=40, trend_window=20)
    sc = next(c for c in climaxes if c.kind == "SC" and c.idx == k)
    ar = detect_automatic_rally(bars, sc, max_bars=20)
    assert ar is not None
    st = detect_secondary_test(bars, sc, ar, tolerance_atr=5.0, max_bars=60)
    assert st is not None
    assert st.kind == "ST"


def test_sow_lpsy_after_range() -> None:
    n_range = 80
    n_after = 30
    n = n_range + n_after
    rng = np.random.default_rng(7)
    closes = np.empty(n)
    closes[:n_range] = np.clip(100.0 + 0.5 * rng.normal(0, 1, n_range), 98.5, 101.5)
    opens = closes.copy()
    highs = closes + 0.2
    lows = closes - 0.2
    vols = rng.lognormal(10.0, 0.1, n)
    k_sow = n_range
    closes[k_sow] = 95.0
    opens[k_sow] = 98.0
    highs[k_sow] = 98.1
    lows[k_sow] = 94.0
    vols[k_sow] *= 20.0
    for j in range(k_sow + 1, k_sow + 6):
        closes[j] = 97.0
        opens[j] = 96.5
        highs[j] = 97.5
        lows[j] = 96.3
        vols[j] = vols[k_sow] * 0.1
    bars = make_ohlcv(opens, highs, lows, closes, vols)
    tr = TradingRange(start_idx=5, end_idx=n_range, support=98.0, resistance=102.0)
    sow, lpsy = detect_sow_lpsy(bars, tr)
    assert any(s.idx == k_sow for s in sow)
    assert len(lpsy) >= 1
