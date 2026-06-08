"""
One-shot opportunity scan after the 2026-06-05 sell-off.

Two independent analyses, both driven by live yfinance data through the
qufin stack:

A. GOOGL long-term accumulation entries
   - trend / momentum context (EMA50/200, RSI, ADX, ATR, supertrend)
   - swing-clustered support levels below spot
   - volume-by-price (POC / value area / high-volume nodes)
   - relative strength vs SPY (+ slope)
   - Wyckoff range / phase read
   - Fibonacci retracement grid

B. "The market goes up" long-call thesis on SPY
   - the systematic EMA-cross long-call signal's current desire (does the
     trend model even want to be long right now?)
   - option-chain greeks at a ~35-DTE target expiry
   - dealer GEX structure: gamma-flip, call/put walls, max pain
   - concrete ATM / OTM call candidates with cost, breakeven, theta, vega

ANALYSIS ONLY. No orders are placed anywhere.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from qufin.data._types import OHLCV
from qufin.indicators import adx, atr, ema, rsi, supertrend
from qufin.indicators.support_resistance import pivot_points, support_resistance_from_swings
from qufin.options import (
    call_wall,
    greeks_for_chain,
    max_pain,
    put_wall,
    zero_gamma_level,
)
from qufin.options.data import load_chain_yfinance
from qufin.strategies.ema_cross_long_call import EmaCrossLongCallParams, current_state
from qufin.wyckoff import (
    classify_phases,
    detect_climax,
    detect_trading_ranges,
    find_swings,
    relative_strength,
    rs_slope,
    volume_profile,
)

# Rough macro assumptions (continuously compounded). Labelled as assumptions
# wherever they feed a number the user might act on.
RISK_FREE = 0.042
SPY_DIV_YIELD = 0.012


def _fetch_ohlcv(symbol: str, *, period: str = "2y") -> tuple[OHLCV, pl.DataFrame]:
    """Return (OHLCV, date+close frame) for ``symbol`` via yfinance -> polars."""
    import yfinance as yf

    pdf = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    if pdf.empty:
        raise RuntimeError(f"no bars for {symbol}")
    pdf = pdf.reset_index()
    date_col = "Date" if "Date" in pdf.columns else "Datetime"
    pdf = pdf.rename(
        columns={
            date_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = pl.from_pandas(pdf[["timestamp", "open", "high", "low", "close", "volume"]])
    ts = df.get_column("timestamp")
    if isinstance(ts.dtype, pl.Datetime) and ts.dtype.time_zone is not None:
        df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    else:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    df = df.with_columns(
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )
    dated = df.select(
        pl.col("timestamp").dt.date().alias("date"), pl.col("close")
    )
    return OHLCV.from_records(df, symbol=symbol), dated


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def analyze_googl() -> None:
    _section("A.  GOOGL — long-term accumulation read")
    bars, googl_dc = _fetch_ohlcv("GOOGL", period="2y")
    _, spy_dc = _fetch_ohlcv("SPY", period="2y")

    high = bars.high()
    low = bars.low()
    close = bars.close()
    spot = float(close[-1])
    n = len(close)

    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    rsi14 = rsi(close, 14)
    atr14 = atr(high, low, close, 14)
    adx_res = adx(high, low, close, 14)
    st = supertrend(high, low, close, 10, 3.0)

    hi_52w = float(np.max(high[-252:])) if n >= 252 else float(np.max(high))
    lo_52w = float(np.min(low[-252:])) if n >= 252 else float(np.min(low))

    print(f"  spot                 = {spot:.2f}")
    print(f"  52w range            = {lo_52w:.2f}  ..  {hi_52w:.2f}"
          f"   (spot is {100 * (spot - lo_52w) / (hi_52w - lo_52w):.0f}% up the range)")
    print(f"  EMA50 / EMA200       = {ema50[-1]:.2f} / {ema200[-1]:.2f}"
          f"   ({'ABOVE' if spot > ema200[-1] else 'BELOW'} 200d, "
          f"{'golden' if ema50[-1] > ema200[-1] else 'death'} cross)")
    print(f"  RSI(14)              = {rsi14[-1]:.1f}"
          f"   ({'oversold' if rsi14[-1] < 30 else 'neutral' if rsi14[-1] < 70 else 'overbought'})")
    print(f"  ADX(14) / +DI / -DI  = {adx_res.adx[-1]:.1f} / "
          f"{adx_res.plus_di[-1]:.1f} / {adx_res.minus_di[-1]:.1f}"
          f"   (trend {'strong' if adx_res.adx[-1] > 25 else 'weak/none'}, "
          f"{'down' if adx_res.minus_di[-1] > adx_res.plus_di[-1] else 'up'}-biased)")
    print(f"  ATR(14)              = {atr14[-1]:.2f}  ({100 * atr14[-1] / spot:.1f}% of spot/day)")
    print(f"  Supertrend           = {'LONG' if st.direction[-1] > 0 else 'SHORT'}"
          f"  line @ {st.line[-1]:.2f}")

    # Swing-clustered support below spot = candidate accumulation rungs.
    # Use a recent window so levels reflect the current price regime, not the
    # year-ago base (GOOGL has ~doubled over the lookback).
    recent_start = max(0, n - 130)
    recent = bars.slice_bars(recent_start, n)
    swings = find_swings(recent, left=3, right=3)
    levels = support_resistance_from_swings(swings, tolerance=0.012)
    below = sorted(
        (lv for lv in levels if lv.price < spot * 0.999),
        key=lambda lv: lv.price,
        reverse=True,
    )
    print("\n  Recent (~6mo) swing supports BELOW spot (nearest first):")
    for lv in below[:6]:
        print(f"    {lv.price:8.2f}   {(lv.price - spot) / spot * 100:+5.1f}%   "
              f"touches={lv.touches:2d}  strength={lv.strength:5.1f}  kind={lv.kind}")

    # Volume-by-price over the last ~year: HVNs are natural demand shelves.
    start = max(0, n - 252)
    vp = volume_profile(bars, n_bins=60, start=start)
    hvn_prices = sorted(
        (float(vp.bin_centres[i]) for i in vp.hvn_idx if vp.bin_centres[i] < spot),
        reverse=True,
    )
    print(f"\n  Volume profile (last {n - start}d):  POC={vp.poc:.2f}  "
          f"VAL={vp.val:.2f}  VAH={vp.vah:.2f}")
    print(f"    high-volume nodes below spot: "
          f"{', '.join(f'{p:.2f}' for p in hvn_prices[:6]) or '(none)'}")

    # Relative strength vs SPY (aligned on date).
    joined = googl_dc.join(spy_dc, on="date", how="inner", suffix="_spy")
    g = joined["close"].to_numpy().astype(np.float64)
    s = joined["close_spy"].to_numpy().astype(np.float64)
    rs = relative_strength(g, s, normalize=True)
    rs_sl = rs_slope(rs, 21)
    print(f"\n  RS vs SPY (rebased)  = {rs[-1]:.3f}   21d slope = {rs_sl[-1]:+.5f}"
          f"   ({'OUT' if rs_sl[-1] > 0 else 'UNDER'}performing lately)")

    # Wyckoff structural read.
    ranges = detect_trading_ranges(bars, min_bars=20)
    climaxes = detect_climax(bars)
    phases = classify_phases(bars, ranges, climaxes, [], [])
    cur_phase = next(
        (p for p in phases if p.start_idx <= n - 1 <= p.end_idx), phases[-1] if phases else None
    )
    last_climax = climaxes[-1] if climaxes else None
    print("\n  Wyckoff:")
    print(f"    trading ranges detected = {len(ranges)}; "
          f"current phase = "
          f"{cur_phase.schematic + '-' + cur_phase.phase if cur_phase else 'none/markdown'}")
    if last_climax is not None:
        cidx = last_climax.idx
        print(f"    last climax = {last_climax.kind} @ idx {cidx} "
              f"(z_vol={last_climax.z_volume:.1f}, price {last_climax.price:.2f})")

    # Fibonacci retracement grid (52w high -> 52w low).
    rng = hi_52w - lo_52w
    print("\n  Fib retracement (52w high->low), as downside reference shelves:")
    for f in (0.382, 0.5, 0.618, 0.786):
        lvl = hi_52w - f * rng
        tag = "  <-- near spot" if abs(lvl - spot) / spot < 0.02 else ""
        print(f"    {int(f * 100):>3}% = {lvl:.2f}{tag}")

    # Near-term floor pivots from the latest completed bar.
    pp = pivot_points(float(high[-1]), float(low[-1]), float(close[-1]))
    print(f"\n  Next-session floor pivots: S1={pp.s1:.2f} S2={pp.s2:.2f} "
          f"S3={pp.s3:.2f} | PP={pp.pp:.2f} | R1={pp.r1:.2f}")


def analyze_spy_calls() -> None:
    _section("B.  SPY — long-call thesis ('market goes up')")
    bars, _ = _fetch_ohlcv("SPY", period="1y")
    close = bars.close()
    spot = float(close[-1])

    # 1. What does the systematic long-call trend model want RIGHT NOW?
    params = EmaCrossLongCallParams(fast_window=12, slow_window=26, target_dte=35, exit_dte=7)
    state = current_state(close, as_of=date.today(), params=params, symbol="SPY")
    print(f"  spot                 = {spot:.2f}")
    print(f"  EMA-cross model      = fast {state.fast_ema:.2f} vs slow {state.slow_ema:.2f}")
    print(f"  systematic desire    = {'LONG CALL' if state.desire_long else 'FLAT (no long)'}"
          f"{'  (fresh cross this bar)' if state.on_fresh_cross else ''}")
    rsi14 = rsi(close, 14)
    print(f"  RSI(14)              = {rsi14[-1]:.1f}")

    # 2. Option chain at the nearest expiry to ~35 DTE, plus near expiries for GEX.
    import yfinance as yf

    all_exp = list(yf.Ticker("SPY").options or ())
    if not all_exp:
        print("  (no option expiries returned; skipping options analysis)")
        return
    today = date.today()

    def _dte(e: str) -> int:
        return (date.fromisoformat(e) - today).days

    target = min(all_exp, key=lambda e: abs(_dte(e) - 35))
    gex_exp = [e for e in all_exp if 1 <= _dte(e) <= 50]
    print(f"\n  target expiry        = {target}  (DTE={_dte(target)})")
    print(f"  loading {len(gex_exp)} near expiries for GEX + greeks...")

    chain = load_chain_yfinance(
        "SPY", expiries=gex_exp, r=RISK_FREE, q=SPY_DIV_YIELD, solve_iv=True, as_of=today
    )

    # Dealer gamma structure.
    try:
        flip = zero_gamma_level(chain)
        cw = call_wall(chain)
        pw = put_wall(chain)
        mp = max_pain(chain)
        print("\n  Dealer GEX structure (near-dated OI):")
        print(f"    gamma flip   = {flip:.2f}" if flip is not None else "    gamma flip   = none in range")
        if flip is not None:
            regime = "NEGATIVE gamma (moves amplified)" if spot < flip else "POSITIVE gamma (moves dampened/pinned)"
            print(f"    -> spot {spot:.2f} is {'below' if spot < flip else 'above'} flip => {regime}")
        print(f"    call wall    = {cw:.2f}   (overhead resistance / pin)")
        print(f"    put wall     = {pw:.2f}   (downside support)")
        print(f"    max pain     = {mp:.2f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (GEX computation skipped: {exc!r})")

    # 3. Call candidates at the target expiry: ATM and OTM rungs.
    tgt = chain.data.filter(
        (pl.col("expiry") == date.fromisoformat(target)) & (pl.col("option_type") == "C")
    ).sort("strike")
    if tgt.height == 0:
        print("  (no calls at target expiry)")
        return

    sub = OptionChainView(chain, tgt, target)
    greeks = sub.greeks()
    strikes = tgt["strike"].to_numpy().astype(np.float64)
    mids = sub.mid()
    ivs = tgt["iv"].to_numpy().astype(np.float64)

    T = (date.fromisoformat(target) - today).days
    print(f"\n  Call candidates @ {target} (DTE={T}, spot {spot:.2f}, "
          f"r={RISK_FREE:.3f}, q={SPY_DIV_YIELD:.3f}):")
    print(f"    {'strike':>7} {'mon%':>6} {'mid':>7} {'delta':>6} {'theta/d':>8} "
          f"{'vega':>6} {'IV%':>6} {'breakeven':>9} {'to BE%':>7} {'cost$':>8}")
    targets_money = [1.00, 1.02, 1.04]  # ATM, ~2% OTM, ~4% OTM
    chosen_idx: list[int] = []
    for m in targets_money:
        want = spot * m
        chosen_idx.append(int(np.argmin(np.abs(strikes - want))))
    for i in sorted(set(chosen_idx)):
        k = strikes[i]
        mid = mids[i]
        be = k + mid
        print(f"    {k:7.0f} {100 * (k / spot - 1):+6.1f} {mid:7.2f} "
              f"{greeks.delta[i]:6.2f} {greeks.theta[i] / 365:8.3f} "
              f"{greeks.vega[i]:6.2f} {100 * ivs[i]:6.1f} {be:9.2f} "
              f"{100 * (be / spot - 1):+7.1f} {mid * chain.multiplier:8.0f}")
    print("\n  (delta = $ per $1 SPY move per share; theta/d = $ decay/day/share;")
    print("   vega = $ per 1.00 vol per share; cost$ = mid x 100 multiplier.)")


class OptionChainView:
    """Thin wrapper to reuse greeks_for_chain / mid on a filtered single-expiry slice."""

    __slots__ = ("_chain",)

    def __init__(self, parent: object, frame: pl.DataFrame, expiry: str) -> None:
        from qufin.options import OptionChain

        self._chain = OptionChain(
            data=frame,
            spot=parent.spot,  # type: ignore[attr-defined]
            as_of=parent.as_of,  # type: ignore[attr-defined]
            underlying=parent.underlying,  # type: ignore[attr-defined]
            r=parent.r,  # type: ignore[attr-defined]
            q=parent.q,  # type: ignore[attr-defined]
            multiplier=parent.multiplier,  # type: ignore[attr-defined]
        )

    def greeks(self):  # type: ignore[no-untyped-def]
        return greeks_for_chain(self._chain)

    def mid(self) -> np.ndarray:
        return self._chain.mid()


def main() -> int:
    print(f"qufin opportunity scan  |  as of {date.today().isoformat()}  |  ANALYSIS ONLY")
    analyze_googl()
    analyze_spy_calls()
    print("\nDone. Nothing was traded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
