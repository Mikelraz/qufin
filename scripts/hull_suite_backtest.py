"""
Backtest the Hull Suite strategy on Alpaca historical bars.

Reuses the project's existing Alpaca connection (credentials from the
project-root ``.env``, same loader as ``check_alpaca_connection.py``) to
fetch historical OHLC bars for one or more symbols, then runs
:func:`qufin.strategies.hull_strategy.generate_signals` over the close
prices and evaluates the resulting position series with
:func:`qufin.strategies.hull_backtest.backtest_hull`.

Outputs a per-symbol summary and compares strategy performance against
buy-and-hold over the same window.

Usage
-----
    uv run python scripts/hull_suite_backtest.py AAPL MSFT
    uv run python scripts/hull_suite_backtest.py SPY --start 2022-01-01 \\
        --timeframe Day --fast 50 --slow 60 --slow-type ehma
    uv run python scripts/hull_suite_backtest.py QQQ --timeframe Hour \\
        --amount 1 --start 2024-06-01

Notes
-----
* The Alpaca free IEX feed is used by default; pass ``--feed sip`` if your
  subscription supports it.
* No look-ahead: the back-tester earns ``signal[t-1] * ret[t]`` so the
  reported numbers are causally realisable.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# .env loader (matches scripts/check_alpaca_connection.py)
# ---------------------------------------------------------------------------


def _load_env(path: Path) -> None:
    """KEY=VALUE per line, ``#`` comments allowed.  Skips silently if absent."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest the Hull Suite strategy on Alpaca historical bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("symbols", nargs="+", help="Ticker(s) to backtest (e.g. AAPL SPY).")
    p.add_argument(
        "--start",
        default=(datetime.now(UTC) - timedelta(days=365 * 3)).date().isoformat(),
        help="ISO date for the first bar. Default: 3 years ago.",
    )
    p.add_argument(
        "--end",
        default=datetime.now(UTC).date().isoformat(),
        help="ISO date for the last bar. Default: today.",
    )
    p.add_argument(
        "--timeframe",
        choices=["Min", "Hour", "Day"],
        default="Day",
        help="Alpaca timeframe unit.",
    )
    p.add_argument("--amount", type=int, default=1, help="Multiplier for the timeframe unit.")
    p.add_argument("--feed", default="iex", choices=["iex", "sip"], help="Alpaca data feed.")
    p.add_argument("--fast", type=int, default=50, help="Fast Hull length.")
    p.add_argument("--slow", type=int, default=60, help="Slow Hull length.")
    p.add_argument(
        "--fast-type",
        choices=["hma", "thma", "ehma"],
        default="hma",
        help="Hull variant for the fast band.",
    )
    p.add_argument(
        "--slow-type",
        choices=["hma", "thma", "ehma"],
        default="ehma",
        help="Hull variant for the slow band.",
    )
    p.add_argument(
        "--length-multiplier",
        type=float,
        default=1.0,
        help="Uniform scale on both lengths (e.g. 2.0 ~ next-higher-timeframe ribbon).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _annualizer(timeframe: str, amount: int) -> float:
    """
    √(bars-per-year) used to annualise Sharpe.

    Conventions: 252 trading days; 6.5h regular session => 390 minutes / 6.5
    hours per day.
    """
    match timeframe:
        case "Day":
            return math.sqrt(252.0 / amount)
        case "Hour":
            return math.sqrt(252.0 * 6.5 / amount)
        case "Min":
            return math.sqrt(252.0 * 390.0 / amount)
        case _:
            return math.sqrt(252.0)


def _buy_and_hold_stats(close: np.ndarray, annualizer: float) -> dict[str, float]:
    """Reference statistics for a 100%-invested buy-and-hold baseline."""
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.diff(np.log(np.maximum(close, 1e-12)))
    equity = float(np.exp(np.sum(log_ret)))
    mean = float(np.mean(log_ret))
    std = float(np.std(log_ret))
    sharpe = float(mean / std * annualizer) if std > 1e-12 else 0.0
    cum = np.exp(np.cumsum(log_ret))
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum / peak - 1.0))
    return {
        "total_return": equity - 1.0,
        "annual_return": mean * (annualizer**2),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# Per-symbol pipeline
# ---------------------------------------------------------------------------


def _run_symbol(symbol: str, args: argparse.Namespace) -> int:
    from qufin.data.vendors.alpaca import load_alpaca_ohlc
    from qufin.strategies.hull_backtest import backtest_hull
    from qufin.strategies.hull_strategy import generate_signals

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

    print(f"\n=== {symbol} ===")
    print(
        f"  window  : {args.start} -> {args.end}  "
        f"timeframe={args.amount}{args.timeframe}  feed={args.feed}"
    )
    try:
        bars = load_alpaca_ohlc(
            symbol,
            start=start,
            end=end,
            amount=args.amount,
            unit=args.timeframe,
            feed=args.feed,
        )
    except Exception as exc:  # noqa: BLE001 — surface Alpaca's actual error
        print(f"  FAILED to load bars for {symbol}: {exc!r}")
        return 1

    frame: pl.DataFrame = bars.data
    if frame.height < args.slow * 3:
        print(
            f"  SKIP - only {frame.height} bars returned; need at least "
            f"{args.slow * 3} for Hull lookback={args.slow}."
        )
        return 1

    print(f"  bars    : {frame.height}")

    sig = generate_signals(
        frame,
        fast_length=args.fast,
        slow_length=args.slow,
        fast_type=args.fast_type,
        slow_type=args.slow_type,
        length_multiplier=args.length_multiplier,
    )
    annualizer = _annualizer(args.timeframe, args.amount)
    report = backtest_hull(frame, sig, annualizer=annualizer)
    bh = _buy_and_hold_stats(bars.close(), annualizer)

    # Exposure stats — fraction of bars actually carrying a position.
    sig_arr = sig.to_numpy()
    long_frac = float(np.mean(sig_arr > 0))
    short_frac = float(np.mean(sig_arr < 0))
    flat_frac = 1.0 - long_frac - short_frac

    print(
        "  Hull Suite  "
        f"fast={args.fast}({args.fast_type})  "
        f"slow={args.slow}({args.slow_type})  "
        f"mult={args.length_multiplier}"
    )
    print("  ----- strategy ----------------------------")
    print(f"   total return  : {report.total_return * 100:8.2f}%")
    print(f"   annual return : {report.annual_return * 100:8.2f}%")
    print(f"   sharpe        : {report.sharpe:8.3f}")
    print(f"   max drawdown  : {report.max_drawdown * 100:8.2f}%")
    print(f"   trades        : {report.n_trades:8d}")
    print(f"   win rate      : {report.win_rate * 100:8.1f}%")
    print(f"   avg trade     : {report.avg_trade * 100:8.3f}%")
    print(f"   exposure      : long={long_frac:.2f}  short={short_frac:.2f}  flat={flat_frac:.2f}")
    print("  ----- buy & hold --------------------------")
    print(f"   total return  : {bh['total_return'] * 100:8.2f}%")
    print(f"   annual return : {bh['annual_return'] * 100:8.2f}%")
    print(f"   sharpe        : {bh['sharpe']:8.3f}")
    print(f"   max drawdown  : {bh['max_drawdown'] * 100:8.2f}%")
    print("  ----- delta vs B&H ------------------------")
    print(f"   total return  : {(report.total_return - bh['total_return']) * 100:+8.2f}%")
    print(f"   sharpe        : {report.sharpe - bh['sharpe']:+8.3f}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    _load_env(Path(__file__).resolve().parent.parent / ".env")
    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_SECRET_KEY"):
        print(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in environment. "
            "Add them to .env at the project root.",
            file=sys.stderr,
        )
        return 1

    args = _parse_args()
    failures = 0
    for sym in args.symbols:
        failures += _run_symbol(sym.upper(), args)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
