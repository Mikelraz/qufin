"""Historical backtest of the Wyckoff-Hull confluence strategy.

Fetches daily OHLC bars for a list of ETFs via yfinance, plugs them into
``qufin.trading.BacktestEngine``, runs ``ConfluenceStrategy``, and writes
the ``BacktestReport`` plus a ``TearSheet`` summary to ``reports/``.

Usage
-----
    uv run python scripts/confluence_backtest.py \\
        SPY QQQ IWM XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC \\
        --start 2018-01-01 --end 2025-12-31 \\
        --report-dir reports/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from qufin.strategies.confluence import ConfluenceParams, ConfluenceStrategy
from qufin.trading import BacktestEngine, PercentSlippage
from qufin.trading.data import load_ohlc_many
from qufin.trading.engine import Clock, EngineConfig, NextBarOpenExecution
from qufin.trading.evaluation import tearsheet
from qufin.trading.strategy.base import StrategyContext

log = logging.getLogger("confluence.backtest")

DEFAULT_UNIVERSE = (
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "symbols",
        nargs="*",
        default=list(DEFAULT_UNIVERSE),
        help="tickers to trade (default: full ETF universe)",
    )
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"))
    p.add_argument("--starting-cash", type=float, default=100_000.0)
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--no-gex", action="store_true", help="disable the GEX overlay")
    p.add_argument("--min-confluences", type=int, default=3)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--heartbeat", type=int, default=50, help="emit a progress line every N bars (0 = silent)"
    )
    return p.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_clock(symbols: list[str], start: str, end: str) -> Clock:
    log.info("loading daily bars for %d symbols  start=%s  end=%s", len(symbols), start, end)
    t0 = time.perf_counter()
    ohlcv = load_ohlc_many(symbols, start=start, end=end, interval="1d")
    bars = {sym: o.data for sym, o in ohlcv.items() if o.data.height > 0}
    if not bars:
        raise SystemExit("no historical bars were loaded for any symbol")
    for sym, df in bars.items():
        log.info(
            "  %-5s  %5d bars  %s -> %s",
            sym,
            df.height,
            df["timestamp"].min(),
            df["timestamp"].max(),
        )
    log.info(
        "data load complete in %.1fs (%d symbols, %d total bars)",
        time.perf_counter() - t0,
        len(bars),
        sum(df.height for df in bars.values()),
    )
    return Clock(bars=bars)


class _LoggingConfluenceStrategy(ConfluenceStrategy):
    """ConfluenceStrategy that emits a heartbeat log line every N bars."""

    heartbeat: int = 50
    _bar_counter: int = 0
    _last_log_t: float = 0.0

    def on_bar(self, ctx: StrategyContext):  # type: ignore[override]
        self._bar_counter += 1
        if self.heartbeat > 0 and self._bar_counter % self.heartbeat == 0:
            now = time.perf_counter()
            dt = now - self._last_log_t if self._last_log_t else 0.0
            self._last_log_t = now
            log.info(
                "  bar %5d  ts=%s  equity=$%s  positions=%d  (last %d bars in %.1fs)",
                self._bar_counter,
                ctx.bar.timestamp,
                f"{ctx.account.equity:,.0f}",
                sum(1 for p in ctx.positions.values() if not p.is_flat),
                self.heartbeat,
                dt,
            )
        return super().on_bar(ctx)


def main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info(
        "confluence backtest starting  symbols=%d  GEX=%s  min_confluences=%d",
        len(args.symbols),
        not args.no_gex,
        args.min_confluences,
    )

    params = ConfluenceParams(
        starting_cash=args.starting_cash,
        use_gex_overlay=not args.no_gex,
        min_confluences=args.min_confluences,
    )
    clock = _build_clock(args.symbols, args.start, args.end)
    strategy = _LoggingConfluenceStrategy(params=params, symbols=list(clock.bars.keys()))
    strategy.heartbeat = args.heartbeat
    engine = BacktestEngine(
        strategy=strategy,
        clock=clock,
        execution=NextBarOpenExecution(slippage=PercentSlippage(bps=params.slippage_bps)),
        config=EngineConfig(starting_cash=params.starting_cash),
    )

    n_steps = len(clock.unique_timestamps())
    log.info(
        "engine.run() starting — %d clock steps, starting cash $%s",
        n_steps,
        f"{params.starting_cash:,.0f}",
    )
    t0 = time.perf_counter()
    report = engine.run()
    log.info(
        "engine.run() finished in %.1fs (%d trades, equity curve %d rows)",
        time.perf_counter() - t0,
        report.trades.height,
        report.equity_curve.height,
    )

    log.info("computing tearsheet…")
    ts = tearsheet(report)

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    eq_path = out / f"confluence_equity_{stamp}.parquet"
    tr_path = out / f"confluence_trades_{stamp}.parquet"
    sm_path = out / f"confluence_summary_{stamp}.json"
    report.equity_curve.write_parquet(eq_path)
    report.trades.write_parquet(tr_path)
    sm_path.write_text(json.dumps(ts.summary, indent=2, default=float))
    log.info("wrote %s", eq_path)
    log.info("wrote %s", tr_path)
    log.info("wrote %s", sm_path)
    print(json.dumps(ts.summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
