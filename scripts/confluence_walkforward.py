"""Walk-forward cross-validation for the confluence strategy.

Trains on a rolling 2-year window, evaluates on the next 6 months, then
steps forward 6 months. Reports per-fold Sharpe, max-drawdown and CAGR
as a polars frame.

The "training" pass here is informational only — the strategy's
parameters are fixed across folds (default ``ConfluenceParams``). The
per-fold backtests still exercise the full pipeline, so degradation
between folds is informative even without hyperparameter search.

Usage
-----
    uv run python scripts/confluence_walkforward.py SPY QQQ IWM \\
        --start 2018-01-01 --end 2025-12-31 \\
        --train-years 2 --test-months 6 --step-months 6
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from qufin.strategies.confluence import ConfluenceParams, ConfluenceStrategy
from qufin.trading import BacktestEngine, PercentSlippage
from qufin.trading.data import load_ohlc_many
from qufin.trading.engine import Clock, EngineConfig, NextBarOpenExecution
from qufin.trading.evaluation import tearsheet

log = logging.getLogger("confluence.walkforward")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass(slots=True)
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("symbols", nargs="*", default=["SPY", "QQQ", "IWM"])
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"))
    p.add_argument("--train-years", type=int, default=2)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--step-months", type=int, default=6)
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _fold_windows(
    start: datetime, end: datetime, train_years: int, test_months: int, step_months: int
) -> list[tuple[datetime, ...]]:
    folds: list[tuple[datetime, ...]] = []
    cursor = start
    train_delta = timedelta(days=365 * train_years)
    test_delta = timedelta(days=30 * test_months)
    step_delta = timedelta(days=30 * step_months)
    while cursor + train_delta + test_delta <= end:
        train_end = cursor + train_delta
        test_end = train_end + test_delta
        folds.append((cursor, train_end, train_end, test_end))
        cursor = cursor + step_delta
    return folds


def _run_window(
    symbols: list[str], start: datetime, end: datetime, params: ConfluenceParams
) -> dict[str, float]:
    log.debug("  loading bars  %s -> %s  for %d symbols", start.date(), end.date(), len(symbols))
    t_load = time.perf_counter()
    ohlcv = load_ohlc_many(symbols, start=start, end=end, interval="1d")
    bars = {sym: o.data for sym, o in ohlcv.items() if o.data.height > 0}
    if not bars:
        log.warning("  no bars returned for window %s..%s", start.date(), end.date())
        return {}
    total_bars = sum(df.height for df in bars.values())
    log.debug(
        "  loaded %d bars across %d symbols in %.1fs",
        total_bars,
        len(bars),
        time.perf_counter() - t_load,
    )
    clock = Clock(bars=bars)
    strategy = ConfluenceStrategy(params=params, symbols=list(bars.keys()))
    engine = BacktestEngine(
        strategy=strategy,
        clock=clock,
        execution=NextBarOpenExecution(slippage=PercentSlippage(bps=params.slippage_bps)),
        config=EngineConfig(starting_cash=params.starting_cash),
    )
    t_run = time.perf_counter()
    report = engine.run()
    log.debug(
        "  engine.run() finished in %.1fs (%d trades)",
        time.perf_counter() - t_run,
        report.trades.height,
    )
    return tearsheet(report).summary


def main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    folds = _fold_windows(start, end, args.train_years, args.test_months, args.step_months)
    if not folds:
        raise SystemExit("no folds fit in the requested window")
    log.info(
        "walk-forward CV  symbols=%d  folds=%d  train=%dy  test=%dmo  step=%dmo",
        len(args.symbols),
        len(folds),
        args.train_years,
        args.test_months,
        args.step_months,
    )

    params = ConfluenceParams()
    results: list[FoldResult] = []
    total_t0 = time.perf_counter()
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        log.info(
            "[fold %d/%d] test window %s..%s  starting", i, len(folds), te_s.date(), te_e.date()
        )
        t0 = time.perf_counter()
        summary = _run_window(args.symbols, te_s, te_e, params)
        dt = time.perf_counter() - t0
        results.append(
            FoldResult(
                fold=i,
                train_start=tr_s.date().isoformat(),
                train_end=tr_e.date().isoformat(),
                test_start=te_s.date().isoformat(),
                test_end=te_e.date().isoformat(),
                cagr=float(summary.get("cagr", float("nan"))),
                sharpe=float(summary.get("sharpe", float("nan"))),
                sortino=float(summary.get("sortino", float("nan"))),
                max_drawdown=float(summary.get("max_drawdown", float("nan"))),
            )
        )
        log.info(
            "[fold %d/%d] done in %.1fs  sharpe=%.2f  cagr=%.2f%%  mdd=%.2f%%",
            i,
            len(folds),
            dt,
            results[-1].sharpe,
            results[-1].cagr * 100.0,
            results[-1].max_drawdown * 100.0,
        )
    log.info("all %d folds finished in %.1fs", len(folds), time.perf_counter() - total_t0)

    df = pl.DataFrame([asdict(r) for r in results])
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    df.write_parquet(out / f"confluence_walkforward_{stamp}.parquet")
    print("\n", df)
    print(
        json.dumps(
            {
                "mean_sharpe": float(df["sharpe"].mean() or 0.0),
                "std_sharpe": float(df["sharpe"].std() or 0.0),
                "min_sharpe": float(df["sharpe"].min() or 0.0),
                "worst_mdd": float(df["max_drawdown"].min() or 0.0),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
