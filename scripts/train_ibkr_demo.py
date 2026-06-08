"""
Train the IBKR-demo EMA-cross + ATR strategy.

Pipeline
--------
1. Fetch ~3 years of daily SPY bars from IBKR (delayed data, paper-safe).
2. Hold out the last 20% as out-of-sample.
3. Grid-search ``fast_window`` × ``slow_window`` × ``atr_window`` × ``atr_mult``
   on the training slice, scoring with Sharpe.
4. Re-run the best params on the held-out slice to report OOS Sharpe.
5. Persist the winning params + metadata to ``artifacts/ema_cross_atr_spy.json``
   so the live runner can load them.

Prerequisites
-------------
* IB Gateway logged into the **paper** account, API enabled.
* ``IBKR_HOST`` / ``IBKR_PORT`` / ``IBKR_CLIENT_ID`` in ``.env``.
* ``uv sync --group trading-live`` has been run.

Usage
-----
    uv run python scripts/train_ibkr_demo.py
    uv run python scripts/train_ibkr_demo.py --symbol QQQ --years 5

Exit 0 on success, non-zero on data or fitting failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from qufin.data._types import BAR_SCHEMA
from qufin.strategies.ema_cross_atr import make_strategy
from qufin.trading._types import BacktestReport
from qufin.trading.engine.clock import Clock
from qufin.trading.engine.engine import BacktestEngine, EngineConfig
from qufin.trading.strategy.base import StrategyBase
from qufin.trading.training.objectives import sharpe_objective
from qufin.trading.training.search import GridSearch


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().split("#", 1)[0].strip())


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _fetch_bars(symbol: str, *, years: int, host: str, port: int, client_id: int) -> pl.DataFrame:
    """Pull daily bars from IBKR and return a BAR_SCHEMA frame."""
    from qufin.data.vendors import IBKRHistoricalOHLC

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=years * 365 + 14)
    src = IBKRHistoricalOHLC(host=host, port=port, client_id=client_id)
    ohlcv = src.fetch(symbol, start=start, end=end, interval="1d")
    if len(ohlcv) == 0:
        raise RuntimeError(f"IBKR returned 0 bars for {symbol}")
    return ohlcv.data


def _split_train_oos(
    frame: pl.DataFrame, *, train_frac: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n = frame.height
    cut = int(n * train_frac)
    return frame.head(cut), frame.tail(n - cut)


def _backtest(
    strategy_params: dict[str, float | int], bars: pl.DataFrame, symbol: str, starting_cash: float
) -> BacktestReport:
    strategy = make_strategy(strategy_params, symbol=symbol)
    clock = Clock(bars={symbol: bars})
    engine = BacktestEngine(
        strategy=strategy,
        clock=clock,
        config=EngineConfig(starting_cash=starting_cash, history_window=0),
    )
    return engine.run()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--symbol", default="SPY", help="Equity ticker (default: SPY)")
    parser.add_argument("--years", type=int, default=3, help="History window in years (default: 3)")
    parser.add_argument(
        "--train-frac", type=float, default=0.8, help="In-sample fraction (default: 0.8)"
    )
    parser.add_argument("--starting-cash", type=float, default=100_000.0)
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the best params (default: artifacts/ema_cross_atr_{symbol}.json)",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = f"artifacts/ema_cross_atr_{args.symbol.lower()}.json"

    repo_root = Path(__file__).resolve().parent.parent
    _load_env(repo_root / ".env")
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = _int_env("IBKR_PORT", 7497)
    # Pull data on a unique client_id so it can coexist with the live runner.
    client_id = _int_env("IBKR_CLIENT_ID", 1) + 20

    print(f"Fetching {args.years}y of daily {args.symbol} bars from IBKR ({host}:{port})…")
    try:
        bars = _fetch_bars(args.symbol, years=args.years, host=host, port=port, client_id=client_id)
    except Exception as exc:  # noqa: BLE001 — surface the underlying error
        print(f"  FAILED: {exc!r}")
        return 1
    # Schema-normalise so Clock accepts the frame (drop any extra columns).
    bars = bars.select(*BAR_SCHEMA.keys())
    print(f"  got {bars.height} bars: {bars['timestamp'][0]} → {bars['timestamp'][-1]}")

    train, oos = _split_train_oos(bars, train_frac=args.train_frac)
    print(f"  train = {train.height} bars, OOS = {oos.height} bars")

    # Hand-picked grid. ~5 × 4 × 3 × 3 = 180 combos; daily backtest is fast.
    grid: dict[str, list[int] | list[float]] = {
        "fast_window": [5, 8, 12, 20, 30],
        "slow_window": [30, 50, 80, 120],
        "atr_window": [10, 14, 21],
        "atr_mult": [2.0, 3.0, 4.0],
    }

    # GridSearch evaluates every Cartesian combo; for combos that violate
    # EmaCrossAtrParams' invariants (e.g. slow_window <= fast_window) we return
    # a no-op strategy so the backtest produces zero trades and scores 0,
    # rather than letting the ValueError tear down the whole search.
    def factory(params):  # type: ignore[no-untyped-def]
        try:
            return make_strategy(dict(params), symbol=args.symbol)
        except ValueError:
            return StrategyBase()

    def bars_factory() -> dict[str, pl.DataFrame]:
        return {args.symbol: train}

    n_pairs = sum(1 for f in grid["fast_window"] for s in grid["slow_window"] if s > f)
    n_combos = n_pairs * len(grid["atr_window"]) * len(grid["atr_mult"])
    print(f"Grid-searching over {n_combos} valid combos…")
    search = GridSearch(
        strategy_factory=factory,
        grid=grid,
        bars_factory=bars_factory,
        objective=sharpe_objective,
        engine_config=EngineConfig(starting_cash=args.starting_cash, history_window=0),
        n_jobs=1,  # Each backtest is fast; keep serial to avoid pickling weirdness on Windows.
    )
    try:
        result = search.run()
    except Exception as exc:  # noqa: BLE001
        print(f"  search FAILED: {exc!r}")
        return 1

    # GridSearch evaluates every Cartesian combo, including ones where the
    # strategy rejects (slow <= fast). Those score 0 (degenerate-Sharpe path
    # in sharpe_objective) so the legitimate best still wins.
    print(f"Best in-sample Sharpe = {result.best_score:.3f}")
    for k, v in result.best_params.items():
        print(f"  {k:>12s} = {v}")

    print("Out-of-sample replay with best params…")
    oos_report = _backtest(result.best_params, oos, args.symbol, args.starting_cash)
    oos_sharpe = sharpe_objective(oos_report)
    oos_trades = oos_report.trades.height
    print(f"  OOS Sharpe   = {oos_sharpe:.3f}")
    print(f"  OOS trades   = {oos_trades}")
    if oos_report.equity_curve.height > 0:
        end_eq = float(oos_report.equity_curve["equity"][-1])
        print(f"  OOS end eq.  = ${end_eq:,.2f}  (started at ${args.starting_cash:,.2f})")

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": args.symbol,
        "interval": "1d",
        "params": result.best_params,
        "metrics": {
            "in_sample_sharpe": float(result.best_score),
            "out_of_sample_sharpe": float(oos_sharpe),
            "out_of_sample_trades": int(oos_trades),
        },
        "data_window": {
            "start": str(bars["timestamp"][0]),
            "end": str(bars["timestamp"][-1]),
            "train_bars": train.height,
            "oos_bars": oos.height,
        },
        "trained_at": datetime.now(tz=UTC).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved params → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
