"""
Train the IBKR options demo: long-call trend follower.

Pipeline
--------
1. Fetch ~3 years of daily underlying bars from IBKR.
2. Estimate the flat-IV assumption used by Black-Scholes marks as the
   annualised realised volatility of daily log returns over a rolling
   window (default 60 days), clamped to [10%, 80%].
3. Hold out the last 20% as out-of-sample.
4. Grid-search ``fast_window x slow_window x strike_moneyness x
   target_dte x exit_dte`` on the training slice, with option marks
   priced by ``FlatIVMarkProvider(iv=realised_vol)``.
5. Re-run the best params on the held-out slice; persist params to
   ``artifacts/ema_cross_long_call_{symbol}.json``.

Notes
-----
* The backtest uses synthetic expiries (``today + target_dte``) and
  strikes rounded to the nearest dollar. This is intentional — the
  goal is to size the *signal*, not to over-fit to listed expiry
  cycles. The live runner picks the closest listed contract.
* Constant IV is a simplification. PnL biases toward overpaying premium
  in low-vol regimes and underpaying in high-vol regimes. For a richer
  model, swap ``FlatIVMarkProvider`` for a custom ``MarkProvider`` that
  takes a vol-surface; the rest of the pipeline stays the same.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from qufin.data._types import BAR_SCHEMA
from qufin.strategies.ema_cross_long_call import make_strategy
from qufin.trading._types import BacktestReport, FixedCommission, PercentSlippage
from qufin.trading.engine.clock import Clock
from qufin.trading.engine.engine import BacktestEngine, EngineConfig
from qufin.trading.engine.execution import OptionAwareExecution
from qufin.trading.engine.options_engine import FlatIVMarkProvider, OptionsEngine
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
    from qufin.data.vendors import IBKRHistoricalOHLC

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=years * 365 + 14)
    src = IBKRHistoricalOHLC(host=host, port=port, client_id=client_id)
    ohlcv = src.fetch(symbol, start=start, end=end, interval="1d")
    if len(ohlcv) == 0:
        raise RuntimeError(f"IBKR returned 0 bars for {symbol}")
    return ohlcv.data


def _realised_vol(bars: pl.DataFrame, *, window: int) -> float:
    """Annualised stdev of daily log returns over the trailing ``window`` bars."""
    close = bars["close"].to_numpy().astype(np.float64, copy=False)
    if len(close) < window + 2:
        raise ValueError(f"need >= {window + 2} bars for realised vol")
    logret = np.diff(np.log(close[-(window + 1):]))
    rv = float(np.std(logret, ddof=1) * np.sqrt(252.0))
    return max(0.10, min(0.80, rv))


def _split_train_oos(
    frame: pl.DataFrame, *, train_frac: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n = frame.height
    cut = int(n * train_frac)
    return frame.head(cut), frame.tail(n - cut)


def _backtest(
    strategy_params: Mapping[str, Any],
    bars: pl.DataFrame,
    *,
    symbol: str,
    starting_cash: float,
    iv: float,
) -> BacktestReport:
    strategy = make_strategy(dict(strategy_params), symbol=symbol)
    clock = Clock(bars={symbol: bars})
    provider = FlatIVMarkProvider(iv=iv)
    engine = BacktestEngine(
        strategy=strategy,
        clock=clock,
        execution=OptionAwareExecution(
            mark_provider=provider,
            slippage=PercentSlippage(bps=5.0),  # 5 bps on the option premium
            commissions=FixedCommission(per_contract=0.65),
        ),
        options_engine=OptionsEngine(mark_provider=provider),
        config=EngineConfig(starting_cash=starting_cash, history_window=0),
    )
    return engine.run()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--symbol", default="AAPL", help="Underlying ticker (default: AAPL)")
    parser.add_argument("--years", type=int, default=3, help="History window in years (default: 3)")
    parser.add_argument(
        "--train-frac", type=float, default=0.8, help="In-sample fraction (default: 0.8)"
    )
    parser.add_argument("--starting-cash", type=float, default=100_000.0)
    parser.add_argument(
        "--vol-window", type=int, default=60, help="Bars used for realised-vol IV (default 60)"
    )
    parser.add_argument(
        "--iv",
        type=float,
        default=None,
        help="Override the realised-vol IV with a hardcoded value (e.g. 0.25)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Where to write the best params"
            " (default: artifacts/ema_cross_long_call_{symbol}.json)"
        ),
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = f"artifacts/ema_cross_long_call_{args.symbol.lower()}.json"

    repo_root = Path(__file__).resolve().parent.parent
    _load_env(repo_root / ".env")
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = _int_env("IBKR_PORT", 7497)
    # Distinct client_id offset so this can coexist with the equity demo's
    # training and live processes.
    client_id = _int_env("IBKR_CLIENT_ID", 1) + 50

    print(f"Fetching {args.years}y of daily {args.symbol} bars from IBKR ({host}:{port})...")
    try:
        bars = _fetch_bars(args.symbol, years=args.years, host=host, port=port, client_id=client_id)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc!r}")
        return 1
    bars = bars.select(*BAR_SCHEMA.keys())
    print(f"  got {bars.height} bars: {bars['timestamp'][0]} -> {bars['timestamp'][-1]}")

    iv = float(args.iv) if args.iv is not None else _realised_vol(bars, window=args.vol_window)
    print(f"  flat IV for backtest = {iv:.3f}  ({'override' if args.iv else 'realised'})")

    train, oos = _split_train_oos(bars, train_frac=args.train_frac)
    print(f"  train = {train.height} bars, OOS = {oos.height} bars")

    grid: dict[str, list[int] | list[float]] = {
        "fast_window": [8, 12, 20],
        "slow_window": [40, 60, 100],
        "strike_moneyness": [0.98, 1.00, 1.02],
        "target_dte": [30, 45, 60],
        "exit_dte": [5, 10],
    }
    n_pairs = sum(1 for f in grid["fast_window"] for s in grid["slow_window"] if s > f)
    n_combos = (
        n_pairs
        * len(grid["strike_moneyness"])
        * len(grid["target_dte"])
        * len(grid["exit_dte"])
    )
    print(f"Grid-searching over {n_combos} valid combos...")

    def factory(params):  # type: ignore[no-untyped-def]
        try:
            return make_strategy(dict(params), symbol=args.symbol)
        except ValueError:
            return StrategyBase()

    def bars_factory() -> dict[str, pl.DataFrame]:
        return {args.symbol: train}

    # GridSearch's default scoring uses the equity-only EngineConfig path —
    # we need the OptionsEngine + OptionAwareExecution wiring, so build the
    # full report ourselves and reduce inside the objective.
    def objective_with_options(report: BacktestReport) -> float:
        return sharpe_objective(report)

    # Custom scoring loop (one-line replacement for GridSearch._score that
    # plugs the options engine in).
    import itertools

    combos = [
        dict(zip(grid.keys(), point, strict=True))
        for point in itertools.product(*(list(v) for v in grid.values()))
    ]
    scored: list[tuple[dict[str, Any], float]] = []
    for combo in combos:
        try:
            report = _backtest(
                combo, train, symbol=args.symbol, starting_cash=args.starting_cash, iv=iv
            )
        except ValueError:
            scored.append((combo, 0.0))
            continue
        scored.append((combo, objective_with_options(report)))

    best_combo, best_score = max(scored, key=lambda kv: kv[1])
    print(f"Best in-sample Sharpe = {best_score:.3f}")
    for k, v in best_combo.items():
        print(f"  {k:>16s} = {v}")

    print("Out-of-sample replay with best params...")
    oos_report = _backtest(
        best_combo, oos, symbol=args.symbol, starting_cash=args.starting_cash, iv=iv
    )
    oos_sharpe = sharpe_objective(oos_report)
    oos_trades = oos_report.trades.height
    print(f"  OOS Sharpe   = {oos_sharpe:.3f}")
    print(f"  OOS trades   = {oos_trades}")
    if oos_report.equity_curve.height > 0:
        end_eq = float(oos_report.equity_curve["equity"][-1])
        print(f"  OOS end eq.  = ${end_eq:,.2f}  (started at ${args.starting_cash:,.2f})")

    # Silence GridSearch unused-import warning when iv override is taken.
    _ = (objective_with_options, GridSearch)

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": args.symbol,
        "interval": "1d",
        "params": best_combo,
        "iv": iv,
        "metrics": {
            "in_sample_sharpe": float(best_score),
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
    print(f"Saved params -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
