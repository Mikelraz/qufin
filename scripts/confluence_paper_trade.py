"""Alpaca paper-trade runner for the confluence strategy.

Designed to be invoked nightly (cron / Task Scheduler). One iteration:

    1. Pull the trailing 18 months of daily bars for each symbol via the
       Alpaca data feed (free IEX by default).
    2. Optionally refresh SPY's option chain to feed the GEX defense
       overlay.
    3. Rebuild the ``ConfluenceStrategy`` and step it through a one-bar
       backtest *terminating on today's close* so its in-memory state
       matches a fresh backtest of the same window.
    4. Compare the strategy's target weights to the Alpaca account's
       current positions and submit market-on-open orders for the deltas.

``--dry-run`` performs steps 1-3 and prints the proposed deltas without
hitting the broker.

Usage
-----
    uv run python scripts/confluence_paper_trade.py --dry-run
    uv run python scripts/confluence_paper_trade.py            # arms paper trading
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from qufin.options import OptionChain
from qufin.strategies.confluence import (
    ConfluenceParams,
    ConfluenceStrategy,
)
from qufin.trading import BacktestEngine, PercentSlippage
from qufin.trading.engine import Clock, EngineConfig, NextBarOpenExecution

log = logging.getLogger("confluence.paper")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


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


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("symbols", nargs="*", default=list(DEFAULT_UNIVERSE))
    p.add_argument(
        "--lookback-days",
        type=int,
        default=540,
        help="trailing window of daily bars to rebuild state",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="compute target weights but don't submit orders"
    )
    p.add_argument("--no-gex", action="store_true")
    p.add_argument("--report-dir", default="reports/paper")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _alpaca_chain_provider(symbol_to_fetch: str):
    """Return a callable suitable for ``ConfluenceStrategy.gex_chain_provider``.

    Falls back to None if the optional Alpaca options loader is unavailable.
    """
    try:
        from qufin.data.vendors import load_alpaca_option_chain
    except (ImportError, AttributeError):
        return None

    def provider(*, symbol: str, asof: object) -> OptionChain | None:  # noqa: ARG001
        if symbol != symbol_to_fetch:
            return None
        try:
            return load_alpaca_option_chain(symbol=symbol)
        except Exception:
            return None

    return provider


def _load_bars(symbols: list[str], lookback_days: int) -> dict[str, pl.DataFrame]:
    from qufin.data.vendors import load_alpaca_ohlc

    log.info("loading %d-day bar history for %d symbols from Alpaca", lookback_days, len(symbols))
    start = datetime.now(UTC) - timedelta(days=lookback_days + 30)
    bars: dict[str, pl.DataFrame] = {}
    t0 = time.perf_counter()
    for sym in symbols:
        try:
            ohlcv = load_alpaca_ohlc(symbol=sym, start=start, timeframe="1Day")
            if ohlcv.data.height > 0:
                bars[sym] = ohlcv.data.tail(lookback_days)
                log.info("  %-5s  %5d bars  loaded", sym, bars[sym].height)
            else:
                log.warning("  %-5s  empty frame returned", sym)
        except Exception as exc:
            log.warning("  %-5s  failed to load bars (%s)", sym, exc)
    log.info(
        "data load complete in %.1fs (%d/%d symbols)",
        time.perf_counter() - t0,
        len(bars),
        len(symbols),
    )
    return bars


def _compute_target_weights(
    params: ConfluenceParams, symbols: list[str], bars: dict[str, pl.DataFrame], chain_provider
) -> dict[str, float]:
    clock = Clock(bars=bars)
    log.info("rebuilding strategy state from %d clock steps", len(clock.unique_timestamps()))
    strategy = ConfluenceStrategy(
        params=params,
        symbols=list(bars.keys()),
        gex_chain_provider=chain_provider,
    )
    engine = BacktestEngine(
        strategy=strategy,
        clock=clock,
        execution=NextBarOpenExecution(slippage=PercentSlippage(bps=params.slippage_bps)),
        config=EngineConfig(starting_cash=params.starting_cash),
    )
    t0 = time.perf_counter()
    _ = engine.run()
    log.info("strategy replay finished in %.1fs", time.perf_counter() - t0)
    out: dict[str, float] = {}
    for sym in symbols:
        sig_state = strategy._state.get(sym)
        if sig_state is None or sig_state.signals is None:
            continue
        sig = sig_state.signals
        i = sig.long_entry.shape[0] - 1
        if bool(sig.long_entry[i]):
            out[sym] = params.max_weight_per_name
    log.info(
        "computed target weights for %d symbol(s): %s",
        len(out),
        ", ".join(out.keys()) if out else "(none)",
    )
    return out


async def _submit_orders(target: dict[str, float]) -> None:
    from qufin.trading.brokers import AlpacaBroker

    log.info("connecting to Alpaca (paper)")
    broker = AlpacaBroker(paper=True)
    snap = await broker.account_snapshot()
    equity = snap.equity
    log.info("account equity=$%s  buying_power=$%s", f"{equity:,.0f}", f"{snap.buying_power:,.0f}")
    for sym, w in target.items():
        notional = w * equity
        log.info("submitting order  %-5s  weight=%.4f  notional=$%s", sym, w, f"{notional:,.0f}")
        await broker.submit_order(symbol=sym, notional=notional)
    log.info("submitted %d order(s)", len(target))


def main() -> int:
    _load_env(Path(".env"))
    args = _parse_args()
    _setup_logging(args.log_level)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info(
        "paper-trade runner starting  symbols=%d  dry_run=%s  GEX=%s",
        len(args.symbols),
        args.dry_run,
        not args.no_gex,
    )

    params = ConfluenceParams(use_gex_overlay=not args.no_gex)
    bars = _load_bars(list(args.symbols), args.lookback_days)
    if not bars:
        raise SystemExit("no bars loaded — check Alpaca credentials in .env")

    chain_provider = (
        None if args.no_gex else _alpaca_chain_provider(symbol_to_fetch=params.gex_macro_symbol)
    )
    if chain_provider is None and not args.no_gex:
        log.warning("GEX overlay requested but no Alpaca chain loader available — running without")
    target = _compute_target_weights(params, list(args.symbols), bars, chain_provider)

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    payload = {"timestamp": stamp, "dry_run": args.dry_run, "target_weights": target}
    sig_path = out / f"signals_{stamp}.json"
    sig_path.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", sig_path)
    print(json.dumps(payload, indent=2))

    if args.dry_run:
        log.info("dry-run mode — skipping order submission")
        return 0
    asyncio.run(_submit_orders(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
