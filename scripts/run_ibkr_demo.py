"""
Live paper-trade runner for the trained EMA-cross + ATR demo strategy.

What it does
------------
On a fixed cadence (default once at startup, optionally looped):

1. Pull the recent daily-bar history for the configured symbol from IBKR.
2. Replay it through ``ema_cross_atr.current_state`` to compute the
   *desired* target weight right now.
3. Read the broker's current position and account snapshot.
4. If the desired position differs from the current position, place a
   single MARKET order to reconcile.

Safety
------
* **Dry-run by default** — without ``--live`` the runner only prints
  the decision it would have submitted.
* ``--max-position-usd`` hard-caps the notional sent to the broker even
  when the strategy says "fully invested".
* Single symbol, long-or-flat only.

Prerequisites
-------------
* ``scripts/train_ibkr_demo.py`` has been run and produced
  ``artifacts/ema_cross_atr_spy.json``.
* IB Gateway logged into the **paper** account.

Usage
-----
    # safe dry-run (prints the decision, places nothing):
    uv run python scripts/run_ibkr_demo.py

    # actually submit paper orders:
    uv run python scripts/run_ibkr_demo.py --live

    # run continuously every 5 minutes:
    uv run python scripts/run_ibkr_demo.py --live --interval 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


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


async def _decide_and_trade(
    *,
    host: str,
    port: int,
    client_id: int,
    symbol: str,
    params_payload: dict,
    max_position_usd: float,
    live: bool,
    history_days: int,
) -> int:
    from qufin.data.vendors import IBKRHistoricalOHLC
    from qufin.strategies.ema_cross_atr import EmaCrossAtrParams, current_state
    from qufin.trading._types import Order, OrderType, TimeInForce
    from qufin.trading.brokers import IBKRBroker

    params = EmaCrossAtrParams(
        fast_window=int(params_payload["fast_window"]),
        slow_window=int(params_payload["slow_window"]),
        atr_window=int(params_payload["atr_window"]),
        atr_mult=float(params_payload["atr_mult"]),
        target_weight=float(params_payload.get("target_weight", 1.0)),
    )

    # 1. History — pulled on a different client_id so it can coexist with the
    #    long-running broker socket below.
    data_src = IBKRHistoricalOHLC(host=host, port=port, client_id=client_id + 10)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=history_days)
    print(f"  pulling {history_days}d of daily {symbol} bars from IBKR…")
    ohlcv = await asyncio.to_thread(data_src.fetch, symbol, start=start, end=end, interval="1d")
    if len(ohlcv) < params.min_bars:
        print(f"  FAIL: got {len(ohlcv)} bars, need {params.min_bars}")
        return 1

    state = current_state(
        bars_close=ohlcv.close(),
        bars_high=ohlcv.high(),
        bars_low=ohlcv.low(),
        params=params,
    )
    print(f"  last close   = {state.last_close:.2f}")
    print(f"  fast EMA     = {state.fast_ema:.2f}")
    print(f"  slow EMA     = {state.slow_ema:.2f}")
    print(f"  ATR          = {state.atr:.2f}")
    if math.isfinite(state.trail_stop):
        print(f"  trail stop   = {state.trail_stop:.2f}  (highest close {state.highest_close:.2f})")
    label = "LONG" if state.target_weight > 0 else "FLAT"
    print(f"  target weight= {state.target_weight:.2f}  ({label})")

    # 2. Broker — account + position.
    broker = IBKRBroker(host=host, port=port, client_id=client_id)
    try:
        await broker.connect()
        account = await broker.account()
        positions = await broker.positions()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL to connect to IBKR: {exc!r}")
        return 1

    try:
        held = next((p for p in positions if str(p.asset) == symbol), None)
        cur_qty = float(held.qty) if held is not None else 0.0
        cur_label = "flat" if cur_qty == 0 else "long" if cur_qty > 0 else "short"
        print(f"  account eq.  = ${account.equity:,.2f}  cash=${account.cash:,.2f}")
        print(f"  current pos  = {cur_qty:.0f} sh  ({cur_label})")

        # 3. Desired share count, capped by max_position_usd.
        target_notional = min(state.target_weight * account.equity, max_position_usd)
        target_qty = math.floor(target_notional / state.last_close) if state.last_close > 0 else 0
        delta = float(target_qty - cur_qty)
        target_notional_actual = target_qty * state.last_close
        print(f"  target pos   = {target_qty:.0f} sh  (notional ${target_notional_actual:,.0f})")

        if abs(delta) < 1.0:
            print("  → no trade (already in target position)")
            return 0

        side = "BUY" if delta > 0 else "SELL"
        print(f"  → {side} {abs(delta):.0f} {symbol} @ MARKET (delta = {delta:+.0f} sh)")
        if not live:
            print("  (dry-run; pass --live to actually submit)")
            return 0

        order = Order(
            asset=symbol,
            qty=delta,
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            tag="ema_cross_atr_demo",
        )
        order_id = await broker.place_order(order)
        print(f"  submitted: order_id={order_id}")
        # Give IBKR a beat to register the order before we disconnect.
        await asyncio.sleep(1.0)
    finally:
        await broker.disconnect()
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--params",
        default="artifacts/ema_cross_atr_spy.json",
        help="Path (relative to repo root) of the params JSON from train_ibkr_demo.py",
    )
    parser.add_argument(
        "--max-position-usd",
        type=float,
        default=10_000.0,
        help="Hard cap on dollar notional of the position (default $10k)",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=400,
        help="Bars of history to pull each loop (default 400)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually submit paper orders. Without this flag, dry-runs.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Seconds between iterations. 0 = run once and exit (default).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    _load_env(repo_root / ".env")
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = _int_env("IBKR_PORT", 7497)
    client_id = _int_env("IBKR_CLIENT_ID", 1) + 30

    params_path = repo_root / args.params
    if not params_path.exists():
        print(f"params file not found: {params_path}")
        print("  run: uv run python scripts/train_ibkr_demo.py")
        return 1
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    symbol = str(payload["symbol"])
    print(f"Loaded params for {symbol} (trained {payload.get('trained_at', '?')})")
    print(f"  in-sample Sharpe   = {payload['metrics']['in_sample_sharpe']:.3f}")
    print(
        f"  out-of-sample      = {payload['metrics']['out_of_sample_sharpe']:.3f}"
        f"  ({payload['metrics']['out_of_sample_trades']} trades)"
    )
    if args.live:
        print("  MODE = LIVE (orders will be sent to IBKR paper account)")
    else:
        print("  MODE = DRY-RUN (no orders will be submitted)")

    while True:
        print(f"\n--- cycle @ {datetime.now(tz=UTC).isoformat(timespec='seconds')} ---")
        rc = await _decide_and_trade(
            host=host,
            port=port,
            client_id=client_id,
            symbol=symbol,
            params_payload=payload["params"],
            max_position_usd=args.max_position_usd,
            live=args.live,
            history_days=args.history_days,
        )
        if args.interval <= 0:
            return rc
        if rc != 0:
            print(f"  cycle returned {rc}; will retry after interval")
        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
