"""
Live paper-trade runner for the long-call options demo strategy.

What it does
------------
On a fixed cadence (default once at startup, optionally looped):

1. Pull recent daily bars of the underlying from IBKR.
2. Replay them through ``ema_cross_long_call.current_state`` to compute
   the *desired* state right now (long a call, or flat).
3. Read the broker's option positions on this underlying.
4. Reconcile:
   * desired=long, holding nothing -> look up the closest listed
     (expiry, strike) to the strategy's target, BUY 1 call.
   * desired=flat (or held call's DTE has fallen below ``exit_dte``)
     while holding a long call -> SELL the held call to close.
   * otherwise -> no action.

Safety
------
* **Dry-run by default** — pass ``--live`` to actually submit orders.
* ``--max-contracts`` caps how many contracts may be opened (default 1).
* Single underlying, long-only-calls.

Usage
-----
    # confirm decision without trading:
    uv run python scripts/run_ibkr_options_demo.py \\
        --params artifacts/ema_cross_long_call_aapl.json

    # actually submit paper orders:
    uv run python scripts/run_ibkr_options_demo.py \\
        --params artifacts/ema_cross_long_call_aapl.json --live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


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


async def _find_listed_contract(
    ib_async_mod: Any,
    ib: Any,
    *,
    underlying: str,
    target_expiry: date,
    target_strike: float,
    option_type: str,
) -> Any:
    """Pick the listed (expiry, strike) closest to the strategy's target.

    Uses ``reqSecDefOptParams`` to enumerate all valid expiries/strikes for
    the underlying, picks the expiry-strike pair with the smallest
    (day-difference, strike-difference) and qualifies it. Returns the
    qualified ``Option`` contract.
    """
    stock = ib_async_mod.Stock(underlying, "SMART", "USD")
    await ib.qualifyContractsAsync(stock)

    params_list = await ib.reqSecDefOptParamsAsync(
        underlying, "", "STK", stock.conId
    )
    if not params_list:
        raise RuntimeError(f"no option chain params returned for {underlying}")
    # Prefer SMART; fall back to the first listed exchange.
    pref = next((p for p in params_list if p.exchange == "SMART"), params_list[0])
    expirations = sorted(pref.expirations)
    strikes = sorted(pref.strikes)
    if not expirations or not strikes:
        raise RuntimeError(f"empty option chain for {underlying}")

    def _to_date(yyyymmdd: str) -> date:
        return date.fromisoformat(f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}")

    today = datetime.now(tz=UTC).date()
    future_exps = [e for e in expirations if _to_date(e) > today]
    if not future_exps:
        raise RuntimeError("no future expirations available")
    chosen_exp_str = min(future_exps, key=lambda e: abs((_to_date(e) - target_expiry).days))
    chosen_strike = float(min(strikes, key=lambda k: abs(k - target_strike)))

    opt = ib_async_mod.Option(
        underlying, chosen_exp_str, chosen_strike, option_type, "SMART"
    )
    await ib.qualifyContractsAsync(opt)
    return opt


async def _decide_and_trade(
    *,
    host: str,
    port: int,
    client_id: int,
    symbol: str,
    params_payload: dict[str, Any],
    iv: float,
    max_contracts: int,
    live: bool,
    history_days: int,
) -> int:
    from qufin.data.vendors import IBKRHistoricalOHLC
    from qufin.options._types import CALL, OptionContract
    from qufin.strategies.ema_cross_long_call import EmaCrossLongCallParams, current_state
    from qufin.trading._types import Order, OrderType, TimeInForce
    from qufin.trading.brokers import IBKRBroker

    params = EmaCrossLongCallParams(
        fast_window=int(params_payload["fast_window"]),
        slow_window=int(params_payload["slow_window"]),
        strike_moneyness=float(params_payload["strike_moneyness"]),
        target_dte=int(params_payload["target_dte"]),
        exit_dte=int(params_payload["exit_dte"]),
        contracts=int(params_payload.get("contracts", 1)),
        strike_step=float(params_payload.get("strike_step", 1.0)),
    )

    # 1. Underlying history -> desired state.
    data_src = IBKRHistoricalOHLC(host=host, port=port, client_id=client_id + 10)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=history_days)
    print(f"  pulling {history_days}d of daily {symbol} bars from IBKR...")
    ohlcv = await asyncio.to_thread(data_src.fetch, symbol, start=start, end=end, interval="1d")
    if len(ohlcv) < params.min_bars:
        print(f"  FAIL: got {len(ohlcv)} bars, need {params.min_bars}")
        return 1

    state = current_state(
        bars_close=ohlcv.close(), as_of=end.date(), params=params, symbol=symbol
    )
    print(f"  last close       = {state.last_close:.2f}")
    print(f"  fast EMA / slow  = {state.fast_ema:.2f} / {state.slow_ema:.2f}")
    print(f"  desire           = {'LONG CALL' if state.desire_long else 'FLAT'}")
    print(f"  flat-IV (backtest)= {iv:.3f}")
    if state.suggested_contract is not None:
        sc = state.suggested_contract
        print(f"  suggested target = {sc.underlying} {sc.expiry} {sc.strike}{sc.option_type}")

    # 2. Connect broker, read positions.
    broker = IBKRBroker(host=host, port=port, client_id=client_id)
    try:
        await broker.connect()
        account = await broker.account()
        positions = await broker.positions()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL to connect to IBKR: {exc!r}")
        return 1

    try:
        held_calls = [
            p
            for p in positions
            if isinstance(p.asset, OptionContract)
            and p.asset.underlying == symbol
            and p.asset.option_type == CALL
            and p.qty > 0
        ]
        print(f"  account eq.      = ${account.equity:,.2f}  cash=${account.cash:,.2f}")
        print(f"  current calls    = {len(held_calls)}")
        for p in held_calls:
            c = p.asset
            assert isinstance(c, OptionContract)
            dte_held = (c.expiry - end.date()).days
            print(f"    {c.underlying} {c.expiry} {c.strike}{c.option_type}  "
                  f"qty={p.qty:.0f}  avg={p.avg_price:.2f}  DTE={dte_held}")

        # 3. Reconcile.
        should_close: list[Any] = []
        for p in held_calls:
            c = p.asset
            assert isinstance(c, OptionContract)
            dte_held = (c.expiry - end.date()).days
            if (not state.desire_long) or dte_held <= params.exit_dte:
                should_close.append(p)

        should_open = state.desire_long and len(held_calls) == 0

        if not should_close and not should_open:
            print("  -> no trade (already in target state)")
            return 0

        # IB module shared across operations to amortise the connect cost.
        from ib_async import IB  # noqa: F401  (only used via broker._ib below)
        ib_mod = sys.modules["ib_async"]

        # 3a. Closes first (frees buying power before opens, if any).
        for p in should_close:
            c = p.asset
            assert isinstance(c, OptionContract)
            qty = -float(p.qty)
            print(f"  -> SELL {abs(qty):.0f} {c.underlying} {c.expiry} {c.strike}C (close)")
            if not live:
                continue
            order = Order(
                asset=c,
                qty=qty,
                order_type=OrderType.MARKET,
                tif=TimeInForce.DAY,
                tag="opt_demo_close",
            )
            order_id = await broker.place_order(order)
            print(f"    submitted: order_id={order_id}")

        # 3b. Open.
        if should_open:
            sc = state.suggested_contract
            assert sc is not None
            qty = float(min(params.contracts, max_contracts))
            opt = await _find_listed_contract(
                ib_mod,
                broker._ib,
                underlying=symbol,
                target_expiry=sc.expiry,
                target_strike=sc.strike,
                option_type=CALL,
            )
            chosen = OptionContract(
                strike=float(opt.strike),
                expiry=date.fromisoformat(
                    f"{opt.lastTradeDateOrContractMonth[:4]}-"
                    f"{opt.lastTradeDateOrContractMonth[4:6]}-"
                    f"{opt.lastTradeDateOrContractMonth[6:8]}"
                ),
                option_type=CALL,
                underlying=symbol,
            )
            chosen_dte = (chosen.expiry - end.date()).days
            print(f"  -> BUY  {qty:.0f} {chosen.underlying} {chosen.expiry} "
                  f"{chosen.strike}C  (target {sc.expiry} {sc.strike}, DTE={chosen_dte})")
            if not live:
                print("  (dry-run; pass --live to actually submit)")
                return 0
            order = Order(
                asset=chosen,
                qty=qty,
                order_type=OrderType.MARKET,
                tif=TimeInForce.DAY,
                tag="opt_demo_open",
            )
            order_id = await broker.place_order(order)
            print(f"    submitted: order_id={order_id}")

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
        default="artifacts/ema_cross_long_call_aapl.json",
        help="Params JSON written by train_ibkr_options_demo.py",
    )
    parser.add_argument(
        "--max-contracts", type=int, default=1, help="Hard cap on contracts opened (default 1)"
    )
    parser.add_argument("--history-days", type=int, default=400)
    parser.add_argument("--live", action="store_true",
                        help="Actually submit paper orders. Without this flag, dry-runs.")
    parser.add_argument("--interval", type=int, default=0,
                        help="Seconds between iterations. 0 = run once and exit (default).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    _load_env(repo_root / ".env")
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = _int_env("IBKR_PORT", 7497)
    client_id = _int_env("IBKR_CLIENT_ID", 1) + 60

    params_path = repo_root / args.params
    if not params_path.exists():
        print(f"params file not found: {params_path}")
        print("  run: uv run python scripts/train_ibkr_options_demo.py")
        return 1
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    symbol = str(payload["symbol"])
    print(f"Loaded params for {symbol} (trained {payload.get('trained_at', '?')})")
    print(f"  in-sample Sharpe   = {payload['metrics']['in_sample_sharpe']:.3f}")
    print(f"  out-of-sample      = {payload['metrics']['out_of_sample_sharpe']:.3f}"
          f"  ({payload['metrics']['out_of_sample_trades']} trades)")
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
            iv=float(payload.get("iv", 0.25)),
            max_contracts=args.max_contracts,
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
