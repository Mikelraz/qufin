"""
Account snapshot for the configured IBKR account.

Prints:
  1. Account summary (cash, equity, buying power, P&L)
  2. Open positions, each refreshed with a delayed mid so unrealised P&L is
     current (not just the last fill). Use ``--skip-quotes`` to skip the refresh.
  3. Working/open orders across all clients that haven't filled or cancelled.

Uses delayed market data; no live data subscription required.

Usage
-----
    uv run python scripts/ibkr_account.py
    uv run python scripts/ibkr_account.py --skip-quotes      # faster, no MTM
    uv run python scripts/ibkr_account.py --check            # connection smoke test only

Exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from _common import ACCOUNT_OFFSET, ibkr_config


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--skip-quotes", action="store_true",
                        help="Skip per-position quote refresh (faster).")
    parser.add_argument("--settle-seconds", type=float, default=3.0,
                        help="Seconds to wait per quote refresh (default 3).")
    parser.add_argument("--check", action="store_true",
                        help="Connection smoke test only: print account + position count, exit.")
    args = parser.parse_args()

    cfg = ibkr_config().offset(ACCOUNT_OFFSET)

    try:
        from ib_async import IB  # noqa: F401
    except ImportError:
        print("ib_async not installed. Run: uv sync --group trading-live")
        return 1

    from qufin.options._types import OptionContract
    from qufin.trading.brokers import IBKRBroker, MarketDataType, quote_option

    broker = IBKRBroker(host=cfg.host, port=cfg.port, client_id=cfg.client_id)
    try:
        await broker.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to connect to IBKR at {cfg.host}:{cfg.port} "
              f"(client_id={cfg.client_id}): {exc!r}")
        print("  - is IB Gateway / TWS running and logged in?")
        print("  - is the API enabled and the socket port matching IBKR_PORT?")
        print("  - is IBKR_CLIENT_ID unique vs other connected clients?")
        return 1

    try:
        snap = await broker.account()
        positions = await broker.positions()

        print(f"=== Account ({cfg.host}:{cfg.port}) ===")
        print(f"  cash         = ${snap.cash:,.2f}")
        print(f"  equity       = ${snap.equity:,.2f}")
        print(f"  buying_power = ${snap.buying_power:,.2f}")
        print(f"  unreal P&L   = ${snap.day_pnl:+,.2f}")
        print(f"  realised P&L = ${snap.total_pnl - snap.day_pnl:+,.2f}")

        if args.check:
            print(f"\nConnection OK. {len(positions)} open position(s).")
            return 0

        print(f"\n=== Positions ({len(positions)}) ===")
        if not positions:
            print("  (none)")
        elif args.skip_quotes:
            for p in positions:
                print(f"  {_describe(p.asset)}  qty={p.qty:+.0f}  avg=${p.avg_price:.2f}")
        else:
            broker._ib.reqMarketDataType(int(MarketDataType.DELAYED_FROZEN))
            today = datetime.now(tz=UTC).date()
            for p in positions:
                if isinstance(p.asset, OptionContract):
                    c = p.asset
                    dte = (c.expiry - today).days
                    avg_per_share = p.avg_price / 100.0  # option avgCost is per contract
                    q = await quote_option(broker._ib, c, settle_seconds=args.settle_seconds)
                    mid = q.mid
                    if mid is not None:
                        mtm = (mid - avg_per_share) * 100 * p.qty
                        ret = (mid / avg_per_share - 1) * 100 if avg_per_share else 0.0
                        print(f"  {_describe(c)}  qty={p.qty:+.0f}  avg=${avg_per_share:.2f}  "
                              f"mid=${mid:.2f}  DTE={dte}  MTM=${mtm:+,.2f} ({ret:+.0f}%)")
                    else:
                        print(f"  {_describe(c)}  qty={p.qty:+.0f}  avg=${avg_per_share:.2f}  "
                              f"mid=n/a  DTE={dte}")
                else:
                    print(f"  {p.asset}  qty={p.qty:+.0f}  avg=${p.avg_price:.2f}")

        # Working orders across all clients (not just this client_id).
        await broker._ib.reqAllOpenOrdersAsync()
        await asyncio.sleep(1.0)
        active = {"PendingSubmit", "PreSubmitted", "Submitted", "ApiPending",
                  "PendingCancel", "Inactive"}
        working = [t for t in broker._ib.trades() if t.orderStatus.status in active]
        print(f"\n=== Working orders ({len(working)}) ===")
        if not working:
            print("  (none)")
        for t in working:
            c = t.contract
            if c.secType == "OPT":
                exp = c.lastTradeDateOrContractMonth
                desc = f"{c.symbol} {exp[:4]}-{exp[4:6]}-{exp[6:8]} {c.strike:g}{c.right}"
            else:
                desc = c.symbol
            o = t.order
            tag = getattr(o, "orderRef", "") or ""
            print(f"  #{o.orderId}  {o.action} {o.totalQuantity:.0f} {desc}  "
                  f"{o.orderType}@${o.lmtPrice:.2f}  {o.tif}  -> {t.orderStatus.status}"
                  + (f"  [{tag}]" if tag else ""))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc!r}")
        return 1
    finally:
        await broker.disconnect()


def _describe(asset: object) -> str:
    from qufin.options._types import OptionContract

    if isinstance(asset, OptionContract):
        return f"{asset.underlying} {asset.expiry} {asset.strike:g}{asset.option_type}"
    return str(asset)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
