"""
Cancel working IBKR orders by id, by symbol, or all of them.

Dry-runs by default: lists the orders that *would* be cancelled. Pass ``--live``
to actually cancel. Selection is one of ``--order-id`` / ``--symbol`` / ``--all``.

Usage
-----
    uv run python scripts/ibkr_cancel.py --all                 # preview every working order
    uv run python scripts/ibkr_cancel.py --symbol CRWV --live  # cancel CRWV working orders
    uv run python scripts/ibkr_cancel.py --order-id 123,124 --live
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from _common import CANCEL_OFFSET, ibkr_config

_ACTIVE = {"PendingSubmit", "PreSubmitted", "Submitted", "ApiPending", "PendingCancel", "Inactive"}


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--order-id", help="Comma-separated order id(s) to cancel.")
    g.add_argument("--symbol", help="Cancel all working orders on this underlying symbol.")
    g.add_argument("--all", action="store_true", help="Cancel every working order.")
    parser.add_argument("--live", action="store_true",
                        help="Actually cancel. Without it, previews.")
    args = parser.parse_args()

    cfg = ibkr_config().offset(CANCEL_OFFSET)

    try:
        from ib_async import IB  # noqa: F401
    except ImportError:
        print("ib_async not installed. Run: uv sync --group trading-live")
        return 1

    from qufin.trading.brokers import IBKRBroker

    wanted_ids = (
        {s.strip() for s in args.order_id.split(",") if s.strip()} if args.order_id else None
    )

    broker = IBKRBroker(host=cfg.host, port=cfg.port, client_id=cfg.client_id)
    try:
        await broker.connect()
        await broker._ib.reqAllOpenOrdersAsync()
        await asyncio.sleep(1.0)
        working = [t for t in broker._ib.trades() if t.orderStatus.status in _ACTIVE]

        selected = [t for t in working if _matches(t, wanted_ids, args.symbol, args.all)]
        print(f"=== {len(selected)} order(s) selected (of {len(working)} working) ===")
        if not selected:
            print("  (none)")
            return 0
        for t in selected:
            print(f"  #{t.order.orderId}  {t.order.action} {t.order.totalQuantity:.0f} "
                  f"{t.contract.symbol}  {t.order.orderType}@${t.order.lmtPrice:.2f}  "
                  f"-> {t.orderStatus.status}")

        if not args.live:
            print("\nMODE = DRY-RUN. Pass --live to cancel.")
            return 0

        for t in selected:
            await broker.cancel_order(str(t.order.orderId))
            print(f"  cancel requested: #{t.order.orderId}")
        await asyncio.sleep(2.0)
        print("\nDone.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc!r}")
        return 1
    finally:
        await broker.disconnect()


def _matches(trade, wanted_ids, symbol, want_all) -> bool:  # noqa: ANN001
    if want_all:
        return True
    if wanted_ids is not None:
        return str(trade.order.orderId) in wanted_ids
    return trade.contract.symbol == symbol


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
