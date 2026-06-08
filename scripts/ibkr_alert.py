"""
Price-trigger alert with optional option-context pricing.

One-shot check intended to be run periodically (cron / Windows Task Scheduler).
Fetches the underlying spot via delayed data, compares it to trigger levels, and
(optionally) prints mid/IV/delta for a watch-list of option strikes so that when
a trigger fires you can see immediately whether premiums moved enough to act.

Never places orders. Safe to run on a live account.

Trigger direction
-----------------
* ``--below`` (default): fire when spot <= a trigger (buy-the-dip watch).
* ``--above``: fire when spot >= a trigger (breakout / take-profit watch).

Exit codes
----------
  0 = no trigger crossed
  1 = at least one trigger crossed (wire into a notifier on exit 1)

Usage
-----
    uv run python scripts/ibkr_alert.py --symbol ONDS --triggers 10,9,8 \\
        --strikes 15,17,20 --expiry 2027-01-15
    uv run python scripts/ibkr_alert.py --symbol NVDA --above --triggers 1200,1300
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from _common import ALERT_OFFSET, fmt, ibkr_config, parse_date


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--triggers", required=True, help="Comma-separated trigger levels.")
    parser.add_argument("--above", action="store_true",
                        help="Fire when spot >= a trigger (default: spot <= trigger).")
    parser.add_argument("--strikes", default=None,
                        help="Optional comma-separated option strikes to price for context.")
    parser.add_argument("--expiry", type=parse_date, default=None,
                        help="Expiry for --strikes context (YYYY-MM-DD).")
    parser.add_argument("--put", action="store_true", help="Price puts (default calls).")
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.strikes and args.expiry is None:
        parser.error("--strikes requires --expiry")

    triggers = sorted((float(t) for t in args.triggers.split(",") if t.strip()),
                      reverse=not args.above)
    cfg = ibkr_config().offset(ALERT_OFFSET)

    try:
        from ib_async import IB  # noqa: F401
    except ImportError:
        print("ib_async not installed. Run: uv sync --group trading-live")
        return 1

    from qufin.options._types import CALL, PUT, OptionContract
    from qufin.trading.brokers import QuoteSession

    async with QuoteSession(host=cfg.host, port=cfg.port, client_id=cfg.client_id) as qs:
        spot_q = await qs.stock_quote(args.symbol, settle_seconds=args.settle_seconds)
        spot = spot_q.last or spot_q.mid or spot_q.bid
        if spot is None:
            print(f"FAIL: could not fetch {args.symbol} spot")
            return 1

        ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"=== {args.symbol} alert @ {ts} ===")
        print(f"  spot     = ${spot:.2f}")
        fired = [t for t in triggers if (spot >= t if args.above else spot <= t)]
        rel = ">=" if args.above else "<="
        print(f"  triggers ({rel}) = {triggers}  fired: "
              + (f"{fired}  ALERT" if fired else "none"))

        if args.strikes:
            strikes = sorted(float(s) for s in args.strikes.split(",") if s.strip())
            right = "P" if args.put else "C"
            dte = (args.expiry - datetime.now(tz=UTC).date()).days
            contracts = [
                OptionContract(strike=k, expiry=args.expiry,
                               option_type=PUT if args.put else CALL, underlying=args.symbol)
                for k in strikes
            ]
            quotes = await qs.option_quotes(contracts, settle_seconds=args.settle_seconds)
            print(f"\n  {args.expiry.isoformat()} {right} (DTE={dte}):")
            print(f"    {'Strike':>7}  {'Bid':>6} {'Ask':>6} {'Mid':>6}  {'IV':>6}  {'Delta':>6}")
            for c in contracts:
                q = quotes[c]
                print(f"    {c.strike:>7.1f}  {fmt(q.bid):>6} {fmt(q.ask):>6} "
                      f"{fmt(q.mid, '${:.2f}'):>6}  {fmt(q.iv, '{:.0%}'):>6}  "
                      f"{fmt(q.delta, '{:+.2f}'):>6}")

    if fired:
        edge = min(fired) if args.above else max(fired)
        print(f"\n  ACTION: spot crossed the ${edge:.2f} trigger.")
        return 1
    nearest = triggers[0]
    print(f"\n  Distance to first trigger (${nearest:.2f}): "
          f"${spot - nearest:+.2f} ({(nearest / spot - 1) * 100:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
