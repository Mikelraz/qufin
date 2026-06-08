"""
One-off delayed quote for a stock or option contract.

Connects a read-only client, prints the underlying spot, and — if option
parameters are given — bid/ask/mid/IV/Greeks for the contract.

Usage
-----
    # stock quote only:
    uv run python scripts/ibkr_quote.py --symbol CRWV

    # option quote (call by default):
    uv run python scripts/ibkr_quote.py --symbol CRWV --expiry 2026-06-18 --strike 110

    # a put:
    uv run python scripts/ibkr_quote.py --symbol SPY --expiry 2026-06-18 --strike 500 --put
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from _common import QUOTE_OFFSET, fmt, ibkr_config, parse_date


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--expiry", type=parse_date, default=None,
                        help="YYYY-MM-DD (enables option quote)")
    parser.add_argument("--strike", type=float, default=None)
    parser.add_argument("--put", action="store_true", help="Quote a put (default is a call).")
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if (args.expiry is None) != (args.strike is None):
        parser.error("--expiry and --strike must be given together for an option quote")

    cfg = ibkr_config().offset(QUOTE_OFFSET)

    try:
        from ib_async import IB  # noqa: F401
    except ImportError:
        print("ib_async not installed. Run: uv sync --group trading-live")
        return 1

    from qufin.options._types import CALL, PUT, OptionContract
    from qufin.trading.brokers import QuoteSession

    async with QuoteSession(host=cfg.host, port=cfg.port, client_id=cfg.client_id) as qs:
        spot = await qs.spot(args.symbol, settle_seconds=args.settle_seconds)
        print(f"{args.symbol} spot (delayed) = {fmt(spot, '${:.2f}')}")

        if args.expiry is None:
            return 0

        contract = OptionContract(
            strike=float(args.strike),
            expiry=args.expiry,
            option_type=PUT if args.put else CALL,
            underlying=args.symbol,
        )
        q = await qs.option_quote(contract, settle_seconds=args.settle_seconds)
        dte = (args.expiry - datetime.now(tz=UTC).date()).days
        right = "P" if args.put else "C"
        print(f"\n{args.symbol} {args.expiry.isoformat()} {args.strike:g}{right}  (DTE={dte})")
        print(f"  bid={fmt(q.bid)}  ask={fmt(q.ask)}  last={fmt(q.last)}  "
              f"mid={fmt(q.mid, '${:.2f}')}  spread={fmt(q.spread_pct, '{:.1f}%')}")
        print(f"  IV={fmt(q.iv, '{:.1%}')}  delta={fmt(q.delta, '{:+.3f}')}  "
              f"gamma={fmt(q.gamma, '{:.3f}')}  theta={fmt(q.theta, '{:+.3f}')}  "
              f"vega={fmt(q.vega, '{:.3f}')}")
        print(f"  OI={fmt(q.open_interest, '{:.0f}')}  vol={fmt(q.volume, '{:.0f}')}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
