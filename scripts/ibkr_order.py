"""
Place a buy or sell order on a stock or option (paper or live IBKR).

Dry-runs by default — prints the proposed order(s), a fresh delayed quote, and
the account snapshot. Pass ``--live`` to actually submit.

Asset
-----
* Stock:  give ``--symbol`` only.
* Option: give ``--symbol --expiry --strike`` (call by default; ``--put`` for a put).

Price (single-leg orders)
-------------------------
* ``--market``        market order.
* ``--limit X``       explicit limit price per share.
* ``--at-ask`` / ``--at-bid``   limit = current ask / bid.
* default             limit = mid + ``--mid-offset`` (offset defaults to 0).

Laddered orders
---------------
``--tiers "1@10.00,1@13.00"`` submits one LIMIT order per tier (qty@price),
ignoring ``--qty`` and the price flags. Handy for scale-out sells or staged
entries; combine with ``--tif gtc`` so the legs rest in the book.

Usage
-----
    # dry-run: buy 1 CRWV call at mid
    uv run python scripts/ibkr_order.py --symbol CRWV --expiry 2026-06-18 --strike 110 --side buy

    # submit a marketable buy of 2 contracts
    uv run python scripts/ibkr_order.py --symbol CRWV --expiry 2026-06-18 --strike 110 \\
        --side buy --qty 2 --at-ask --live

    # two-tier GTC scale-out sell
    uv run python scripts/ibkr_order.py --symbol CRWV --expiry 2026-06-18 --strike 110 \\
        --side sell --tiers "1@10.00,1@13.00" --tif gtc --live

    # buy 100 shares of SPY at market
    uv run python scripts/ibkr_order.py --symbol SPY --side buy --qty 100 --market --live
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

from _common import QUOTE_OFFSET, fmt, ibkr_config, parse_date, print_ib_errors


@dataclass(slots=True)
class Leg:
    qty: int
    limit_price: float | None  # None => use price flags / market
    label: str


def _parse_tiers(spec: str) -> list[Leg]:
    legs: list[Leg] = []
    for i, part in enumerate(p.strip() for p in spec.split(",") if p.strip()):
        qty_s, _, price_s = part.partition("@")
        if not price_s:
            raise SystemExit(f"--tiers entry {part!r} must be 'qty@price'")
        legs.append(Leg(qty=int(qty_s), limit_price=float(price_s), label=f"tier{i + 1}"))
    return legs


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", choices=("buy", "sell"), required=True)
    parser.add_argument("--expiry", type=parse_date, default=None, help="YYYY-MM-DD (=> option)")
    parser.add_argument("--strike", type=float, default=None)
    parser.add_argument("--put", action="store_true", help="Option is a put (default call).")
    parser.add_argument("--qty", type=int, default=1, help="Units: shares, or contracts.")
    parser.add_argument("--tiers", type=str, default=None, help="Laddered legs 'qty@price,...'.")
    parser.add_argument("--limit", type=float, default=None, help="Explicit limit price per share.")
    parser.add_argument("--market", action="store_true", help="Market order.")
    parser.add_argument("--at-ask", action="store_true", help="Limit = current ask.")
    parser.add_argument("--at-bid", action="store_true", help="Limit = current bid.")
    parser.add_argument("--mid-offset", type=float, default=0.0, help="Limit = mid + offset.")
    parser.add_argument("--tif", choices=("day", "gtc", "ioc", "fok"), default="day")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--live", action="store_true", help="Submit. Without it, dry-runs.")
    args = parser.parse_args()

    if (args.expiry is None) != (args.strike is None):
        parser.error("--expiry and --strike must be given together for an option")

    cfg = ibkr_config()

    try:
        from ib_async import IB  # noqa: F401
    except ImportError:
        print("ib_async not installed. Run: uv sync --group trading-live")
        return 1

    from qufin.options._types import CALL, PUT, OptionContract
    from qufin.trading._types import Order, OrderRejectedError, OrderType, TimeInForce
    from qufin.trading.brokers import IBKRBroker, QuoteSession

    is_option = args.expiry is not None
    if is_option:
        asset: object = OptionContract(
            strike=float(args.strike),
            expiry=args.expiry,
            option_type=PUT if args.put else CALL,
            underlying=args.symbol,
        )
        multiplier = 100.0
        right = "P" if args.put else "C"
        dte = (args.expiry - datetime.now(tz=UTC).date()).days
        desc = f"{args.symbol} {args.expiry.isoformat()} {args.strike:g}{right} (DTE={dte})"
    else:
        asset = args.symbol
        multiplier = 1.0
        desc = args.symbol

    sign = 1 if args.side == "buy" else -1
    legs = _parse_tiers(args.tiers) if args.tiers else [Leg(args.qty, None, "order")]
    total_qty = sum(leg.qty for leg in legs)
    print(f"Asset: {desc}")
    print(
        f"Action: {args.side.upper()}  total units: {total_qty}  "
        f"(notional {total_qty * multiplier:g})"
    )

    # 1. Refresh a delayed quote for pricing + display.
    qcfg = cfg.offset(QUOTE_OFFSET)
    async with QuoteSession(host=qcfg.host, port=qcfg.port, client_id=qcfg.client_id) as qs:
        spot = await qs.spot(args.symbol, settle_seconds=args.settle_seconds)
        if is_option:
            q = await qs.option_quote(asset, settle_seconds=args.settle_seconds)  # type: ignore[arg-type]
        else:
            q = await qs.stock_quote(args.symbol, settle_seconds=args.settle_seconds)
    bid, ask, mid = q.bid, q.ask, q.mid
    print(f"\nFresh quote (delayed):  {args.symbol} spot={fmt(spot, '${:.2f}')}")
    print(f"  bid={fmt(bid)}  ask={fmt(ask)}  last={fmt(q.last)}  mid={fmt(mid, '${:.2f}')}")
    if is_option and q.iv is not None:
        print(
            f"  IV={fmt(q.iv, '{:.1%}')}  delta={fmt(q.delta, '{:+.2f}')}  "
            f"theta={fmt(q.theta, '{:+.3f}')}/day"
        )

    # 2. Resolve order type + price for each leg.
    order_type = OrderType.MARKET if (args.market and not args.tiers) else OrderType.LIMIT
    resolved: list[tuple[Leg, float | None]] = []
    for leg in legs:
        if leg.limit_price is not None:
            price = leg.limit_price
        elif order_type == OrderType.MARKET:
            price = None
        else:
            price = _resolve_price(args, bid=bid, ask=ask, mid=mid)
            if price is None:
                print(
                    "\nFAIL: cannot determine a limit price (bid/ask missing). "
                    "Pass --limit X or --market."
                )
                return 1
        resolved.append((leg, price))

    # 3. Print the proposed orders.
    print(f"\nProposed order(s)  [{order_type.value.upper()}, {args.tif.upper()}]:")
    total_value = 0.0
    for leg, price in resolved:
        if price is None:
            print(f"  {leg.label}: {args.side.upper()} {leg.qty} @ MARKET")
        else:
            value = price * multiplier * leg.qty
            total_value += value
            extra = ""
            if is_option and args.side == "buy":
                extra = f"  breakeven=${float(args.strike) + price:.2f}"
            print(
                f"  {leg.label}: {args.side.upper()} {leg.qty} @ ${price:.2f}  "
                f"value=${value:,.2f}{extra}"
            )
    if total_value:
        kind = "max cost/loss" if args.side == "buy" else "proceeds"
        print(f"  total {kind} = ${total_value:,.2f}")

    # 4. Connect broker, check account + holdings.
    broker = IBKRBroker(host=cfg.host, port=cfg.port, client_id=cfg.client_id)
    print(f"\nConnecting broker (client_id={cfg.client_id})...")
    try:
        await broker.connect()
        snap = await broker.account()
        print(
            f"  cash=${snap.cash:,.2f}  equity=${snap.equity:,.2f}  "
            f"buying_power=${snap.buying_power:,.2f}"
        )

        if args.side == "buy" and total_value > snap.cash * 0.5:
            print(f"  WARN: cost ${total_value:,.0f} > 50% of cash ${snap.cash:,.0f}.")
        if args.side == "sell":
            held = await _held_qty(broker, asset, is_option)
            print(f"  current holding: {held:+.0f} unit(s)")
            if held < total_qty:
                print(
                    f"\nFAIL: hold {held:.0f} but selling {total_qty}. "
                    "Reduce --qty/--tiers or verify the position."
                )
                return 1

        if not args.live:
            print("\nMODE = DRY-RUN. Pass --live to submit.")
            return 0

        tif = TimeInForce[args.tif.upper()]
        order_ids: list[str] = []
        for leg, price in resolved:
            order = Order(
                asset=asset,  # type: ignore[arg-type]
                qty=float(sign * leg.qty),
                order_type=order_type,
                tif=tif,
                limit_price=price,
                tag=args.tag or leg.label,
            )
            order_id = await broker.place_order(order)
            order_ids.append(order_id)
            print(f"  submitted {leg.label}: order_id={order_id}")
            await asyncio.sleep(0.5)

        print("\nOrder status:")
        for order_id in order_ids:
            try:
                status = await broker.wait_for_status(order_id, timeout=4.0)
            except OrderRejectedError as exc:
                reason = exc.status.reject_reason or "no reason reported"
                print(f"  {order_id}: REJECTED -> {reason}")
                continue
            total = status.filled + status.remaining
            fill = f"  avgFill={status.avg_fill_price:.2f}" if status.avg_fill_price else ""
            print(f"  {order_id}: {status.status}  filled={status.filled:.0f}/{total:.0f}{fill}")

        # Surface any non-benign IBKR messages (delayed-data, rejects, etc.).
        print_ib_errors(broker.errors)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {exc!r}")
        return 1
    finally:
        await broker.disconnect()
    return 0


def _resolve_price(args: argparse.Namespace, *, bid, ask, mid) -> float | None:  # noqa: ANN001
    if args.limit is not None:
        return float(args.limit)
    if args.at_ask:
        return float(ask) if ask else None
    if args.at_bid:
        return float(bid) if bid else None
    if mid is None:
        return None
    return math.floor((mid + args.mid_offset) * 100 + 0.5) / 100  # round to cent


async def _held_qty(broker, asset, is_option: bool) -> float:  # noqa: ANN001
    from qufin.options._types import OptionContract

    for p in await broker.positions():
        if is_option and isinstance(p.asset, OptionContract) and isinstance(asset, OptionContract):
            if (
                p.asset.underlying == asset.underlying
                and p.asset.expiry == asset.expiry
                and p.asset.strike == asset.strike
                and p.asset.option_type == asset.option_type
            ):
                return p.qty
        elif not is_option and p.asset == asset:
            return p.qty
    return 0.0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
