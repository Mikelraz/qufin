"""
Option-chain explorer for any symbol.

Pulls the chain via ``reqSecDefOptParams``, narrows to a DTE window and a
strike band (around a target strike, defaulting to spot), then fetches delayed
quotes + Greeks for each contract and prints one table per expiry.

Usage
-----
    uv run python scripts/ibkr_chain.py --symbol CRWV
    uv run python scripts/ibkr_chain.py --symbol SPY --min-dte 0 --max-dte 14 --side put
    uv run python scripts/ibkr_chain.py --symbol NOW --target-strike 900 --strike-band 50
    uv run python scripts/ibkr_chain.py --symbol CRWV --strikes 100,110,120
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import UTC, date, datetime

from _common import CHAIN_OFFSET, fmt, ibkr_config


def _ymd(s: str) -> date:
    return date.fromisoformat(f"{s[:4]}-{s[4:6]}-{s[6:8]}")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", choices=("call", "put", "both"), default="call")
    parser.add_argument("--target-strike", type=float, default=None,
                        help="Centre of the strike band (default: spot).")
    parser.add_argument("--strike-band", type=float, default=None,
                        help="Half-width of the strike band in dollars (default: 15%% of spot).")
    parser.add_argument("--strikes", type=str, default=None,
                        help="Explicit comma-separated strikes (overrides target/band).")
    parser.add_argument("--min-dte", type=int, default=1)
    parser.add_argument("--max-dte", type=int, default=30)
    parser.add_argument("--max-strikes-per-expiry", type=int, default=9)
    parser.add_argument("--settle-seconds", type=float, default=4.0)
    args = parser.parse_args()

    cfg = ibkr_config().offset(CHAIN_OFFSET)

    try:
        from ib_async import Stock
    except ImportError:
        print("ib_async not installed. Run: uv sync --group trading-live")
        return 1

    from qufin.options._types import CALL, PUT, OptionContract
    from qufin.trading.brokers import QuoteSession

    sides = {"call": [CALL], "put": [PUT], "both": [CALL, PUT]}[args.side]

    async with QuoteSession(host=cfg.host, port=cfg.port, client_id=cfg.client_id) as qs:
        ib = qs.ib
        stock = Stock(args.symbol, "SMART", "USD")
        await ib.qualifyContractsAsync(stock)
        spot = await qs.spot(args.symbol, settle_seconds=args.settle_seconds)
        print(f"Underlying {args.symbol}: delayed last/close = {fmt(spot, '${:.2f}')}\n")

        params = await ib.reqSecDefOptParamsAsync(args.symbol, "", "STK", stock.conId)
        if not params:
            print(f"No option chain returned for {args.symbol}. "
                  "Not listed, or the account lacks options permissions.")
            return 1
        chain = next((p for p in params if p.exchange == "SMART"), params[0])
        all_expiries = sorted(_ymd(e) for e in chain.expirations)
        all_strikes = sorted(chain.strikes)
        today = datetime.now(tz=UTC).date()

        in_window = [e for e in all_expiries if args.min_dte <= (e - today).days <= args.max_dte]
        if not in_window:
            dtes = [(e - today).days for e in all_expiries[:10]]
            print(f"No expiries in [{args.min_dte}, {args.max_dte}] DTE. Available DTEs: {dtes}...")
            return 1

        ref = args.target_strike if args.target_strike is not None else (spot or all_strikes[0])
        band_strikes = _select_strikes(args, all_strikes, ref)
        if not band_strikes:
            print(f"No strikes selected. Available range: "
                  f"${all_strikes[0]:.1f}-${all_strikes[-1]:.1f}")
            return 1
        print(f"Expiries: {[f'{e.isoformat()} (DTE={(e - today).days})' for e in in_window]}")
        print(f"Strikes:  {band_strikes}\n")

        contracts = [
            OptionContract(strike=k, expiry=exp, option_type=side, underlying=args.symbol)
            for exp in in_window for k in band_strikes for side in sides
        ]
        total = len(contracts)
        print(f"Fetching delayed quotes + Greeks for {total} contracts "
              f"(~{total * args.settle_seconds / max(1, total // 60 + 1):.0f}s)...\n")
        quotes = await qs.option_quotes(contracts, settle_seconds=args.settle_seconds)

    _print_tables(contracts, quotes, today, ref)
    print("\nReminder: 1 contract = 100 shares. Quotes are DELAYED (15 min) on paper.")
    return 0


def _select_strikes(args: argparse.Namespace, all_strikes: list[float], ref: float) -> list[float]:
    if args.strikes is not None:
        wanted = sorted(float(x.strip()) for x in args.strikes.split(",") if x.strip())
        avail = set(all_strikes)
        chosen = [k for k in wanted if k in avail]
        missing = [k for k in wanted if k not in avail]
        if missing:
            print(f"WARN: requested strikes not listed and dropped: {missing}")
        return chosen
    band = args.strike_band if args.strike_band is not None else ref * 0.15
    lo, hi = ref - band, ref + band
    chosen = [k for k in all_strikes if lo <= k <= hi]
    if len(chosen) > args.max_strikes_per_expiry:
        chosen = sorted(sorted(chosen, key=lambda k: abs(k - ref))[: args.max_strikes_per_expiry])
    return chosen


def _print_tables(
    contracts: list, quotes: dict, today: date, ref: float
) -> None:
    rows = sorted(
        ((c, quotes[c]) for c in contracts),
        key=lambda cq: (cq[0].expiry, cq[0].option_type, cq[0].strike),
    )
    header = (f"{'Strike':>7} {'Side':>4}  {'Bid':>6} {'Ask':>6} {'Mid':>6} {'Spr%':>5}  "
              f"{'IV':>6}  {'Delta':>6} {'Gamma':>6} {'Theta':>7} {'Vega':>6}  "
              f"{'OI':>6} {'Vol':>6}")
    cur: date | None = None
    for c, q in rows:
        if c.expiry != cur:
            cur = c.expiry
            print(f"\n=== {c.expiry.isoformat()} (DTE={(c.expiry - today).days}) ===")
            print(header)
            print("-" * len(header))
        print(f"{c.strike:>7.1f} {c.option_type:>4}  "
              f"{fmt(q.bid):>6} {fmt(q.ask):>6} {fmt(q.mid):>6} {fmt(q.spread_pct, '{:.1f}'):>5}  "
              f"{fmt(q.iv, '{:.2%}'):>6}  {fmt(q.delta, '{:+.3f}'):>6} {fmt(q.gamma, '{:.3f}'):>6} "
              f"{fmt(q.theta, '{:+.3f}'):>7} {fmt(q.vega, '{:.3f}'):>6}  "
              f"{fmt(q.open_interest, '{:.0f}'):>6} {fmt(q.volume, '{:.0f}'):>6}")

    print("\n--- nearest-the-money per expiry ---")
    by_exp: dict[date, list] = defaultdict(list)
    for c, q in rows:
        by_exp[c.expiry].append((c, q))
    for exp, items in sorted(by_exp.items()):
        c, q = min(items, key=lambda cq: abs(cq[0].strike - ref))
        print(f"  {exp.isoformat()}: {c.strike:.1f}{c.option_type}  mid={fmt(q.mid, '${:.2f}')}  "
              f"delta={fmt(q.delta, '{:+.2f}')}  IV={fmt(q.iv, '{:.0%}')}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
