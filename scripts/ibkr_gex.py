"""
Compute dealer GEX/VEX/walls/flip for a symbol from a live IBKR chain.

Pulls the chain via ``qufin.options.data.IBKRChainLoader``, runs it through the
``qufin.options.gex`` toolkit, and prints aggregate metrics, magnets
(call/put wall, max-pain, gamma flip), and a per-strike table.

Usage
-----
    uv run python scripts/ibkr_gex.py --symbol CRWV
    uv run python scripts/ibkr_gex.py --symbol NOW --strike-band 0.15 --max-dte 30
    uv run python scripts/ibkr_gex.py --symbol SPY --out artifacts/spy_gex.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from _common import GEX_OFFSET, fmt_dollars, ibkr_config


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--min-dte", type=int, default=0)
    parser.add_argument("--max-dte", type=int, default=60,
                        help="Cap on DTE. Closer = more 'intraday' signal (default 60).")
    parser.add_argument("--max-expiries", type=int, default=6,
                        help="Hard cap on expiries (closest-first, default 6).")
    parser.add_argument("--strike-band", type=float, default=0.25,
                        help="Fraction of spot for strike window (default 0.25 = +/-25%%).")
    parser.add_argument("--settle-seconds", type=float, default=10.0,
                        help="Seconds to wait per batch for delayed quotes (default 10).")
    parser.add_argument("--batch-size", type=int, default=60,
                        help="Max concurrent reqMktData subscriptions per batch (default 60).")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of per-strike rows to display (default 20).")
    parser.add_argument("--solve-iv", action="store_true",
                        help="Re-solve IV from mid prices (slower; usually unnecessary).")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional parquet path to dump per-strike exposures.")
    args = parser.parse_args()

    cfg = ibkr_config().offset(GEX_OFFSET)

    from qufin.options.data import IBKRChainLoader
    from qufin.options.gex import (
        aggregate_exposure,
        call_wall,
        max_pain,
        put_wall,
        zero_gamma_level,
    )

    print(f"Loading {args.symbol} chain from IBKR...")
    loader = IBKRChainLoader(
        host=cfg.host, port=cfg.port, client_id=cfg.client_id,
        settle_seconds=args.settle_seconds, batch_size=args.batch_size,
    )
    chain = await loader.load(
        args.symbol,
        min_dte=args.min_dte, max_dte=args.max_dte, max_expiries=args.max_expiries,
        strike_pct_band=args.strike_band, include_puts=True, solve_iv=args.solve_iv,
    )
    print(f"\nChain: {len(chain.data)} contracts after filtering")
    print(f"  spot         = ${chain.spot:.2f}")
    print(f"  as_of        = {chain.as_of}")
    print(f"  expiries     = {sorted(chain.data['expiry'].unique().to_list())}")

    exposure = aggregate_exposure(chain)
    df = exposure.to_dataframe()
    total_gex = exposure.notes["total_gex"]
    total_dex = exposure.notes["total_dex"]
    total_vex = float(exposure.vex.sum())
    total_charm = float(exposure.charm.sum())
    regime = "POSITIVE (mean-reverting)" if total_gex > 0 else "NEGATIVE (trend-amplifying)"

    cw = _safe_call(call_wall, chain)
    pw = _safe_call(put_wall, chain)
    mp = _safe_call(max_pain, chain)
    flip = zero_gamma_level(chain)

    print(f"\n=== Aggregate ({args.symbol} @ ${chain.spot:.2f}) ===")
    print(f"  Total GEX    = {fmt_dollars(total_gex)}  ({regime})")
    print(f"  Total VEX    = {fmt_dollars(total_vex)}  (dealer vanna $)")
    print(f"  Total DEX    = {fmt_dollars(total_dex)}  (dealer delta $)")
    print(f"  Total charm  = {fmt_dollars(total_charm)}")

    print("\n=== Magnets ===")
    _print_magnet("Call wall", cw, chain.spot)
    _print_magnet("Put wall", pw, chain.spot)
    _print_magnet("Max-pain", mp, chain.spot)
    if flip is not None:
        print(f"  Gamma flip   = ${flip:.2f}  ({(flip / chain.spot - 1) * 100:+.1f}% from spot)")
    else:
        print("  Gamma flip   = no sign change in search window")

    import polars as pl
    df_sorted = (
        df.with_columns(pl.col("gex").abs().alias("abs_gex"))
        .sort("abs_gex", descending=True).head(args.top).sort("strike").drop("abs_gex")
    )
    print(f"\n=== Per-strike exposures (top {args.top} by |GEX|, sorted by strike) ===")
    print(f"{'Strike':>8}  {'GEX':>10}  {'VEX':>10}  {'DEX':>10}  {'Call OI':>9}  {'Put OI':>9}")
    print("-" * 70)
    for row in df_sorted.iter_rows(named=True):
        print(f"{row['strike']:>8.2f}  {fmt_dollars(row['gex']):>10}  "
              f"{fmt_dollars(row['vex']):>10}  {fmt_dollars(row['dex']):>10}  "
              f"{int(row['call_oi']):>9,}  {int(row['put_oi']):>9,}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        print(f"\nWrote per-strike frame to {out_path}")

    print("\n=== Reading the signal ===")
    if total_gex > 0:
        print("  POSITIVE GEX: dealers net-long gamma -> sell rallies, buy dips -> price\n"
              "  mean-reverts, realised vol < implied. Walls are strong magnets. Favours\n"
              "  premium-selling over long calls.")
    else:
        print("  NEGATIVE GEX: dealers net-short gamma -> buy rallies, sell dips -> moves\n"
              "  amplify, realised vol > implied. Trends extend. Favours long calls/puts.")
    if flip is not None:
        side = "ABOVE" if chain.spot > flip else "BELOW"
        next_regime = "PINNING" if total_gex < 0 else "AMPLIFICATION"
        print(f"  Spot ${chain.spot:.2f} is {side} the gamma flip ${flip:.2f}. "
              f"Crossing it would shift the regime to {next_regime}.")
    return 0


def _safe_call(fn, chain) -> float:  # noqa: ANN001
    try:
        return fn(chain)
    except Exception:  # noqa: BLE001
        return float("nan")


def _print_magnet(label: str, value: float, spot: float) -> None:
    if value == value:  # not NaN
        print(f"  {label:<12} = ${value:.2f}  ({(value / spot - 1) * 100:+.1f}% from spot)")
    else:
        print(f"  {label:<12} = n/a")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
