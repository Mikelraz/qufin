"""
Fetch historical OHLC bars from IBKR and optionally write them to Parquet.

Wraps ``qufin.data.vendors.IBKRHistoricalOHLC``. Uses delayed data (free on
paper accounts). Bars match the project ``BAR_SCHEMA``.

Usage
-----
    # last 30 calendar days of daily SPY bars to stdout
    uv run python scripts/ibkr_history.py --symbol SPY --days 30

    # 5-minute bars over an explicit window, written to parquet
    uv run python scripts/ibkr_history.py --symbol CRWV --interval 5m \\
        --start 2026-05-01 --end 2026-05-28 --out artifacts/crwv_5m.parquet

    # several symbols at once (writes one parquet each into --out-dir)
    uv run python scripts/ibkr_history.py --symbols SPY,QQQ,IWM --days 90 --out-dir artifacts/
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _common import HISTORY_OFFSET, ibkr_config, parse_date


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--symbol", help="Single ticker.")
    g.add_argument("--symbols", help="Comma-separated tickers.")
    parser.add_argument("--interval", default="1d",
                        help="1s,5s,15s,30s,1m,2m,5m,15m,30m,1h,2h,4h,1d,1w,1mo (default 1d).")
    parser.add_argument("--days", type=int, default=None, help="Window = last N days up to now.")
    parser.add_argument("--start", type=parse_date, default=None, help="YYYY-MM-DD (with --end).")
    parser.add_argument("--end", type=parse_date, default=None, help="YYYY-MM-DD (default now).")
    parser.add_argument("--out", type=str, default=None, help="Parquet path (single symbol).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to write one <symbol>.parquet per symbol.")
    args = parser.parse_args()

    if args.days is None and args.start is None:
        parser.error("provide either --days or --start")

    end = (datetime.combine(args.end, datetime.min.time(), tzinfo=UTC)
           if args.end else datetime.now(tz=UTC))
    if args.days is not None:
        start = end - timedelta(days=args.days)
    else:
        start = datetime.combine(args.start, datetime.min.time(), tzinfo=UTC)

    symbols = ([args.symbol] if args.symbol
               else [s.strip() for s in args.symbols.split(",") if s.strip()])
    cfg = ibkr_config().offset(HISTORY_OFFSET)

    try:
        from qufin.data.vendors import IBKRHistoricalOHLC
    except ImportError:
        print("ib_async not installed. Run: uv sync --group trading-live")
        return 1

    src = IBKRHistoricalOHLC(host=cfg.host, port=cfg.port, client_id=cfg.client_id)
    print(f"Fetching {args.interval} bars for {symbols} "
          f"[{start.date()} .. {end.date()}] from {cfg.host}:{cfg.port}...")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if len(symbols) == 1:
        ohlc = src.fetch(symbols[0], start=start, end=end, interval=args.interval)
        frames = {symbols[0]: ohlc}
    else:
        frames = src.fetch_many(symbols, start=start, end=end, interval=args.interval)

    for sym, ohlc in frames.items():
        df = ohlc.data
        print(f"\n=== {sym}: {df.height} bars ===")
        print(df.head(3))
        if df.height > 3:
            print("...")
            print(df.tail(2))
        path: Path | None = None
        if out_dir:
            path = out_dir / f"{sym}.parquet"
        elif args.out and len(symbols) == 1:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
        if path is not None:
            df.write_parquet(path)
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
