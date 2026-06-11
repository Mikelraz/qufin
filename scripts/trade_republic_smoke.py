"""Read-only live smoke test for :class:`TradeRepublicBroker`.

Connects to a real Trade Republic account through the qufin broker adapter,
reads cash / equity / open positions / one live quote, and prints them. It
**never** places, modifies, or cancels an order.

The 2FA code Trade Republic texts to the phone is entered in a small Tkinter
popup; if no GUI is available it falls back to a console prompt.

Run it::

    uv run --group trading-live python scripts/trade_republic_smoke.py [ISIN]

Credentials come from ``TR_PHONE_NUMBER`` / ``TR_PIN`` (and optional
``TR_LOCALE``), loaded from ``qufin/.env`` or the sibling
``trade-republic-api/.env`` if present.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _load_env() -> None:
    """Populate os.environ from qufin/.env, then the sibling TR .env (no override)."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / ".env",  # qufin/.env
        here.parents[2] / "trade-republic-api" / ".env",  # sibling client repo
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _popup_token_provider(prompt: str) -> str:
    """Ask for the 2FA code in a Tkinter popup; fall back to console input."""
    try:
        import tkinter as tk
        from tkinter import simpledialog
    except Exception:  # noqa: BLE001 - no Tk build: use the console
        return input(prompt).strip()

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        code = simpledialog.askstring("Trade Republic 2FA", prompt, parent=root)
    finally:
        root.destroy()
    if not code:
        raise RuntimeError("no 2FA code entered")
    return code.strip()


async def main(isin: str) -> int:
    _load_env()
    from qufin.trading.brokers import TradeRepublicBroker

    broker = TradeRepublicBroker(token_provider=_popup_token_provider)
    print(f"Connecting as {broker.phone_number} (locale={broker.locale}) ...", flush=True)
    await broker.connect()
    print("Login OK\n", flush=True)
    try:
        snap = await broker.account()
        positions = await broker.positions()
        print(f"Cash:        {snap.cash:>14.2f} {broker.currency}")
        print(f"Equity:      {snap.equity:>14.2f} {broker.currency}")
        print(f"Buying pwr:  {snap.buying_power:>14.2f} {broker.currency}")
        print(f"Positions:   {len(positions):>14d}\n")
        if positions:
            print(f"  {'ISIN':<14} {'Qty':>10}  {'Avg buy-in':>12}  {'Mark':>12}")
            print(f"  {'-' * 14} {'-' * 10}  {'-' * 12}  {'-' * 12}")
            for p in positions:
                print(f"  {p.asset:<14} {p.qty:>10.4f}  {p.avg_price:>12.2f}  {p.last_mark:>12.2f}")

        # One live quote via the bar stream (tick-derived bar), then stop.
        print(f"\nLive quote for {isin} on {broker.exchange}:")
        stream = broker.stream_bars([isin])
        try:
            bar = await asyncio.wait_for(anext(stream), timeout=broker.timeout)
            print(f"  {bar.symbol}  close={bar.close:.4f}  @ {bar.timestamp:%H:%M:%S} UTC")
        except (StopAsyncIteration, TimeoutError):
            print("  (no quote received)")
        except Exception as exc:  # noqa: BLE001 - a quote hiccup must not fail the smoke test
            print(f"  (quote error: {type(exc).__name__}: {exc})")
        finally:
            await stream.aclose()
    finally:
        await broker.disconnect()
    print("\nDisconnected. (read-only — no orders were placed)")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "US0378331005"  # Apple
    raise SystemExit(asyncio.run(main(target)))
