"""
One-shot smoke test for the Alpaca paper-trading connection.

Loads credentials from the project-root ``.env`` (gitignored), opens an
``AlpacaBroker``, queries the account + positions, and prints a small
summary. Exits 0 on success, 1 on any failure. Safe to re-run.

Usage::

    uv run python scripts/check_alpaca_connection.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _load_env(path: Path) -> None:
    """Minimal ``.env`` loader — KEY=VALUE per line, ``#`` comments allowed."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


async def main() -> int:
    _load_env(Path(__file__).resolve().parent.parent / ".env")

    from qufin.trading.brokers import AlpacaBroker

    broker = AlpacaBroker(paper=True)
    try:
        await broker.connect()
        snap = await broker.account()
        positions = await broker.positions()
    except Exception as exc:  # noqa: BLE001 — surface the actual error to the user
        print(f"FAILED to query Alpaca paper account: {exc!r}")
        return 1
    finally:
        await broker.disconnect()

    print("Alpaca paper connection OK")
    print(f"  cash         = ${snap.cash:,.2f}")
    print(f"  equity       = ${snap.equity:,.2f}")
    print(f"  buying_power = ${snap.buying_power:,.2f}")
    print(f"  positions    = {len(positions)} open")
    for p in positions:
        print(f"    {p.asset}: qty={p.qty}, avg_price={p.avg_price}, mark={p.last_mark}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
