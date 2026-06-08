"""
Shared CLI glue for the ``ibkr_*.py`` command-line tools in this directory.

Keeps the thin scripts free of the copy-pasted ``.env`` loader, client-id
arithmetic, and number formatting. The reusable connection/quote logic lives in
the library (``qufin.trading.brokers.quotes`` and ``qufin.trading.brokers``);
only the genuinely script-level glue lives here.

Client-id offsets
-----------------
Order-placing tools connect their ``IBKRBroker`` on the base ``IBKR_CLIENT_ID``.
Read-only tools (quotes, chains, alerts) connect on ``base + <offset>`` so they
never collide with the broker connection or with each other when two tools run
at once. The offsets below are arbitrary-but-distinct.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qufin.data.vendors import IBKRErrorListener

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Distinct client-id offsets per read-only tool (added to IBKR_CLIENT_ID).
ACCOUNT_OFFSET = 95
QUOTE_OFFSET = 90
CHAIN_OFFSET = 96
GEX_OFFSET = 97
ALERT_OFFSET = 99
HISTORY_OFFSET = 11
CANCEL_OFFSET = 94


def load_env(path: Path | None = None) -> None:
    """Minimal ``.env`` loader — ``KEY=VALUE`` per line, ``#`` comments allowed."""
    path = path or (_PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().split("#", 1)[0].strip())


def int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(slots=True, frozen=True)
class IBKRConfig:
    """Resolved IBKR connection settings."""

    host: str
    port: int
    client_id: int

    def offset(self, by: int) -> IBKRConfig:
        """Return a copy with ``client_id`` shifted (for a disjoint read-only client)."""
        return IBKRConfig(self.host, self.port, self.client_id + by)


def ibkr_config(*, default_port: int = 4002) -> IBKRConfig:
    """Load IBKR connection settings from the project ``.env`` / environment.

    Default port 4002 = paper IB Gateway. Override with ``IBKR_PORT`` (7497 =
    paper TWS, 4001 = live Gateway, 7496 = live TWS).
    """
    load_env()
    return IBKRConfig(
        host=os.environ.get("IBKR_HOST", "127.0.0.1"),
        port=int_env("IBKR_PORT", default_port),
        client_id=int_env("IBKR_CLIENT_ID", 1),
    )


def parse_date(s: str) -> date:
    """Parse an ISO ``YYYY-MM-DD`` date for argparse ``type=`` use."""
    return date.fromisoformat(s)


def fmt(v: float | None, spec: str = "{:.2f}", dash: str = "  -  ") -> str:
    """Format an optional float, rendering ``None`` as a dash placeholder."""
    return dash if v is None else spec.format(v)


def print_ib_errors(listener: IBKRErrorListener, *, only_problems: bool = True) -> None:
    """Print classified IBKR messages captured by an ``IBKRErrorListener``.

    With ``only_problems`` (default) the benign data-farm/notice chatter is
    skipped and only data-subscription, connection, contract, and order-reject
    messages are shown.
    """
    records = listener.problems() if only_problems else listener.errors()
    if not records:
        return
    print("\nIBKR messages:")
    for rec in records:
        print(f"  - {rec}")


def fmt_dollars(x: float) -> str:
    """Compact signed dollar formatting: ``$1.23B`` / ``$456.0M`` / ``$7.8K``."""
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.1f}K"
    return f"{sign}${a:,.0f}"
