"""
Shared low-level Interactive Brokers plumbing.

Provider-agnostic glue used by every IBKR adapter in qufin — the order broker
(``trading.brokers.ibkr``), the quote session (``trading.brokers.quotes``), the
historical-bar loader (``data.vendors.ibkr``), and the option-chain loader
(``options.data.ibkr``). It lives under ``data.vendors`` because ``data`` is the
lowest layer in the package (both ``options`` and ``trading`` already depend on
it), so all four call sites import from here without creating import cycles.

The single most important thing this module adds over raw ``ib_async`` use is
**error visibility**: :class:`IBKRErrorListener` subscribes to ``errorEvent`` and
classifies every code so callers can tell a benign market-data-farm notice from
a real subscription gap, a connection drop, or an order rejection.

``ib_async`` is imported lazily inside functions so importing this module never
requires the optional ``trading-live`` dependency group.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum, IntEnum
from typing import Any


class MarketDataType(IntEnum):
    """IBKR ``reqMarketDataType`` codes.

    ``DELAYED_FROZEN`` returns last-known delayed values even when the feed has
    gone idle (lunchtime, just after the open), which is what paper accounts
    without a real-time subscription want most of the time.
    """

    REALTIME = 1
    FROZEN = 2
    DELAYED = 3
    DELAYED_FROZEN = 4


def safe_float(x: Any) -> float | None:
    """``ib_async`` reports missing fields as NaN; coerce those to ``None``."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def ymd_to_date(s: str) -> date:
    """Parse IBKR's ``YYYYMMDD`` contract-month string into a ``date``."""
    return date.fromisoformat(f"{s[:4]}-{s[4:6]}-{s[6:8]}")


# ---------------------------------------------------------------------------
# Error classification


class IBKRErrorCategory(Enum):
    """Actionable bucket for an IBKR error/notification code."""

    INFO = "info"  # benign status notices (data-farm connected, etc.)
    DATA = "data"  # market-data subscription / delayed-feed issues
    CONNECTION = "connection"  # socket / gateway / clientId problems
    CONTRACT = "contract"  # contract not found / not permitted
    ORDER_REJECT = "order_reject"  # order rejected or cancelled by the system
    ORDER_WARNING = "order_warning"  # order accepted with an ignored attribute
    UNKNOWN = "unknown"


# Well-known IBKR codes -> category. Not exhaustive by design: the goal is to
# turn the codes we actually encounter into actionable buckets; anything
# unmapped falls through to UNKNOWN. References: IBKR "Message Codes" docs.
_INFO_CODES = frozenset({1101, 1102, 2100, 2104, 2106, 2107, 2108, 2119, 2150, 2158})
_ORDER_WARNING_CODES = frozenset({399, 2109, 2137, 2148})
_DATA_CODES = frozenset({162, 354, 10089, 10091, 10167, 10168, 10197})
_CONNECTION_CODES = frozenset({326, 502, 503, 504, 1100, 1300, 2110})
_CONTRACT_CODES = frozenset({200, 203, 321, 322, 478})
_ORDER_REJECT_CODES = frozenset({201, 202, 10147, 10148})


def classify_error(code: int) -> IBKRErrorCategory:
    """Map an IBKR error code to an :class:`IBKRErrorCategory`."""
    if code in _INFO_CODES:
        return IBKRErrorCategory.INFO
    if code in _ORDER_WARNING_CODES:
        return IBKRErrorCategory.ORDER_WARNING
    if code in _DATA_CODES:
        return IBKRErrorCategory.DATA
    if code in _CONNECTION_CODES:
        return IBKRErrorCategory.CONNECTION
    if code in _CONTRACT_CODES:
        return IBKRErrorCategory.CONTRACT
    if code in _ORDER_REJECT_CODES:
        return IBKRErrorCategory.ORDER_REJECT
    return IBKRErrorCategory.UNKNOWN


@dataclass(slots=True, frozen=True)
class IBKRError:
    """A single classified message from IBKR's ``errorEvent``."""

    code: int
    message: str
    category: IBKRErrorCategory
    req_id: int = -1
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    contract_repr: str | None = None

    @property
    def is_benign(self) -> bool:
        """True for informational notices and accepted-with-warning orders."""
        return self.category in (IBKRErrorCategory.INFO, IBKRErrorCategory.ORDER_WARNING)

    def __str__(self) -> str:
        ctx = f" [{self.contract_repr}]" if self.contract_repr else ""
        return f"({self.category.value}) {self.code}: {self.message}{ctx}"


def _contract_label(contract: Any) -> str | None:
    if contract is None:
        return None
    sym = getattr(contract, "localSymbol", None) or getattr(contract, "symbol", None)
    return str(sym) if sym else None


@dataclass(slots=True)
class IBKRErrorListener:
    """Captures and classifies messages from an ``ib_async.IB`` ``errorEvent``.

    Attach to a connected (or about-to-connect) ``IB`` instance with
    :meth:`attach`; thereafter every error/notification is recorded as an
    :class:`IBKRError`. The bounded buffer keeps the most recent ``max_records``.
    """

    max_records: int = 200
    _records: list[IBKRError] = field(default_factory=lambda: list[IBKRError](), init=False)
    _ib: Any = field(default=None, init=False, repr=False)

    def attach(self, ib: Any) -> None:
        self._ib = ib
        ib.errorEvent += self.handle

    def detach(self) -> None:
        if self._ib is not None:
            self._ib.errorEvent -= self.handle
            self._ib = None

    def handle(
        self, req_id: int, code: int, message: str, contract: Any = None, *_extra: Any
    ) -> None:
        """``errorEvent`` callback; also callable directly (tests, replay)."""
        record = IBKRError(
            code=int(code),
            message=str(message),
            category=classify_error(int(code)),
            req_id=int(req_id),
            contract_repr=_contract_label(contract),
        )
        self._records.append(record)
        if len(self._records) > self.max_records:
            del self._records[: len(self._records) - self.max_records]

    def errors(self, category: IBKRErrorCategory | None = None) -> list[IBKRError]:
        if category is None:
            return list(self._records)
        return [e for e in self._records if e.category == category]

    def problems(self) -> list[IBKRError]:
        """All non-benign records (everything except INFO / ORDER_WARNING)."""
        return [e for e in self._records if not e.is_benign]

    def subscription_warnings(self) -> list[IBKRError]:
        return self.errors(IBKRErrorCategory.DATA)

    @property
    def has_subscription_issue(self) -> bool:
        return any(e.category == IBKRErrorCategory.DATA for e in self._records)

    def last(self, category: IBKRErrorCategory | None = None) -> IBKRError | None:
        items = self.errors(category)
        return items[-1] if items else None

    def clear(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# Connection + market-data snapshots


async def connect_ib(
    host: str,
    port: int,
    client_id: int,
    *,
    timeout: float = 8.0,
    market_data_type: MarketDataType | int | None = None,
    listener: IBKRErrorListener | None = None,
) -> Any:
    """Connect a fresh ``ib_async.IB`` with friendly errors and optional plumbing.

    Attaches ``listener`` *before* connecting (so connect-time errors such as a
    duplicate ``clientId`` are captured), applies ``market_data_type`` if given,
    and translates the common failure modes into a clear :class:`ConnectionError`.
    """
    try:
        from ib_async import IB
    except ImportError as e:
        raise ImportError(
            "ib_async is required for IBKR connectivity. Run: uv sync --group trading-live"
        ) from e

    ib = IB()
    if listener is not None:
        listener.attach(ib)
    try:
        await ib.connectAsync(host, port, clientId=client_id, timeout=timeout)
    except TimeoutError as e:
        raise ConnectionError(
            f"timed out after {timeout:.0f}s connecting to IBKR at {host}:{port} "
            "(is TWS/Gateway running with the API enabled on that port?)"
        ) from e
    except (ConnectionRefusedError, OSError) as e:
        raise ConnectionError(
            f"could not reach IBKR at {host}:{port} "
            "(is TWS/Gateway running with the API enabled on that port?)"
        ) from e
    if market_data_type is not None:
        ib.reqMarketDataType(int(market_data_type))
    return ib


def _is_quotable(ticker: Any) -> bool:
    bid = safe_float(getattr(ticker, "bid", None))
    ask = safe_float(getattr(ticker, "ask", None))
    return bid is not None and ask is not None and bid > 0.0 and ask > 0.0


async def snapshot(
    ib: Any,
    contract: Any,
    *,
    generic_ticks: str = "",
    settle_seconds: float = 4.0,
    await_quotable: bool = False,
    poll_interval: float = 0.1,
) -> Any:
    """Subscribe, wait for the feed to land ticks, read, unsubscribe; return the ticker.

    With ``await_quotable`` the wait ends as soon as a two-sided quote is present
    (or ``settle_seconds`` elapses, whichever comes first) — strictly never slower
    than the blind sleep, usually faster.
    """
    ticker = ib.reqMktData(contract, generic_ticks, False, False)
    if await_quotable:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settle_seconds
        while loop.time() < deadline and not _is_quotable(ticker):
            await asyncio.sleep(poll_interval)
    else:
        await asyncio.sleep(settle_seconds)
    ib.cancelMktData(contract)
    return ticker


async def snapshot_many(
    ib: Any,
    contracts: Sequence[Any],
    *,
    generic_ticks: str = "",
    settle_seconds: float = 8.0,
    batch_size: int = 60,
    pacing_seconds: float = 1.0,
) -> list[Any]:
    """Snapshot many contracts in concurrent batches; returns tickers input-aligned.

    Subscribes up to ``batch_size`` contracts at once, waits ``settle_seconds``
    for the (typically delayed) feed to land, reads, then unsubscribes before the
    next batch. The returned list is positionally aligned with ``contracts``.
    """
    indexed = list(enumerate(contracts))
    tickers: list[Any] = [None] * len(indexed)
    for i in range(0, len(indexed), batch_size):
        batch = indexed[i : i + batch_size]
        live = [(idx, c, ib.reqMktData(c, generic_ticks, False, False)) for idx, c in batch]
        await asyncio.sleep(settle_seconds)
        for idx, c, ticker in live:
            tickers[idx] = ticker
            ib.cancelMktData(c)
        if i + batch_size < len(indexed):
            await asyncio.sleep(pacing_seconds)
    return tickers
