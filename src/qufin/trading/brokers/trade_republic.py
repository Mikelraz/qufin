"""Trade Republic broker adapter.

Wraps the ``trade-republic-api`` (``trapi``) client behind the :class:`Broker`
protocol so qufin strategies can read balance / positions and place limit and
stop orders on a real Trade Republic account. The qufin ``asset`` symbol is
treated as a Trade Republic **ISIN**.

``trapi`` is imported lazily so the trading subpackage stays importable without
the optional ``trading-live`` dependency.

Authentication / credentials
----------------------------
Reads ``TR_PHONE_NUMBER`` / ``TR_PIN`` from the environment unless passed
explicitly; the device key file is ``TR_KEY_FILE`` (or trapi's ``./key``
default). Device pairing is a **one-time interactive step** (it prompts for an
SMS token) — run it separately. ``connect`` only logs in and raises if no key is
present; it never prompts.

Known limitations
-----------------
* Trade Republic exposes **ticks**, not OHLCV bars: ``stream_bars`` polls
  ``ticker`` and emits tick-derived bars (``open == high == low == close``).
* Trade Republic has **no native fills push channel**: ``stream_fills`` polls
  terminated ``orders`` and is best-effort — the orders payload shape is not yet
  verified against the live API.
* Options are not supported.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from ...options._types import OptionContract
from .._types import (
    AccountSnapshot,
    AssetKind,
    BarEvent,
    Fill,
    Order,
    OrderId,
    OrderType,
    Position,
    TimeInForce,
    new_order_id,
)


def _pairing_disabled(_prompt: str) -> str:
    """Token provider that refuses interactive pairing inside the broker."""
    raise RuntimeError(
        "Trade Republic device is not paired; run device pairing separately "
        "before connecting (it prompts for an SMS token)."
    )


@dataclass(slots=True)
class TradeRepublicBroker:
    """Trade Republic live broker.

    Parameters
    ----------
    phone_number, pin  Default to ``TR_PHONE_NUMBER`` / ``TR_PIN``.
    key_path           Device key PEM path; defaults to ``TR_KEY_FILE`` then ``./key``.
    locale             Server response language (``"de"`` / ``"en"``).
    exchange           Trade Republic exchange id for orders/quotes (e.g. ``"LSX"``).
    currency           Account currency used to pick the cash balance.
    timeout            Seconds to wait for each one-shot WebSocket response.
    poll_interval      Seconds between polls for ``stream_bars`` / ``stream_fills``.
    """

    phone_number: str | None = None
    pin: str | None = None
    key_path: str | None = None
    locale: str = "de"
    exchange: str = "LSX"
    currency: str = "EUR"
    timeout: float = 20.0
    poll_interval: float = 2.0
    _client: Any = field(default=None, init=False, repr=False)
    _trapi: Any = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.phone_number = self.phone_number or os.environ.get("TR_PHONE_NUMBER")
        self.pin = self.pin or os.environ.get("TR_PIN")
        self.key_path = self.key_path or os.environ.get("TR_KEY_FILE")
        if not self.phone_number or not self.pin:
            raise ValueError(
                "TR_PHONE_NUMBER / TR_PIN are required. Pass them explicitly or set the env vars."
            )

    # ------------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        trapi = self._import_trapi()
        from trapi.config import resolve_key_path

        key = resolve_key_path(self.key_path)
        if not key.is_file():
            raise RuntimeError(
                f"no Trade Republic device key at {key}; pair the device first "
                "(pairing prompts for an SMS token)."
            )
        self._trapi = trapi
        self._client = trapi.TradeRepublic(
            self.phone_number,
            self.pin,
            self.locale,
            key_path=self.key_path,
            token_provider=_pairing_disabled,
        )
        # login() is synchronous (blocking HTTP); keep the event loop free.
        await asyncio.to_thread(self._client.login)
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------ reads

    async def account(self) -> AccountSnapshot:
        self._require_connected()
        cash = self._trapi.Cash.from_response(
            await self._read_one(self._client.cash()), self.currency
        )
        portfolio = self._trapi.Portfolio.from_response(
            await self._read_one(self._client.compact_portfolio())
        )
        net = portfolio.net_value or 0.0
        return AccountSnapshot(
            timestamp=datetime.now(tz=UTC),
            cash=cash.amount,
            equity=cash.amount + net,
            buying_power=cash.amount,
        )

    async def positions(self) -> list[Position]:
        self._require_connected()
        portfolio = self._trapi.Portfolio.from_response(
            await self._read_one(self._client.compact_portfolio())
        )
        out: list[Position] = []
        for p in portfolio.positions:
            mark = (p.net_value / p.quantity) if (p.net_value and p.quantity) else p.average_buy_in
            out.append(
                Position(asset=p.isin, qty=p.quantity, avg_price=p.average_buy_in, last_mark=mark)
            )
        return out

    # ------------------------------------------------------------------ orders

    async def place_order(self, order: Order) -> OrderId:
        self._require_connected()
        request = self._to_trapi_order(order)
        client_id = order.client_id or new_order_id("tr")
        result = self._trapi.OrderResult.from_response(
            await self._read_one(self._client.order(request, client_id=client_id))
        )
        return result.order_id or client_id

    async def cancel_order(self, order_id: OrderId) -> None:
        self._require_connected()
        await self._read_one(self._client.cancel_order(order_id))

    # ------------------------------------------------------------------ streams

    def stream_bars(self, symbols: Sequence[str]) -> AsyncIterator[BarEvent]:
        return self._bar_stream(symbols)

    def stream_fills(self) -> AsyncIterator[Fill]:
        return self._fill_stream()

    async def _bar_stream(self, symbols: Sequence[str]) -> AsyncIterator[BarEvent]:
        while self._connected:
            for symbol in symbols:
                quote = self._trapi.Quote.from_response(
                    await self._read_one(self._client.ticker(symbol, self.exchange))
                )
                price = quote.last if quote.last is not None else quote.mid
                if price is not None:
                    now = datetime.now(tz=UTC)
                    yield BarEvent(
                        symbol=symbol,
                        timestamp=now,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=0.0,
                        index=-1,
                    )
            await asyncio.sleep(self.poll_interval)

    async def _fill_stream(self) -> AsyncIterator[Fill]:
        seen: set[str] = set()
        while self._connected:
            raw = await self._read_one(self._client.orders(terminated=True))
            for fill in self._extract_fills(raw, seen):
                yield fill
            await asyncio.sleep(self.poll_interval)

    # ------------------------------------------------------------------ helpers

    def _to_trapi_order(self, order: Order) -> Any:
        if order.asset_kind is AssetKind.OPTION or isinstance(order.asset, OptionContract):
            raise ValueError("Trade Republic does not support option orders")
        trapi = self._trapi
        mode = {
            OrderType.MARKET: trapi.OrderMode.MARKET,
            OrderType.LIMIT: trapi.OrderMode.LIMIT,
            OrderType.STOP: trapi.OrderMode.STOP_MARKET,
            OrderType.STOP_LIMIT: trapi.OrderMode.STOP_LIMIT,
        }[order.order_type]
        side = trapi.Side.BUY if order.qty > 0 else trapi.Side.SELL
        return trapi.OrderRequest(
            isin=str(order.asset),
            side=side,
            size=abs(order.qty),
            mode=mode,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            expiry=self._to_expiry(order.tif),
            exchange=self.exchange,
        )

    def _to_expiry(self, tif: TimeInForce) -> Any:
        match tif:
            case TimeInForce.DAY:
                return self._trapi.Expiry.GFD
            case TimeInForce.GTC:
                return self._trapi.Expiry.GTC
            case _:
                raise ValueError(f"Trade Republic has no equivalent for {tif}")

    def _extract_fills(self, payload: Any, seen: set[str]) -> list[Fill]:
        """Best-effort parse of executed orders into fills (payload shape unverified)."""
        if isinstance(payload, list):
            entries: list[Any] = cast("list[Any]", payload)
        else:
            entries = cast("list[Any]", payload.get("orders", []))
        fills: list[Fill] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            entry = cast("dict[str, Any]", item)
            order_id = entry.get("orderId") or entry.get("id")
            executed = entry.get("executedSize") or entry.get("filledSize")
            price = entry.get("executedPrice") or entry.get("price")
            isin = entry.get("instrumentId")
            if not (order_id and executed and price and isin) or order_id in seen:
                continue
            seen.add(str(order_id))
            sign = -1.0 if entry.get("type") == "sell" else 1.0
            fills.append(
                Fill(
                    order_id=str(order_id),
                    asset=str(isin),
                    timestamp=datetime.now(tz=UTC),
                    qty=sign * float(executed),
                    price=float(price),
                )
            )
        return fills

    async def _read_one(self, subscribe: Any) -> Any:
        """Drive one subscription to a single response (subscribe, then receive one)."""
        await subscribe
        return await asyncio.wait_for(self._client.start(receive_one=True), timeout=self.timeout)

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("TradeRepublicBroker is not connected; await broker.connect() first")

    @staticmethod
    def _import_trapi() -> Any:
        try:
            import trapi
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "TradeRepublicBroker requires the 'trade-republic-api' package; "
                "install qufin's 'trading-live' dependency group."
            ) from exc
        return trapi
