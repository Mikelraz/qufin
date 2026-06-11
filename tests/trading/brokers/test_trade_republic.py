"""Tests for the Trade Republic broker adapter.

The unit tests drive the broker with a fake trapi client (no network) and assert
the qufin ``Order`` → trapi payload mapping and the account/positions parsing.
They are skipped when the optional ``trade-republic-api`` package is not
installed. A separate credential-gated test reaches the live account read-only.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from qufin.trading import Order, OrderType
from qufin.trading.brokers import TradeRepublicBroker

# The adapter and its models need the optional dependency.
trapi = pytest.importorskip("trapi")


class FakeClient:
    """Mimics trapi's subscribe-then-receive-one flow without a WebSocket."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self._pending: Any = None
        self.orders_sent: list[tuple[Any, str | None]] = []
        self.cancelled: str | None = None

    async def cash(self) -> None:
        self._pending = self._responses["cash"]

    async def account_pairs(self) -> None:
        self._pending = self._responses.get(
            "account_pairs", {"accounts": [{"securitiesAccountNumber": "SEC1"}]}
        )

    async def compact_portfolio_by_type(self, sec_acc_no: str | None = None) -> None:
        self._pending = self._responses["portfolio"]

    async def ticker(self, isin: str, exchange: str) -> None:
        self._pending = self._responses["ticker"]

    async def orders(self, terminated: bool = False) -> None:
        self._pending = self._responses.get("orders", [])

    async def order(self, request: Any, client_id: str | None = None) -> str:
        self.orders_sent.append((request, client_id))
        self._pending = self._responses.get("order", {})
        return client_id or "generated"

    async def cancel_order(self, order_id: str) -> None:
        self.cancelled = order_id
        self._pending = {}

    async def start(self, receive_one: bool = False) -> Any:
        return self._pending


def _broker(responses: dict[str, Any]) -> tuple[TradeRepublicBroker, FakeClient]:
    broker = TradeRepublicBroker(phone_number="+490000000000", pin="0000")
    client = FakeClient(responses)
    broker._trapi = trapi
    broker._client = client
    broker._connected = True
    return broker, client


def test_account_equity_uses_live_marks() -> None:
    broker, _ = _broker(
        {
            "cash": [{"currencyId": "EUR", "amount": "1000.0"}],
            "portfolio": {
                "positions": [
                    {"instrumentId": "US0378331005", "netSize": "4", "averageBuyIn": "100.0"},
                ]
            },
            "ticker": {
                "last": {"price": "110.0"},
                "bid": {"price": "109.0"},
                "ask": {"price": "111.0"},
            },
        }
    )
    snap = asyncio.run(broker.account())
    assert snap.cash == 1000.0
    assert snap.equity == 1000.0 + 4 * 110.0  # cash + mark * qty
    assert snap.buying_power == 1000.0


def test_positions_use_live_ticker_mark() -> None:
    broker, _ = _broker(
        {
            "portfolio": {
                "positions": [
                    {"instrumentId": "US0378331005", "netSize": "4", "averageBuyIn": "100.0"},
                ]
            },
            "ticker": {"last": {"price": "110.0"}},
        }
    )
    positions = asyncio.run(broker.positions())
    assert len(positions) == 1
    assert positions[0].asset == "US0378331005"
    assert positions[0].qty == 4.0
    assert positions[0].avg_price == 100.0
    assert positions[0].last_mark == 110.0  # from the live ticker


def test_position_mark_falls_back_to_avg_buy_in_without_quote() -> None:
    broker, _ = _broker(
        {
            "portfolio": {
                "positions": [
                    {"instrumentId": "X", "netSize": "2", "averageBuyIn": "50.0"},
                ]
            },
            "ticker": {},  # no bid/ask/last -> no mark -> fall back to avg buy-in
        }
    )
    positions = asyncio.run(broker.positions())
    assert positions[0].last_mark == 50.0


def test_place_limit_order_maps_to_trapi_request() -> None:
    broker, client = _broker({"order": {"orderId": "tr-7"}})
    order = Order(
        asset="US0378331005",
        qty=2.0,
        order_type=OrderType.LIMIT,
        limit_price=150.0,
        client_id="cid-1",
    )
    order_id = asyncio.run(broker.place_order(order))
    assert order_id == "tr-7"
    request, client_id = client.orders_sent[0]
    assert client_id == "cid-1"
    assert request.isin == "US0378331005"
    assert request.side is trapi.Side.BUY
    assert request.mode is trapi.OrderMode.LIMIT
    assert request.size == 2.0
    assert request.limit_price == 150.0


def test_sell_stop_order_maps_to_stop_market() -> None:
    broker, client = _broker({"order": {}})
    order = Order(asset="X", qty=-3.0, order_type=OrderType.STOP, stop_price=90.0)
    asyncio.run(broker.place_order(order))
    request, _ = client.orders_sent[0]
    assert request.side is trapi.Side.SELL
    assert request.mode is trapi.OrderMode.STOP_MARKET
    assert request.stop_price == 90.0
    assert request.size == 3.0


def test_cancel_order_forwards_id() -> None:
    broker, client = _broker({})
    asyncio.run(broker.cancel_order("ord-42"))
    assert client.cancelled == "ord-42"


def test_not_connected_raises() -> None:
    broker = TradeRepublicBroker(phone_number="+490000000000", pin="0000")

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="not connected"):
            await broker.account()

    asyncio.run(scenario())


def test_missing_credentials_raise() -> None:
    saved = {k: os.environ.pop(k, None) for k in ("TR_PHONE_NUMBER", "TR_PIN")}
    try:
        with pytest.raises(ValueError, match="TR_PHONE_NUMBER"):
            TradeRepublicBroker()
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


# -- credential-gated live read (never places an order) ---------------------


@pytest.mark.skipif(
    not (os.environ.get("TR_PHONE_NUMBER") and os.environ.get("TR_PIN")),
    reason="Trade Republic credentials not set",
)
def test_trade_republic_account_is_reachable_live() -> None:
    async def scenario() -> None:
        broker = TradeRepublicBroker()
        await broker.connect()
        try:
            snap = await broker.account()
            positions = await broker.positions()
        finally:
            await broker.disconnect()
        assert snap.equity >= 0.0
        assert isinstance(positions, list)

    asyncio.run(scenario())
