"""PaperBroker smoke tests — async surface works against the same execution model."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from qufin.trading import BarEvent, Order, OrderType, PaperBroker


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_paper_broker_market_order_fills_at_next_bar(event_loop):
    broker = PaperBroker(starting_cash=10_000.0)

    async def scenario():
        await broker.connect()
        order = Order(asset="XYZ", qty=10.0, order_type=OrderType.MARKET, client_id="cid-1")
        oid = await broker.place_order(order)
        assert oid == "cid-1"
        bar = BarEvent(
            symbol="XYZ",
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
            open=100.0, high=101.0, low=99.0, close=100.5,
            volume=1_000.0, index=0,
        )
        await broker.ingest_bar(bar)
        positions = await broker.positions()
        assert len(positions) == 1
        assert positions[0].qty == 10.0
        assert positions[0].avg_price == 100.0
        await broker.disconnect()

    event_loop.run_until_complete(scenario())


def test_paper_broker_not_connected_raises(event_loop):
    broker = PaperBroker()

    async def scenario():
        with pytest.raises(RuntimeError, match="not connected"):
            await broker.account()

    event_loop.run_until_complete(scenario())
