"""Live smoke test against the Alpaca paper endpoint.

Skipped automatically when ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` are
absent so CI runs without credentials. Locally, the credentials are read
from the project-root ``.env`` (gitignored) by ``conftest.py``.

The test only *queries* the account — it never places an order. Order
placement / cancel round-trips live in ``test_alpaca_paper_order.py``
(also credential-gated and marked ``slow``).
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")),
    reason="Alpaca credentials not set",
)


def test_alpaca_paper_account_is_reachable():
    from qufin.trading.brokers import AlpacaBroker

    async def scenario():
        broker = AlpacaBroker(paper=True)
        await broker.connect()
        try:
            snap = await broker.account()
            positions = await broker.positions()
        finally:
            await broker.disconnect()
        # Paper accounts always carry equity ≥ 0; buying power ≥ cash for margin enabled.
        assert snap.equity >= 0.0
        assert snap.buying_power >= snap.cash
        # positions is a list (possibly empty) of Position objects.
        assert isinstance(positions, list)

    asyncio.run(scenario())
