"""Pure-logic tests for the IBKR broker adapter — no live gateway required.

Order-status classification, the rejection exception, the quote container, and
the order-translation totality are all exercised without connecting to TWS.
``_ib_order`` building is guarded with ``importorskip`` since it instantiates
``ib_async`` order objects.
"""

from __future__ import annotations

import pytest

from qufin.trading._types import Order, OrderRejectedError, OrderStatus, OrderType
from qufin.trading.brokers.quotes import Quote


def test_order_status_classification() -> None:
    working = OrderStatus("1", "Submitted", filled=0.0, remaining=1.0)
    filled = OrderStatus("2", "Filled", filled=1.0, remaining=0.0, avg_fill_price=1.48)
    rejected = OrderStatus("3", "Inactive", reject_reason="insufficient buying power")
    pending = OrderStatus("4", "PendingSubmit")

    assert working.is_working and not working.is_done
    assert filled.is_filled and filled.is_done
    assert rejected.is_rejected and rejected.is_done
    # Transient states must be neither working nor done so wait_for_status keeps polling.
    assert not pending.is_working and not pending.is_done


def test_order_status_rejected_by_reason() -> None:
    st = OrderStatus("5", "Cancelled", reject_reason="price out of band")
    assert st.is_rejected


def test_order_rejected_error_carries_status() -> None:
    st = OrderStatus("6", "Inactive", reject_reason="no shortable shares")
    err = OrderRejectedError(st)
    assert err.status is st
    assert "no shortable shares" in str(err)


def test_quote_mid_spread_price() -> None:
    q = Quote(bid=1.43, ask=1.53, last=1.47)
    assert q.mid == pytest.approx(1.48)
    assert q.spread_pct == pytest.approx((q.ask - q.bid) / q.mid * 100.0)
    assert q.price == pytest.approx(1.48)


def test_quote_falls_back_to_last_without_two_sided_market() -> None:
    q = Quote(bid=0.0, ask=0.0, last=2.0)
    assert q.mid is None
    assert q.spread_pct is None
    assert q.price == 2.0


@pytest.mark.parametrize(
    ("order_type", "kwargs", "action", "qty"),
    [
        (OrderType.MARKET, {}, "BUY", 1.0),
        (OrderType.LIMIT, {"limit_price": 1.48}, "BUY", 1.0),
        (OrderType.STOP, {"stop_price": 1.40}, "BUY", 1.0),
        (OrderType.STOP_LIMIT, {"limit_price": 1.48, "stop_price": 1.40}, "BUY", 1.0),
    ],
)
def test_ib_order_builds_for_each_type(
    order_type: OrderType, kwargs: dict[str, float], action: str, qty: float
) -> None:
    pytest.importorskip("ib_async")
    from qufin.trading.brokers.ibkr import _ib_order

    order = Order(asset="SPY", qty=1.0, order_type=order_type, **kwargs)
    ib_order = _ib_order(order)
    assert ib_order.action == action
    assert float(ib_order.totalQuantity) == qty


def test_ib_order_sell_sign_and_abs_qty() -> None:
    pytest.importorskip("ib_async")
    from qufin.trading.brokers.ibkr import _ib_order

    order = Order(asset="SPY", qty=-3.0, order_type=OrderType.MARKET)
    ib_order = _ib_order(order)
    assert ib_order.action == "SELL"
    assert float(ib_order.totalQuantity) == 3.0
