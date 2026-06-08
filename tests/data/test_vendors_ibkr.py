"""Pure-logic tests for the shared IBKR plumbing and the historical-bars helpers.

None of these touch a live gateway — they exercise error classification, NaN
coercion, the bounded error listener, and the request-string builders directly.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from qufin.data.vendors._ib_common import (
    IBKRError,
    IBKRErrorCategory,
    IBKRErrorListener,
    classify_error,
    safe_float,
    ymd_to_date,
)
from qufin.data.vendors.ibkr import _duration_string, _interval_to_bar_size


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (10091, IBKRErrorCategory.DATA),
        (10167, IBKRErrorCategory.DATA),
        (2109, IBKRErrorCategory.ORDER_WARNING),
        (2104, IBKRErrorCategory.INFO),
        (200, IBKRErrorCategory.CONTRACT),
        (201, IBKRErrorCategory.ORDER_REJECT),
        (326, IBKRErrorCategory.CONNECTION),
        (9999, IBKRErrorCategory.UNKNOWN),
    ],
)
def test_classify_error(code: int, expected: IBKRErrorCategory) -> None:
    assert classify_error(code) is expected


def test_ibkr_error_is_benign() -> None:
    info = IBKRError(2104, "data farm ok", IBKRErrorCategory.INFO)
    warn = IBKRError(2109, "outside rth ignored", IBKRErrorCategory.ORDER_WARNING)
    data = IBKRError(10091, "delayed", IBKRErrorCategory.DATA)
    assert info.is_benign
    assert warn.is_benign
    assert not data.is_benign


def test_listener_records_and_filters() -> None:
    listener = IBKRErrorListener()
    listener.handle(1, 2104, "Market data farm connection is OK")
    listener.handle(6, 10091, "requires subscription; displaying delayed")
    listener.handle(20, 2109, "Outside RTH attribute ignored")

    assert len(listener.errors()) == 3
    assert len(listener.problems()) == 1  # only the DATA record is non-benign
    assert listener.has_subscription_issue
    subs = listener.subscription_warnings()
    assert len(subs) == 1
    assert subs[0].code == 10091
    last = listener.last(IBKRErrorCategory.DATA)
    assert last is not None
    assert last.code == 10091


def test_listener_is_bounded() -> None:
    listener = IBKRErrorListener(max_records=5)
    for i in range(20):
        listener.handle(i, 2104, "ok")
    assert len(listener.errors()) == 5


def test_listener_extracts_contract_label() -> None:
    contract = SimpleNamespace(localSymbol="ACHR  270115C00006000", symbol="ACHR")
    listener = IBKRErrorListener()
    listener.handle(20, 201, "order rejected", contract)
    rec = listener.last()
    assert rec is not None
    assert rec.contract_repr == "ACHR  270115C00006000"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (float("nan"), None),
        (math.inf, None),
        (-math.inf, None),
        ("1.5", 1.5),
        (3, 3.0),
        (2.5, 2.5),
    ],
)
def test_safe_float(value: object, expected: float | None) -> None:
    assert safe_float(value) == expected


def test_ymd_to_date() -> None:
    d = ymd_to_date("20270115")
    assert (d.year, d.month, d.day) == (2027, 1, 15)


def test_duration_string_buckets() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    assert _duration_string(base, base.replace(hour=12)).endswith(" S")
    assert _duration_string(base, datetime(2026, 1, 20, tzinfo=UTC)).endswith(" D")
    assert _duration_string(base, datetime(2026, 3, 1, tzinfo=UTC)).endswith(" W")
    assert _duration_string(base, datetime(2027, 6, 1, tzinfo=UTC)).endswith(" Y")
    with pytest.raises(ValueError, match="after start"):
        _duration_string(base, base)


def test_interval_to_bar_size() -> None:
    assert _interval_to_bar_size("1d") == "1 day"
    assert _interval_to_bar_size("5m") == "5 mins"
    with pytest.raises(ValueError, match="unsupported"):
        _interval_to_bar_size("3y")
