"""Adjustment tests: splits, cash dividends, total-return."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from qufin.data.adjustments import (
    ACTIONS_SCHEMA,
    CorporateAction,
    apply_splits,
    back_adjust,
    total_return_index,
    total_return_series,
)
from qufin.data.adjustments.actions import actions_frame

from .conftest import make_ohlcv


def _action_at(day: int, **kw: object) -> CorporateAction:
    return CorporateAction(
        timestamp=datetime(2024, 1, day, tzinfo=UTC),
        **kw,  # type: ignore[arg-type]
    )


def test_apply_splits_returns_input_when_no_actions() -> None:
    bars = make_ohlcv(10, symbol="AAPL")
    empty = pl.DataFrame(schema=ACTIONS_SCHEMA)
    out = apply_splits(bars, empty)
    assert out.data.equals(bars.data)


def test_apply_splits_two_for_one_halves_prior_prices() -> None:
    bars = make_ohlcv(10, symbol="AAPL", seed=0)
    pre_close = bars.close()[2]
    pre_vol = bars.volume()[2]
    actions = actions_frame(
        [_action_at(5, symbol="AAPL", kind="split", ratio=2.0)]
    )
    adj = apply_splits(bars, actions)
    np.testing.assert_allclose(adj.close()[2], pre_close / 2.0)
    np.testing.assert_allclose(adj.volume()[2], pre_vol * 2.0)
    # Bar on the ex-date itself is NOT adjusted
    np.testing.assert_allclose(adj.close()[4], bars.close()[4])
    # Bar after ex-date is unchanged
    np.testing.assert_allclose(adj.close()[6], bars.close()[6])


def test_apply_splits_compound_two_splits() -> None:
    bars = make_ohlcv(20, symbol="AAPL", seed=1)
    pre_close = bars.close()[0]
    actions = actions_frame(
        [
            _action_at(5, symbol="AAPL", kind="split", ratio=2.0),
            _action_at(10, symbol="AAPL", kind="split", ratio=3.0),
        ]
    )
    adj = apply_splits(bars, actions)
    np.testing.assert_allclose(adj.close()[0], pre_close / 6.0)
    # Between the two splits, only the second applies
    between = bars.close()[7]
    np.testing.assert_allclose(adj.close()[7], between / 3.0)


def test_apply_splits_reverse_split() -> None:
    bars = make_ohlcv(10, symbol="AAPL", seed=2)
    pre_close = bars.close()[2]
    actions = actions_frame(
        [_action_at(5, symbol="AAPL", kind="split", ratio=0.1)]
    )
    adj = apply_splits(bars, actions)
    np.testing.assert_allclose(adj.close()[2], pre_close * 10.0)


def test_apply_splits_ignores_other_symbols() -> None:
    bars = make_ohlcv(10, symbol="AAPL", seed=3)
    actions = actions_frame(
        [_action_at(5, symbol="MSFT", kind="split", ratio=2.0)]
    )
    adj = apply_splits(bars, actions)
    assert adj.data.equals(bars.data)


def test_total_return_series_no_dividends_matches_plain_log_returns() -> None:
    bars = make_ohlcv(20, symbol="AAPL", seed=4)
    empty = pl.DataFrame(schema=ACTIONS_SCHEMA)
    s = total_return_series(bars, empty).to_numpy()
    expected = np.zeros_like(s)
    closes = bars.close()
    expected[1:] = np.log(closes[1:] / closes[:-1])
    np.testing.assert_allclose(s, expected)


def test_total_return_series_adds_dividend_to_ex_date_return() -> None:
    bars = make_ohlcv(10, symbol="AAPL", seed=5)
    closes = bars.close()
    actions = actions_frame(
        [_action_at(5, symbol="AAPL", kind="cash_div", cash=0.5)]
    )
    s = total_return_series(bars, actions).to_numpy()
    # Bar index 4 corresponds to day 5 (start = Jan 1 + 4 days = Jan 5).
    expected_ret_4 = np.log((closes[4] + 0.5) / closes[3])
    np.testing.assert_allclose(s[4], expected_ret_4)
    # Other bars unchanged
    for i in (1, 2, 3, 6, 7):
        np.testing.assert_allclose(s[i], np.log(closes[i] / closes[i - 1]))


def test_total_return_index_is_cumulative_exp_of_log_returns() -> None:
    bars = make_ohlcv(15, symbol="AAPL", seed=6)
    actions = actions_frame(
        [
            _action_at(5, symbol="AAPL", kind="cash_div", cash=0.25),
            _action_at(10, symbol="AAPL", kind="cash_div", cash=0.30),
        ]
    )
    idx = total_return_index(bars, actions, base=100.0).to_numpy()
    log_ret = total_return_series(bars, actions).to_numpy()
    np.testing.assert_allclose(idx, 100.0 * np.exp(np.cumsum(log_ret)))


def test_back_adjust_combines_splits_and_dividends() -> None:
    bars = make_ohlcv(20, symbol="AAPL", seed=7)
    pre_close = bars.close()[0]
    actions = actions_frame(
        [
            _action_at(5, symbol="AAPL", kind="cash_div", cash=0.40),
            _action_at(10, symbol="AAPL", kind="split", ratio=2.0),
        ]
    )
    adj = back_adjust(bars, actions)
    # Earliest bar has both adjustments applied; final ratio is bounded above
    # by the split factor alone (so adj < pre/2.0).
    assert adj.close()[0] < pre_close / 2.0
    # After the last action, prices are unchanged
    np.testing.assert_allclose(adj.close()[15], bars.close()[15])


def test_apply_splits_preserves_schema_and_symbol() -> None:
    bars = make_ohlcv(5, symbol="NVDA")
    actions = actions_frame(
        [_action_at(3, symbol="NVDA", kind="split", ratio=10.0)]
    )
    adj = apply_splits(bars, actions)
    assert adj.symbol == "NVDA"
    assert adj.data.schema == bars.data.schema


def test_corporate_action_validates_ratio_and_cash() -> None:
    import pytest

    with pytest.raises(ValueError):
        CorporateAction(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="A",
            kind="split",
            ratio=0.0,
        )
    with pytest.raises(ValueError):
        CorporateAction(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="A",
            kind="cash_div",
            cash=-1.0,
        )
