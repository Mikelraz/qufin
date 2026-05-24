"""Point-in-time universe tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from qufin.data.universe import (
    MEMBERSHIP_SCHEMA,
    UniverseSnapshot,
    asof_universe,
    detect_potential_survivorship_bias,
    load_membership_csv,
    membership_from_records,
    pit_join,
)

NY = ZoneInfo("America/New_York")


def _record(
    sym: str, idx: str, start: datetime, end: datetime
) -> dict[str, object]:
    return {"symbol": sym, "index": idx, "start": start, "end": end}


def _sample_membership() -> pl.DataFrame:
    return membership_from_records(
        [
            _record(
                "AAPL", "SP500", datetime(2000, 1, 1, tzinfo=UTC),
                datetime(2200, 1, 1, tzinfo=UTC),
            ),
            _record(
                "MSFT", "SP500", datetime(2000, 1, 1, tzinfo=UTC),
                datetime(2200, 1, 1, tzinfo=UTC),
            ),
            _record(
                "ENRON", "SP500", datetime(1996, 1, 1, tzinfo=UTC),
                datetime(2001, 11, 28, tzinfo=UTC),
            ),
            _record(
                "TSLA", "NDX", datetime(2013, 5, 22, tzinfo=UTC),
                datetime(2200, 1, 1, tzinfo=UTC),
            ),
        ]
    )


def test_membership_from_records_validates_schema() -> None:
    with pytest.raises(ValueError):
        membership_from_records([{"symbol": "X", "index": "I"}])


def test_membership_empty_returns_typed_empty_frame() -> None:
    df = membership_from_records([])
    assert df.is_empty()
    assert df.schema == MEMBERSHIP_SCHEMA


def test_asof_universe_returns_active_members() -> None:
    m = _sample_membership()
    snap = asof_universe(m, index="SP500", asof=datetime(2000, 6, 1, tzinfo=UTC))
    assert isinstance(snap, UniverseSnapshot)
    assert set(snap.symbols) == {"AAPL", "MSFT", "ENRON"}


def test_asof_universe_excludes_pre_start_and_post_end() -> None:
    m = _sample_membership()
    snap_before = asof_universe(m, index="SP500", asof=datetime(1995, 1, 1, tzinfo=UTC))
    assert "ENRON" not in snap_before.symbols
    snap_after = asof_universe(m, index="SP500", asof=datetime(2002, 1, 1, tzinfo=UTC))
    assert "ENRON" not in snap_after.symbols
    assert "AAPL" in snap_after.symbols


def test_asof_universe_filters_by_index() -> None:
    m = _sample_membership()
    snap = asof_universe(m, index="NDX", asof=datetime(2020, 1, 1, tzinfo=UTC))
    assert snap.symbols == ("TSLA",)


def test_asof_universe_requires_tz_aware() -> None:
    m = _sample_membership()
    with pytest.raises(ValueError):
        asof_universe(m, index="SP500", asof=datetime(2020, 1, 1))


def test_asof_universe_in_local_tz() -> None:
    """A non-UTC tz-aware asof should still resolve correctly."""
    m = _sample_membership()
    asof_local = datetime(2000, 6, 1, 9, 30, tzinfo=NY)
    snap = asof_universe(m, index="SP500", asof=asof_local)
    assert "AAPL" in snap.symbols


def test_pit_join_filters_features_by_membership_window() -> None:
    m = _sample_membership()
    features = pl.DataFrame(
        {
            "symbol": ["AAPL", "ENRON", "ENRON", "TSLA"],
            "timestamp": [
                datetime(2010, 1, 1, tzinfo=UTC),
                datetime(2000, 6, 1, tzinfo=UTC),  # in
                datetime(2002, 6, 1, tzinfo=UTC),  # out (post-bankruptcy)
                datetime(2012, 1, 1, tzinfo=UTC),  # before TSLA join
            ],
            "value": [1.0, 2.0, 3.0, 4.0],
        },
        schema={
            "symbol": pl.Utf8,
            "timestamp": pl.Datetime("ns", time_zone="UTC"),
            "value": pl.Float64,
        },
    )
    sp = pit_join(features, m, index="SP500")
    assert sp["value"].to_list() == [1.0, 2.0]
    ndx = pit_join(features, m, index="NDX")
    assert ndx.is_empty()


def test_pit_join_requires_symbol_and_timestamp_columns() -> None:
    m = _sample_membership()
    bad = pl.DataFrame({"x": [1]})
    with pytest.raises(ValueError):
        pit_join(bad, m, index="SP500")


def test_pit_join_returns_empty_when_index_unknown() -> None:
    m = _sample_membership()
    features = pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)],
        },
        schema={
            "symbol": pl.Utf8,
            "timestamp": pl.Datetime("ns", time_zone="UTC"),
        },
    )
    assert pit_join(features, m, index="UNKNOWN").is_empty()


def test_detect_survivorship_bias_flags_missing_members() -> None:
    m = _sample_membership()
    features = pl.DataFrame({"symbol": ["AAPL", "MSFT"]})
    report = detect_potential_survivorship_bias(features, m, index="SP500")
    assert report.ever_members == 3
    assert "ENRON" in report.ever_but_missing
    assert 0.0 < report.bias_ratio < 1.0


def test_detect_survivorship_bias_clean_features_have_zero_ratio() -> None:
    m = _sample_membership()
    features = pl.DataFrame({"symbol": ["AAPL", "MSFT", "ENRON"]})
    report = detect_potential_survivorship_bias(features, m, index="SP500")
    assert report.bias_ratio == 0.0
    assert report.ever_but_missing == ()


def test_load_membership_csv_round_trips(tmp_path: Path) -> None:
    csv_path = tmp_path / "sp500.csv"
    csv_path.write_text(
        "symbol,index,start,end\n"
        "AAPL,SP500,2000-01-01T00:00:00+00:00,\n"
        "ENRON,SP500,1996-01-01T00:00:00+00:00,2001-11-28T00:00:00+00:00\n"
    )
    df = load_membership_csv(csv_path)
    assert df.schema == MEMBERSHIP_SCHEMA
    assert df.height == 2
    snap = asof_universe(df, index="SP500", asof=datetime(2000, 6, 1, tzinfo=UTC))
    assert set(snap.symbols) == {"AAPL", "ENRON"}


def test_load_membership_csv_missing_columns_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("symbol,index\nAAPL,SP500\n")
    with pytest.raises(ValueError):
        load_membership_csv(csv_path)
