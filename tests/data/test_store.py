"""End-to-end tests for the partitioned Parquet Store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from qufin.data.store import Store
from qufin.data.store.partition import partition_path

from .conftest import make_ohlcv


def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_get_returns_none_for_unknown_symbol(tmp_path: Path) -> None:
    store = Store.open(tmp_path)
    assert store.get("AAPL", "1d", _dt(2020), _dt(2024)) is None


def test_put_then_get_round_trips_a_single_year(tmp_path: Path) -> None:
    store = Store.open(tmp_path)
    bars = make_ohlcv(252, start=_dt(2023, 1, 3), symbol="AAPL")
    store.put(bars, "1d")

    got = store.get("AAPL", "1d", _dt(2023), _dt(2024))
    assert got is not None
    assert got.symbol == "AAPL"
    assert got.n_bars == 252


def test_put_creates_one_partition_per_year(tmp_path: Path) -> None:
    store = Store.open(tmp_path)
    bars = make_ohlcv(500, start=_dt(2022, 12, 15), symbol="AAPL")
    store.put(bars, "1d")

    assert partition_path(tmp_path, "AAPL", "1d", 2022).exists()
    assert partition_path(tmp_path, "AAPL", "1d", 2023).exists()
    assert partition_path(tmp_path, "AAPL", "1d", 2024).exists()


def test_get_filters_by_range_across_partitions(tmp_path: Path) -> None:
    store = Store.open(tmp_path)
    bars = make_ohlcv(800, start=_dt(2022, 1, 3), symbol="SPY")
    store.put(bars, "1d")

    sliced = store.get("SPY", "1d", _dt(2023), _dt(2024))
    assert sliced is not None
    ts = sliced.data["timestamp"]
    assert ts.min() >= _dt(2023)  # type: ignore[operator]
    assert ts.max() < _dt(2024)  # type: ignore[operator]


def test_upsert_replaces_overlapping_timestamps(tmp_path: Path) -> None:
    store = Store.open(tmp_path)
    first = make_ohlcv(10, start=_dt(2024, 1, 1), symbol="AAPL", seed=1)
    store.put(first, "1d")

    overlap_start = _dt(2024, 1, 5)
    second = make_ohlcv(10, start=overlap_start, symbol="AAPL", seed=2)
    store.put(second, "1d")

    got = store.get("AAPL", "1d", _dt(2024, 1, 1), _dt(2024, 2, 1))
    assert got is not None
    assert got.n_bars == 14  # 4 from first + 10 from second
    # The overlapped bars should carry the second frame's values.
    closes = got.data.sort("timestamp")["close"].to_list()
    second_first_close = second.data["close"].to_list()[0]
    overlap_row = got.data.filter(
        got.data["timestamp"] == overlap_start
    )["close"][0]
    assert overlap_row == second_first_close
    assert len(closes) == 14


def test_missing_ranges_reflect_manifest_coverage(tmp_path: Path) -> None:
    store = Store.open(tmp_path)
    bars = make_ohlcv(31, start=_dt(2023, 6, 1), symbol="AAPL")
    store.put(bars, "1d")

    gaps = store.missing_ranges("AAPL", "1d", _dt(2023, 1, 1), _dt(2024, 1, 1))
    assert len(gaps) == 2
    assert gaps[0].start == _dt(2023, 1, 1)
    assert gaps[0].end == _dt(2023, 6, 1)
    assert gaps[1].start.year == 2023 and gaps[1].start.month == 7
    assert gaps[1].end == _dt(2024, 1, 1)


def test_invalidate_removes_partitions_and_coverage(tmp_path: Path) -> None:
    store = Store.open(tmp_path)
    bars = make_ohlcv(100, start=_dt(2024, 1, 1), symbol="AAPL")
    store.put(bars, "1d")
    assert store.get("AAPL", "1d", _dt(2024), _dt(2025)) is not None

    store.invalidate("AAPL", "1d")
    assert store.get("AAPL", "1d", _dt(2024), _dt(2025)) is None
    assert store.manifest.coverage_for("AAPL", "1d").intervals == []


def test_manifest_persists_across_store_open(tmp_path: Path) -> None:
    store = Store.open(tmp_path)
    bars = make_ohlcv(60, start=_dt(2024, 1, 1), symbol="QQQ")
    store.put(bars, "1d")
    del store

    reopened = Store.open(tmp_path)
    cov = reopened.manifest.coverage_for("QQQ", "1d")
    assert len(cov.intervals) == 1
    got = reopened.get("QQQ", "1d", _dt(2024), _dt(2025))
    assert got is not None and got.n_bars == 60


def test_put_with_intraday_step_works(tmp_path: Path) -> None:
    store = Store.open(tmp_path)
    bars = make_ohlcv(
        390, start=_dt(2024, 6, 3), step=timedelta(minutes=1), symbol="AAPL"
    )
    store.put(bars, "1m")
    got = store.get("AAPL", "1m", _dt(2024, 6, 3), _dt(2024, 6, 4))
    assert got is not None and got.n_bars == 390
