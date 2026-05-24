"""CsvOHLC + protocol-conformance tests (no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from qufin.data.vendors import CsvOHLC, OHLCSource, YFinanceOHLC
from qufin.data.vendors.alpaca import AlpacaOHLC, _parse_interval


def _write_csv(path: Path, n: int = 30) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    ts = [start + timedelta(days=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "Date": ts,
            "Open": [100.0 + i * 0.1 for i in range(n)],
            "High": [100.5 + i * 0.1 for i in range(n)],
            "Low": [99.5 + i * 0.1 for i in range(n)],
            "Close": [100.2 + i * 0.1 for i in range(n)],
            "Volume": [1_000_000.0 + i for i in range(n)],
        }
    )
    df.write_csv(path)


def test_csv_vendor_round_trips(tmp_path: Path) -> None:
    _write_csv(tmp_path / "AAPL.csv", n=30)
    vendor = CsvOHLC(root=tmp_path)
    bars = vendor.fetch(
        "AAPL",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        interval="1d",
    )
    assert bars.symbol == "AAPL"
    assert bars.n_bars == 30
    assert bars.data.schema["timestamp"].time_zone == "UTC"  # type: ignore[union-attr]


def test_csv_vendor_filters_by_range(tmp_path: Path) -> None:
    _write_csv(tmp_path / "AAPL.csv", n=30)
    vendor = CsvOHLC(root=tmp_path)
    bars = vendor.fetch(
        "AAPL",
        start=datetime(2024, 1, 10, tzinfo=UTC),
        end=datetime(2024, 1, 20, tzinfo=UTC),
        interval="1d",
    )
    assert bars.n_bars == 10


def test_csv_vendor_missing_file_raises(tmp_path: Path) -> None:
    vendor = CsvOHLC(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        vendor.fetch(
            "MISSING",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 2, 1, tzinfo=UTC),
            interval="1d",
        )


def test_csv_vendor_fetch_many(tmp_path: Path) -> None:
    _write_csv(tmp_path / "AAPL.csv")
    _write_csv(tmp_path / "MSFT.csv")
    vendor = CsvOHLC(root=tmp_path)
    out = vendor.fetch_many(
        ["AAPL", "MSFT"],
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        interval="1d",
    )
    assert set(out.keys()) == {"AAPL", "MSFT"}


def test_concrete_vendors_satisfy_protocol(tmp_path: Path) -> None:
    assert isinstance(CsvOHLC(root=tmp_path), OHLCSource)
    assert isinstance(YFinanceOHLC(), OHLCSource)
    assert isinstance(AlpacaOHLC(), OHLCSource)


def test_alpaca_interval_parser() -> None:
    assert _parse_interval("1d") == (1, "Day")
    assert _parse_interval("5m") == (5, "Min")
    assert _parse_interval("1h") == (1, "Hour")
    assert _parse_interval("15min") == (15, "Min")
    with pytest.raises(ValueError):
        _parse_interval("1week")
