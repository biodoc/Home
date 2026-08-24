"""Data sources: the CSV path must be as trustworthy as the download path.

Files people export from brokers, vendors, and spreadsheets are messy in
predictable ways -- lowercase headers, "Adj Close", "Vol", a date column called
anything at all. The reader is forgiving about those and strict about the one
thing that matters, which is that a Close column exists.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qsb import data, sources
from qsb.config import Config


# -- registry ---------------------------------------------------------------
def test_builtin_sources_are_registered():
    assert "yfinance" in sources.available()
    assert "csv" in sources.available()


def test_unknown_source_lists_the_alternatives():
    with pytest.raises(KeyError, match="Available"):
        sources.build("bloomberg")


def test_csv_source_without_a_directory_says_what_to_pass():
    with pytest.raises(ValueError, match="--csv-dir"):
        sources.build("csv")


def test_sources_satisfy_the_protocol(tmp_path):
    (tmp_path / "X.csv").write_text("Date,Close\n2024-01-02,10\n")
    for src in (sources.build("yfinance"), sources.build("csv", csv_dir=tmp_path)):
        assert isinstance(src, sources.DataSource)
        assert isinstance(src.available(), tuple)


def test_csv_dir_can_come_from_config(tmp_path):
    (tmp_path / "X.csv").write_text("Date,Close\n2024-01-02,10\n")
    cfg = Config.load(overrides={"source": "csv", "csv_dir": str(tmp_path)})
    src = sources.build(cfg.source, cfg)
    assert src.available()[0]


# -- availability -----------------------------------------------------------
def test_missing_csv_directory_is_reported_not_raised(tmp_path):
    ok, why = sources.build("csv", csv_dir=tmp_path / "nope").available()
    assert not ok
    assert "does not exist" in why


def test_empty_csv_directory_is_reported(tmp_path):
    ok, why = sources.build("csv", csv_dir=tmp_path).available()
    assert not ok
    assert "no .csv" in why


def test_populated_csv_directory_is_available(tmp_path):
    (tmp_path / "AAA.csv").write_text("Date,Close\n2024-01-02,10\n")
    ok, why = sources.build("csv", csv_dir=tmp_path).available()
    assert ok
    assert "1 file" in why


# -- file discovery ---------------------------------------------------------
def _write(tmp_path, name, body="Date,Close\n2024-01-02,10\n2024-01-03,11\n"):
    (tmp_path / name).write_text(body)


def test_interval_specific_filename_wins(tmp_path):
    _write(tmp_path, "AAA.csv", "Date,Close\n2024-01-02,1\n")
    _write(tmp_path, "AAA_1d.csv", "Date,Close\n2024-01-02,99\n")
    src = sources.build("csv", csv_dir=tmp_path)
    assert src.fetch("AAA", "2024-01-01", None, "1d")["Close"].iloc[0] == 99


def test_bare_ticker_filename_is_accepted(tmp_path):
    _write(tmp_path, "AAA.csv")
    assert len(sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d")) == 2


def test_filename_matching_is_case_insensitive(tmp_path):
    _write(tmp_path, "aaa_1d.csv")
    assert not sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d").empty


def test_txt_extension_is_accepted(tmp_path):
    _write(tmp_path, "AAA.txt")
    assert not sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d").empty


def test_a_missing_ticker_lists_what_is_present(tmp_path):
    _write(tmp_path, "AAA.csv")
    with pytest.raises(FileNotFoundError, match="AAA"):
        sources.build("csv", csv_dir=tmp_path).fetch("ZZZ", "2024-01-01", None, "1d")


# -- header tolerance -------------------------------------------------------
@pytest.mark.parametrize("header", [
    "Date,Open,High,Low,Close,Volume",
    "date,open,high,low,close,volume",
    "DATE,OPEN,HIGH,LOW,CLOSE,VOLUME",
    "timestamp,Open,High,Low,Adj Close,Vol",
    "Datetime,open,high,low,adj_close,volume",
])
def test_common_header_spellings_are_understood(tmp_path, header):
    (tmp_path / "AAA.csv").write_text(f"{header}\n2024-01-02,1,2,0.5,1.5,100\n")
    out = sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d")
    assert "Close" in out.columns
    assert out["Close"].iloc[0] == 1.5


def test_a_real_close_is_preferred_over_adj_close(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "Date,Close,Adj Close\n2024-01-02,10.0,9.0\n"
    )
    out = sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d")
    assert out["Close"].iloc[0] == 10.0


def test_close_only_files_are_usable(tmp_path):
    """Open/High/Low are optional; the engine trades on Close."""
    _write(tmp_path, "AAA.csv", "Date,Close\n2024-01-02,10\n2024-01-03,11\n")
    out = sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d")
    assert list(out.columns) == ["Close"]


def test_an_unnamed_first_date_column_is_detected(tmp_path):
    (tmp_path / "AAA.csv").write_text(",Close\n2024-01-02,10\n2024-01-03,11\n")
    out = sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d")
    assert isinstance(out.index, pd.DatetimeIndex)


def test_a_file_without_a_date_column_names_what_it_found(tmp_path):
    (tmp_path / "AAA.csv").write_text("ticker,price\nAAA,10\n")
    with pytest.raises(ValueError, match="date column"):
        sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d")


def test_a_file_without_a_close_column_says_so(tmp_path):
    (tmp_path / "AAA.csv").write_text("Date,Open,High,Low\n2024-01-02,1,2,0.5\n")
    with pytest.raises(ValueError, match="Close"):
        sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d")


def test_an_empty_file_is_reported_clearly(tmp_path):
    (tmp_path / "AAA.csv").write_text("Date,Close\n")
    with pytest.raises(ValueError):
        sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d")


# -- date handling ----------------------------------------------------------
def test_rows_are_trimmed_to_the_requested_window(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "Date,Close\n2023-01-02,1\n2024-01-02,2\n2025-01-02,3\n"
    )
    out = sources.build("csv", csv_dir=tmp_path).fetch(
        "AAA", "2024-01-01", "2024-12-31", "1d"
    )
    assert len(out) == 1
    assert out["Close"].iloc[0] == 2


def test_an_unsorted_file_is_sorted(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "Date,Close\n2024-03-01,3\n2024-01-02,1\n2024-02-01,2\n"
    )
    out = sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2024-01-01", None, "1d")
    assert out.index.is_monotonic_increasing
    assert out["Close"].tolist() == [1, 2, 3]


def test_a_window_with_no_rows_reports_the_available_range(tmp_path):
    (tmp_path / "AAA.csv").write_text("Date,Close\n2024-01-02,1\n")
    with pytest.raises(ValueError, match="covers"):
        sources.build("csv", csv_dir=tmp_path).fetch("AAA", "2030-01-01", None, "1d")


# -- integration with the loader -------------------------------------------
def test_csv_source_flows_through_cleaning_and_caching(tmp_path):
    """Exported files get the same validation as downloads, and are cached."""
    csv_dir = tmp_path / "exports"
    csv_dir.mkdir()
    # Deliberately messy: duplicate row, a negative price, unsorted.
    (csv_dir / "AAA_1d.csv").write_text(
        "date,close,volume\n"
        "2024-01-04,12,1000\n"
        "2024-01-02,10,1000\n"
        "2024-01-02,10.5,1000\n"
        "2024-01-03,-1,1000\n"
    )
    cfg = Config.load(overrides={
        "source": "csv",
        "csv_dir": str(csv_dir),
        "cache_dir": str(tmp_path / "cache"),
        "min_avg_dollar_volume": 0.0,
    })

    df, report = data.load_prices("AAA", cfg)

    assert report.duplicate_timestamps == 1
    assert report.nonpositive_prices == 1
    # The CSV source sorts on read, so the ordering is already correct by the
    # time cleaning sees it; what matters is the outcome, not which layer did it.
    assert df.index.is_monotonic_increasing
    assert "csv" in report.source

    # The fetch is cached, so a later offline run needs no source at all.
    cached, cached_report = data.load_prices("AAA", cfg, offline=True)
    assert cached_report.source == "cache"
    assert len(cached) == len(df)


def test_offline_never_consults_the_source(tmp_path, monkeypatch):
    cfg = Config.load(overrides={"cache_dir": str(tmp_path)})

    def explode(*a, **k):
        raise AssertionError("offline load attempted a fetch")

    monkeypatch.setattr(sources.YFinanceSource, "fetch", explode)
    with pytest.raises(FileNotFoundError):
        data.load_prices("AAA", cfg, offline=True)


def test_unusable_source_is_reported_before_fetching(tmp_path):
    cfg = Config.load(overrides={"cache_dir": str(tmp_path)})
    empty = sources.CsvDirectorySource(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="unusable"):
        data.fetch_prices("AAA", "2024-01-01", None, "1d", source=empty)
