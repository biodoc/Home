"""Data validation and cleaning: every repair must be counted, not silent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_frame
from qsb import data


def test_clean_frame_needs_no_repairs():
    df = make_frame(np.linspace(10, 20, 50))
    out, report = data.validate_and_clean(df, "CLEAN")
    assert report.clean
    assert report.rows_in == report.rows_out == 50
    assert list(out.columns) == data.OHLCV


def test_duplicate_timestamps_are_dropped_and_counted():
    idx = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
    df = make_frame([10.0, 11.0, 12.0], index=idx)
    out, report = data.validate_and_clean(df, "DUP")
    assert report.duplicate_timestamps == 1
    assert len(out) == 2
    # keep="last" -- the later row wins
    assert out["Close"].iloc[0] == 11.0


def test_nonpositive_prices_are_dropped_and_counted():
    df = make_frame([10.0, 0.0, -5.0, 12.0])
    out, report = data.validate_and_clean(df, "BAD")
    assert report.nonpositive_prices == 2
    assert len(out) == 2
    assert (out["Close"] > 0).all()


def test_out_of_order_index_is_sorted_and_flagged():
    idx = pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"])
    df = make_frame([12.0, 10.0, 11.0], index=idx)
    out, report = data.validate_and_clean(df, "UNSORTED")
    assert report.reordered
    assert out.index.is_monotonic_increasing
    assert out["Close"].tolist() == [10.0, 11.0, 12.0]


def test_nan_volume_is_filled_and_counted():
    df = make_frame([10.0, 11.0, 12.0])
    df.loc[df.index[1], "Volume"] = np.nan
    out, report = data.validate_and_clean(df, "VOL")
    assert report.nan_volume_filled == 1
    assert out["Volume"].iloc[1] == 0.0


def test_missing_close_rows_are_dropped():
    df = make_frame([10.0, 11.0, 12.0])
    df.loc[df.index[1], "Close"] = np.nan
    out, report = data.validate_and_clean(df, "GAP")
    assert report.missing_close == 1
    assert len(out) == 2


def test_calendar_gaps_are_reported_not_filled():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"])
    df = make_frame([10.0, 11.0, 12.0], index=idx)
    out, report = data.validate_and_clean(df, "HOLE")
    # The hole is reported...
    assert report.calendar_gaps > 15
    # ...but never papered over with synthetic bars.
    assert len(out) == 3


def test_suspect_split_move_is_flagged_but_kept():
    df = make_frame([10.0, 10.1, 100.0, 101.0])
    out, report = data.validate_and_clean(df, "SPLIT")
    assert report.suspect_moves == 1
    assert len(out) == 4, "suspect moves are flagged, not deleted"


def test_multiindex_columns_are_flattened():
    df = make_frame([10.0, 11.0, 12.0])
    df.columns = pd.MultiIndex.from_product([df.columns, ["XYZ"]])
    out, report = data.validate_and_clean(df, "MI")
    assert list(out.columns) == data.OHLCV
    assert any("MultiIndex" in n for n in report.notes)


def test_all_rows_invalid_raises_rather_than_returning_empty():
    df = make_frame([0.0, -1.0, -2.0])
    with pytest.raises(ValueError, match="no usable rows"):
        data.validate_and_clean(df, "DEAD")


def test_missing_close_column_raises():
    df = make_frame([10.0, 11.0]).drop(columns=["Close"])
    with pytest.raises(ValueError, match="no Close column"):
        data.validate_and_clean(df, "NOCLOSE")


def test_timezone_aware_index_is_normalized():
    idx = pd.date_range("2024-01-02", periods=3, tz="America/New_York")
    df = make_frame([10.0, 11.0, 12.0], index=idx)
    out, _ = data.validate_and_clean(df, "TZ")
    assert out.index.tz is None


# -- liquidity --------------------------------------------------------------
def test_thin_days_are_marked_untradable_not_dropped():
    df = make_frame([10.0] * 5, volume=1000.0)
    df.loc[df.index[2], "Volume"] = 1.0
    out = data.annotate_liquidity(df, min_dollar_volume=5_000.0)
    assert len(out) == 5, "thin days stay in the series"
    assert not out["tradable"].iloc[2]
    assert out["tradable"].iloc[0]


def test_zero_threshold_disables_the_liquidity_filter():
    df = make_frame([10.0] * 3, volume=1.0)
    out = data.annotate_liquidity(df, min_dollar_volume=0.0)
    assert out["tradable"].all()


def test_liquidity_profile_reports_dollar_volume():
    df = data.annotate_liquidity(make_frame([10.0] * 10, volume=1000.0), 5_000.0)
    profile = data.liquidity_profile(df, 5_000.0)
    assert profile["avg_dollar_volume"] == pytest.approx(10_000.0)
    assert profile["untradable_days"] == 0


def test_cache_path_respects_format():
    p = data.cache_path("UAMY", "1d", "/tmp/cache", "parquet")
    assert p.name == "UAMY_1d.parquet"
    assert data.cache_path("uamy", "1d", "/tmp/cache", "csv").name == "UAMY_1d.csv"


def test_offline_without_cache_raises_clearly(cfg, tmp_path):
    cfg.cache_dir = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="offline"):
        data.load_prices("NOPE", cfg, offline=True)


def test_a_bad_close_drops_the_row_but_a_bad_open_does_not():
    """Only Close disqualifies a session -- the engine trades on Close alone.

    Broker exports and close-only vendor files routinely carry zero or blank
    Open/High/Low. Dropping those sessions would discard good data over a
    column nothing reads.
    """
    df = make_frame([10.0, 11.0, 12.0, 13.0])
    df.loc[df.index[1], ["Open", "High", "Low"]] = 0.0      # survivable
    df.loc[df.index[2], "Close"] = -1.0                     # fatal

    out, report = data.validate_and_clean(df, "MIXED")

    assert report.nonpositive_prices == 1, "only the bad Close counts"
    assert len(out) == 3, "the zero-Open row must survive"
    assert out["Close"].tolist() == [10.0, 11.0, 13.0]
    # The zeroed columns are backfilled from Close, not left at zero.
    assert out.loc[out.index[1], "Open"] == 11.0
    assert report.missing_ohlc_filled == 3


def test_a_close_only_file_needs_no_ohl_columns():
    """A frame carrying nothing but Close must load and backfill cleanly."""
    df = make_frame([10.0, 11.0, 12.0])[["Close", "Volume"]]
    out, report = data.validate_and_clean(df, "CLOSEONLY")
    assert len(out) == 3
    assert (out["Open"] == out["Close"]).all()
    assert any("Open absent" in n for n in report.notes)


# -- filename portability ---------------------------------------------------
# Cache files should be legal on Windows as well as POSIX, so a data_cache
# copied between machines stays usable.
@pytest.mark.parametrize("ticker,expected", [
    ("UAMY", "UAMY_1d.csv"),
    ("brk-b", "BRK-B_1d.csv"),          # normalised to upper case
    ("^GSPC", "^GSPC_1d.csv"),          # caret is legal on Windows
    ("EURUSD=X", "EURUSD=X_1d.csv"),    # equals is legal
    ("ABC.TO", "ABC.TO_1d.csv"),        # exchange suffix
    ("A/B", "A-B_1d.csv"),              # slash is illegal everywhere
    ("A:B", "A-B_1d.csv"),              # colon is illegal on Windows
    ("A*B?", "A-B-_1d.csv"),            # wildcards are illegal on Windows
    ('A"B', "A-B_1d.csv"),              # quote is illegal on Windows
])
def test_cache_filenames_are_windows_legal(ticker, expected):
    assert data.cache_path(ticker, "1d", "cache").name == expected


def test_no_cache_filename_contains_a_windows_illegal_character():
    for ticker in ["A/B", "A:B", "A*B", "A?B", 'A"B', "A<B", "A>B", "A|B", "A\\B"]:
        stem = data.safe_filename_stem(ticker, "1d")
        assert not set(stem) & set('<>:"/\\|?*'), f"{ticker} -> {stem}"


@pytest.mark.parametrize("ticker", ["CON", "PRN", "AUX", "NUL", "COM1", "LPT9"])
def test_windows_reserved_device_names_are_not_produced(ticker):
    """Windows refuses to create CON.csv, PRN.csv, and friends."""
    stem = data.safe_filename_stem(ticker, "1d")
    assert stem.split(".")[0].upper() not in data._RESERVED_STEMS
    # The interval suffix already does the work; assert it, do not assume it.
    assert stem == f"{ticker}_1d"


def test_trailing_dots_and_spaces_are_stripped():
    """Windows silently drops them, which would collide two distinct tickers."""
    assert data.safe_filename_stem("ABC.", "1d") == "ABC_1d"
    assert data.safe_filename_stem("ABC ", "1d") == "ABC_1d"


@pytest.mark.parametrize("junk", ["///", "...", "   ", "***", "?"])
def test_a_ticker_with_no_usable_characters_is_rejected(junk):
    """Pure punctuation is a typo or a mangled config entry, not a symbol."""
    with pytest.raises(ValueError, match="no usable filename"):
        data.safe_filename_stem(junk, "1d")
