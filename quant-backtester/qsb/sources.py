"""
Pluggable price data sources.

The backtester should not care where bars come from, and hardcoding one vendor
is a real fragility rather than a hypothetical one: yfinance scrapes endpoints
Yahoo never promised to keep stable, so it breaks on Yahoo's schedule, not
yours. Isolating that behind a small interface means a vendor change is a new
30-line class instead of surgery on the loader.

It also makes the offline path first class. Running from CSVs you exported
yourself is not a degraded fallback -- it is a source like any other, and on a
machine with no network access to market data it is the only one that works.

Every source returns a raw OHLCV frame. None of them clean it: cleaning belongs
to `data.validate_and_clean`, so that the same validation, the same repair
counting, and the same cleaning report apply no matter where the bars came from.

Adding a source:

    class MySource:
        name = "mine"
        def available(self) -> tuple[bool, str]: ...
        def fetch(self, ticker, start, end, interval) -> pd.DataFrame: ...

    register(MySource())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)

# Column spellings seen in the wild, mapped to what this project uses.
_COLUMN_ALIASES = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adjclose": "Close",
    "adj_close": "Close",
    "adjustedclose": "Close",
    "volume": "Volume",
    "vol": "Volume",
}

# Index/date column spellings seen in the wild.
_DATE_COLUMNS = ("date", "datetime", "timestamp", "time", "index", "unnamed:0")


@runtime_checkable
class DataSource(Protocol):
    """Anything that can hand back raw OHLCV bars for one ticker."""

    name: str

    def available(self) -> tuple[bool, str]:
        """(usable_here, human explanation) -- checked before fetching."""
        ...

    def fetch(
        self, ticker: str, start: str, end: str | None, interval: str
    ) -> pd.DataFrame:
        """Return raw bars. Raise with a clear message on failure."""
        ...


# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------
class YFinanceSource:
    """Yahoo Finance via the `yfinance` package.

    Free and convenient, with the caveats that implies: undocumented endpoints,
    occasional silent data quirks, and no uptime guarantee. Good enough for
    exploratory research, which is what this project is.
    """

    name = "yfinance"

    def available(self) -> tuple[bool, str]:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False, (
                "yfinance is not installed. Run `pip install -r requirements.txt`, "
                "or use --source csv to work from exported files."
            )
        return True, "yfinance is installed"

    def fetch(
        self, ticker: str, start: str, end: str | None, interval: str
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(self.available()[1]) from exc

        logger.info("downloading %s %s bars %s -> %s", ticker, interval, start, end)
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,      # split/dividend adjusted OHLC
            progress=False,
            threads=False,
            multi_level_index=False,
        )
        if df is None or df.empty:
            raise ValueError(
                f"yfinance returned no rows for {ticker} "
                f"({start} -> {end}, interval={interval}).\n"
                "Common causes: a delisted or mistyped symbol, a date range with "
                "no sessions, an interval Yahoo does not serve that far back "
                "(intraday history is limited), rate limiting, or no network "
                "route to Yahoo. Run `python backtest.py --doctor` to tell those "
                "apart."
            )
        return df


# ---------------------------------------------------------------------------
# CSV directory
# ---------------------------------------------------------------------------
class CsvDirectorySource:
    """Bars read from a directory of CSV files you provide.

    Deliberately forgiving about format, because these files come from wherever
    you could get them: a broker export, a paid vendor, a spreadsheet, another
    machine that *can* reach Yahoo. Column names are matched case-insensitively
    against common spellings, the date column is auto-detected, and only Close
    is truly required.

    File naming, most specific first:
        {TICKER}_{interval}.csv     e.g. UAMY_1d.csv
        {TICKER}.csv                e.g. UAMY.csv
    Case-insensitive, and .txt is accepted alongside .csv.
    """

    name = "csv"

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()

    def available(self) -> tuple[bool, str]:
        if not self.directory.exists():
            return False, f"CSV directory does not exist: {self.directory}"
        if not self.directory.is_dir():
            return False, f"CSV path is not a directory: {self.directory}"
        n = len(self._all_files())
        if n == 0:
            return False, f"no .csv or .txt files found in {self.directory}"
        return True, f"{n} file(s) in {self.directory}"

    def _all_files(self) -> list[Path]:
        out: list[Path] = []
        for pattern in ("*.csv", "*.CSV", "*.txt", "*.TXT"):
            out.extend(self.directory.glob(pattern))
        return sorted(set(out))

    def find_file(self, ticker: str, interval: str) -> Path | None:
        """Locate the file for one ticker, preferring an interval-specific name."""
        wanted = [
            f"{ticker}_{interval}".lower(),
            ticker.lower(),
        ]
        by_stem = {p.stem.lower(): p for p in self._all_files()}
        for name in wanted:
            if name in by_stem:
                return by_stem[name]
        return None

    def fetch(
        self, ticker: str, start: str, end: str | None, interval: str
    ) -> pd.DataFrame:
        path = self.find_file(ticker, interval)
        if path is None:
            existing = ", ".join(sorted(p.stem for p in self._all_files())) or "(none)"
            raise FileNotFoundError(
                f"no CSV for {ticker} in {self.directory}. Expected "
                f"{ticker}_{interval}.csv or {ticker}.csv. Found: {existing}"
            )

        df = read_price_csv(path)
        return _slice_dates(df, start, end)


def read_price_csv(path: str | Path) -> pd.DataFrame:
    """Read one price CSV, normalising its date index and column names.

    Raises a message naming the actual columns found when it cannot make sense
    of a file -- a silent empty frame here would surface much later as a
    baffling "no usable rows" error.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{path.name} is empty")

    lowered = {str(c).strip().lower(): c for c in raw.columns}

    # -- date column --------------------------------------------------------
    date_col = next((lowered[c] for c in _DATE_COLUMNS if c in lowered), None)
    if date_col is None:
        # Fall back to the first column if it parses as dates.
        first = raw.columns[0]
        parsed = pd.to_datetime(raw[first], errors="coerce")
        if parsed.notna().mean() > 0.8:
            date_col = first
        else:
            raise ValueError(
                f"{path.name}: could not find a date column. Looked for one of "
                f"{_DATE_COLUMNS}; the file has {list(raw.columns)}"
            )

    index = pd.to_datetime(raw[date_col], errors="coerce")
    df = raw.drop(columns=[date_col])
    df.index = index
    df.index.name = "Date"

    # -- value columns ------------------------------------------------------
    renamed: dict = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "").replace("-", "_")
        if key in _COLUMN_ALIASES:
            target = _COLUMN_ALIASES[key]
            # A real Close beats an Adj Close alias for the same slot.
            if target == "Close" and "Close" in renamed.values() and key != "close":
                continue
            renamed[col] = target
    if not renamed:
        raise ValueError(
            f"{path.name}: no recognisable OHLCV columns. Found "
            f"{list(raw.columns)}; expected some of Open/High/Low/Close/Volume"
        )

    df = df.rename(columns=renamed)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if "Close" not in keep:
        raise ValueError(
            f"{path.name}: no Close (or Adj Close) column. Found {list(raw.columns)}"
        )
    return df[keep]


def _slice_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Trim to the requested window, tolerating an unsorted index."""
    out = df.sort_index()
    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index <= pd.Timestamp(end)]
    if out.empty:
        raise ValueError(
            f"no rows left after trimming to {start} -> {end}. The file covers "
            f"{df.index.min()} -> {df.index.max()}"
        )
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_BUILTIN = {"yfinance": YFinanceSource}


def build(name: str, cfg=None, csv_dir: str | Path | None = None) -> DataSource:
    """Construct a source by name.

    `csv` needs a directory, taken from the explicit argument, then the config's
    `csv_dir`, and failing both it is an error worth stating plainly rather than
    defaulting to somewhere surprising.
    """
    key = (name or "yfinance").strip().lower()

    if key in _BUILTIN:
        return _BUILTIN[key]()

    if key == "csv":
        directory = csv_dir or getattr(cfg, "csv_dir", None)
        if not directory:
            raise ValueError(
                "--source csv needs a directory: pass --csv-dir PATH or set "
                '"csv_dir" in your config file.'
            )
        return CsvDirectorySource(directory)

    raise KeyError(f"unknown source {name!r}. Available: {', '.join(available())}")


def register(source: DataSource) -> None:
    """Add a custom source class to the registry."""
    _BUILTIN[source.name] = type(source)


def available() -> list[str]:
    return sorted(set(_BUILTIN) | {"csv"})
