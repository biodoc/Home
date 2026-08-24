"""
Data loading, validation, cleaning, caching, and liquidity profiling.

yfinance data is convenient and free, which means it is also occasionally wrong.
Observed failure modes this module defends against:

  * duplicate timestamps for the same session
  * zero, negative, or NaN prices
  * missing sessions (holidays are fine; multi-day holes are not)
  * split/adjustment artifacts that show up as absurd single-day moves
  * a Volume column that is NaN rather than 0

Everything cleaned is *counted and logged*. Silent repair is how a backtest ends
up measuring a data bug instead of a signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import sources

logger = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# A single-session move larger than this is flagged as a suspected split or
# adjustment artifact. It is reported, not removed -- small caps genuinely do
# move like this, and dropping real moves would flatter every mean-reversion
# signal in the library.
SUSPECT_MOVE = 0.60


@dataclass
class CleaningReport:
    """A record of exactly what was changed between download and backtest."""

    ticker: str
    rows_in: int = 0
    rows_out: int = 0
    duplicate_timestamps: int = 0
    nonpositive_prices: int = 0
    missing_close: int = 0
    missing_ohlc_filled: int = 0
    nan_volume_filled: int = 0
    reordered: bool = False
    calendar_gaps: int = 0
    largest_gap_days: int = 0
    suspect_moves: int = 0
    source: str = "unknown"
    notes: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def clean(self) -> bool:
        """True when nothing at all had to be repaired."""
        return (
            self.rows_dropped == 0
            and self.missing_ohlc_filled == 0
            and self.nan_volume_filled == 0
            and self.suspect_moves == 0
        )

    def summary(self) -> str:
        """One-block human-readable summary, always printed with the report."""
        lines = [
            f"  source              : {self.source}",
            f"  rows in / out       : {self.rows_in} -> {self.rows_out} "
            f"({self.rows_dropped} dropped)",
        ]
        if self.duplicate_timestamps:
            lines.append(
                f"  duplicate timestamps: {self.duplicate_timestamps} (kept last)"
            )
        if self.nonpositive_prices:
            lines.append(
                f"  zero/negative prices: {self.nonpositive_prices} rows dropped"
            )
        if self.missing_close:
            lines.append(f"  missing close       : {self.missing_close} rows dropped")
        if self.missing_ohlc_filled:
            lines.append(
                f"  missing O/H/L filled: {self.missing_ohlc_filled} "
                "(back-filled from Close)"
            )
        if self.nan_volume_filled:
            lines.append(f"  NaN volume -> 0     : {self.nan_volume_filled}")
        if self.reordered:
            lines.append("  index               : was out of order, sorted")
        if self.calendar_gaps:
            lines.append(
                f"  missing sessions    : {self.calendar_gaps} business days absent "
                f"(largest run {self.largest_gap_days}d) -- holidays included"
            )
        if self.suspect_moves:
            lines.append(
                f"  suspect moves       : {self.suspect_moves} day(s) moved >"
                f"{SUSPECT_MOVE:.0%} -- possible split/adjustment artifact, KEPT"
            )
        for note in self.notes:
            lines.append(f"  note                : {note}")
        if self.clean and not self.notes:
            lines.append("  no repairs needed")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Download + cache
# ---------------------------------------------------------------------------
def cache_path(ticker: str, interval: str, cache_dir: str | Path,
               fmt: str = "csv") -> Path:
    """Location of the local cache file for one ticker/interval."""
    suffix = "parquet" if fmt == "parquet" else "csv"
    safe = ticker.upper().replace("/", "-")
    return Path(cache_dir) / f"{safe}_{interval}.{suffix}"


def _read_cache(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path)
            return
        except (ImportError, ValueError) as exc:  # pragma: no cover - env dependent
            logger.warning(
                "parquet write failed (%s); falling back to CSV. "
                "Install pyarrow or set cache_format='csv'.", exc
            )
            path = path.with_suffix(".csv")
    df.to_csv(path)


def fetch_prices(
    ticker: str,
    start: str,
    end: str | None,
    interval: str = "1d",
    source: "sources.DataSource | None" = None,
) -> pd.DataFrame:
    """Fetch raw OHLCV bars from a data source.

    Defaults to yfinance when no source is given, which keeps the simple case
    simple. The source is constructed lazily so that the rest of the package --
    and the whole test suite -- works with no network and no yfinance installed.
    """
    src = source or sources.build("yfinance")

    ok, why = src.available()
    if not ok:
        raise RuntimeError(f"data source {src.name!r} is unusable: {why}")

    return src.fetch(ticker, start, end, interval)


def load_prices(
    ticker: str,
    cfg,
    force_refresh: bool = False,
    offline: bool = False,
    source: "sources.DataSource | None" = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Return cleaned bars for one ticker, using the local cache when possible.

    Args:
        ticker: symbol to load.
        cfg: a `config.Config`.
        force_refresh: ignore any cached file and re-fetch.
        offline: never fetch; use the cache only, and fail if it is missing.
        source: where to fetch from when the cache misses. Defaults to the
            source named in the config (yfinance unless changed).

    Returns:
        (dataframe, cleaning_report). The dataframe has a sorted, unique,
        tz-naive DatetimeIndex and columns Open/High/Low/Close/Volume, plus
        derived columns `dollar_volume` and `tradable`.

    Note that whatever the source, the bars go through the same
    `validate_and_clean` pass. A broker export gets the same scrutiny as a
    download.
    """
    path = cache_path(ticker, cfg.interval, cfg.cache_dir, cfg.cache_format)
    source_label = "cache"
    raw: pd.DataFrame | None = None

    if not force_refresh and path.exists():
        try:
            raw = _read_cache(path)
            logger.info("loaded %s from cache (%s)", ticker, path)
        except Exception as exc:  # corrupt cache should never be fatal
            logger.warning("cache read failed for %s (%s); re-downloading", ticker, exc)
            raw = None

    if raw is None:
        if offline:
            raise FileNotFoundError(
                f"--offline was requested but no cache exists for {ticker} at "
                f"{path}. Run once without --offline to populate the cache, or "
                "point --source csv at a directory of exported files."
            )
        src = source or sources.build(getattr(cfg, "source", "yfinance"), cfg)
        raw = fetch_prices(ticker, cfg.start, cfg.end, cfg.interval, source=src)
        source_label = f"{src.name} (fresh fetch)"
        _write_cache(raw, path)

    df, report = validate_and_clean(raw, ticker)
    report.source = source_label

    df = annotate_liquidity(df, cfg.min_avg_dollar_volume)
    return df, report


# ---------------------------------------------------------------------------
# Validation and cleaning
# ---------------------------------------------------------------------------
def validate_and_clean(
    raw: pd.DataFrame, ticker: str = "?"
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw OHLCV frame, counting every repair made.

    This never fills prices forward across missing sessions. A gap stays a gap;
    it is reported, and the return series simply spans it. Forward-filling
    prices would manufacture zero-return days and inflate every Sharpe ratio in
    the report.
    """
    report = CleaningReport(ticker=ticker, rows_in=len(raw))
    df = raw.copy()

    # -- columns ------------------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns a MultiIndex when several tickers are requested.
        # Flatten by taking the level that carries the OHLCV names.
        level = 0 if "Close" in df.columns.get_level_values(0) else 1
        df.columns = df.columns.get_level_values(level)
        report.notes.append("flattened MultiIndex columns")

    df.columns = [str(c).strip().title().replace(" ", "") for c in df.columns]
    if "Adjclose" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adjclose": "Close"})

    missing = [c for c in OHLCV if c not in df.columns]
    if "Close" in missing:
        raise ValueError(f"{ticker}: data has no Close column (got {list(df.columns)})")
    for col in missing:
        # Open/High/Low missing is survivable -- the engine trades on Close.
        df[col] = np.nan
        report.notes.append(f"column {col} absent; created as NaN")
    df = df[OHLCV]

    # -- index --------------------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    bad_index = df.index.isna()
    if bad_index.any():
        report.notes.append(f"{int(bad_index.sum())} unparseable timestamps dropped")
        df = df[~bad_index]

    dupes = int(df.index.duplicated(keep="last").sum())
    if dupes:
        report.duplicate_timestamps = dupes
        df = df[~df.index.duplicated(keep="last")]

    if not df.index.is_monotonic_increasing:
        report.reordered = True
        df = df.sort_index()

    # -- prices -------------------------------------------------------------
    for col in OHLCV:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_close = int(df["Close"].isna().sum())
    if missing_close:
        report.missing_close = missing_close
        df = df[df["Close"].notna()]

    # Only a bad CLOSE disqualifies a session. The engine trades on Close, and
    # plenty of legitimate sources leave Open/High/Low blank or zero -- broker
    # exports and close-only vendor files routinely do. Dropping those rows
    # would throw away perfectly good sessions over a column nothing reads.
    nonpositive = df["Close"] <= 0
    n_bad = int(nonpositive.sum())
    if n_bad:
        report.nonpositive_prices = n_bad
        df = df[~nonpositive]

    # Non-positive O/H/L is treated as absent rather than fatal, then filled
    # from Close below along with genuine NaNs.
    for col in ["Open", "High", "Low"]:
        bad = df[col] <= 0
        if bad.any():
            df[col] = df[col].where(~bad)

    # Backfill absent O/H/L from Close so downstream code can rely on the
    # columns existing. Close is never synthesised.
    for col in ["Open", "High", "Low"]:
        gaps = int(df[col].isna().sum())
        if gaps:
            report.missing_ohlc_filled += gaps
            df[col] = df[col].fillna(df["Close"])

    nan_vol = int(df["Volume"].isna().sum())
    if nan_vol:
        report.nan_volume_filled = nan_vol
        df["Volume"] = df["Volume"].fillna(0.0)
    df["Volume"] = df["Volume"].clip(lower=0.0)

    # -- structural checks (reported, not repaired) -------------------------
    if len(df) >= 2:
        expected = pd.bdate_range(df.index.min(), df.index.max())
        absent = expected.difference(df.index)
        report.calendar_gaps = int(len(absent))
        deltas = df.index.to_series().diff().dt.days.dropna()
        report.largest_gap_days = int(deltas.max()) if len(deltas) else 0

        moves = df["Close"].pct_change(fill_method=None).abs()
        report.suspect_moves = int((moves > SUSPECT_MOVE).sum())

    report.rows_out = len(df)
    if report.rows_out == 0:
        raise ValueError(
            f"{ticker}: no usable rows survived cleaning "
            f"({report.rows_in} rows in). See the cleaning report."
        )
    return df, report


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------
def annotate_liquidity(df: pd.DataFrame, min_dollar_volume: float) -> pd.DataFrame:
    """Add `dollar_volume` and a boolean `tradable` column.

    Thin days are not dropped -- dropping them would splice unrelated prices
    together and create phantom returns. They are marked untradable, and the
    engine forces the position flat on those days instead.
    """
    out = df.copy()
    out["dollar_volume"] = out["Close"] * out["Volume"]
    if min_dollar_volume and min_dollar_volume > 0:
        out["tradable"] = out["dollar_volume"] >= float(min_dollar_volume)
    else:
        out["tradable"] = True
    return out


def liquidity_profile(df: pd.DataFrame, min_dollar_volume: float = 0.0) -> dict:
    """Summarise how tradable a name actually was over the sample."""
    dv = df.get("dollar_volume")
    if dv is None:
        dv = df["Close"] * df["Volume"]
    dv = dv.replace([np.inf, -np.inf], np.nan).dropna()
    if dv.empty:
        return {"days": 0}

    tradable = df["tradable"] if "tradable" in df else pd.Series(True, index=df.index)
    profile = {
        "days": int(len(dv)),
        "avg_dollar_volume": float(dv.mean()),
        "median_dollar_volume": float(dv.median()),
        "p10_dollar_volume": float(dv.quantile(0.10)),
        "min_dollar_volume": float(dv.min()),
        "threshold": float(min_dollar_volume),
        "untradable_days": int((~tradable).sum()),
        "untradable_pct": float((~tradable).mean() * 100.0),
        "first": df.index.min(),
        "last": df.index.max(),
    }
    return profile
