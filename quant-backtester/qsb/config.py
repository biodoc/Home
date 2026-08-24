"""
Configuration loading.

The ticker universe, cost assumptions, and walk-forward window sizes all live
here rather than being baked into signal or backtest logic. That is a hard rule
for this project: no ticker list, no cost number, and no window length may be
hardcoded anywhere in `signals.py`, `engine.py`, or `metrics.py`.

Config resolution order (later wins):
    1. DEFAULTS in this module
    2. a JSON config file, if one is supplied
    3. explicit CLI overrides

JSON is used rather than YAML purely to avoid a dependency; the brief asks for a
minimal dependency footprint.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# These are starting assumptions, not measurements. The cost numbers in
# particular should be replaced with figures observed on real fills before any
# result here is taken seriously.
DEFAULTS: dict[str, Any] = {
    # Development universe. Results here are the *in-development* set: whatever
    # looks good on these names must still be re-tested on `validation_universe`
    # before it counts as evidence of anything.
    "universe": ["UAMY", "SMR", "BBAI", "AEM"],

    # A deliberately diverse hold-out set: different sectors, market caps, and
    # liquidity regimes, none of which were used to develop any signal.
    "validation_universe": [
        "AAPL",   # mega-cap tech
        "JNJ",    # mega-cap healthcare
        "XOM",    # mega-cap energy
        "KO",     # mega-cap staples
        "CAT",    # large-cap industrials
        "SCHW",   # large-cap financials
        "ETSY",   # mid-cap consumer discretionary
        "PLUG",   # small-cap, high volatility
        "RIG",    # small-cap, high volatility energy
        "GOLD",   # large-cap miner (liquidity contrast to AEM)
    ],

    "start": "2015-01-01",
    "end": None,               # None -> today
    "interval": "1d",          # yfinance interval string

    "cache_dir": "data_cache",
    "cache_format": "csv",     # "csv" (no extra deps) or "parquet" (needs pyarrow)

    # Days whose dollar volume falls below this are marked untradable and forced
    # flat, rather than being silently backtested as if a fill were available.
    # Set to 0 to disable the filter entirely.
    "min_avg_dollar_volume": 1_000_000.0,

    # Walk-forward windows, in trading days. Parameters are selected on `train`
    # and then applied, untouched, to the immediately following `test` window.
    "walk_forward": {
        "train_days": 378,      # ~18 months
        "test_days": 126,       # ~6 months
        "min_train_days": 252,  # refuse to fit on less than ~1 year
        "min_train_trades": 5,  # below this, parameter selection is not trusted
    },

    # Split cost model. Stock legs carry NO commission -- spread and slippage
    # only. Option legs carry a per-contract commission on top of a wider
    # spread and slippage. See costs.py for how these are applied.
    "costs": {
        "stock": {
            "half_spread_bps": 5.0,       # half the quoted bid/ask, in bps of price
            "slippage_bps": 2.0,          # additional adverse fill, in bps
            "commission_per_contract": 0.0,   # stocks: zero, by construction
            "contract_multiplier": 1,
            "premium_pct_of_underlying": 1.0,
        },
        "option": {
            "half_spread_bps": 150.0,     # options quote far wider than the stock
            "slippage_bps": 50.0,
            "commission_per_contract": 0.65,
            "contract_multiplier": 100,
            # Used only to convert a per-contract commission into a fraction of
            # notional. This is a crude stand-in for real chain data; see the
            # PROXY warning in costs.py.
            "premium_pct_of_underlying": 0.05,
        },
    },

    # Taxes are assumed to be ZERO for this phase. This is a simplification for
    # comparing signals against each other, not a real tax assumption. See the
    # README. There is deliberately no tax setting to change.
    "trading_days_per_year": 252,
    "risk_free_rate": 0.0,     # annualized, used for Sharpe

    # Random-walk diagnostics
    "diagnostics": {
        "acf_max_lag": 10,
        "variance_ratio_horizons": [2, 4, 8, 16],
    },
}


@dataclass(frozen=True)
class CostSpec:
    """Cost parameters for a single instrument type."""

    half_spread_bps: float
    slippage_bps: float
    commission_per_contract: float = 0.0
    contract_multiplier: int = 1
    premium_pct_of_underlying: float = 1.0

    def __post_init__(self) -> None:
        if self.half_spread_bps < 0 or self.slippage_bps < 0:
            raise ValueError("spread and slippage must be non-negative")
        if self.commission_per_contract < 0:
            raise ValueError("commission must be non-negative")
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive")
        if not (0 < self.premium_pct_of_underlying <= 1.0):
            raise ValueError("premium_pct_of_underlying must be in (0, 1]")


@dataclass(frozen=True)
class WalkForwardSpec:
    """Rolling train/test geometry, in trading days."""

    train_days: int = 378
    test_days: int = 126
    min_train_days: int = 252
    min_train_trades: int = 5

    def __post_init__(self) -> None:
        if self.test_days <= 0 or self.train_days <= 0:
            raise ValueError("train_days and test_days must be positive")
        if self.train_days < self.min_train_days:
            raise ValueError("train_days must be >= min_train_days")


@dataclass
class Config:
    """Fully resolved run configuration."""

    universe: list[str]
    validation_universe: list[str]
    start: str
    end: str | None
    interval: str
    cache_dir: str
    cache_format: str
    min_avg_dollar_volume: float
    walk_forward: WalkForwardSpec
    costs: dict[str, CostSpec]
    trading_days_per_year: int
    risk_free_rate: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "Config":
        """Build a Config from DEFAULTS, an optional JSON file, and overrides."""
        raw = copy.deepcopy(DEFAULTS)

        if path is not None:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"config file not found: {p}")
            with p.open("r", encoding="utf-8") as fh:
                file_cfg = json.load(fh)
            if not isinstance(file_cfg, dict):
                raise ValueError(f"config file {p} must contain a JSON object")
            raw = _deep_merge(raw, file_cfg)

        if overrides:
            # Drop None values so "not supplied on the CLI" never clobbers a
            # value that came from the config file.
            clean = {k: v for k, v in overrides.items() if v is not None}
            raw = _deep_merge(raw, clean)

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        costs = {
            name: CostSpec(**spec) for name, spec in raw["costs"].items()
        }
        if "stock" not in costs or "option" not in costs:
            raise ValueError("costs must define both 'stock' and 'option'")
        if costs["stock"].commission_per_contract != 0.0:
            raise ValueError(
                "stock commission must be 0.0 -- this project assumes a "
                "commission-free equity broker. Model equity cost as spread "
                "plus slippage only."
            )

        universe = _as_ticker_list(raw["universe"], "universe")
        validation = _as_ticker_list(
            raw["validation_universe"], "validation_universe"
        )

        overlap = sorted(set(universe) & set(validation))
        if overlap:
            raise ValueError(
                "validation_universe must not share tickers with universe "
                f"(overlap: {', '.join(overlap)}). A hold-out set that "
                "contains development names is not a hold-out set."
            )

        cache_format = str(raw["cache_format"]).lower()
        if cache_format not in {"csv", "parquet"}:
            raise ValueError("cache_format must be 'csv' or 'parquet'")

        return cls(
            universe=universe,
            validation_universe=validation,
            start=str(raw["start"]),
            end=raw["end"],
            interval=str(raw["interval"]),
            cache_dir=str(raw["cache_dir"]),
            cache_format=cache_format,
            min_avg_dollar_volume=float(raw["min_avg_dollar_volume"]),
            walk_forward=WalkForwardSpec(**raw["walk_forward"]),
            costs=costs,
            trading_days_per_year=int(raw["trading_days_per_year"]),
            risk_free_rate=float(raw["risk_free_rate"]),
            diagnostics=dict(raw.get("diagnostics", {})),
        )

    # -- helpers ------------------------------------------------------------
    def cost_spec(self, instrument: str) -> CostSpec:
        """Look up the cost parameters for 'stock' or 'option'."""
        key = instrument.lower()
        if key not in self.costs:
            raise KeyError(
                f"unknown instrument {instrument!r}; "
                f"known: {sorted(self.costs)}"
            )
        return self.costs[key]


def _as_ticker_list(value: Any, label: str) -> list[str]:
    """Normalize and de-duplicate a ticker list while preserving order."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list of tickers")

    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        ticker = str(item).strip().upper()
        if not ticker:
            raise ValueError(f"{label} contains an empty ticker")
        if ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
