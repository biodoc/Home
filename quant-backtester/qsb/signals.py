"""
The signal library.

Every signal is a pure function of a price frame that returns a *target position*
series taking values in {-1, 0, +1}:

    +1  long one unit
     0  flat
    -1  short one unit

Two rules hold for every signal in this module, without exception:

  1. CAUSALITY. The value at bar *t* may depend only on data up to and including
     bar *t*. No `.shift(-n)`, no centred windows, no full-sample statistics
     (a full-sample mean or quantile leaks the future into the past). Rolling
     windows and expanding windows are fine; `.quantile()` over the whole series
     is not.

  2. INTERPRETATION. The position at bar *t* is a decision made on the close of
     bar *t*, and it is earned over bar *t+1*. The engine applies that one-bar
     shift, so signals here must NOT pre-shift. Shifting in both places would
     hide a real edge; shifting in neither would manufacture a fake one.

Each signal also declares a parameter grid. The walk-forward engine searches
that grid on each training window and applies the winner, untouched, to the
following test window. Grids are kept small on purpose: a grid with hundreds of
combinations will find something that "works" on any training window at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalDef:
    """A named signal plus the parameter grid the engine is allowed to search."""

    name: str
    func: Callable[..., pd.Series]
    param_grid: dict[str, list] = field(default_factory=dict)
    description: str = ""
    plain_english: str = ""

    def default_params(self) -> dict:
        """The first value of each grid axis -- used as a safe fallback."""
        return {k: v[0] for k, v in self.param_grid.items()}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df:
        raise KeyError("signal input frame needs a 'Close' column")
    return df["Close"].astype("float64")


def _state_machine(
    index: pd.Index,
    go_long: pd.Series,
    go_short: pd.Series,
    go_flat: pd.Series,
) -> pd.Series:
    """Build a position series with hysteresis: hold until told otherwise.

    Precedence on any given bar is flat < long < short-if-both (the entry
    conditions in this library are mutually exclusive by construction, so the
    tie-break never actually fires). Bars where no condition holds inherit the
    previous position, which is what makes an entry/exit band behave like a real
    position rather than flickering on and off at the threshold.
    """
    state = pd.Series(np.nan, index=index, dtype="float64")
    state[go_flat.fillna(False).to_numpy()] = 0.0
    state[go_long.fillna(False).to_numpy()] = 1.0
    state[go_short.fillna(False).to_numpy()] = -1.0
    return state.ffill().fillna(0.0)


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """(x - rolling mean) / rolling stdev, using only trailing data."""
    mean = series.rolling(lookback, min_periods=lookback).mean()
    std = series.rolling(lookback, min_periods=lookback).std(ddof=1)
    # A zero-variance window (a halted or fully flat stock) is not a 0-sigma
    # event, it is an undefined one. NaN keeps it out of the signal.
    std = std.where(std > 0)
    return (series - mean) / std


def wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Classic Wilder RSI, 0..100, computed with an exponential average."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing is an EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken run of gains -> RSI 100 by definition.
    rsi = rsi.where(avg_loss > 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    rsi[avg_gain.isna() | avg_loss.isna()] = np.nan
    return rsi


def realized_vol(series: pd.Series, lookback: int, ppy: int = 252) -> pd.Series:
    """Annualized trailing realized volatility of log returns."""
    logret = np.log(series).diff()
    return logret.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(ppy)


# ---------------------------------------------------------------------------
# Signal 1 -- short-term mean reversion via z-score
# ---------------------------------------------------------------------------
def mean_reversion_zscore(
    df: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
) -> pd.Series:
    """Fade stretches away from a trailing mean.

    Go long when price is `entry_z` standard deviations *below* its trailing
    mean, short when it is that far above, and flatten once it has come back
    inside `exit_z`. The entry/exit band is what stops the position from
    churning every time the z-score jitters across a single threshold.
    """
    if exit_z >= entry_z:
        raise ValueError("exit_z must be tighter than entry_z")

    z = rolling_zscore(_close(df), lookback)
    return _state_machine(
        df.index,
        go_long=z <= -entry_z,
        go_short=z >= entry_z,
        go_flat=z.abs() <= exit_z,
    )


# ---------------------------------------------------------------------------
# Signal 2 -- RSI extremes
# ---------------------------------------------------------------------------
def rsi_reversion(
    df: pd.DataFrame,
    period: int = 14,
    lower: float = 30.0,
    upper: float = 70.0,
) -> pd.Series:
    """Buy oversold, sell overbought, exit on the return to the midline.

    A second, independent test of the same mean-reversion premise as the z-score
    signal. If one works and the other does not on the same ticker, that is
    evidence about the specific threshold, not about mean reversion.
    """
    if not 0 < lower < 50 < upper < 100:
        raise ValueError("need 0 < lower < 50 < upper < 100")

    rsi = wilder_rsi(_close(df), period)
    return _state_machine(
        df.index,
        go_long=rsi <= lower,
        go_short=rsi >= upper,
        # Crossing back through the midline closes either side.
        go_flat=(rsi > 45.0) & (rsi < 55.0),
    )


# ---------------------------------------------------------------------------
# Signal 3 -- moving-average crossover
# ---------------------------------------------------------------------------
def ma_crossover(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    long_only: bool = False,
) -> pd.Series:
    """Trend continuation: long above the crossover, short (or flat) below."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")

    close = _close(df)
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()

    pos = pd.Series(0.0, index=df.index, dtype="float64")
    valid = fast_ma.notna() & slow_ma.notna()
    pos[valid & (fast_ma > slow_ma)] = 1.0
    pos[valid & (fast_ma <= slow_ma)] = 0.0 if long_only else -1.0
    return pos


# ---------------------------------------------------------------------------
# Signal 4 -- rate-of-change momentum
# ---------------------------------------------------------------------------
def momentum_roc(
    df: pd.DataFrame,
    lookback: int = 60,
    threshold: float = 0.0,
) -> pd.Series:
    """Hold the sign of the trailing return, when it is large enough to bother.

    `threshold` is a fraction (0.02 = 2%). A dead band around zero keeps the
    position from flipping on noise in a sideways tape.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    close = _close(df)
    roc = close.pct_change(lookback, fill_method=None)

    pos = pd.Series(0.0, index=df.index, dtype="float64")
    pos[roc > threshold] = 1.0
    pos[roc < -threshold] = -1.0
    pos[roc.isna()] = 0.0
    return pos


# ---------------------------------------------------------------------------
# Signal 5 -- volatility regime / Bollinger squeeze breakout
# ---------------------------------------------------------------------------
def vol_regime_breakout(
    df: pd.DataFrame,
    lookback: int = 20,
    n_std: float = 2.0,
    squeeze_lookback: int = 126,
    squeeze_pct: float = 0.25,
) -> pd.Series:
    """Trade breakouts only while volatility is compressed.

    Bollinger bandwidth is compared against its OWN trailing distribution -- an
    expanding/rolling quantile, never a full-sample one, since a full-sample
    quantile would let the signal know how volatile the stock would later become.

    When bandwidth sits in the bottom `squeeze_pct` of its trailing history and
    price closes outside the band, take the direction of the break; flatten when
    price returns inside the band or the squeeze ends.
    """
    if not 0 < squeeze_pct < 1:
        raise ValueError("squeeze_pct must be in (0, 1)")

    close = _close(df)
    mid = close.rolling(lookback, min_periods=lookback).mean()
    sd = close.rolling(lookback, min_periods=lookback).std(ddof=1)
    upper = mid + n_std * sd
    lower = mid - n_std * sd

    bandwidth = (upper - lower) / mid.where(mid > 0)
    # Trailing quantile of bandwidth: strictly backward looking.
    squeeze_level = bandwidth.rolling(
        squeeze_lookback, min_periods=max(lookback, squeeze_lookback // 4)
    ).quantile(squeeze_pct)
    squeezed = bandwidth <= squeeze_level

    return _state_machine(
        df.index,
        go_long=squeezed & (close > upper),
        go_short=squeezed & (close < lower),
        go_flat=(close <= upper) & (close >= lower),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
REGISTRY: dict[str, SignalDef] = {
    "mean_reversion": SignalDef(
        name="mean_reversion",
        func=mean_reversion_zscore,
        param_grid={
            "lookback": [10, 20, 40],
            "entry_z": [1.0, 1.5, 2.0],
            "exit_z": [0.25, 0.5],
        },
        description="z-score of close vs rolling mean, faded at the extremes",
        plain_english=(
            "Assume an unusually large move away from the recent average tends "
            "to snap back. Buy when the stock is far below its own recent "
            "average, sell it short when it is far above, and step aside once "
            "it has returned to normal."
        ),
    ),
    "rsi_reversion": SignalDef(
        name="rsi_reversion",
        func=rsi_reversion,
        param_grid={
            "period": [7, 14, 21],
            "lower": [20.0, 30.0],
            "upper": [70.0, 80.0],
        },
        description="Wilder RSI extremes, exit on return to the midline",
        plain_english=(
            "The same 'it snaps back' idea as mean reversion, but measured with "
            "RSI, which compares the size of recent up moves to recent down "
            "moves. Buy when RSI says oversold, short when it says overbought."
        ),
    ),
    "ma_crossover": SignalDef(
        name="ma_crossover",
        func=ma_crossover,
        param_grid={
            "fast": [10, 20, 50],
            "slow": [50, 100, 200],
        },
        description="fast/slow moving-average crossover, trend following",
        plain_english=(
            "Assume trends persist. Hold long while a short-term average sits "
            "above a long-term one, and flip short when it drops below. This is "
            "the opposite bet to the mean-reversion signals; on any given "
            "ticker at most one of them should look good, and often neither does."
        ),
    ),
    "momentum_roc": SignalDef(
        name="momentum_roc",
        func=momentum_roc,
        param_grid={
            "lookback": [20, 60, 120],
            "threshold": [0.0, 0.02, 0.05],
        },
        description="sign of trailing rate of change, with a dead band",
        plain_english=(
            "Look at how much the stock has moved over the last N days. If it "
            "is up by more than a set amount, go long; down by more than that, "
            "go short; otherwise stand aside."
        ),
    ),
    "vol_regime": SignalDef(
        name="vol_regime",
        func=vol_regime_breakout,
        param_grid={
            "lookback": [20, 40],
            "n_std": [2.0, 2.5],
            "squeeze_pct": [0.2, 0.35],
        },
        description="Bollinger-bandwidth squeeze followed by a band breakout",
        plain_english=(
            "Quiet periods often precede large moves. Wait until the stock's "
            "recent trading range is unusually narrow compared with its own "
            "history, then follow the direction in which it eventually breaks "
            "out of that range."
        ),
    ),
}


def get(name: str) -> SignalDef:
    """Look up a signal by name, with a helpful error for typos."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in REGISTRY:
        return REGISTRY[key]
    raise KeyError(
        f"unknown signal {name!r}. Available: {', '.join(sorted(REGISTRY))}"
    )


def available() -> list[str]:
    return sorted(REGISTRY)


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a parameter grid, as a list of kwargs dicts."""
    if not grid:
        return [{}]
    import itertools

    keys = list(grid)
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def valid_combos(sig: SignalDef) -> list[dict]:
    """Grid combinations that the signal actually accepts.

    Some grids generate nonsense pairs (fast=50 with slow=50, exit_z above
    entry_z). Rather than special-casing the grid, invalid combinations are
    detected by asking the signal itself and dropping whatever it rejects.
    """
    probe = pd.DataFrame(
        {
            "Open": np.linspace(10, 11, 5),
            "High": np.linspace(10, 11, 5),
            "Low": np.linspace(10, 11, 5),
            "Close": np.linspace(10, 11, 5),
            "Volume": np.full(5, 1e6),
        },
        index=pd.bdate_range("2020-01-01", periods=5),
    )
    good = []
    for params in expand_grid(sig.param_grid):
        try:
            sig.func(probe, **params)
        except ValueError:
            continue  # structurally invalid combination
        except Exception:
            continue
        good.append(params)
    return good
