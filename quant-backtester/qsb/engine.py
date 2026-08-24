"""
The walk-forward backtest engine.

A single in-sample fit over the whole history tells you almost nothing: with a
parameter grid and enough history, something always "works". This engine instead
rolls a train/test split forward through time:

    |---- train 1 ----|-- test 1 --|
              |---- train 2 ----|-- test 2 --|
                        |---- train 3 ----|-- test 3 --|

Parameters are chosen on each TRAIN window and then applied, frozen, to the
TEST window that immediately follows it. Only the concatenated test windows are
reported as the out-of-sample result. Train-window performance is recorded too,
but purely so the gap between the two is visible -- a signal whose train Sharpe
is 1.8 and whose test Sharpe is 0.1 is telling you it is fitting noise.

Signals are computed once over the full price series and then sliced. That is
safe because every signal in `signals.py` is causal (its value at bar t uses
only bars <= t), and it is realistic: on a given day you really would have had
the preceding history available to warm up a 200-day moving average.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import metrics, signals
from .costs import CostModel

logger = logging.getLogger(__name__)


@dataclass
class PeriodResult:
    """One train/test window of the walk."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict
    train_stats: dict
    test_stats: dict
    n_candidates: int = 0


@dataclass
class BacktestResult:
    """Everything one (ticker, signal, instrument) run produced."""

    ticker: str
    signal: str
    instrument: str
    periods: list[PeriodResult] = field(default_factory=list)
    oos_returns: pd.DataFrame | None = None
    aggregate: dict = field(default_factory=dict)
    stability: dict = field(default_factory=dict)
    verdict_code: str = "NO_DATA"
    verdict: str = ""
    cost_description: str = ""
    is_cost_proxy: bool = False
    liquidity: dict = field(default_factory=dict)
    cleaning: Any = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.oos_returns is not None and not self.oos_returns.empty

    def chosen_params(self) -> list[dict]:
        return [p.params for p in self.periods]

    def params_stable(self) -> bool:
        """True when the grid search picked the same parameters every window.

        Parameters that jump around from window to window mean the search is
        chasing noise, and the reported out-of-sample number is an average over
        several different strategies rather than one.
        """
        if len(self.periods) < 2:
            return True
        first = self.periods[0].params
        return all(p.params == first for p in self.periods[1:])


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------
def make_windows(n: int, train: int, test: int) -> list[tuple[int, int, int, int]]:
    """Non-overlapping test windows rolling forward through `n` bars.

    Returns a list of (train_start, train_end, test_start, test_end) positional
    slices, half-open on the right. Test windows never overlap, so concatenating
    them produces a continuous out-of-sample series with no bar counted twice.
    """
    windows = []
    start = 0
    while start + train + test <= n:
        windows.append((start, start + train, start + train, start + train + test))
        start += test

    # Trailing partial window: include it only if it is at least half a test
    # window long, so the last few months of data are not silently discarded.
    if windows:
        last_test_end = windows[-1][3]
        remaining = n - last_test_end
        if remaining >= max(2, test // 2):
            t_start = last_test_end - train
            if t_start >= 0:
                windows.append((t_start, last_test_end, last_test_end, n))
    return windows


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def _returns_for(
    df: pd.DataFrame,
    positions: pd.Series,
    cost_model: CostModel,
    sl: slice,
) -> pd.DataFrame:
    """Build the return frame for one parameter set over one slice."""
    window = df.iloc[sl]
    return metrics.compute_returns(
        close=window["Close"],
        positions=positions.iloc[sl],
        cost_fraction=cost_model.cost_fraction(window["Close"]),
        tradable=window.get("tradable"),
    )


def _evaluate(
    df: pd.DataFrame,
    positions: pd.Series,
    cost_model: CostModel,
    sl: slice,
    ppy: int,
    risk_free: float,
) -> tuple[pd.DataFrame, dict]:
    """Full statistics for one parameter set over one slice of the series."""
    returns = _returns_for(df, positions, cost_model, sl)
    return returns, metrics.performance(returns, ppy=ppy, risk_free=risk_free)


def _selection_score(n_trades: int, net_sharpe: float, min_trades: int) -> float:
    """Rank a candidate parameter set on its TRAIN window.

    Net Sharpe, not gross: selecting on gross would pick the highest-turnover
    parameters every time and then hand them to the test window to pay for.
    Candidates that barely traded are rejected outright -- a 3-trade Sharpe of
    4.0 is not information.

    Note that because selection uses NET Sharpe, the cost model legitimately
    influences which parameters win. Under option-level frictions the search
    will favour lower-turnover settings, which is the realistic behaviour: you
    would not run a 200-trade-a-year rule through a 500 bps round trip.
    """
    if n_trades < min_trades:
        return float("-inf")
    if not np.isfinite(net_sharpe):
        return float("-inf")
    return float(net_sharpe)


def run_walk_forward(
    df: pd.DataFrame,
    signal_def: signals.SignalDef,
    cost_model: CostModel,
    cfg,
    ticker: str = "?",
    param_grid: dict[str, list] | None = None,
) -> BacktestResult:
    """Run the full walk-forward for one ticker/signal/instrument combination."""
    wf = cfg.walk_forward
    ppy = cfg.trading_days_per_year
    rf = cfg.risk_free_rate

    result = BacktestResult(
        ticker=ticker,
        signal=signal_def.name,
        instrument=cost_model.instrument,
        cost_description=cost_model.describe(df["Close"]),
        is_cost_proxy=cost_model.is_proxy,
    )

    n = len(df)
    windows = make_windows(n, wf.train_days, wf.test_days)
    if not windows:
        need = wf.train_days + wf.test_days
        result.verdict_code = "INSUFFICIENT_HISTORY"
        result.verdict = (
            f"Not enough history to walk forward: {n} bars available, "
            f"{need} needed for one {wf.train_days}/{wf.test_days} "
            "train/test window. No out-of-sample result was produced. "
            "Nothing here should be interpreted as evidence either way."
        )
        result.warnings.append(f"only {n} bars; need >= {need}")
        return result

    # Pre-compute every candidate's positions once over the full series.
    combos = signals.valid_combos(signal_def)
    if param_grid is not None:
        combos = signals.expand_grid(param_grid)
    if not combos:
        combos = [signal_def.default_params()]

    candidates: list[tuple[dict, pd.Series]] = []
    for params in combos:
        try:
            pos = signal_def.func(df, **params)
        except Exception as exc:
            logger.debug("%s %s rejected params %s (%s)",
                         ticker, signal_def.name, params, exc)
            continue
        candidates.append((params, pos.astype("float64")))

    if not candidates:
        result.verdict_code = "NO_VALID_PARAMS"
        result.verdict = "No parameter combination could be evaluated."
        return result

    oos_frames: list[pd.DataFrame] = []
    for i, (tr0, tr1, te0, te1) in enumerate(windows):
        train_slice = slice(tr0, tr1)
        test_slice = slice(te0, te1)

        # Search the grid on the TRAINING window using only the two numbers
        # selection needs; full statistics are computed once, for the winner.
        best_params, best_score = None, float("-inf")
        for params, pos in candidates:
            train_returns = _returns_for(df, pos, cost_model, train_slice)
            n_tr, shp = metrics.quick_score(train_returns, ppy, rf)
            score = _selection_score(n_tr, shp, wf.min_train_trades)
            if score > best_score:
                best_params, best_score = params, score

        if best_params is None or best_score == float("-inf"):
            # Nothing traded enough on this training window to justify a choice.
            # Fall back to the signal's documented defaults rather than picking
            # the least-bad noise fit, and say so.
            best_params = signal_def.default_params()
            pos = dict(
                (tuple(sorted(p.items())), s) for p, s in candidates
            ).get(tuple(sorted(best_params.items())))
            if pos is None:
                pos = candidates[0][1]
                best_params = candidates[0][0]
            result.warnings.append(
                f"window {i}: no candidate met the {wf.min_train_trades}-trade "
                "minimum in training; fell back to default parameters"
            )
        else:
            pos = next(p for par, p in candidates if par == best_params)

        _, best_train_stats = _evaluate(
            df, pos, cost_model, train_slice, ppy, rf
        )
        test_returns, test_stats = _evaluate(
            df, pos, cost_model, test_slice, ppy, rf
        )
        oos_frames.append(test_returns)

        result.periods.append(
            PeriodResult(
                index=i,
                train_start=df.index[tr0],
                train_end=df.index[tr1 - 1],
                test_start=df.index[te0],
                test_end=df.index[te1 - 1],
                params=best_params,
                train_stats=best_train_stats,
                test_stats=test_stats,
                n_candidates=len(candidates),
            )
        )

    oos = pd.concat(oos_frames).sort_index()
    # Test windows are non-overlapping, but the trailing partial window can
    # re-cover a bar or two; keep the first occurrence so nothing is double counted.
    oos = oos[~oos.index.duplicated(keep="first")]
    # Rebuild the equity curves across the stitched series.
    oos["equity_gross"] = (1.0 + oos["gross"]).cumprod()
    oos["equity_net"] = (1.0 + oos["net"]).cumprod()

    result.oos_returns = oos
    result.aggregate = metrics.performance(oos, ppy=ppy, risk_free=rf)
    result.stability = summarize_stability(result.periods)
    code, sentence = metrics.edge_verdict(result.aggregate)
    result.verdict_code, result.verdict = code, sentence

    if not result.params_stable():
        result.warnings.append(
            "parameters changed between walk-forward windows -- the aggregate "
            "figure blends several different parameter sets"
        )
    return result


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
def summarize_stability(periods: list[PeriodResult]) -> dict:
    """Measure how consistent the out-of-sample result is across sub-periods.

    An aggregate Sharpe of 0.8 built from windows of [2.9, -0.4, -0.2, 0.3] is
    not the same finding as one built from [0.7, 0.9, 0.8, 0.8], and reporting
    only the average would hide the difference. This is the overfitting guard
    the brief asks for, applied to the reporting itself.
    """
    if not periods:
        return {}

    net = np.array(
        [p.test_stats.get("net_sharpe", np.nan) for p in periods], dtype="float64"
    )
    gross = np.array(
        [p.test_stats.get("gross_sharpe", np.nan) for p in periods], dtype="float64"
    )
    net_ret = np.array(
        [p.test_stats.get("net_annual_return", np.nan) for p in periods],
        dtype="float64",
    )
    train_net = np.array(
        [p.train_stats.get("net_sharpe", np.nan) for p in periods], dtype="float64"
    )

    finite = net[np.isfinite(net)]
    out = {
        "n_periods": len(periods),
        "net_sharpe_mean": float(np.nanmean(net)) if finite.size else float("nan"),
        "net_sharpe_std": float(np.nanstd(net, ddof=1)) if finite.size > 1 else float("nan"),
        "net_sharpe_min": float(np.nanmin(net)) if finite.size else float("nan"),
        "net_sharpe_max": float(np.nanmax(net)) if finite.size else float("nan"),
        "gross_sharpe_mean": float(np.nanmean(gross)) if np.isfinite(gross).any() else float("nan"),
        "positive_periods": int(np.nansum(net_ret > 0)),
        "negative_periods": int(np.nansum(net_ret <= 0)),
    }
    out["pct_positive"] = (
        out["positive_periods"] / len(periods) * 100.0 if periods else float("nan")
    )

    # Train-minus-test Sharpe: the overfitting tax, made explicit.
    both = np.isfinite(train_net) & np.isfinite(net)
    out["train_minus_test_sharpe"] = (
        float(np.mean(train_net[both] - net[both])) if both.any() else float("nan")
    )

    flags = []
    if finite.size > 1 and np.isfinite(out["net_sharpe_std"]):
        if out["net_sharpe_std"] > 1.0:
            flags.append(
                "UNSTABLE: out-of-sample Sharpe varies by more than 1.0 across "
                "sub-periods. The aggregate figure is an average over materially "
                "different regimes."
            )
    if out["negative_periods"] and out["positive_periods"]:
        if out["pct_positive"] < 50.0:
            flags.append(
                f"MAJORITY NEGATIVE: only {out['pct_positive']:.0f}% of "
                "out-of-sample windows were profitable net of costs."
            )
    if np.isfinite(out["train_minus_test_sharpe"]) and out["train_minus_test_sharpe"] > 1.0:
        flags.append(
            f"OVERFIT GAP: training Sharpe exceeds test Sharpe by "
            f"{out['train_minus_test_sharpe']:.2f} on average. The parameter "
            "search is fitting the training window rather than finding a "
            "durable effect."
        )
    out["flags"] = flags
    return out
