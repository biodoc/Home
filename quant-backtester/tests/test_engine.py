"""Walk-forward engine behaviour, including the edge cases the brief names."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_frame
from qsb import engine, signals
from qsb.config import Config
from qsb.costs import CostModel


@pytest.fixture
def stock_costs(cfg):
    return CostModel.from_config(cfg, "stock")


# -- window construction ----------------------------------------------------
def test_windows_roll_forward_without_overlapping_test_periods():
    windows = engine.make_windows(1000, train=378, test=126)
    assert windows
    test_spans = [(t0, t1) for _, _, t0, t1 in windows]
    for (a0, a1), (b0, b1) in zip(test_spans, test_spans[1:]):
        assert a1 <= b0, "test windows must not overlap or bars are double counted"


def test_every_test_window_follows_its_own_training_window():
    for tr0, tr1, te0, te1 in engine.make_windows(1000, 378, 126):
        assert tr1 == te0, "the test window must start where training ended"
        assert tr1 - tr0 == 378
        assert te0 < te1


def test_no_windows_when_history_is_too_short():
    assert engine.make_windows(100, train=378, test=126) == []


def test_trailing_partial_window_is_kept_when_substantial():
    windows = engine.make_windows(600, train=378, test=126)
    assert windows[-1][3] == 600, "the last months of data should not be discarded"


def test_trailing_stub_is_discarded_when_trivial():
    windows = engine.make_windows(505, train=378, test=126)
    assert windows[-1][3] == 504, "a 1-bar stub is not a test window"


# -- end to end -------------------------------------------------------------
def test_walk_forward_produces_out_of_sample_results(random_walk_frame, cfg, stock_costs):
    result = engine.run_walk_forward(
        random_walk_frame, signals.get("mean_reversion"), stock_costs, cfg, "TEST"
    )
    assert result.ok
    assert len(result.periods) > 1
    assert result.aggregate["bars"] > 0
    assert result.verdict_code in {
        "SURVIVES_COSTS", "COSTS_KILL_IT", "NO_GROSS_EDGE", "NO_TRADES"
    }


def test_reported_results_cover_only_test_windows(random_walk_frame, cfg, stock_costs):
    """The aggregate must never include a bar the parameters were fitted on."""
    result = engine.run_walk_forward(
        random_walk_frame, signals.get("momentum_roc"), stock_costs, cfg, "TEST"
    )
    first_test_start = result.periods[0].test_start
    assert result.oos_returns.index.min() >= first_test_start
    # Training starts before the first test bar, and none of it is reported.
    assert result.periods[0].train_start < first_test_start


def test_no_bar_is_counted_twice_in_the_stitched_series(random_walk_frame, cfg, stock_costs):
    result = engine.run_walk_forward(
        random_walk_frame, signals.get("ma_crossover"), stock_costs, cfg, "TEST"
    )
    assert not result.oos_returns.index.duplicated().any()
    assert result.oos_returns.index.is_monotonic_increasing


def test_parameters_are_frozen_within_each_test_window(random_walk_frame, cfg, stock_costs):
    result = engine.run_walk_forward(
        random_walk_frame, signals.get("mean_reversion"), stock_costs, cfg, "TEST"
    )
    for period in result.periods:
        assert isinstance(period.params, dict) and period.params
        assert set(period.params) == set(signals.get("mean_reversion").param_grid)


def test_insufficient_history_is_reported_not_silently_fitted(cfg, stock_costs):
    """Too little data must produce a clear refusal, never an in-sample result."""
    short = make_frame(np.linspace(10.0, 12.0, 100))
    result = engine.run_walk_forward(
        short, signals.get("mean_reversion"), stock_costs, cfg, "SHORT"
    )
    assert not result.ok
    assert result.verdict_code == "INSUFFICIENT_HISTORY"
    assert "No out-of-sample result was produced" in result.verdict
    assert result.aggregate == {}


def test_a_signal_that_never_triggers_is_handled(cfg, stock_costs):
    """Zero trades must be a reported verdict, not a division by zero."""
    flat = make_frame([25.0] * 400)
    result = engine.run_walk_forward(
        flat, signals.get("mean_reversion"), stock_costs, cfg, "FLAT"
    )
    assert result.ok
    assert result.aggregate["n_trades"] == 0
    assert result.verdict_code == "NO_TRADES"
    assert np.isnan(result.aggregate["gross_hit_rate"])


def test_missing_days_do_not_break_the_engine(cfg, stock_costs):
    """A series with holes must still backtest; gaps are spanned, not filled."""
    rng = np.random.default_rng(11)
    df = make_frame(40 * np.exp(np.cumsum(rng.normal(0, 0.02, 600))))
    df = df.drop(df.index[100:140])           # a two-month hole
    df = df.drop(df.index[::17])              # scattered single-day holes
    result = engine.run_walk_forward(
        df, signals.get("momentum_roc"), stock_costs, cfg, "HOLES"
    )
    assert result.ok
    assert np.isfinite(result.aggregate["net_annual_return"])


def test_thin_liquidity_forces_positions_flat(cfg, stock_costs):
    rng = np.random.default_rng(5)
    df = make_frame(30 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))), volume=1e6)
    df["dollar_volume"] = df["Close"] * df["Volume"]
    df["tradable"] = False                     # nothing is tradable
    result = engine.run_walk_forward(
        df, signals.get("mean_reversion"), stock_costs, cfg, "THIN"
    )
    assert result.aggregate["n_trades"] == 0
    assert (result.oos_returns["held"] == 0).all()


def test_costs_reduce_net_performance_for_identical_positions(random_walk_frame, cfg):
    """With the parameter search pinned, option frictions must cost more.

    The grid is collapsed to a single combination so both runs trade exactly the
    same positions. That isolates the cost model, which is the thing under test.
    """
    fixed = {"lookback": [20], "entry_z": [1.5], "exit_z": [0.5]}
    runs = {}
    for instrument in ("stock", "option"):
        runs[instrument] = engine.run_walk_forward(
            random_walk_frame, signals.get("mean_reversion"),
            CostModel.from_config(cfg, instrument), cfg, "T", param_grid=fixed,
        )

    stock, option = runs["stock"], runs["option"]
    assert stock.aggregate["gross_annual_return"] == pytest.approx(
        option.aggregate["gross_annual_return"]
    ), "gross performance cannot depend on the cost model"
    assert option.aggregate["total_cost"] > stock.aggregate["total_cost"]
    assert option.aggregate["net_annual_return"] < stock.aggregate["net_annual_return"]
    assert option.is_cost_proxy
    assert not stock.is_cost_proxy


def test_cost_regime_may_change_which_parameters_win(random_walk_frame, cfg):
    """Selection uses NET Sharpe, so frictions feed back into parameter choice.

    This is intended: under 500 bps round trips the search should prefer
    lower-turnover settings. The test documents the behaviour rather than
    asserting a particular winner, since which parameters win is data-dependent.
    """
    results = {
        instrument: engine.run_walk_forward(
            random_walk_frame, signals.get("mean_reversion"),
            CostModel.from_config(cfg, instrument), cfg, "T",
        )
        for instrument in ("stock", "option")
    }
    # Gross turnover under the option regime should not exceed the stock regime
    # by a wide margin -- the cost-aware search has no reason to churn more.
    assert results["option"].aggregate["turnover_annual"] <= (
        results["stock"].aggregate["turnover_annual"] * 1.5 + 1.0
    )


def test_a_random_walk_does_not_yield_a_surviving_edge(random_walk_frame, cfg, stock_costs):
    """The framework must not manufacture an edge out of noise.

    This is the acid test. Data with no exploitable structure, run through a
    parameter search across five signals, must not come out the other side
    looking profitable after costs. If it does, the walk-forward split is
    leaking.
    """
    survived = []
    for name in signals.available():
        result = engine.run_walk_forward(
            random_walk_frame, signals.get(name), stock_costs, cfg, "NOISE"
        )
        if result.verdict_code == "SURVIVES_COSTS":
            survived.append((name, result.aggregate["net_sharpe"]))

    assert not [s for s in survived if s[1] > 1.0], (
        f"a random walk produced a high-Sharpe 'edge': {survived}"
    )


# -- stability --------------------------------------------------------------
def test_stability_flags_wildly_varying_periods():
    periods = [
        engine.PeriodResult(
            index=i, train_start=None, train_end=None,
            test_start=None, test_end=None, params={},
            train_stats={"net_sharpe": 2.0},
            test_stats={"net_sharpe": s, "net_annual_return": s / 10},
        )
        for i, s in enumerate([3.0, -2.0, 2.5, -1.8])
    ]
    stability = engine.summarize_stability(periods)
    assert stability["net_sharpe_std"] > 1.0
    assert any("UNSTABLE" in f for f in stability["flags"])


def test_stability_flags_a_train_test_gap():
    periods = [
        engine.PeriodResult(
            index=i, train_start=None, train_end=None,
            test_start=None, test_end=None, params={},
            train_stats={"net_sharpe": 2.5},
            test_stats={"net_sharpe": 0.1, "net_annual_return": 0.01},
        )
        for i in range(4)
    ]
    stability = engine.summarize_stability(periods)
    assert stability["train_minus_test_sharpe"] == pytest.approx(2.4)
    assert any("OVERFIT GAP" in f for f in stability["flags"])


def test_stability_flags_majority_negative_windows():
    periods = [
        engine.PeriodResult(
            index=i, train_start=None, train_end=None,
            test_start=None, test_end=None, params={},
            train_stats={"net_sharpe": 0.5},
            test_stats={"net_sharpe": s, "net_annual_return": r},
        )
        for i, (s, r) in enumerate([(0.4, 0.05), (-0.3, -0.04), (-0.2, -0.03),
                                    (-0.1, -0.01)])
    ]
    stability = engine.summarize_stability(periods)
    assert any("MAJORITY NEGATIVE" in f for f in stability["flags"])


def test_stability_of_no_periods_is_empty():
    assert engine.summarize_stability([]) == {}


def test_unstable_parameters_produce_a_warning(random_walk_frame, cfg, stock_costs):
    result = engine.run_walk_forward(
        random_walk_frame, signals.get("mean_reversion"), stock_costs, cfg, "T"
    )
    if not result.params_stable():
        assert any("parameters changed" in w for w in result.warnings)
