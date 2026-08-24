"""Return construction, trade extraction, and performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsb import metrics


def _series(values, start="2020-01-01"):
    return pd.Series(
        values, index=pd.bdate_range(start, periods=len(values)), dtype="float64"
    )


# -- the execution shift ----------------------------------------------------
def test_position_is_held_one_bar_after_the_decision():
    close = _series([100.0, 110.0, 120.0])
    pos = _series([1.0, 0.0, 0.0])
    out = metrics.compute_returns(close, pos, 0.0)
    # Decided on bar 0, earned over bar 1.
    assert out["held"].tolist() == [0.0, 1.0, 0.0]
    assert out["gross"].iloc[1] == pytest.approx(0.10)
    assert out["gross"].iloc[0] == 0.0


def test_no_lookahead_bias_in_return_construction():
    """A signal that 'knows' the current bar's return must NOT be profitable.

    `perfect` is the sign of each bar's own return -- an oracle. Because the
    engine shifts positions by one bar, the oracle only ever gets to act on
    information it already had, so it must not print a suspiciously perfect
    equity curve. If this test fails, the one-bar shift has been lost and every
    result the framework produces is worthless.
    """
    rng = np.random.default_rng(7)
    close = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))
    perfect = np.sign(close.pct_change(fill_method=None).fillna(0.0))

    out = metrics.compute_returns(close, perfect, 0.0)
    hit_rate = (out["gross"] > 0).sum() / (out["gross"] != 0).sum()
    assert hit_rate < 0.65, "the oracle is being paid for information it lacked"


def test_untradable_days_are_forced_flat():
    close = _series([100.0, 110.0, 120.0, 130.0])
    pos = _series([1.0, 1.0, 1.0, 1.0])
    tradable = pd.Series([True, True, False, True], index=close.index)
    out = metrics.compute_returns(close, pos, 0.0, tradable=tradable)
    assert out["position"].iloc[2] == 0.0
    assert out["held"].iloc[3] == 0.0, "no fill on the day after a thin session"


# -- costs ------------------------------------------------------------------
def test_costs_are_charged_on_turnover_only():
    close = _series([100.0] * 5)
    pos = _series([1.0, 1.0, 1.0, 0.0, 0.0])
    out = metrics.compute_returns(close, pos, 0.001)
    # One entry (bar 1) and one exit (bar 4) -> exactly two charges.
    assert (out["cost"] > 0).sum() == 2
    assert out["cost"].sum() == pytest.approx(0.002)


def test_a_direction_flip_is_charged_twice():
    close = _series([100.0] * 4)
    pos = _series([1.0, -1.0, -1.0, -1.0])
    out = metrics.compute_returns(close, pos, 0.001)
    # held goes 0 -> 1 -> -1: the flip moves two units of notional.
    assert out["turnover"].max() == pytest.approx(2.0)


def test_net_is_never_above_gross_when_costs_are_positive():
    rng = np.random.default_rng(3)
    close = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200))))
    pos = _series(rng.choice([-1.0, 0.0, 1.0], 200), start="2020-01-01")
    pos.index = close.index
    out = metrics.compute_returns(close, pos, 0.001)
    assert (out["net"] <= out["gross"] + 1e-12).all()


def test_per_bar_cost_series_is_applied_bar_by_bar():
    close = _series([100.0] * 4)
    pos = _series([1.0, 1.0, 0.0, 0.0])
    varying = pd.Series([0.01, 0.02, 0.03, 0.04], index=close.index)
    out = metrics.compute_returns(close, pos, varying)
    assert out["cost"].iloc[1] == pytest.approx(0.02)  # entry at bar 1's rate
    assert out["cost"].iloc[3] == pytest.approx(0.04)  # exit at bar 3's rate


# -- trades -----------------------------------------------------------------
def test_trades_are_maximal_runs_of_constant_position():
    close = _series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    pos = _series([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    trades = metrics.extract_trades(metrics.compute_returns(close, pos, 0.0))
    assert len(trades) == 2
    assert trades["direction"].tolist() == [1, 1]


def test_short_trades_profit_when_price_falls():
    close = _series([100.0, 90.0, 80.0])
    pos = _series([-1.0, -1.0, 0.0])
    trades = metrics.extract_trades(metrics.compute_returns(close, pos, 0.0))
    assert len(trades) == 1
    assert trades["direction"].iloc[0] == -1
    assert trades["gross_return"].iloc[0] > 0


def test_a_signal_that_never_triggers_yields_no_trades():
    close = _series([100.0, 101.0, 102.0])
    out = metrics.compute_returns(close, _series([0.0, 0.0, 0.0]), 0.001)
    trades = metrics.extract_trades(out)
    assert trades.empty
    assert out["cost"].sum() == 0.0
    assert out["net"].sum() == 0.0


def test_trade_net_return_includes_the_unwind_cost():
    close = _series([100.0, 100.0, 100.0])
    pos = _series([1.0, 0.0, 0.0])
    trades = metrics.extract_trades(metrics.compute_returns(close, pos, 0.001))
    # Flat price, so the entire trade P&L is the round trip's cost.
    assert trades["net_return"].iloc[0] == pytest.approx(-0.002, abs=1e-5)


# -- statistics -------------------------------------------------------------
def test_max_drawdown_matches_hand_calculation():
    assert metrics.max_drawdown(pd.Series([1.0, 1.5, 0.75, 1.2])) == pytest.approx(-0.5)


def test_max_drawdown_of_a_rising_curve_is_zero():
    assert metrics.max_drawdown(pd.Series([1.0, 1.1, 1.2])) == pytest.approx(0.0)


def test_sharpe_of_a_constant_return_series_is_undefined():
    assert np.isnan(metrics.sharpe(pd.Series([0.01] * 50)))


def test_sharpe_scales_with_the_annualization_factor():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.01] * 20)
    assert metrics.sharpe(r, ppy=252) == pytest.approx(
        metrics.sharpe(r, ppy=63) * np.sqrt(4), rel=1e-6
    )


def test_sharpe_needs_at_least_two_observations():
    assert np.isnan(metrics.sharpe(pd.Series([0.01])))


def test_annualized_return_of_a_doubling_over_one_year():
    r = pd.Series([2.0 ** (1 / 252) - 1.0] * 252)
    assert metrics.annualized_return(r, ppy=252) == pytest.approx(1.0, rel=1e-6)


def test_performance_always_reports_gross_and_net():
    close = _series([100.0, 101.0, 102.0, 101.0, 103.0])
    stats = metrics.performance(
        metrics.compute_returns(close, _series([1.0] * 5), 0.001)
    )
    for key in ("total_return", "annual_return", "sharpe", "max_drawdown",
                "hit_rate", "avg_win", "avg_loss"):
        assert f"gross_{key}" in stats, f"gross_{key} missing"
        assert f"net_{key}" in stats, f"net_{key} missing"


def test_performance_on_zero_trades_is_nan_not_a_crash():
    close = _series([100.0, 101.0, 102.0])
    stats = metrics.performance(metrics.compute_returns(close, _series([0.0] * 3), 0.001))
    assert stats["n_trades"] == 0
    assert np.isnan(stats["gross_hit_rate"])
    assert np.isnan(stats["net_hit_rate"])


def test_breakeven_edge_is_expressed_per_unit_traded():
    close = _series([100.0, 110.0, 110.0])
    pos = _series([1.0, 0.0, 0.0])
    stats = metrics.performance(metrics.compute_returns(close, pos, 0.001))
    # 10% gross gain over 2.0 units of turnover = 500 bps per unit.
    assert stats["gross_edge_bps_per_unit_traded"] == pytest.approx(500.0, rel=1e-3)
    assert stats["assumed_cost_bps_per_unit_traded"] == pytest.approx(10.0)


# -- verdicts ---------------------------------------------------------------
def test_verdict_reports_no_trades():
    close = _series([100.0, 101.0, 102.0])
    stats = metrics.performance(metrics.compute_returns(close, _series([0.0] * 3), 0.0))
    assert metrics.edge_verdict(stats)[0] == "NO_TRADES"


def test_verdict_flags_a_gross_edge_destroyed_by_costs():
    """The headline case the brief asks to be called out explicitly."""
    # Price zig-zags up and down; the position is held only across the up legs,
    # so the gross edge is real. It just requires a round trip every two bars.
    close = _series([100.0, 101.0] * 50)
    pos = _series([1.0, 0.0] * 50)
    pos.index = close.index
    stats = metrics.performance(metrics.compute_returns(close, pos, 0.0))
    assert stats["gross_annual_return"] > 0, "fixture should have a gross edge"

    # Now charge a cost far above the per-trade edge: gross positive, net negative.
    stats = metrics.performance(metrics.compute_returns(close, pos, 0.02))
    code, message = metrics.edge_verdict(stats)
    assert code == "COSTS_KILL_IT"
    assert "EDGE DOES NOT SURVIVE COSTS" in message
    assert "bps per unit traded" in message


def test_verdict_flags_the_absence_of_any_gross_edge():
    close = _series(np.linspace(100.0, 60.0, 100))
    stats = metrics.performance(metrics.compute_returns(close, _series([1.0] * 100), 0.0))
    code, message = metrics.edge_verdict(stats)
    assert code == "NO_GROSS_EDGE"
    assert "costs are not the problem" in message


def test_verdict_hedges_even_when_a_signal_survives():
    close = _series(np.linspace(100.0, 200.0, 300))
    stats = metrics.performance(metrics.compute_returns(close, _series([1.0] * 300), 1e-5))
    code, message = metrics.edge_verdict(stats)
    assert code == "SURVIVES_COSTS"
    assert "not a validated edge" in message
