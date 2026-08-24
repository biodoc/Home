"""Reporting rules that the brief makes non-negotiable.

Two properties are enforced structurally rather than by convention:
gross and net always appear together, and an unprofitable signal is described
in words, not only implied by its numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_frame
from qsb import engine, metrics, reporting, signals
from qsb.costs import CostModel


@pytest.fixture
def result(random_walk_frame, cfg):
    from qsb import data

    df = data.annotate_liquidity(random_walk_frame, 0.0)
    res = engine.run_walk_forward(
        df, signals.get("mean_reversion"),
        CostModel.from_config(cfg, "stock"), cfg, "TEST",
    )
    res.liquidity = data.liquidity_profile(df, 0.0)
    return res


def test_report_shows_gross_and_net_together(result):
    text = reporting.format_result(result)
    assert "GROSS" in text and "NET" in text
    # Every headline metric appears with both columns on one line.
    for line in text.splitlines():
        if line.strip().startswith("Sharpe ratio"):
            numbers = [t for t in line.split() if t.replace("-", "").replace(".", "").isdigit()]
            assert len(numbers) >= 2, "Sharpe must be shown gross AND net"


def test_report_states_the_cost_hurdle_explicitly(result):
    text = reporting.format_result(result)
    assert "COST HURDLE" in text
    assert "bps per unit traded" in text


def test_report_includes_the_per_period_breakdown(result):
    """Aggregates can hide an unstable signal; the breakdown must be shown."""
    text = reporting.format_result(result)
    assert "PER-PERIOD WALK-FORWARD BREAKDOWN" in text
    assert "out-of-sample test windows only" in text
    for period in result.periods:
        assert period.test_start.strftime("%Y-%m-%d") in text


def test_report_shows_parameters_chosen_for_each_window(result):
    text = reporting.format_result(result)
    assert "parameters" in text
    assert "lookback=" in text


def test_report_carries_the_tax_disclaimer(result):
    text = reporting.format_result(result)
    assert "TAXES ARE MODELLED AS ZERO" in text
    assert "not financial advice" in text


def _squash(text: str) -> str:
    """Collapse whitespace so word-wrapped prose can be matched verbatim."""
    return " ".join(text.split())


def test_report_states_a_verdict_in_words(result):
    text = reporting.format_result(result)
    assert "VERDICT:" in text
    # The verdict is word-wrapped in the report, so compare on squashed text.
    assert _squash(result.verdict) in _squash(text)
    # And it must be prose, not just a code.
    assert len(result.verdict.split()) > 8


def test_a_cost_killed_signal_says_so_in_words():
    """The brief's key requirement: do not merely imply it through numbers."""
    close = pd.Series(
        [100.0, 101.0] * 50, index=pd.bdate_range("2020-01-01", periods=100)
    )
    pos = pd.Series([1.0, 0.0] * 50, index=close.index)
    stats = metrics.performance(metrics.compute_returns(close, pos, 0.02))

    res = engine.BacktestResult(ticker="X", signal="s", instrument="stock")
    res.aggregate = stats
    res.oos_returns = metrics.compute_returns(close, pos, 0.02)
    res.verdict_code, res.verdict = metrics.edge_verdict(stats)

    text = reporting.format_result(res)
    assert "EDGE DOES NOT SURVIVE COSTS" in text
    assert "COSTS_KILL_IT" in text


def test_option_reports_carry_the_proxy_warning(random_walk_frame, cfg):
    res = engine.run_walk_forward(
        random_walk_frame, signals.get("mean_reversion"),
        CostModel.from_config(cfg, "option"), cfg, "OPT",
    )
    text = reporting.format_result(res)
    assert "PROXY" in text
    assert "not an options backtest" in text.lower()


def test_insufficient_history_report_does_not_show_performance(cfg):
    res = engine.run_walk_forward(
        make_frame(np.linspace(10.0, 12.0, 50)), signals.get("mean_reversion"),
        CostModel.from_config(cfg, "stock"), cfg, "SHORT",
    )
    text = reporting.format_result(res)
    assert "INSUFFICIENT_HISTORY" in text
    assert "AGGREGATE OUT-OF-SAMPLE PERFORMANCE" not in text


def test_summary_lists_gross_and_net_for_every_row(result):
    text = reporting.format_summary([result, result])
    assert "gross Shp" in text and "net Shp" in text
    assert "gross ret" in text and "net ret" in text


def test_summary_calls_out_a_clean_sweep_of_failures():
    res = engine.BacktestResult(ticker="X", signal="s", instrument="stock")
    res.verdict_code = "COSTS_KILL_IT"
    res.aggregate = {"n_trades": 5}
    text = reporting.format_summary([res])
    assert "No combination survived costs" in text
    assert "frictions, not the ideas" in text


def test_results_export_to_a_flat_frame(result):
    frame = reporting.results_to_frame([result])
    assert len(frame) == 1
    for column in ("ticker", "signal", "verdict", "gross_sharpe", "net_sharpe"):
        assert column in frame.columns


def test_export_never_contains_net_without_gross(result):
    frame = reporting.results_to_frame([result])
    net_cols = {c for c in frame.columns if c.startswith("net_")}
    for col in net_cols:
        assert f"gross_{col[4:]}" in frame.columns, f"{col} has no gross counterpart"


def test_random_walk_report_renders(random_walk_frame):
    from qsb import diagnostics

    rep = diagnostics.random_walk_report(random_walk_frame["Close"], "TEST")
    text = reporting.format_random_walk(rep)
    assert "AUTOCORRELATION" in text
    assert "VARIANCE RATIO" in text
    assert "VR = 1 is a random walk" in text
    assert rep.verdict.split(".")[0] in text


def test_nan_values_render_as_na_not_a_crash():
    """A signal that never traded has undefined statistics; they must print."""
    close = pd.Series(
        [100.0] * 30, index=pd.bdate_range("2020-01-01", periods=30)
    )
    never = pd.Series(0.0, index=close.index)
    returns = metrics.compute_returns(close, never, 0.001)

    res = engine.BacktestResult(ticker="X", signal="s", instrument="stock")
    res.oos_returns = returns
    res.aggregate = metrics.performance(returns)
    res.verdict_code, res.verdict = metrics.edge_verdict(res.aggregate)

    assert res.aggregate["n_trades"] == 0
    assert np.isnan(res.aggregate["gross_hit_rate"])

    text = reporting.format_result(res)
    assert "n/a" in text, "undefined statistics must render, not crash or show 0"
    assert "NO_TRADES" in text
