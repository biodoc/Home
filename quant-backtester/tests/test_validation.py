"""Broader-universe validation and the overfit judgements it produces."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from qsb import validation


@dataclass
class FakeResult:
    """A stand-in for BacktestResult carrying only what `compare` reads."""

    ticker: str
    signal: str = "mean_reversion"
    net_sharpe: float = 0.0
    gross_sharpe: float = 0.0
    net_annual_return: float = 0.0
    gross_annual_return: float = 0.0
    verdict_code: str = "SURVIVES_COSTS"
    ok: bool = True

    @property
    def aggregate(self):
        return {
            "n_trades": 20,
            "net_sharpe": self.net_sharpe,
            "gross_sharpe": self.gross_sharpe,
            "net_annual_return": self.net_annual_return,
            "gross_annual_return": self.gross_annual_return,
        }


def make(tickers, sharpe, ret=None):
    ret = sharpe / 10 if ret is None else ret
    return [
        FakeResult(ticker=t, net_sharpe=sharpe, gross_sharpe=sharpe + 0.2,
                   net_annual_return=ret, gross_annual_return=ret + 0.02)
        for t in tickers
    ]


DEV = ["UAMY", "SMR", "BBAI", "AEM"]
VAL = ["AAPL", "JNJ", "XOM", "KO"]


def test_strong_on_both_sets_is_confirmed():
    cmp = validation.compare("s", make(DEV, 1.0), make(VAL, 0.9))
    assert cmp.code == "CONFIRMED"
    assert "HOLDS UP OUT OF SAMPLE" in cmp.message


def test_confirmation_still_refuses_to_call_it_proof():
    cmp = validation.compare("s", make(DEV, 1.0), make(VAL, 0.9))
    assert "long way short of proof" in cmp.message


def test_positive_on_dev_and_flat_on_holdout_is_overfit():
    cmp = validation.compare("s", make(DEV, 1.2), make(VAL, -0.05, ret=-0.005))
    assert cmp.code == "OVERFIT"
    assert "LIKELY OVERFIT" in cmp.message


def test_positive_on_dev_and_negative_on_holdout_is_reversed():
    cmp = validation.compare("s", make(DEV, 1.2), make(VAL, -0.8, ret=-0.08))
    assert cmp.code == "REVERSED"
    assert "LIKELY CURVE FIT" in cmp.message
    assert "property of those specific tickers" in cmp.message


def test_a_weakened_but_surviving_signal_is_labelled_as_such():
    cmp = validation.compare("s", make(DEV, 2.0), make(VAL, 0.35))
    assert cmp.code == "WEAKENED"
    assert "Size any expectation to the hold-out number" in cmp.message


def test_no_development_edge_means_nothing_to_validate():
    cmp = validation.compare("s", make(DEV, -0.5, ret=-0.05), make(VAL, 0.8))
    assert cmp.code == "NO_DEV_EDGE"
    assert "nothing to validate" in cmp.message


def test_missing_holdout_results_produce_no_conclusion():
    cmp = validation.compare("s", make(DEV, 1.0), [])
    assert cmp.code == "INSUFFICIENT_DATA"
    assert "No conclusion can be drawn" in cmp.message


def test_the_two_sets_are_never_pooled_into_one_number():
    """The comparison must keep both sets addressable and separate."""
    cmp = validation.compare("s", make(DEV, 1.0), make(VAL, 0.2))
    assert cmp.dev_stats["n_usable"] == 4
    assert cmp.val_stats["n_usable"] == 4
    assert set(cmp.dev_stats["tickers"]) == set(DEV)
    assert set(cmp.val_stats["tickers"]) == set(VAL)
    # No key anywhere holds a combined statistic.
    assert "combined" not in cmp.dev_stats and "combined" not in cmp.val_stats


def test_per_ticker_detail_is_retained_for_both_sets():
    cmp = validation.compare("s", make(DEV, 1.0), make(VAL, 0.5))
    assert set(cmp.dev_stats["per_ticker"]) == set(DEV)
    for detail in cmp.val_stats["per_ticker"].values():
        assert "net_sharpe" in detail and "gross_sharpe" in detail


def test_mixed_holdout_results_use_the_median_not_the_best():
    """One lucky hold-out name must not rescue a failing signal."""
    holdout = (
        make(["AAPL"], 3.0)          # one spectacular result
        + make(["JNJ", "XOM", "KO"], -0.4, ret=-0.04)  # three failures
    )
    cmp = validation.compare("s", make(DEV, 1.2), holdout)
    assert cmp.code in {"OVERFIT", "REVERSED"}


def test_formatted_output_shows_both_sets_separately():
    cmp = validation.compare("s", make(DEV, 1.0), make(VAL, 0.9))
    text = validation.format_comparison(cmp)
    assert "DEVELOPMENT SET" in text
    assert "HOLD-OUT SET" in text
    assert "never combined into" in text
    for ticker in DEV + VAL:
        assert ticker in text
