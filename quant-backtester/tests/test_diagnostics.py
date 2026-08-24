"""Random-walk diagnostics, checked against processes with known properties.

The variance-ratio tests here are the strongest correctness evidence in the
suite: for an AR(1) process on returns with coefficient phi, theory gives
VR(2) = 1 + phi exactly. If the implementation is wrong, that number will not
come out right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsb import diagnostics


def ar1_returns(phi: float, n: int = 4000, seed: int = 0, sigma: float = 0.02):
    """Returns with a known first-order autocorrelation of `phi`."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    return pd.Series(r)


def prices_from(returns: pd.Series, start: float = 100.0) -> pd.Series:
    return start * np.exp(returns.cumsum())


# -- normal distribution helpers -------------------------------------------
def test_two_sided_p_matches_known_normal_quantiles():
    assert diagnostics.two_sided_p(1.959964) == pytest.approx(0.05, abs=1e-5)
    assert diagnostics.two_sided_p(0.0) == pytest.approx(1.0)
    assert diagnostics.two_sided_p(2.575829) == pytest.approx(0.01, abs=1e-5)


def test_binomial_guard_matches_hand_calculation():
    # P(X >= 1) for n=1, p=0.05
    assert diagnostics.binomial_sf(1, 1, 0.05) == pytest.approx(0.05)
    # Every trial significant is (0.05)^3
    assert diagnostics.binomial_sf(3, 3, 0.05) == pytest.approx(0.05**3)
    assert diagnostics.binomial_sf(0, 10, 0.05) == 1.0


# -- variance ratio ---------------------------------------------------------
def test_variance_ratio_of_a_random_walk_is_about_one():
    rng = np.random.default_rng(99)
    r = pd.Series(rng.normal(0, 0.02, 6000))
    vr = diagnostics.variance_ratio_test(r, q=2)
    assert vr.vr == pytest.approx(1.0, abs=0.1)
    assert vr.p_value > 0.05
    assert vr.interpretation == "consistent with a random walk"


def test_variance_ratio_recovers_the_theoretical_value_for_mean_reversion():
    """For AR(1) returns, VR(2) = 1 + phi. With phi = -0.4, VR(2) = 0.6."""
    vr = diagnostics.variance_ratio_test(ar1_returns(-0.4, seed=1), q=2)
    assert vr.vr == pytest.approx(0.6, abs=0.06)
    assert vr.p_value < 0.01
    assert vr.interpretation == "mean reverting"


def test_variance_ratio_recovers_the_theoretical_value_for_momentum():
    """With phi = +0.3, VR(2) = 1.3."""
    vr = diagnostics.variance_ratio_test(ar1_returns(0.3, seed=2), q=2)
    assert vr.vr == pytest.approx(1.3, abs=0.06)
    assert vr.p_value < 0.01
    assert vr.interpretation == "trending"


def test_variance_ratio_needs_enough_observations():
    vr = diagnostics.variance_ratio_test(pd.Series([0.01, 0.02]), q=8)
    assert np.isnan(vr.vr)
    assert vr.interpretation == "undefined"


def test_variance_ratio_of_a_constant_series_is_undefined():
    vr = diagnostics.variance_ratio_test(pd.Series([0.0] * 500), q=2)
    assert np.isnan(vr.vr)


# -- autocorrelation --------------------------------------------------------
def test_autocorrelation_detects_a_known_lag_one_relationship():
    acf = diagnostics.autocorrelation(ar1_returns(-0.4, seed=3), max_lag=5)
    lag1 = acf.loc[acf["lag"] == 1, "acf"].iloc[0]
    assert lag1 == pytest.approx(-0.4, abs=0.05)
    assert acf.loc[acf["lag"] == 1, "significant"].iloc[0]


def test_autocorrelation_of_white_noise_is_near_zero():
    rng = np.random.default_rng(4)
    acf = diagnostics.autocorrelation(pd.Series(rng.normal(0, 1, 5000)), max_lag=5)
    assert acf["acf"].abs().max() < 0.06


def test_autocorrelation_confidence_band_scales_with_sample_size():
    rng = np.random.default_rng(5)
    small = diagnostics.autocorrelation(pd.Series(rng.normal(0, 1, 100)), 3)
    large = diagnostics.autocorrelation(pd.Series(rng.normal(0, 1, 10000)), 3)
    assert small["conf95"].iloc[0] > large["conf95"].iloc[0]


def test_autocorrelation_of_a_tiny_series_is_empty_not_an_error():
    assert diagnostics.autocorrelation(pd.Series([0.01, 0.02]), 5).empty


# -- full report ------------------------------------------------------------
def test_report_calls_a_mean_reverting_series_a_deviation():
    prices = prices_from(ar1_returns(-0.4, seed=6))
    report = diagnostics.random_walk_report(prices, "MR")
    assert "DEVIATION DETECTED" in report.verdict
    assert "mean reverting" in report.verdict
    assert report.significant_tests > report.tests_run * 0.4


def test_report_does_not_claim_structure_in_a_random_walk():
    """Across several seeds, a random walk must not routinely be flagged.

    A 5% test fires on ~5% of null samples by construction, so this checks the
    rate, not any single draw.
    """
    flagged = 0
    for seed in range(12):
        rng = np.random.default_rng(1000 + seed)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 2500))))
        if "DEVIATION DETECTED" in diagnostics.random_walk_report(prices, "RW").verdict:
            flagged += 1
    assert flagged <= 3, f"{flagged}/12 random walks were called structured"


def test_report_always_warns_against_overreading_a_positive():
    prices = prices_from(ar1_returns(-0.5, seed=8))
    verdict = diagnostics.random_walk_report(prices, "MR").verdict
    assert "NOT a tradable edge" in verdict


def test_report_carries_the_multiple_testing_p_value():
    prices = prices_from(ar1_returns(0.0, seed=9))
    report = diagnostics.random_walk_report(prices, "RW")
    assert 0.0 <= report.chance_p_value <= 1.0
    assert report.tests_run == len(report.acf) + len(report.variance_ratios)


def test_report_handles_a_short_series_without_crashing():
    report = diagnostics.random_walk_report(pd.Series([10.0, 11.0, 12.0]), "TINY")
    assert report.verdict
