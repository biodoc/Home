"""
Random-walk diagnostics.

This module tests the premise the whole project rests on. If a ticker's returns
are indistinguishable from a random walk, then no signal computed from that
price series alone can have an edge, and any backtest that appears to show one
is measuring the parameter search rather than the market.

Two independent tests are run:

  AUTOCORRELATION -- are returns at lag k correlated with returns at lag 0?
    Under a random walk, all autocorrelations are zero and the sample estimates
    fall inside roughly +/- 1.96/sqrt(n).

  VARIANCE RATIO (Lo & MacKinlay, 1988) -- does variance scale linearly with
    the holding period? Under a random walk, the variance of q-period returns is
    q times the variance of 1-period returns, so VR(q) = 1.
      VR(q) > 1  ->  positive serial correlation (trending / momentum)
      VR(q) < 1  ->  negative serial correlation (mean reverting)
    The heteroskedasticity-robust z-statistic is used, because equity returns
    are emphatically not homoskedastic; the homoskedastic version would reject
    the random walk far too often on volatile small caps.

The normal CDF is computed from `math.erfc` so that scipy is not a dependency.

A caution that belongs next to every output of this module: these tests are run
on the same history the signals are fitted on, and running enough of them
guarantees some will look significant. A single p-value below 0.05 across four
horizons and ten lags is what you would expect from noise alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _norm_sf(x: float) -> float:
    """Upper-tail probability of the standard normal, via erfc."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def two_sided_p(z: float) -> float:
    """Two-sided p-value for a standard normal statistic."""
    if not np.isfinite(z):
        return float("nan")
    return float(2.0 * _norm_sf(abs(z)))


def binomial_sf(k: int, n: int, p: float = 0.05) -> float:
    """P(X >= k) for X ~ Binomial(n, p).

    Used as a multiple-testing guard. Running fourteen tests at the 5% level and
    finding two "significant" results is not a discovery -- it is arithmetic.
    This asks the only question that matters: is the NUMBER of rejections itself
    surprising for a series with no structure at all?

    The tests being counted are not independent (neighbouring ACF lags and
    nested variance-ratio horizons share data), so this is an approximation. It
    errs toward calling things chance, which is the right direction to err.
    """
    if n <= 0 or k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(
        sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k, n + 1))
    )


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------
def autocorrelation(returns: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    """Sample autocorrelation with approximate 95% random-walk bands."""
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(r)
    if n < 3:
        return pd.DataFrame(columns=["lag", "acf", "conf95", "significant"])

    band = 1.96 / math.sqrt(n)
    rows = []
    for lag in range(1, min(max_lag, n - 2) + 1):
        a, b = r.iloc[lag:], r.iloc[:-lag]
        if a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
            ac = float("nan")
        else:
            ac = float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])
        rows.append(
            {
                "lag": lag,
                "acf": ac,
                "conf95": band,
                "significant": bool(np.isfinite(ac) and abs(ac) > band),
            }
        )
    return pd.DataFrame(rows)


def ljung_box_stat(acf_values: np.ndarray, n: int) -> float:
    """Ljung-Box Q statistic (the chi-square p-value is deliberately omitted).

    Reported as a magnitude only. Turning Q into a p-value needs a chi-square
    CDF, which would mean a scipy dependency for one number.
    """
    a = np.asarray(acf_values, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0 or n <= a.size:
        return float("nan")
    lags = np.arange(1, a.size + 1)
    return float(n * (n + 2) * np.sum(a**2 / (n - lags)))


# ---------------------------------------------------------------------------
# Variance ratio
# ---------------------------------------------------------------------------
@dataclass
class VarianceRatio:
    q: int
    vr: float
    z: float
    p_value: float
    n: int

    @property
    def interpretation(self) -> str:
        if not np.isfinite(self.vr):
            return "undefined"
        if self.p_value >= 0.05 or not np.isfinite(self.p_value):
            return "consistent with a random walk"
        return "trending" if self.vr > 1.0 else "mean reverting"


def variance_ratio_test(returns: pd.Series, q: int) -> VarianceRatio:
    """Lo-MacKinlay overlapping variance ratio with a robust z-statistic.

    `returns` should be LOG returns; overlapping q-period log returns are simple
    rolling sums, which is what makes the estimator tractable.
    """
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype="float64")
    n = r.size
    if q < 2 or n < q + 2:
        return VarianceRatio(q=q, vr=float("nan"), z=float("nan"),
                             p_value=float("nan"), n=n)

    mu = r.mean()
    dev = r - mu

    # 1-period variance (unbiased)
    var_1 = np.sum(dev**2) / (n - 1)
    if var_1 <= 0:
        return VarianceRatio(q=q, vr=float("nan"), z=float("nan"),
                             p_value=float("nan"), n=n)

    # q-period variance from overlapping sums, with the Lo-MacKinlay
    # bias-correcting denominator m.
    q_sums = np.convolve(r, np.ones(q, dtype="float64"), mode="valid")
    m = q * (n - q + 1) * (1.0 - q / n)
    if m <= 0:
        return VarianceRatio(q=q, vr=float("nan"), z=float("nan"),
                             p_value=float("nan"), n=n)
    var_q = np.sum((q_sums - q * mu) ** 2) / m

    vr = float(var_q / var_1)

    # Heteroskedasticity-robust standard error.
    denom = float(np.sum(dev**2)) ** 2
    if denom <= 0:
        return VarianceRatio(q=q, vr=vr, z=float("nan"), p_value=float("nan"), n=n)

    phi = 0.0
    for j in range(1, q):
        num = float(np.sum((dev[j:] ** 2) * (dev[:-j] ** 2)))
        delta_j = num / denom
        weight = 2.0 * (q - j) / q
        phi += (weight**2) * delta_j

    if phi <= 0:
        return VarianceRatio(q=q, vr=vr, z=float("nan"), p_value=float("nan"), n=n)

    z = (vr - 1.0) / math.sqrt(phi)
    return VarianceRatio(q=q, vr=vr, z=float(z), p_value=two_sided_p(z), n=n)


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------
@dataclass
class RandomWalkReport:
    ticker: str
    n_observations: int
    acf: pd.DataFrame
    variance_ratios: list[VarianceRatio]
    ljung_box_q: float
    verdict: str = ""
    tests_run: int = 0
    significant_tests: int = 0
    chance_p_value: float = float("nan")
    notes: list[str] = field(default_factory=list)


def random_walk_report(
    close: pd.Series,
    ticker: str = "?",
    max_lag: int = 10,
    horizons: list[int] | None = None,
) -> RandomWalkReport:
    """Run both diagnostics on one price series and summarise the result."""
    horizons = horizons or [2, 4, 8, 16]

    simple = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    logret = np.log(close.astype("float64")).diff().replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    acf = autocorrelation(simple, max_lag=max_lag)
    vrs = [variance_ratio_test(logret, q) for q in horizons]
    q_stat = ljung_box_stat(acf["acf"].to_numpy() if len(acf) else np.array([]), len(simple))

    n_sig_acf = int(acf["significant"].sum()) if len(acf) else 0
    n_sig_vr = sum(1 for v in vrs if np.isfinite(v.p_value) and v.p_value < 0.05)
    tests_run = len(acf) + len(vrs)
    n_sig = n_sig_acf + n_sig_vr

    # How many "significant" results pure noise would be expected to produce,
    # and how surprising the observed count actually is.
    expected_false = 0.05 * tests_run
    chance_p = binomial_sf(n_sig, tests_run, 0.05)

    if n_sig == 0:
        verdict = (
            "INDISTINGUISHABLE FROM A RANDOM WALK. No autocorrelation lag and no "
            "variance-ratio horizon rejected the random-walk null at the 5% "
            "level. A price-only signal has nothing to exploit here; treat any "
            "positive backtest on this ticker as noise until proven otherwise."
        )
    elif chance_p >= 0.05:
        verdict = (
            f"NOT MEANINGFUL. {n_sig} of {tests_run} tests were significant at "
            f"5%, against ~{expected_false:.1f} expected by chance alone "
            f"(p={chance_p:.2f} that noise produces at least this many). This "
            "is not evidence of structure."
        )
    else:
        directions = {
            v.interpretation for v in vrs
            if np.isfinite(v.p_value) and v.p_value < 0.05
        }
        direction = ", ".join(sorted(directions)) if directions else "unclear direction"
        verdict = (
            f"DEVIATION DETECTED. {n_sig} of {tests_run} tests were significant "
            f"at 5% against ~{expected_false:.1f} expected by chance "
            f"(p={chance_p:.3f}; {direction}). This is a reason to look "
            "further, NOT a tradable edge: statistical detectability and "
            "post-cost profitability are different questions, and this test "
            "used the same history any signal was fitted on."
        )

    return RandomWalkReport(
        ticker=ticker,
        n_observations=len(simple),
        acf=acf,
        variance_ratios=vrs,
        ljung_box_q=q_stat,
        verdict=verdict,
        tests_run=tests_run,
        significant_tests=n_sig,
        chance_p_value=chance_p,
    )
