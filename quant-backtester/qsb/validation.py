"""
Broader-universe validation.

A signal that works on UAMY, SMR, BBAI, and AEM has been tested on four names
that were chosen *because they were already being watched*. That is a selected
sample, and a result on it is a hypothesis, not a finding.

This module re-runs a signal on a hold-out universe -- different sectors,
different market caps, different liquidity regimes, none of them used to develop
the signal -- and compares the two sets. Crucially, the two are reported
SEPARATELY. They are never pooled into one statistic: pooling would let four
lucky small caps carry ten unrelated names, or vice versa, and would destroy the
only thing this comparison is for.

The comparison produces one of four judgements:

  CONFIRMED       positive on both sets, in the same direction
  OVERFIT         positive on development names, absent or negative on hold-out
  REVERSED        positive on development names, significantly negative on
                  hold-out -- worse than mere absence, it suggests the original
                  result was a artifact of those specific names
  NO_DEV_EDGE     nothing to validate; the signal did not work on the
                  development set either
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationComparison:
    """Development set vs hold-out set for one signal."""

    signal: str
    dev_tickers: list[str] = field(default_factory=list)
    val_tickers: list[str] = field(default_factory=list)
    dev_stats: dict = field(default_factory=dict)
    val_stats: dict = field(default_factory=dict)
    code: str = "NO_DATA"
    message: str = ""


def _aggregate_across_tickers(results: list) -> dict:
    """Summarise a set of per-ticker results WITHOUT pooling their returns.

    Each ticker keeps its own identity; what is aggregated is the distribution
    of per-ticker outcomes (how many worked, median Sharpe, and so on). This is
    deliberately not a combined equity curve -- combining them would imply a
    portfolio that was never specified, sized, or rebalanced.
    """
    usable = [r for r in results if r.ok and r.aggregate.get("n_trades", 0) > 0]
    if not usable:
        return {"n_tickers": len(results), "n_usable": 0}

    net_sharpe = np.array(
        [r.aggregate.get("net_sharpe", np.nan) for r in usable], dtype="float64"
    )
    gross_sharpe = np.array(
        [r.aggregate.get("gross_sharpe", np.nan) for r in usable], dtype="float64"
    )
    net_ret = np.array(
        [r.aggregate.get("net_annual_return", np.nan) for r in usable], dtype="float64"
    )
    gross_ret = np.array(
        [r.aggregate.get("gross_annual_return", np.nan) for r in usable], dtype="float64"
    )

    return {
        "n_tickers": len(results),
        "n_usable": len(usable),
        "tickers": [r.ticker for r in usable],
        "median_net_sharpe": float(np.nanmedian(net_sharpe)),
        "mean_net_sharpe": float(np.nanmean(net_sharpe)),
        "median_gross_sharpe": float(np.nanmedian(gross_sharpe)),
        "median_net_annual_return": float(np.nanmedian(net_ret)),
        "median_gross_annual_return": float(np.nanmedian(gross_ret)),
        "n_positive_net": int(np.nansum(net_ret > 0)),
        "n_positive_gross": int(np.nansum(gross_ret > 0)),
        "pct_positive_net": float(np.nansum(net_ret > 0) / len(usable) * 100.0),
        "n_survived": sum(1 for r in usable if r.verdict_code == "SURVIVES_COSTS"),
        "per_ticker": {
            r.ticker: {
                "net_sharpe": r.aggregate.get("net_sharpe"),
                "gross_sharpe": r.aggregate.get("gross_sharpe"),
                "net_annual_return": r.aggregate.get("net_annual_return"),
                "gross_annual_return": r.aggregate.get("gross_annual_return"),
                "verdict": r.verdict_code,
            }
            for r in usable
        },
    }


def compare(
    signal_name: str,
    dev_results: list,
    val_results: list,
) -> ValidationComparison:
    """Judge whether a development-set result survives on the hold-out set."""
    dev = _aggregate_across_tickers(dev_results)
    val = _aggregate_across_tickers(val_results)

    cmp = ValidationComparison(
        signal=signal_name,
        dev_tickers=[r.ticker for r in dev_results],
        val_tickers=[r.ticker for r in val_results],
        dev_stats=dev,
        val_stats=val,
    )

    if not dev.get("n_usable") or not val.get("n_usable"):
        cmp.code = "INSUFFICIENT_DATA"
        cmp.message = (
            "Not enough usable results on one or both sets to compare. "
            "No conclusion can be drawn."
        )
        return cmp

    dev_sharpe = dev["median_net_sharpe"]
    val_sharpe = val["median_net_sharpe"]
    dev_positive = dev["median_net_annual_return"] > 0 and dev_sharpe > 0

    if not dev_positive:
        cmp.code = "NO_DEV_EDGE"
        cmp.message = (
            f"The signal did not work net of costs on the development set "
            f"(median net Sharpe {dev_sharpe:.2f}), so there is nothing to "
            f"validate. For reference, the hold-out set's median net Sharpe was "
            f"{val_sharpe:.2f}."
        )
        return cmp

    if val_sharpe <= -0.25:
        cmp.code = "REVERSED"
        cmp.message = (
            f"LIKELY CURVE FIT. The signal looked positive on the "
            f"{dev['n_usable']} development names (median net Sharpe "
            f"{dev_sharpe:.2f}) but REVERSED on the {val['n_usable']} hold-out "
            f"names (median net Sharpe {val_sharpe:.2f}). A relationship that "
            "inverts on names it was not built from is a property of those "
            "specific tickers, not a statistical edge."
        )
    elif val_sharpe <= 0 or val["pct_positive_net"] < 40.0:
        cmp.code = "OVERFIT"
        cmp.message = (
            f"LIKELY OVERFIT. Positive on the development set (median net "
            f"Sharpe {dev_sharpe:.2f}) but not on the hold-out set (median "
            f"{val_sharpe:.2f}; only {val['pct_positive_net']:.0f}% of hold-out "
            "names were profitable net of costs). Treat the development-set "
            "result as fitted to those four names until it reproduces "
            "elsewhere."
        )
    elif val_sharpe >= 0.3 * dev_sharpe:
        cmp.code = "CONFIRMED"
        cmp.message = (
            f"HOLDS UP OUT OF SAMPLE. Positive on both sets: median net Sharpe "
            f"{dev_sharpe:.2f} on the {dev['n_usable']} development names and "
            f"{val_sharpe:.2f} on the {val['n_usable']} hold-out names "
            f"({val['pct_positive_net']:.0f}% of hold-out names profitable net "
            "of costs). This is the strongest evidence this tool can produce, "
            "which is still a long way short of proof: it is one signal, on one "
            "history, with modelled rather than observed costs."
        )
    else:
        cmp.code = "WEAKENED"
        cmp.message = (
            f"SURVIVES BUT WEAKENS. Median net Sharpe falls from "
            f"{dev_sharpe:.2f} on the development names to {val_sharpe:.2f} on "
            "the hold-out names. Some of the original result was specific to "
            "those tickers. Size any expectation to the hold-out number, not "
            "the development one."
        )
    return cmp


def format_comparison(cmp: ValidationComparison) -> str:
    """Render the two sets side by side, never blended into one figure."""
    from .reporting import RULE, THIN, _num, _pct, _wrap

    lines = [
        RULE,
        f"BROADER-UNIVERSE VALIDATION: {cmp.signal}",
        RULE,
        "The two sets below are reported separately and are never combined into",
        "a single statistic. Blending them would let one set carry the other.",
        "",
    ]

    for label, stats, tickers in (
        ("DEVELOPMENT SET (signal was built looking at these)",
         cmp.dev_stats, cmp.dev_tickers),
        ("HOLD-OUT SET (never used to develop the signal)",
         cmp.val_stats, cmp.val_tickers),
    ):
        lines.append(label)
        if not stats.get("n_usable"):
            lines.append(f"  no usable results ({len(tickers)} tickers attempted)")
            lines.append("")
            continue
        lines.append(
            f"  tickers             : {stats['n_usable']}/{stats['n_tickers']} usable "
            f"({', '.join(stats['tickers'])})"
        )
        lines.append(f"  median gross Sharpe : {_num(stats['median_gross_sharpe'], 2, 0)}")
        lines.append(f"  median net Sharpe   : {_num(stats['median_net_sharpe'], 2, 0)}")
        lines.append(f"  median gross return : {_pct(stats['median_gross_annual_return'], 1)}/yr")
        lines.append(f"  median net return   : {_pct(stats['median_net_annual_return'], 1)}/yr")
        lines.append(
            f"  profitable net      : {stats['n_positive_net']}/{stats['n_usable']} "
            f"({stats['pct_positive_net']:.0f}%)"
        )
        lines.append("")
        lines.append(f"  {'ticker':<8}{'gross Shp':>11}{'net Shp':>9}{'net ret':>10}  verdict")
        for tk, s in stats["per_ticker"].items():
            lines.append(
                f"  {tk:<8}{_num(s['gross_sharpe'], 2, 11)}"
                f"{_num(s['net_sharpe'], 2, 9)}"
                f"{_pct(s['net_annual_return'], 1):>10}  {s['verdict']}"
            )
        lines.append("")

    lines.append(THIN)
    lines.append(f"JUDGEMENT: {cmp.code}")
    lines.append("")
    lines.append(_wrap(cmp.message, 2))
    return "\n".join(lines)
