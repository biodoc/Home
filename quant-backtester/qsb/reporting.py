"""
Report formatting.

Two rules are enforced structurally here rather than left to discipline:

  1. GROSS AND NET ALWAYS APPEAR TOGETHER. `_perf_block` renders both columns
     from the same stats dict. There is no code path in this module that can
     print one without the other.

  2. THE CONCLUSION IS WRITTEN IN WORDS. A reader should not have to compare two
     Sharpe ratios to discover that a signal is unprofitable; the verdict line
     says so in a sentence.

The per-period table exists for the same reason: an aggregate that averages
[+2.9, -0.4, -0.2, +0.3] into "0.65" is technically true and practically a lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULE = "=" * 78
THIN = "-" * 78

DISCLAIMER = (
    "This is an exploratory backtest, not financial advice and not a trading\n"
    "recommendation. Past statistical behaviour does not persist by default.\n"
    "TAXES ARE MODELLED AS ZERO -- these are pre-tax figures and are NOT\n"
    "real-world after-tax returns."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _pct(x, places: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "     n/a"
    return f"{x * 100:+8.{places}f}%"


def _num(x, places: int = 2, width: int = 9) -> str:
    if x is None or not np.isfinite(x):
        return "n/a".rjust(width)
    return f"{x:{width}.{places}f}"


def _date(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def _params_str(params: dict) -> str:
    if not params:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


# ---------------------------------------------------------------------------
# Performance block -- gross and net, side by side, always
# ---------------------------------------------------------------------------
def _perf_block(stats: dict, title: str = "PERFORMANCE") -> str:
    rows = [
        ("Total return",      "gross_total_return",  "net_total_return",  "pct"),
        ("Annualized return", "gross_annual_return", "net_annual_return", "pct"),
        ("Sharpe ratio",      "gross_sharpe",        "net_sharpe",        "num"),
        ("Annualized vol",    "gross_volatility",    "net_volatility",    "pct"),
        ("Max drawdown",      "gross_max_drawdown",  "net_max_drawdown",  "pct"),
        ("Hit rate",          "gross_hit_rate",      "net_hit_rate",      "raw_pct"),
        ("Average win",       "gross_avg_win",       "net_avg_win",       "pct"),
        ("Average loss",      "gross_avg_loss",      "net_avg_loss",      "pct"),
        ("Average trade",     "gross_avg_trade",     "net_avg_trade",     "pct"),
        ("Win/loss ratio",    "gross_win_loss_ratio", "net_win_loss_ratio", "num"),
        ("Best trade",        "gross_best_trade",    "net_best_trade",    "pct"),
        ("Worst trade",       "gross_worst_trade",   "net_worst_trade",   "pct"),
    ]

    out = [title, f"  {'':<22}{'GROSS':>12}{'NET':>12}   {'(net - gross)':>14}"]
    for label, gkey, nkey, kind in rows:
        g, n = stats.get(gkey, float("nan")), stats.get(nkey, float("nan"))
        if kind == "pct":
            gs, ns = _pct(g).rjust(12), _pct(n).rjust(12)
            delta = _pct(n - g).rjust(14) if np.isfinite(g) and np.isfinite(n) else "".rjust(14)
        elif kind == "raw_pct":
            gs = (f"{g:11.1f}%" if np.isfinite(g) else "n/a".rjust(12))
            ns = (f"{n:11.1f}%" if np.isfinite(n) else "n/a".rjust(12))
            delta = (f"{n - g:13.1f}%" if np.isfinite(g) and np.isfinite(n) else "".rjust(14))
        else:
            gs, ns = _num(g).rjust(12), _num(n).rjust(12)
            delta = _num(n - g).rjust(14) if np.isfinite(g) and np.isfinite(n) else "".rjust(14)
        out.append(f"  {label:<22}{gs}{ns}   {delta}")

    out.append("")
    out.append(f"  Trades              : {stats.get('n_trades', 0)}")
    out.append(f"  Bars                : {stats.get('bars', 0)} "
               f"({stats.get('years', float('nan')):.2f} years)")
    out.append(f"  Time in market      : {stats.get('time_in_market_pct', float('nan')):.1f}%")
    out.append(f"  Turnover (units/yr) : {_num(stats.get('turnover_annual'), 2, 0)}")
    return "\n".join(out)


def _cost_hurdle_block(stats: dict) -> str:
    """The single most important comparison in the whole report."""
    edge = stats.get("gross_edge_bps_per_unit_traded", float("nan"))
    hurdle = stats.get("assumed_cost_bps_per_unit_traded", float("nan"))
    drag = stats.get("cost_drag_annual", float("nan"))

    lines = ["COST HURDLE"]
    lines.append(f"  Gross edge earned   : {_num(edge, 1, 9)} bps per unit traded")
    lines.append(f"  Assumed cost        : {_num(hurdle, 1, 9)} bps per unit traded")
    if np.isfinite(edge) and np.isfinite(hurdle):
        margin = edge - hurdle
        lines.append(f"  Margin              : {_num(margin, 1, 9)} bps "
                     f"({'clears' if margin > 0 else 'FAILS'} the hurdle)")
        if np.isfinite(hurdle) and hurdle > 0:
            lines.append(f"  Cost is             : {edge / hurdle if hurdle else float('nan'):9.2f}x "
                         "covered by gross edge (need > 1.0)")
    lines.append(f"  Annual cost drag    : {_pct(drag)}")
    return "\n".join(lines)


def _period_table(result) -> str:
    """Per-window out-of-sample breakdown -- the anti-overfitting exhibit."""
    if not result.periods:
        return "PER-PERIOD BREAKDOWN\n  (none)"

    lines = [
        "PER-PERIOD WALK-FORWARD BREAKDOWN (out-of-sample test windows only)",
        f"  {'#':<3}{'test window':<25}{'trades':>7}{'gross Shp':>11}"
        f"{'net Shp':>9}{'net ret':>10}   parameters",
        f"  {THIN[:74]}",
    ]
    for p in result.periods:
        ts = p.test_stats
        window = f"{_date(p.test_start)} -> {_date(p.test_end)}"
        lines.append(
            f"  {p.index:<3}{window:<25}{ts.get('n_trades', 0):>7}"
            f"{_num(ts.get('gross_sharpe'), 2, 11)}"
            f"{_num(ts.get('net_sharpe'), 2, 9)}"
            f"{_pct(ts.get('net_annual_return'), 1):>10}   "
            f"{_params_str(p.params)}"
        )

    st = result.stability
    if st:
        lines.append(f"  {THIN[:74]}")
        lines.append(
            f"  Net Sharpe across {st.get('n_periods', 0)} windows: "
            f"mean {_num(st.get('net_sharpe_mean'), 2, 0)}, "
            f"sd {_num(st.get('net_sharpe_std'), 2, 0)}, "
            f"range [{_num(st.get('net_sharpe_min'), 2, 0)}, "
            f"{_num(st.get('net_sharpe_max'), 2, 0)}]"
        )
        lines.append(
            f"  Profitable windows   : {st.get('positive_periods', 0)}/"
            f"{st.get('n_periods', 0)} ({st.get('pct_positive', float('nan')):.0f}%)"
        )
        lines.append(
            f"  Train minus test Shp : {_num(st.get('train_minus_test_sharpe'), 2, 0)} "
            "(large positive = fitting the training window)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full single-result report
# ---------------------------------------------------------------------------
def format_result(result, show_cleaning: bool = True) -> str:
    """Render one BacktestResult as a complete text report."""
    head = (
        f"{RULE}\n"
        f"{result.ticker}  |  signal: {result.signal}  |  "
        f"instrument: {result.instrument}\n{RULE}"
    )
    parts = [head]

    if show_cleaning and result.cleaning is not None:
        parts.append("DATA\n" + result.cleaning.summary())

    if result.liquidity:
        lq = result.liquidity
        parts.append(
            "LIQUIDITY\n"
            f"  period              : {_date(lq.get('first'))} -> {_date(lq.get('last'))}"
            f"  ({lq.get('days', 0)} sessions)\n"
            f"  avg dollar volume   : ${lq.get('avg_dollar_volume', float('nan')):,.0f}\n"
            f"  median              : ${lq.get('median_dollar_volume', float('nan')):,.0f}\n"
            f"  10th percentile     : ${lq.get('p10_dollar_volume', float('nan')):,.0f}\n"
            f"  below ${lq.get('threshold', 0):,.0f} threshold: "
            f"{lq.get('untradable_days', 0)} sessions "
            f"({lq.get('untradable_pct', 0):.1f}%) forced flat"
        )

    parts.append("COST MODEL\n" + result.cost_description)

    if not result.ok:
        parts.append(f"RESULT\n  {result.verdict_code}\n\n  {_wrap(result.verdict)}")
        parts.append(DISCLAIMER)
        return "\n\n".join(parts)

    parts.append(_perf_block(result.aggregate,
                             "AGGREGATE OUT-OF-SAMPLE PERFORMANCE"))
    parts.append(_cost_hurdle_block(result.aggregate))
    parts.append(_period_table(result))

    flags = list(result.stability.get("flags", [])) + list(result.warnings)
    if flags:
        parts.append("FLAGS\n" + "\n".join(f"  [!] {_wrap(f, 6)}" for f in flags))

    parts.append(f"VERDICT: {result.verdict_code}\n\n  {_wrap(result.verdict, 2)}")

    if result.is_cost_proxy:
        parts.append(
            "  [!] " + _wrap(
                "Option costs were applied to the UNDERLYING's return series. "
                "This is a cost sensitivity check, not an options backtest. "
                "It ignores delta, theta, and every other greek.", 6
            )
        )

    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def _wrap(text: str, indent: int = 2, width: int = 76) -> str:
    import textwrap

    return textwrap.fill(
        text, width=width, initial_indent="", subsequent_indent=" " * indent
    )


# ---------------------------------------------------------------------------
# Cross-ticker summary
# ---------------------------------------------------------------------------
def format_summary(results: list, title: str = "SUMMARY") -> str:
    """One line per (ticker, signal), gross and net together."""
    lines = [
        RULE,
        title,
        RULE,
        f"{'ticker':<8}{'signal':<16}{'trades':>7}{'gross Shp':>11}{'net Shp':>9}"
        f"{'gross ret':>11}{'net ret':>10}  verdict",
        THIN,
    ]
    for r in results:
        a = r.aggregate or {}
        lines.append(
            f"{r.ticker:<8}{r.signal:<16}{a.get('n_trades', 0):>7}"
            f"{_num(a.get('gross_sharpe'), 2, 11)}"
            f"{_num(a.get('net_sharpe'), 2, 9)}"
            f"{_pct(a.get('gross_annual_return'), 1):>11}"
            f"{_pct(a.get('net_annual_return'), 1):>10}  {r.verdict_code}"
        )
    lines.append(THIN)

    survived = [r for r in results if r.verdict_code == "SURVIVES_COSTS"]
    killed = [r for r in results if r.verdict_code == "COSTS_KILL_IT"]
    lines.append(
        f"{len(survived)}/{len(results)} combinations were positive net of costs; "
        f"{len(killed)} had a gross edge that costs destroyed."
    )
    if not survived and results:
        lines.append(
            "No combination survived costs. That is the expected outcome and is "
            "informative: it says the frictions, not the ideas, are the binding "
            "constraint at retail scale."
        )
    return "\n".join(lines)


def format_random_walk(report) -> str:
    """Render the autocorrelation / variance-ratio diagnostics."""
    lines = [
        RULE,
        f"RANDOM-WALK DIAGNOSTICS: {report.ticker}  "
        f"({report.n_observations} observations)",
        RULE,
        "AUTOCORRELATION OF DAILY RETURNS",
        f"  {'lag':<6}{'acf':>9}{'95% band':>11}   significant?",
    ]
    for _, row in report.acf.iterrows():
        mark = "YES" if row["significant"] else "no"
        lines.append(
            f"  {int(row['lag']):<6}{row['acf']:>9.4f}"
            f"{row['conf95']:>11.4f}   {mark}"
        )
    lines.append(f"  Ljung-Box Q         : {report.ljung_box_q:.2f} "
                 "(magnitude only; larger = more serial dependence)")
    lines.append("")
    lines.append("VARIANCE RATIO (Lo-MacKinlay, heteroskedasticity-robust)")
    lines.append(f"  {'horizon':<9}{'VR':>8}{'z':>9}{'p':>9}   interpretation")
    for vr in report.variance_ratios:
        lines.append(
            f"  {vr.q:<9}{vr.vr:>8.3f}{vr.z:>9.2f}{vr.p_value:>9.3f}   "
            f"{vr.interpretation}"
        )
    lines.append("  VR = 1 is a random walk; > 1 trends, < 1 mean reverts.")
    lines.append("")
    lines.append(_wrap(f"VERDICT: {report.verdict}", 2))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Machine-readable export
# ---------------------------------------------------------------------------
def results_to_frame(results: list) -> pd.DataFrame:
    """Flatten results into a DataFrame for CSV export."""
    rows = []
    for r in results:
        row = {
            "ticker": r.ticker,
            "signal": r.signal,
            "instrument": r.instrument,
            "verdict": r.verdict_code,
            "verdict_text": r.verdict,
            "params_stable": r.params_stable(),
            "n_periods": len(r.periods),
            "cost_proxy": r.is_cost_proxy,
        }
        row.update({k: v for k, v in (r.aggregate or {}).items()
                    if not isinstance(v, (list, dict))})
        row.update({f"stability_{k}": v for k, v in (r.stability or {}).items()
                    if not isinstance(v, (list, dict))})
        row["flags"] = " | ".join(
            list((r.stability or {}).get("flags", [])) + list(r.warnings)
        )
        rows.append(row)
    return pd.DataFrame(rows)
