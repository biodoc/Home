"""
Return construction, trade extraction, and performance statistics.

This is the module where a backtest usually lies to you, so the timing
convention is stated once and enforced in one place:

    positions[t]  is the decision made on the CLOSE of bar t
    held[t]       = positions[t-1]        <- the one-bar execution shift
    gross[t]      = held[t] * return[t]
    turnover[t]   = |held[t] - held[t-1]|
    cost[t]       = turnover[t] * cost_fraction[t]
    net[t]        = gross[t] - cost[t]

`compute_returns` is the ONLY place the shift happens. Signals must not shift,
and the engine must not shift. Test `test_no_lookahead_bias` in the suite exists
specifically to catch a regression here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Return construction
# ---------------------------------------------------------------------------
def compute_returns(
    close: pd.Series,
    positions: pd.Series,
    cost_fraction: pd.Series | float,
    tradable: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the gross/cost/net return frame for one signal on one ticker.

    Args:
        close: price series.
        positions: target position decided at each bar's close, in {-1,0,+1}.
        cost_fraction: cost of trading one unit of notional, as a fraction.
            Either a constant (stocks) or a per-bar series (options).
        tradable: optional boolean mask. Bars marked False are forced flat,
            which is how the thin-liquidity filter is applied -- we do not
            pretend to get a fill on a day the name barely traded.

    Returns:
        DataFrame indexed like `close` with columns:
        close, ret, position, held, turnover, gross, cost, net,
        equity_gross, equity_net.
    """
    close = close.astype("float64")
    positions = positions.reindex(close.index).astype("float64").fillna(0.0)

    if tradable is not None:
        mask = tradable.reindex(close.index).fillna(False).astype(bool)
        positions = positions.where(mask, 0.0)

    # Simple (not log) returns: they compound the way an account does.
    ret = close.pct_change(fill_method=None)

    # THE shift. Decided at t, earned over t+1.
    held = positions.shift(1).fillna(0.0)

    turnover = held.diff().abs()
    turnover.iloc[0] = abs(held.iloc[0])  # opening the first position is a trade

    if isinstance(cost_fraction, pd.Series):
        cf = cost_fraction.reindex(close.index).astype("float64")
        cf = cf.ffill().bfill()
    else:
        cf = pd.Series(float(cost_fraction), index=close.index, dtype="float64")

    gross = (held * ret).fillna(0.0)
    cost = (turnover * cf).fillna(0.0)
    net = gross - cost

    out = pd.DataFrame(
        {
            "close": close,
            "ret": ret,
            "position": positions,
            "held": held,
            "turnover": turnover.fillna(0.0),
            "cost_fraction": cf,
            "gross": gross,
            "cost": cost,
            "net": net,
        }
    )
    out["equity_gross"] = (1.0 + out["gross"]).cumprod()
    out["equity_net"] = (1.0 + out["net"]).cumprod()
    return out


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------
def extract_trades(returns: pd.DataFrame) -> pd.DataFrame:
    """Collapse the bar-by-bar frame into discrete round-trip trades.

    A trade is a maximal run of bars over which `held` is constant and non-zero.
    Its net return compounds the net series over the run and then subtracts the
    cost of unwinding, which lands on the first bar *after* the run.
    """
    held = returns["held"]
    if held.empty:
        return _empty_trades()

    nonzero = held != 0
    # A new trade starts whenever the held value changes to a new non-zero value.
    group_id = (held != held.shift(1)).cumsum()

    rows = []
    for _, block in returns.groupby(group_id, sort=True):
        if block["held"].iloc[0] == 0:
            continue
        start, end = block.index[0], block.index[-1]

        gross_r = float((1.0 + block["gross"]).prod() - 1.0)
        net_r = float((1.0 + block["net"]).prod() - 1.0)

        # Unwind cost: charged on the bar after the block, if there is one.
        pos_end = returns.index.get_loc(end)
        if pos_end + 1 < len(returns):
            exit_cost = float(returns["cost"].iloc[pos_end + 1])
            net_r = (1.0 + net_r) * (1.0 - exit_cost) - 1.0

        rows.append(
            {
                "entry": start,
                "exit": end,
                "direction": int(block["held"].iloc[0]),
                "bars": int(len(block)),
                "gross_return": gross_r,
                "net_return": net_r,
            }
        )

    if not rows:
        return _empty_trades()
    return pd.DataFrame(rows)


def count_trades(returns: pd.DataFrame) -> int:
    """Number of round-trip trades, computed vectorized.

    Equivalent to `len(extract_trades(...))` but without materialising each
    trade. The walk-forward search evaluates every parameter combination on
    every training window, so this is called thousands of times per run and the
    groupby in `extract_trades` dominates the runtime otherwise.
    """
    held = returns["held"]
    if held.empty:
        return 0
    return int(((held != held.shift(1)) & (held != 0)).sum())


def quick_score(returns: pd.DataFrame, ppy: int = 252, risk_free: float = 0.0) -> tuple[int, float]:
    """(n_trades, net_sharpe) only -- the two numbers parameter selection needs."""
    return count_trades(returns), sharpe(returns["net"], ppy, risk_free)


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["entry", "exit", "direction", "bars", "gross_return", "net_return"]
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def sharpe(returns: pd.Series, ppy: int = 252, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio. NaN when it is not defined."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - (risk_free / ppy)
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ppy))


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline of an equity curve, as a negative fraction."""
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def annualized_return(returns: pd.Series, ppy: int = 252) -> float:
    """Geometric annualized return of a bar-level return series."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    total = float((1.0 + r).prod())
    if total <= 0:
        return -1.0
    years = len(r) / ppy
    if years <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def performance(
    returns: pd.DataFrame,
    ppy: int = 252,
    risk_free: float = 0.0,
) -> dict:
    """Full statistics block for one return frame, gross and net side by side.

    Every performance number in this project is produced here, and every one of
    them exists in a gross and a net variant. There is deliberately no way to
    ask this function for net alone.
    """
    trades = extract_trades(returns)
    n = len(returns)
    years = n / ppy if ppy else float("nan")

    stats: dict = {
        "bars": n,
        "years": years,
        "first": returns.index.min() if n else None,
        "last": returns.index.max() if n else None,
        "n_trades": int(len(trades)),
        "time_in_market_pct": float((returns["held"] != 0).mean() * 100.0) if n else 0.0,
        # Turnover as units traded per year -- 1.0 means one full round trip/yr.
        "turnover_annual": float(returns["turnover"].sum() / years) if years else float("nan"),
        "total_cost": float(returns["cost"].sum()),
        "avg_cost_bps_per_trade": (
            float(returns["cost"].sum() / len(trades) * 1e4) if len(trades) else float("nan")
        ),
    }

    for label in ("gross", "net"):
        r = returns[label]
        eq = returns[f"equity_{label}"]
        stats[f"{label}_total_return"] = float(eq.iloc[-1] - 1.0) if n else float("nan")
        stats[f"{label}_annual_return"] = annualized_return(r, ppy)
        stats[f"{label}_sharpe"] = sharpe(r, ppy, risk_free)
        stats[f"{label}_max_drawdown"] = max_drawdown(eq)
        stats[f"{label}_volatility"] = (
            float(r.std(ddof=1) * np.sqrt(ppy)) if len(r.dropna()) > 1 else float("nan")
        )

        col = f"{label}_return"
        if len(trades):
            wins = trades.loc[trades[col] > 0, col]
            losses = trades.loc[trades[col] <= 0, col]
            stats[f"{label}_hit_rate"] = float(len(wins) / len(trades) * 100.0)
            stats[f"{label}_avg_win"] = float(wins.mean()) if len(wins) else 0.0
            stats[f"{label}_avg_loss"] = float(losses.mean()) if len(losses) else 0.0
            stats[f"{label}_avg_trade"] = float(trades[col].mean())
            stats[f"{label}_best_trade"] = float(trades[col].max())
            stats[f"{label}_worst_trade"] = float(trades[col].min())
            avg_win = stats[f"{label}_avg_win"]
            avg_loss = stats[f"{label}_avg_loss"]
            stats[f"{label}_win_loss_ratio"] = (
                float(avg_win / abs(avg_loss)) if avg_loss < 0 else float("nan")
            )
        else:
            for key in (
                "hit_rate", "avg_win", "avg_loss", "avg_trade",
                "best_trade", "worst_trade", "win_loss_ratio",
            ):
                stats[f"{label}_{key}"] = float("nan")

    # -- the cost hurdle ----------------------------------------------------
    # Gross profit earned per unit of turnover, in bps. This is the honest
    # comparison against the round-trip cost: a signal that earns 4 bps per unit
    # traded cannot survive a 14 bps round trip, whatever its Sharpe looks like.
    total_turnover = float(returns["turnover"].sum())
    gross_pnl = float(returns["gross"].sum())
    stats["total_turnover"] = total_turnover
    stats["gross_edge_bps_per_unit_traded"] = (
        gross_pnl / total_turnover * 1e4 if total_turnover > 0 else float("nan")
    )
    stats["assumed_cost_bps_per_unit_traded"] = (
        float(returns["cost"].sum() / total_turnover * 1e4)
        if total_turnover > 0
        else float("nan")
    )
    stats["cost_drag_annual"] = (
        stats["gross_annual_return"] - stats["net_annual_return"]
        if np.isfinite(stats["gross_annual_return"])
        and np.isfinite(stats["net_annual_return"])
        else float("nan")
    )
    return stats


def edge_verdict(stats: dict) -> tuple[str, str]:
    """Classify a result and say so in words, not just numbers.

    Returns (code, sentence). The codes are:
        NO_TRADES        the signal never fired
        NO_GROSS_EDGE    it loses money even before costs
        COSTS_KILL_IT    a real gross edge that does not survive frictions
        SURVIVES_COSTS   still positive after costs (NOT a recommendation)
    """
    if not stats.get("n_trades"):
        return (
            "NO_TRADES",
            "The signal never triggered on this data. There is nothing to evaluate.",
        )

    gross_s = stats.get("gross_sharpe", float("nan"))
    net_s = stats.get("net_sharpe", float("nan"))
    gross_r = stats.get("gross_annual_return", float("nan"))
    net_r = stats.get("net_annual_return", float("nan"))
    edge = stats.get("gross_edge_bps_per_unit_traded", float("nan"))
    hurdle = stats.get("assumed_cost_bps_per_unit_traded", float("nan"))

    if not (np.isfinite(gross_r) and gross_r > 0 and np.isfinite(gross_s) and gross_s > 0):
        return (
            "NO_GROSS_EDGE",
            "No edge before costs. This signal loses money on this ticker even "
            "with zero frictions, so transaction costs are not the problem.",
        )

    if not (np.isfinite(net_r) and net_r > 0):
        return (
            "COSTS_KILL_IT",
            "EDGE DOES NOT SURVIVE COSTS. Gross performance is positive "
            f"({gross_r:+.1%}/yr, Sharpe {gross_s:.2f}) but net is not "
            f"({net_r:+.1%}/yr, Sharpe {net_s:.2f}). The signal earns "
            f"{edge:.1f} bps per unit traded against a {hurdle:.1f} bps cost. "
            "This is the ordinary outcome for retail statistical arbitrage and "
            "is the single most common reason such attempts fail.",
        )

    return (
        "SURVIVES_COSTS",
        f"Positive after costs: {net_r:+.1%}/yr, Sharpe {net_s:.2f} "
        f"(gross {gross_r:+.1%}/yr, Sharpe {gross_s:.2f}). The signal earns "
        f"{edge:.1f} bps per unit traded against a {hurdle:.1f} bps cost. "
        "This is a survived test, not a validated edge -- check the per-period "
        "stability table and the broader-universe validation before believing it.",
    )
