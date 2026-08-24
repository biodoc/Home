"""
The split transaction-cost model.

Retail equity and retail options are two different cost regimes, and blending
them into one average is the fastest way to make a marginal signal look viable.
They are modelled separately here:

  STOCK   cost = half-spread + slippage, as a fraction of notional traded.
          Commission is zero, by construction -- `config.Config` refuses to load
          a stock spec with a non-zero commission.

  OPTION  cost = (wider half-spread + slippage) as a fraction of option premium,
          PLUS a fixed commission per contract. Because the commission is a flat
          dollar amount, its drag as a percentage depends on the premium: the
          same $0.65 is ~1% of a $65 contract and ~13% of a $5 contract. The
          model reflects that -- option cost is a *series*, varying with price,
          not a constant.

Costs are charged on turnover: |position_t - position_{t-1}|. A round trip
(0 -> +1 -> 0) is charged twice, once on the way in and once on the way out,
which is correct. A flip (+1 -> -1) is charged twice in a single bar, which is
also correct: it is two trades.

------------------------------------------------------------------------------
OPTIONS ARE A PROXY IN THIS PHASE
------------------------------------------------------------------------------
There is no options chain data in this build. Option costs are applied to a
backtest whose returns come from the *underlying's* price series. That is not
the same thing as backtesting an option position: it ignores delta, gamma,
theta, vega, assignment, and the fact that a 1% move in the stock is not a 1%
move in the contract. `is_proxy` is True for options for exactly this reason,
and every report that uses it carries a warning. Do not draw conclusions about
an options strategy from this mode until real chain data lands (wishlist item 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostSpec

BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    """Turns a `CostSpec` into a per-unit-of-turnover cost fraction."""

    instrument: str
    spec: CostSpec

    @classmethod
    def from_config(cls, cfg, instrument: str = "stock") -> "CostModel":
        return cls(instrument=instrument.lower(), spec=cfg.cost_spec(instrument))

    # -- properties ---------------------------------------------------------
    @property
    def is_proxy(self) -> bool:
        """True when this cost model is applied to a return series it does not
        actually describe (see the module docstring)."""
        return self.instrument == "option"

    @property
    def spread_and_slippage_fraction(self) -> float:
        """The price-proportional part of the cost, as a fraction of notional."""
        return (self.spec.half_spread_bps + self.spec.slippage_bps) * BPS

    # -- core ---------------------------------------------------------------
    def cost_fraction(self, price: pd.Series | float) -> pd.Series | float:
        """Cost of trading one unit, as a fraction of the position's notional.

        For stocks this is a constant. For options it varies with price, because
        the flat per-contract commission is a larger percentage of a cheap
        contract than of an expensive one.
        """
        base = self.spread_and_slippage_fraction

        if self.spec.commission_per_contract == 0.0:
            if isinstance(price, pd.Series):
                return pd.Series(base, index=price.index, dtype="float64")
            return base

        # Notional of one contract, approximated from the underlying price.
        premium = self._premium(price)
        contract_notional = premium * self.spec.contract_multiplier

        if isinstance(contract_notional, pd.Series):
            commission_frac = self.spec.commission_per_contract / contract_notional.where(
                contract_notional > 0
            )
            return (base + commission_frac).astype("float64")

        if contract_notional <= 0:
            return float("nan")
        return base + self.spec.commission_per_contract / contract_notional

    def _premium(self, price: pd.Series | float):
        """Estimated option premium from the underlying price."""
        return price * self.spec.premium_pct_of_underlying

    def round_trip_bps(self, price: pd.Series | float) -> float:
        """Cost of a full round trip, in basis points -- the headline number.

        This is the hurdle a signal has to clear per trade. If a signal's average
        gross gain per round trip is below this, it loses money in practice no
        matter how good the hit rate looks.
        """
        frac = self.cost_fraction(price)
        if isinstance(frac, pd.Series):
            frac = float(frac.mean())
        return float(frac) * 2.0 / BPS

    # -- reporting ----------------------------------------------------------
    def describe(self, price: pd.Series | float | None = None) -> str:
        """Human-readable statement of exactly what is being charged."""
        s = self.spec
        lines = [f"  instrument          : {self.instrument}"]
        lines.append(
            f"  half spread         : {s.half_spread_bps:.1f} bps"
        )
        lines.append(f"  slippage            : {s.slippage_bps:.1f} bps")
        if s.commission_per_contract:
            lines.append(
                f"  commission          : ${s.commission_per_contract:.2f} per "
                f"contract ({s.contract_multiplier} shares/contract)"
            )
            lines.append(
                f"  assumed premium     : {s.premium_pct_of_underlying:.1%} of "
                "underlying price"
            )
        else:
            lines.append("  commission          : $0.00 (commission-free equities)")

        if price is not None:
            lines.append(
                f"  => round trip cost  : {self.round_trip_bps(price):.1f} bps "
                "of notional"
            )
        if self.is_proxy:
            lines.append(
                "  !! PROXY            : option costs applied to UNDERLYING "
                "returns. No chain data, no greeks. Not an options backtest."
            )
        return "\n".join(lines)
