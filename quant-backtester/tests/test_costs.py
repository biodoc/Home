"""The split cost model: stock and option regimes must stay distinct."""

from __future__ import annotations

import pandas as pd
import pytest

from qsb.config import Config, CostSpec
from qsb.costs import CostModel


def test_stock_cost_is_spread_plus_slippage_only(cfg):
    cm = CostModel.from_config(cfg, "stock")
    # 5 bps half spread + 2 bps slippage = 7 bps one way.
    assert cm.cost_fraction(50.0) == pytest.approx(7e-4)
    assert cm.round_trip_bps(50.0) == pytest.approx(14.0)


def test_stock_cost_does_not_vary_with_price(cfg):
    cm = CostModel.from_config(cfg, "stock")
    assert cm.cost_fraction(5.0) == cm.cost_fraction(500.0)


def test_stock_commission_is_structurally_zero(cfg):
    assert cfg.cost_spec("stock").commission_per_contract == 0.0


def test_config_rejects_a_nonzero_stock_commission():
    with pytest.raises(ValueError, match="stock commission must be 0"):
        Config.load(overrides={"costs": {"stock": {
            "half_spread_bps": 5.0, "slippage_bps": 2.0,
            "commission_per_contract": 0.65,
        }}})


def test_option_cost_exceeds_stock_cost(cfg):
    stock = CostModel.from_config(cfg, "stock")
    option = CostModel.from_config(cfg, "option")
    assert option.round_trip_bps(50.0) > stock.round_trip_bps(50.0) * 10


def test_option_commission_drag_shrinks_as_premium_rises(cfg):
    """A flat per-contract fee is a bigger percentage of a cheap contract."""
    cm = CostModel.from_config(cfg, "option")
    cheap = cm.cost_fraction(10.0)    # premium 0.50 -> $50 notional
    rich = cm.cost_fraction(200.0)    # premium 10.00 -> $1000 notional
    assert cheap > rich


def test_option_commission_is_computed_per_contract_notional(cfg):
    cm = CostModel.from_config(cfg, "option")
    # price 10 -> premium 5% = 0.50 -> contract notional 0.50 * 100 = $50
    # commission fraction = 0.65 / 50 = 0.013; plus (150+50)bps = 0.02
    assert cm.cost_fraction(10.0) == pytest.approx(0.013 + 0.02)


def test_cost_fraction_on_a_series_returns_a_series(cfg):
    cm = CostModel.from_config(cfg, "option")
    prices = pd.Series([10.0, 50.0, 200.0])
    out = cm.cost_fraction(prices)
    assert isinstance(out, pd.Series)
    assert len(out) == 3
    assert out.is_monotonic_decreasing


def test_stock_series_cost_is_constant(cfg):
    cm = CostModel.from_config(cfg, "stock")
    out = cm.cost_fraction(pd.Series([10.0, 50.0, 200.0]))
    assert out.nunique() == 1


def test_option_model_is_marked_as_a_proxy(cfg):
    assert CostModel.from_config(cfg, "option").is_proxy is True
    assert CostModel.from_config(cfg, "stock").is_proxy is False


def test_option_description_carries_the_proxy_warning(cfg):
    text = CostModel.from_config(cfg, "option").describe(pd.Series([50.0]))
    assert "PROXY" in text
    assert "Not an options backtest" in text


def test_stock_description_states_zero_commission(cfg):
    text = CostModel.from_config(cfg, "stock").describe(pd.Series([50.0]))
    assert "commission-free" in text
    assert "PROXY" not in text


def test_unknown_instrument_raises(cfg):
    with pytest.raises(KeyError, match="unknown instrument"):
        CostModel.from_config(cfg, "futures")


def test_cost_spec_rejects_negative_inputs():
    with pytest.raises(ValueError):
        CostSpec(half_spread_bps=-1.0, slippage_bps=0.0)
    with pytest.raises(ValueError):
        CostSpec(half_spread_bps=1.0, slippage_bps=0.0, commission_per_contract=-1.0)


def test_zero_cost_model_is_expressible():
    """A frictionless run must be possible, for isolating cost effects."""
    cm = CostModel("stock", CostSpec(half_spread_bps=0.0, slippage_bps=0.0))
    assert cm.cost_fraction(100.0) == 0.0
