"""Configuration: the ticker list and cost numbers must never be hardcoded."""

from __future__ import annotations

import json

import pytest

from qsb.config import Config, CostSpec, WalkForwardSpec


def test_defaults_load():
    cfg = Config.load()
    assert cfg.universe == ["UAMY", "SMR", "BBAI", "AEM"]
    assert cfg.interval == "1d"


def test_universe_comes_from_a_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"universe": ["SPY", "QQQ"]}))
    assert Config.load(path).universe == ["SPY", "QQQ"]


def test_cli_overrides_beat_the_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"universe": ["SPY"], "start": "2020-01-01"}))
    cfg = Config.load(path, {"universe": ["IWM"]})
    assert cfg.universe == ["IWM"]
    assert cfg.start == "2020-01-01", "unrelated file settings must survive"


def test_none_overrides_do_not_clobber_file_values(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"start": "2019-01-01"}))
    cfg = Config.load(path, {"start": None, "end": None})
    assert cfg.start == "2019-01-01"


def test_nested_overrides_merge_rather_than_replace():
    cfg = Config.load(overrides={"walk_forward": {"test_days": 21}})
    assert cfg.walk_forward.test_days == 21
    assert cfg.walk_forward.train_days == 378, "untouched keys must survive"


def test_tickers_are_normalized_and_deduplicated():
    cfg = Config.load(overrides={"universe": [" spy ", "SPY", "qqq"]})
    assert cfg.universe == ["SPY", "QQQ"]


def test_a_single_ticker_string_is_accepted():
    assert Config.load(overrides={"universe": "SPY"}).universe == ["SPY"]


def test_empty_universe_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        Config.load(overrides={"universe": []})


def test_holdout_universe_may_not_overlap_the_development_universe():
    """A hold-out set containing development names is not a hold-out set."""
    with pytest.raises(ValueError, match="overlap"):
        Config.load(overrides={
            "universe": ["AAPL", "SMR"],
            "validation_universe": ["AAPL", "XOM"],
        })


def test_default_holdout_universe_is_disjoint_and_diverse():
    cfg = Config.load()
    assert not set(cfg.universe) & set(cfg.validation_universe)
    assert len(cfg.validation_universe) >= 8


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        Config.load("/nonexistent/config.json")


def test_bad_cache_format_is_rejected():
    with pytest.raises(ValueError, match="cache_format"):
        Config.load(overrides={"cache_format": "hdf5"})


def test_walk_forward_rejects_a_train_window_below_the_minimum():
    with pytest.raises(ValueError, match="min_train_days"):
        WalkForwardSpec(train_days=100, test_days=60, min_train_days=252)


def test_walk_forward_rejects_nonpositive_windows():
    with pytest.raises(ValueError):
        WalkForwardSpec(train_days=100, test_days=0, min_train_days=10)


def test_cost_spec_rejects_an_impossible_premium():
    with pytest.raises(ValueError, match="premium"):
        CostSpec(half_spread_bps=1.0, slippage_bps=1.0, premium_pct_of_underlying=0.0)


def test_both_instrument_regimes_must_be_defined():
    with pytest.raises(ValueError, match="both"):
        Config.from_dict({
            **json.loads(json.dumps({
                k: v for k, v in Config.load().__dict__.items()
                if k not in {"walk_forward", "costs"}
            }, default=str)),
            "walk_forward": {"train_days": 378, "test_days": 126,
                             "min_train_days": 252, "min_train_trades": 5},
            "costs": {"stock": {"half_spread_bps": 5.0, "slippage_bps": 2.0}},
        })


def test_taxes_are_not_a_configurable_knob():
    """Taxes are zero by construction in this phase, not a setting to forget."""
    cfg = Config.load()
    assert not any("tax" in k.lower() for k in vars(cfg))
