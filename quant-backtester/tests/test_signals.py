"""Signal correctness against known and synthetic inputs.

The critical property tested here is CAUSALITY: a signal's value at bar t must
not change when bars after t change. `test_signals_are_causal` checks that
directly for every signal in the registry, which is the one bug class that would
silently invalidate every result the framework produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_frame
from qsb import signals


# -- helper maths -----------------------------------------------------------
def test_rolling_zscore_matches_hand_calculation():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = signals.rolling_zscore(s, 3)
    # Window [1,2,3]: mean 2, sd 1 -> z of 3 is +1.0
    assert z.iloc[2] == pytest.approx(1.0)
    # Insufficient history must be NaN, never 0.
    assert z.iloc[:2].isna().all()


def test_rolling_zscore_is_nan_on_zero_variance():
    z = signals.rolling_zscore(pd.Series([5.0] * 10), 3)
    assert z.iloc[3:].isna().all(), "a flat window is undefined, not a 0-sigma event"


def test_rsi_is_100_for_an_unbroken_advance():
    rsi = signals.wilder_rsi(pd.Series(np.arange(1.0, 40.0)), period=14)
    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_for_an_unbroken_decline():
    rsi = signals.wilder_rsi(pd.Series(np.arange(40.0, 1.0, -1.0)), period=14)
    assert rsi.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rsi_stays_within_bounds_on_noise():
    rng = np.random.default_rng(0)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
    rsi = signals.wilder_rsi(px, 14).dropna()
    assert rsi.between(0.0, 100.0).all()


def test_rsi_warmup_is_nan():
    rsi = signals.wilder_rsi(pd.Series(np.arange(1.0, 30.0)), period=14)
    assert rsi.iloc[:13].isna().all()


# -- individual signals -----------------------------------------------------
def test_mean_reversion_shorts_a_stretched_advance(sawtooth_frame):
    pos = signals.mean_reversion_zscore(sawtooth_frame, lookback=10, entry_z=1.0,
                                        exit_z=0.25)
    assert set(pos.unique()) <= {-1.0, 0.0, 1.0}
    # An oscillator should produce both directions.
    assert (pos == 1.0).any() and (pos == -1.0).any()


def test_mean_reversion_fades_the_direction_of_the_move():
    # Price steps sharply above its own mean on the last bar -> expect a short.
    close = [20.0] * 30 + [20.05, 19.95] * 10 + [26.0]
    df = make_frame(close)
    pos = signals.mean_reversion_zscore(df, lookback=20, entry_z=1.5, exit_z=0.5)
    assert pos.iloc[-1] == -1.0


def test_mean_reversion_holds_position_inside_the_band():
    """Hysteresis: once in, stay in until the exit band, not on the first wobble."""
    close = [20.0] * 25 + [17.0] + [19.0] * 5
    df = make_frame(close)
    pos = signals.mean_reversion_zscore(df, lookback=20, entry_z=1.5, exit_z=0.5)
    entry = pos.index.get_loc(pos[pos == 1.0].index[0])
    assert pos.iloc[entry + 1] == 1.0, "position dropped before the exit band"


def test_mean_reversion_rejects_an_exit_band_wider_than_entry():
    with pytest.raises(ValueError, match="exit_z"):
        signals.mean_reversion_zscore(make_frame([1.0] * 30), entry_z=1.0, exit_z=2.0)


def test_ma_crossover_is_long_in_an_uptrend(trending_frame):
    pos = signals.ma_crossover(trending_frame, fast=10, slow=50)
    assert pos.iloc[-1] == 1.0
    assert (pos.iloc[100:] == 1.0).all()


def test_ma_crossover_is_short_in_a_downtrend():
    df = make_frame(np.linspace(30.0, 10.0, 300))
    pos = signals.ma_crossover(df, fast=10, slow=50)
    assert (pos.iloc[100:] == -1.0).all()


def test_ma_crossover_long_only_never_shorts():
    df = make_frame(np.linspace(30.0, 10.0, 300))
    pos = signals.ma_crossover(df, fast=10, slow=50, long_only=True)
    assert (pos >= 0).all()


def test_ma_crossover_rejects_inverted_windows():
    with pytest.raises(ValueError, match="fast window"):
        signals.ma_crossover(make_frame([1.0] * 100), fast=50, slow=20)


def test_momentum_follows_the_trend(trending_frame):
    pos = signals.momentum_roc(trending_frame, lookback=20, threshold=0.0)
    assert pos.iloc[-1] == 1.0


def test_momentum_dead_band_suppresses_small_moves():
    # A 1% total drift over the lookback, with a 5% threshold -> stay flat.
    df = make_frame(np.linspace(100.0, 101.0, 100))
    pos = signals.momentum_roc(df, lookback=20, threshold=0.05)
    assert (pos == 0.0).all()


def test_momentum_warmup_is_flat_not_long():
    df = make_frame(np.linspace(10.0, 20.0, 100))
    pos = signals.momentum_roc(df, lookback=60)
    assert (pos.iloc[:60] == 0.0).all()


def test_vol_regime_returns_valid_positions(random_walk_frame):
    pos = signals.vol_regime_breakout(random_walk_frame)
    assert set(pos.unique()) <= {-1.0, 0.0, 1.0}
    assert len(pos) == len(random_walk_frame)


def test_rsi_reversion_buys_oversold():
    # A long decline drives RSI below 30 -> expect a long (fade the selloff).
    df = make_frame(np.linspace(50.0, 25.0, 60))
    pos = signals.rsi_reversion(df, period=14, lower=30.0, upper=70.0)
    assert pos.iloc[-1] == 1.0


def test_rsi_reversion_rejects_impossible_thresholds():
    with pytest.raises(ValueError):
        signals.rsi_reversion(make_frame([1.0] * 50), lower=60.0, upper=70.0)


# -- properties that must hold for EVERY signal -----------------------------
@pytest.mark.parametrize("name", signals.available())
def test_signals_return_valid_positions(name, random_walk_frame):
    sd = signals.get(name)
    pos = sd.func(random_walk_frame, **sd.default_params())
    assert isinstance(pos, pd.Series)
    assert len(pos) == len(random_walk_frame)
    assert pos.index.equals(random_walk_frame.index)
    assert set(pos.unique()) <= {-1.0, 0.0, 1.0}
    assert not pos.isna().any(), "a NaN position is an unhandled warm-up period"


@pytest.mark.parametrize("name", signals.available())
def test_signals_are_causal(name, random_walk_frame):
    """The value at bar t must not depend on anything after bar t.

    This is the single most important test in the suite. A signal that peeks --
    a centred rolling window, a full-sample quantile, a negative shift -- will
    produce a beautiful backtest and a worthless one. Here the tail of the
    series is replaced with wildly different prices; every position at or before
    the cut must be byte-identical.
    """
    sd = signals.get(name)
    cut = 700

    full = sd.func(random_walk_frame, **sd.default_params())

    tampered = random_walk_frame.copy()
    tampered.iloc[cut:, :4] *= 3.0  # rewrite the future, leave the past alone
    altered = sd.func(tampered, **sd.default_params())

    pd.testing.assert_series_equal(
        full.iloc[:cut], altered.iloc[:cut],
        check_names=False,
        obj=f"{name} leaked future information into the past",
    )


@pytest.mark.parametrize("name", signals.available())
def test_signals_survive_a_short_series(name):
    """Too little history must produce a flat position, not an exception."""
    sd = signals.get(name)
    pos = sd.func(make_frame([10.0, 10.1, 10.2]), **sd.default_params())
    assert (pos == 0.0).all()


@pytest.mark.parametrize("name", signals.available())
def test_signals_survive_a_constant_price(name):
    """A halted stock has no z-score and no RSI; it must not blow up."""
    sd = signals.get(name)
    pos = sd.func(make_frame([10.0] * 300), **sd.default_params())
    assert not pos.isna().any()


# -- registry ---------------------------------------------------------------
def test_registry_lookup_is_forgiving_about_formatting():
    assert signals.get("Mean-Reversion").name == "mean_reversion"
    assert signals.get("MEAN REVERSION").name == "mean_reversion"


def test_unknown_signal_lists_the_alternatives():
    with pytest.raises(KeyError, match="Available"):
        signals.get("nope")


def test_valid_combos_filters_structurally_invalid_pairs():
    sd = signals.get("ma_crossover")
    combos = signals.valid_combos(sd)
    assert all(c["fast"] < c["slow"] for c in combos)
    assert len(combos) < len(signals.expand_grid(sd.param_grid))


def test_expand_grid_covers_the_cartesian_product():
    grid = {"a": [1, 2], "b": [3, 4, 5]}
    assert len(signals.expand_grid(grid)) == 6


def test_every_registered_signal_is_documented():
    for name in signals.available():
        sd = signals.get(name)
        assert sd.description and sd.plain_english, f"{name} is undocumented"
        assert sd.param_grid, f"{name} has no parameter grid to search"
