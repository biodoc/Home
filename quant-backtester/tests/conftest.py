"""Shared fixtures. No test in this suite touches the network."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the package importable without installing it.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qsb.config import Config  # noqa: E402


def make_frame(close, volume=1e7, index=None) -> pd.DataFrame:
    """Build a minimal OHLCV frame from a close series."""
    close = pd.Series(close, dtype="float64")
    if index is None:
        index = pd.bdate_range("2015-01-01", periods=len(close))
    close.index = index
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": np.full(len(close), float(volume)),
        },
        index=index,
    )


@pytest.fixture
def cfg():
    """A config with short walk-forward windows so tests stay fast."""
    return Config.load(
        overrides={
            "walk_forward": {
                "train_days": 120,
                "test_days": 60,
                "min_train_days": 60,
                "min_train_trades": 2,
            },
            "min_avg_dollar_volume": 0.0,
        }
    )


@pytest.fixture
def random_walk_frame():
    """A 1200-bar geometric random walk -- no exploitable structure by design."""
    rng = np.random.default_rng(20240101)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 1200)))
    return make_frame(close)


@pytest.fixture
def trending_frame():
    """A clean upward ramp -- trend signals must be long, reversion signals short."""
    return make_frame(np.linspace(10.0, 30.0, 600))


@pytest.fixture
def sawtooth_frame():
    """A deterministic oscillation -- strongly mean reverting by construction."""
    t = np.arange(600)
    return make_frame(20.0 + 3.0 * np.sin(2 * np.pi * t / 20.0))
