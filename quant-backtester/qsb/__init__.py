"""
qsb -- Quant Signal Backtester.

A small, deliberately transparent framework for testing statistical signals on
daily equity bars at retail scale.

Nothing in this package is financial advice, and nothing here is a production
trading system. It exists to answer one narrow question honestly: does a given
rule, applied to a given ticker, deviate from a random walk by enough to survive
realistic retail transaction costs?

Module map (kept deliberately small and separable):
    config      -- configuration loading and cost/walk-forward specs
    data        -- download, validation, cleaning, caching, liquidity profiling
    signals     -- the signal library, each a pure function of past bars
    costs       -- split cost model: stock (spread+slippage) vs option (+commission)
    metrics     -- return construction, trade extraction, performance statistics
    engine      -- walk-forward backtest driver
    diagnostics -- autocorrelation and variance-ratio random-walk tests
    validation  -- broader-universe out-of-sample checks and overfit flags
    reporting   -- human-readable report formatting
"""

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "signals",
    "costs",
    "metrics",
    "engine",
    "diagnostics",
    "validation",
    "reporting",
]
