#!/usr/bin/env python3
"""
Quant Signal Backtester -- runnable entry point.

Kept at the project root so the tool works straight from a clone with no install
step:

    python backtest.py --doctor
    python backtest.py --tickers UAMY SMR --signals mean_reversion

The implementation lives in `qsb/cli.py` so that the same code is reachable as
the `qsb-backtest` console script after `pip install -e .`. The re-exports below
keep `import backtest` working as a stable surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a clone without installing the package.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qsb import data, diagnostics, doctor, engine, reporting, signals, sources, validation  # noqa: E402,F401
from qsb.cli import build_parser, list_signals, load_all, main, run_one  # noqa: E402,F401

__all__ = [
    "main", "build_parser", "load_all", "run_one", "list_signals",
    "data", "diagnostics", "doctor", "engine", "reporting", "signals",
    "sources", "validation",
]

if __name__ == "__main__":
    raise SystemExit(main())
