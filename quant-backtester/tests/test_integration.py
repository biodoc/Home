"""End-to-end runs through the CLI, using a seeded cache and no network.

These tests write synthetic price data into a cache directory and then invoke
`backtest.py --offline`, which exercises config resolution, data loading,
cleaning, the walk-forward engine, reporting, and the CSV/JSON exports as one
pipeline. If yfinance is ever called here, `--offline` turns that into a loud
failure rather than a silent network dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest  # noqa: E402


def seed_cache(cache_dir: Path, ticker: str, n: int = 900, seed: int = 0,
               drift: float = 0.0) -> None:
    """Write a synthetic price history where the loader expects to find one."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    close = 40.0 * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))
    index = pd.bdate_range("2016-01-04", periods=n)
    pd.DataFrame(
        {
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.uniform(5e5, 5e6, n),
        },
        index=index,
    ).rename_axis("Date").to_csv(cache_dir / f"{ticker}_1d.csv")


@pytest.fixture
def cache(tmp_path):
    d = tmp_path / "cache"
    for i, ticker in enumerate(["AAA", "BBB"]):
        seed_cache(d, ticker, seed=i)
    return d


def run_cli(argv, capsys):
    code = backtest.main(argv)
    return code, capsys.readouterr().out


def test_end_to_end_run_produces_a_report(cache, capsys):
    code, out = run_cli([
        "--tickers", "AAA", "--signals", "mean_reversion",
        "--cache-dir", str(cache), "--offline",
        "--train-days", "252", "--test-days", "126",
    ], capsys)

    assert code == 0
    assert "AAA" in out
    assert "AGGREGATE OUT-OF-SAMPLE PERFORMANCE" in out
    assert "GROSS" in out and "NET" in out
    assert "COST HURDLE" in out
    assert "PER-PERIOD WALK-FORWARD BREAKDOWN" in out
    assert "VERDICT:" in out
    assert "TAXES ARE MODELLED AS ZERO" in out


def test_multiple_tickers_and_signals_are_reported_separately(cache, capsys):
    """Per the brief: results are never pooled across tickers."""
    code, out = run_cli([
        "--tickers", "AAA", "BBB",
        "--signals", "mean_reversion", "momentum_roc",
        "--cache-dir", str(cache), "--offline",
        "--train-days", "252", "--test-days", "126",
    ], capsys)

    assert code == 0
    assert out.count("AGGREGATE OUT-OF-SAMPLE PERFORMANCE") == 4
    assert "DEVELOPMENT SET SUMMARY" in out
    for ticker in ("AAA", "BBB"):
        for signal in ("mean_reversion", "momentum_roc"):
            assert f"{ticker}  |  signal: {signal}" in out


def test_option_instrument_switches_the_cost_regime(cache, capsys):
    code, out = run_cli([
        "--tickers", "AAA", "--signals", "mean_reversion",
        "--cache-dir", str(cache), "--offline", "--instrument", "option",
        "--train-days", "252", "--test-days", "126",
    ], capsys)

    assert code == 0
    assert "per contract" in out
    assert "PROXY" in out


def test_diagnostics_flag_runs_the_random_walk_tests(cache, capsys):
    code, out = run_cli([
        "--tickers", "AAA", "--signals", "mean_reversion",
        "--cache-dir", str(cache), "--offline", "--diagnostics",
        "--train-days", "252", "--test-days", "126",
    ], capsys)

    assert code == 0
    assert "RANDOM-WALK DIAGNOSTICS" in out
    assert "VARIANCE RATIO" in out


def test_validation_flag_reports_the_two_sets_separately(cache, capsys):
    for i, ticker in enumerate(["VVV", "WWW"]):
        seed_cache(cache, ticker, seed=100 + i)

    code, out = run_cli([
        "--tickers", "AAA", "BBB", "--signals", "mean_reversion",
        "--cache-dir", str(cache), "--offline", "--validate",
        "--train-days", "252", "--test-days", "126",
        "--config", str(_write_cfg(cache)),
    ], capsys)

    assert code == 0
    assert "DEVELOPMENT SET SUMMARY" in out
    assert "HOLD-OUT SET SUMMARY" in out
    assert "BROADER-UNIVERSE VALIDATION" in out
    assert "never combined into" in out
    assert "JUDGEMENT:" in out


def _write_cfg(cache: Path) -> Path:
    path = cache / "cfg.json"
    path.write_text(json.dumps({
        "universe": ["AAA", "BBB"],
        "validation_universe": ["VVV", "WWW"],
        "min_avg_dollar_volume": 0.0,
    }))
    return path


def test_csv_and_json_exports_are_written(cache, tmp_path, capsys):
    csv_out = tmp_path / "out" / "run.csv"
    json_out = tmp_path / "out" / "run.json"

    code, _ = run_cli([
        "--tickers", "AAA", "--signals", "mean_reversion",
        "--cache-dir", str(cache), "--offline",
        "--train-days", "252", "--test-days", "126",
        "--out", str(csv_out), "--json", str(json_out),
    ], capsys)

    assert code == 0
    frame = pd.read_csv(csv_out)
    assert len(frame) == 1
    assert {"ticker", "signal", "gross_sharpe", "net_sharpe", "verdict"} <= set(frame.columns)

    payload = json.loads(json_out.read_text())
    assert payload["config"]["universe"] == ["AAA"]
    assert len(payload["results"]) == 1


def test_offline_mode_never_reaches_the_network(cache, capsys, monkeypatch):
    """Guard against a cached run silently falling through to a download."""
    def explode(*args, **kwargs):
        raise AssertionError("offline run attempted a network fetch")

    monkeypatch.setattr(backtest.data, "fetch_prices", explode)
    code, _ = run_cli([
        "--tickers", "AAA", "--signals", "mean_reversion",
        "--cache-dir", str(cache), "--offline",
        "--train-days", "252", "--test-days", "126",
    ], capsys)
    assert code == 0


def test_a_missing_ticker_is_logged_without_killing_the_run(cache, capsys):
    code, out = run_cli([
        "--tickers", "AAA", "MISSING", "--signals", "mean_reversion",
        "--cache-dir", str(cache), "--offline",
        "--train-days", "252", "--test-days", "126",
    ], capsys)
    assert code == 0
    assert "AGGREGATE OUT-OF-SAMPLE PERFORMANCE" in out


def test_all_tickers_missing_exits_nonzero(tmp_path, capsys):
    code, _ = run_cli([
        "--tickers", "NOPE", "--signals", "mean_reversion",
        "--cache-dir", str(tmp_path), "--offline",
    ], capsys)
    assert code == 1


def test_unknown_signal_exits_nonzero(cache, capsys):
    code, _ = run_cli([
        "--tickers", "AAA", "--signals", "not_a_signal",
        "--cache-dir", str(cache), "--offline",
    ], capsys)
    assert code == 2


def test_overlapping_holdout_universe_is_rejected(cache, capsys):
    path = cache / "bad.json"
    path.write_text(json.dumps({
        "universe": ["AAA"], "validation_universe": ["AAA", "VVV"],
    }))
    code, _ = run_cli(["--config", str(path), "--offline"], capsys)
    assert code == 2


def test_list_signals_describes_every_signal(capsys):
    code, out = run_cli(["--list-signals"], capsys)
    assert code == 0
    from qsb import signals
    for name in signals.available():
        assert name in out
    assert "In plain language" in out


def test_signals_all_expands_to_the_whole_registry(cache, capsys):
    from qsb import signals

    code, out = run_cli([
        "--tickers", "AAA", "--signals", "all",
        "--cache-dir", str(cache), "--offline",
        "--train-days", "252", "--test-days", "126",
    ], capsys)
    assert code == 0
    assert out.count("AGGREGATE OUT-OF-SAMPLE PERFORMANCE") == len(signals.available())


def test_thin_liquidity_filter_is_applied_from_the_cli(cache, capsys):
    code, out = run_cli([
        "--tickers", "AAA", "--signals", "mean_reversion",
        "--cache-dir", str(cache), "--offline",
        "--min-dollar-volume", "1e12",     # nothing can clear this
        "--train-days", "252", "--test-days", "126",
    ], capsys)
    assert code == 0
    assert "forced flat" in out
    assert "NO_TRADES" in out


# ---------------------------------------------------------------------------
# Local-run surface: CSV source, doctor, and the installed entry point
# ---------------------------------------------------------------------------
def test_csv_source_runs_a_full_backtest_without_a_cache(tmp_path, capsys):
    """The offline path is a first-class source, not a degraded fallback."""
    exports = tmp_path / "exports"
    exports.mkdir()
    rng = np.random.default_rng(21)
    close = 25.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 900)))
    pd.DataFrame(
        {"close": close, "volume": rng.uniform(1e6, 5e6, 900)},
        index=pd.bdate_range("2016-01-04", periods=900),
    ).rename_axis("date").to_csv(exports / "LOCAL_1d.csv")

    code, out = run_cli([
        "--tickers", "LOCAL", "--signals", "mean_reversion",
        "--source", "csv", "--csv-dir", str(exports),
        "--cache-dir", str(tmp_path / "cache"),
        "--train-days", "252", "--test-days", "126",
    ], capsys)

    assert code == 0
    assert "data source          : csv" in out
    assert "AGGREGATE OUT-OF-SAMPLE PERFORMANCE" in out
    assert "VERDICT:" in out
    # The fetch is cached on the way through, so a later run needs no exports.
    assert (tmp_path / "cache" / "LOCAL_1d.csv").exists()


def test_csv_source_without_a_directory_exits_nonzero(capsys):
    code, _ = run_cli(["--tickers", "AAA", "--source", "csv"], capsys)
    assert code == 2


def test_csv_source_pointed_at_nothing_exits_nonzero(tmp_path, capsys):
    code, _ = run_cli([
        "--tickers", "AAA", "--source", "csv", "--csv-dir", str(tmp_path / "empty"),
    ], capsys)
    assert code == 2


def test_unknown_source_exits_nonzero(cache, capsys):
    code, _ = run_cli([
        "--tickers", "AAA", "--signals", "mean_reversion",
        "--source", "bloomberg", "--cache-dir", str(cache),
    ], capsys)
    assert code == 2


def test_doctor_runs_and_reports(capsys):
    code, out = run_cli(["--doctor", "--no-network"], capsys)
    assert "ENVIRONMENT CHECK" in out
    assert "python" in out
    assert "qsb package" in out
    assert code in (0, 1)


def test_doctor_exit_code_reflects_blocking_problems(capsys, monkeypatch):
    from qsb import doctor as doctor_mod

    def broken(report, *a, **k):
        report.add("python", doctor_mod.FAIL, "too old", "upgrade")

    monkeypatch.setattr(doctor_mod, "check_python", broken)
    code, out = run_cli(["--doctor", "--no-network"], capsys)
    assert code == 1
    assert "blocking problem" in out


def test_doctor_reports_the_configured_cache(tmp_path, capsys):
    (tmp_path / "UAMY_1d.csv").write_text("Date,Close\n2024-01-02,10\n")
    code, out = run_cli([
        "--doctor", "--no-network", "--cache-dir", str(tmp_path),
    ], capsys)
    assert "UAMY" in out


def test_installed_console_script_is_importable():
    """`qsb-backtest` resolves to the same entry point as backtest.py."""
    from qsb.cli import main as package_main

    assert backtest.main is package_main


def test_offline_run_needs_no_source_configuration(cache, capsys):
    """--offline must not require a usable source; the cache is enough."""
    code, out = run_cli([
        "--tickers", "AAA", "--signals", "mean_reversion",
        "--cache-dir", str(cache), "--offline", "--source", "csv",
        "--train-days", "252", "--test-days", "126",
    ], capsys)
    assert code == 0
    assert "cache only (--offline)" in out
