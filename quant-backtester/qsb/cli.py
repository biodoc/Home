"""
Quant Signal Backtester -- command line interface.

Invoked either as `python backtest.py ...` from the project directory or as
`qsb-backtest ...` once installed with `pip install -e .`.

Examples
--------
Check that this machine is set up correctly:
    python backtest.py --doctor

List what is available:
    python backtest.py --list-signals

Run two signals on two tickers (the brief's example):
    python backtest.py --tickers UAMY SMR --signals mean_reversion

Everything in the configured universe, all signals, with the random-walk
diagnostics and the broader-universe validation:
    python backtest.py --signals all --diagnostics --validate

Re-run without touching the network, using the local cache:
    python backtest.py --tickers UAMY --signals momentum_roc --offline

Check how the same signal fares under options frictions:
    python backtest.py --tickers AEM --signals mean_reversion --instrument option

Work from exported CSVs instead of downloading:
    python backtest.py --source csv --csv-dir ~/market-data --signals all

Save machine-readable output:
    python backtest.py --signals all --out results/run.csv

Nothing this program prints is financial advice. Taxes are modelled as zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import data, diagnostics, doctor, engine, reporting, signals, sources, validation
from .config import Config
from .costs import CostModel

logger = logging.getLogger("backtest")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backtest.py",
        description=(
            "Walk-forward backtester for statistical signals on daily equity "
            "bars. Exploratory and educational; not financial advice."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--tickers", nargs="+", metavar="SYM",
        help="tickers to test (default: the universe in the config file)",
    )
    p.add_argument(
        "--signals", nargs="+", metavar="NAME", default=["mean_reversion"],
        help="signal names, or 'all' (default: mean_reversion)",
    )
    p.add_argument("--config", metavar="PATH", help="JSON config file")
    p.add_argument("--start", metavar="YYYY-MM-DD", help="first date to load")
    p.add_argument("--end", metavar="YYYY-MM-DD", help="last date to load")
    p.add_argument("--interval", metavar="STR", help="bar interval (default 1d)")

    p.add_argument(
        "--instrument", choices=["stock", "option"], default="stock",
        help="which cost regime to charge (default: stock)",
    )
    p.add_argument(
        "--min-dollar-volume", type=float, metavar="USD", dest="min_avg_dollar_volume",
        help="force flat on sessions below this dollar volume (0 disables)",
    )

    p.add_argument("--train-days", type=int, help="walk-forward training window")
    p.add_argument("--test-days", type=int, help="walk-forward test window")

    p.add_argument(
        "--diagnostics", action="store_true",
        help="run the autocorrelation / variance-ratio random-walk tests",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="re-test each signal on the hold-out universe and flag overfitting",
    )

    p.add_argument(
        "--source", metavar="NAME",
        help=f"where to fetch bars from: {', '.join(sources.available())} "
             "(default: yfinance)",
    )
    p.add_argument(
        "--csv-dir", metavar="PATH", dest="csv_dir",
        help="directory of exported CSVs, for --source csv",
    )
    p.add_argument("--offline", action="store_true", help="use only cached data")
    p.add_argument(
        "--force-refresh", action="store_true", help="ignore the cache and re-download"
    )
    p.add_argument("--cache-dir", help="where cached price data lives")

    p.add_argument("--out", metavar="PATH", help="write a summary CSV here")
    p.add_argument("--json", metavar="PATH", dest="json_out",
                   help="write full results as JSON here")
    p.add_argument(
        "--list-signals", action="store_true",
        help="describe the available signals and exit",
    )
    p.add_argument(
        "--doctor", action="store_true",
        help="check that this machine is set up correctly, and exit",
    )
    p.add_argument(
        "--no-network", action="store_true",
        help="with --doctor, skip the connectivity probes",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="less logging")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def list_signals() -> None:
    print(reporting.RULE)
    print("AVAILABLE SIGNALS")
    print(reporting.RULE)
    for name in signals.available():
        sd = signals.get(name)
        combos = len(signals.valid_combos(sd))
        print(f"\n{name}")
        print(f"  {sd.description}")
        print(f"  grid: {combos} valid combinations from {sd.param_grid}")
        print(reporting._wrap(f"  In plain language: {sd.plain_english}", 4))
    print()


def load_all(cfg: Config, tickers: list[str], args, source=None) -> dict:
    """Load and clean every ticker up front, reporting failures without dying.

    One unavailable ticker should not abort a run over a dozen; each failure is
    logged with its reason and the rest continue.
    """
    loaded = {}
    for ticker in tickers:
        try:
            df, report = data.load_prices(
                ticker, cfg,
                force_refresh=args.force_refresh,
                offline=args.offline,
                source=source,
            )
            loaded[ticker] = (df, report)
        except Exception as exc:
            logger.error("could not load %s: %s", ticker, exc)
    return loaded


def run_one(df, report, ticker, signal_name, cost_model, cfg):
    """Backtest one ticker/signal pair and attach its data provenance."""
    sig = signals.get(signal_name)
    result = engine.run_walk_forward(
        df=df, signal_def=sig, cost_model=cost_model, cfg=cfg, ticker=ticker
    )
    result.cleaning = report
    result.liquidity = data.liquidity_profile(df, cfg.min_avg_dollar_volume)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else
              (logging.WARNING if args.quiet else logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.doctor:
        report = doctor.run(
            config_path=args.config,
            cache_dir=args.cache_dir,
            skip_network=args.no_network,
        )
        print(report.render())
        return 1 if report.failed else 0

    if args.list_signals:
        list_signals()
        return 0

    # -- configuration ------------------------------------------------------
    overrides = {
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "cache_dir": args.cache_dir,
        "min_avg_dollar_volume": args.min_avg_dollar_volume,
        "source": args.source,
        "csv_dir": args.csv_dir,
    }
    if args.tickers:
        overrides["universe"] = args.tickers
    wf = {k: v for k, v in
          {"train_days": args.train_days, "test_days": args.test_days}.items()
          if v is not None}
    if wf:
        overrides["walk_forward"] = wf

    try:
        cfg = Config.load(args.config, overrides)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    # -- signal selection ---------------------------------------------------
    names = args.signals
    if len(names) == 1 and names[0].lower() == "all":
        names = signals.available()
    try:
        names = [signals.get(n).name for n in names]
    except KeyError as exc:
        print(f"signal error: {exc}", file=sys.stderr)
        return 2

    cost_model = CostModel.from_config(cfg, args.instrument)

    # Build the data source once. Offline runs never fetch, so a source that is
    # unusable there is not worth complaining about.
    source = None
    if not args.offline:
        try:
            source = sources.build(cfg.source, cfg, args.csv_dir)
        except (KeyError, ValueError) as exc:
            print(f"data source error: {exc}", file=sys.stderr)
            return 2

        ok, why = source.available()
        if not ok:
            print(f"data source {cfg.source!r} is unusable: {why}", file=sys.stderr)
            print("Run `python backtest.py --doctor` for a full check.",
                  file=sys.stderr)
            return 2

    # -- development set ----------------------------------------------------
    print(reporting.RULE)
    print("QUANT SIGNAL BACKTESTER")
    print(reporting.RULE)
    print(f"development universe : {', '.join(cfg.universe)}")
    print(f"signals              : {', '.join(names)}")
    print(f"instrument           : {args.instrument}")
    print(f"data source          : "
          f"{'cache only (--offline)' if args.offline else cfg.source}")
    print(f"walk-forward         : {cfg.walk_forward.train_days}d train / "
          f"{cfg.walk_forward.test_days}d test, rolling")
    print(f"period               : {cfg.start} -> {cfg.end or 'today'}")
    print()

    loaded = load_all(cfg, cfg.universe, args, source)
    if not loaded:
        print("No tickers could be loaded. Nothing to do.", file=sys.stderr)
        return 1

    dev_results = []
    for ticker, (df, report) in loaded.items():
        for name in names:
            result = run_one(df, report, ticker, name, cost_model, cfg)
            dev_results.append(result)
            print(reporting.format_result(result))
            print()

    print(reporting.format_summary(dev_results, "DEVELOPMENT SET SUMMARY"))
    print()

    # -- random-walk diagnostics -------------------------------------------
    if args.diagnostics:
        dcfg = cfg.diagnostics
        for ticker, (df, _) in loaded.items():
            rep = diagnostics.random_walk_report(
                df["Close"], ticker,
                max_lag=int(dcfg.get("acf_max_lag", 10)),
                horizons=list(dcfg.get("variance_ratio_horizons", [2, 4, 8, 16])),
            )
            print(reporting.format_random_walk(rep))
            print()

    # -- broader-universe validation ---------------------------------------
    comparisons = []
    val_results = []
    if args.validate:
        print(reporting.RULE)
        print("Loading hold-out universe for validation...")
        print(reporting.RULE)
        val_loaded = load_all(cfg, cfg.validation_universe, args, source)
        if not val_loaded:
            print("Hold-out universe could not be loaded; validation skipped.",
                  file=sys.stderr)
        else:
            for ticker, (df, report) in val_loaded.items():
                for name in names:
                    val_results.append(
                        run_one(df, report, ticker, name, cost_model, cfg)
                    )

            print(reporting.format_summary(val_results, "HOLD-OUT SET SUMMARY"))
            print()

            for name in names:
                cmp = validation.compare(
                    name,
                    [r for r in dev_results if r.signal == name],
                    [r for r in val_results if r.signal == name],
                )
                comparisons.append(cmp)
                print(validation.format_comparison(cmp))
                print()

    # -- exports ------------------------------------------------------------
    all_results = dev_results + val_results
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = reporting.results_to_frame(dev_results)
        frame["set"] = "development"
        if val_results:
            vframe = reporting.results_to_frame(val_results)
            vframe["set"] = "holdout"
            frame = pd.concat([frame, vframe], ignore_index=True)
        frame.to_csv(out, index=False)
        print(f"wrote {out}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "universe": cfg.universe,
                "signals": names,
                "instrument": args.instrument,
                "start": cfg.start,
                "end": cfg.end,
            },
            "results": json.loads(
                reporting.results_to_frame(all_results).to_json(
                    orient="records", date_format="iso"
                )
            ),
            "validation": [
                {"signal": c.signal, "code": c.code, "message": c.message}
                for c in comparisons
            ],
        }
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
