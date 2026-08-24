# Retail Quant Signal Backtester

A walk-forward backtesting framework for statistical signals on daily equity
bars, built to test whether simple price-derived rules deviate from a random
walk by enough to survive realistic retail transaction costs.

**This is exploratory and educational software. It is not financial advice, not
a trading recommendation, and not a production trading system. It places no
orders and connects to no broker.**

---

## The honest framing

Renaissance Technologies' edge came from proprietary datasets, microsecond
execution, and the scale to combine thousands of individually worthless signals
into a portfolio. None of that is available at retail size, and no framework can
manufacture it.

What *is* transferable is the method: test whether a price series deviates from a
random walk, validate out of sample, and be ruthlessly honest about costs. That
is what this tool does. It is designed to tell you "no" clearly and often,
because "no" is the correct answer most of the time.

Three design choices follow from that, and they are enforced in code rather than
left to discipline:

1. **Every report shows gross and net side by side.** There is no code path in
   `reporting.py` that can print one without the other.
2. **A signal that fails to clear its costs says so in a sentence**, not just
   through numbers the reader has to compare.
3. **The per-period breakdown is always printed.** An aggregate Sharpe of 0.4
   assembled from windows of `[-2.8, +1.2, -0.4, +3.2]` is not a finding, and
   showing only the average would hide that.

---

## Setup

**Requires Python 3.11 or newer** — the pinned pandas and numpy do not build on
anything older. Check with `python3 --version`; if it's too old, install a
current Python (macOS: `brew install python@3.13`, Windows: the python.org
installer, or `pyenv` anywhere) before continuing.

### macOS / Linux

```bash
cd quant-backtester
make setup          # creates .venv, installs everything, runs the health check
```

Or without `make`:

```bash
cd quant-backtester
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python backtest.py --doctor
```

### Windows (PowerShell)

```powershell
cd quant-backtester
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python backtest.py --doctor
```

`pip install -e .` is optional but convenient: it puts a `qsb-backtest` command
on your PATH that works from any directory. Without it, run `python backtest.py`
from inside `quant-backtester/`.

### Check it worked

```bash
python backtest.py --doctor
```

This is the first thing to run, and the first thing to re-run when something
misbehaves. It verifies your Python version, that every dependency is present
and new enough, that the package imports, that your config parses, what's in
your cache, and whether market data is actually reachable from this machine —
each failure naming the command that fixes it.

```
[ ok ] python          3.13.2 (CPython)
[ ok ] virtualenv      active (.venv)
[ ok ] pandas          3.0.5
[warn] pyarrow         not installed -- only needed for cache_format "parquet"
[ ok ] qsb package     imports cleanly, 5 signals registered
[ ok ] yahoo finance   query1.finance.yahoo.com and fc.yahoo.com reachable
```

The network check is more careful than it looks. "No data" has several causes
that need completely different fixes, so it distinguishes a machine that is
fully offline, a machine where the internet works but Yahoo specifically is
blocked (corporate network, VPN, DNS filter, egress policy), and the case where
Yahoo's data host is reachable but `fc.yahoo.com` is not — that last one matters
because yfinance uses it for a cookie handshake and fails confusingly without it.
When a proxy is configured it probes *through* the proxy, since a direct socket
test would answer the wrong question.

---

## Quick start

```bash
# See what signals exist and what each one assumes
python backtest.py --list-signals

# The brief's example: two tickers, one signal
python backtest.py --tickers UAMY SMR --signals mean_reversion

# Everything, with the random-walk diagnostics and the hold-out validation
python backtest.py --signals all --diagnostics --validate

# Re-run from cache without touching the network
python backtest.py --tickers UAMY --signals momentum_roc --offline
```

With `make`: `make run`, `make full`, `make test`, `make signals`, `make doctor`.
Override the defaults inline — `make run TICKERS="AEM GOLD" SIGNALS=momentum_roc`.

Copy `config.example.json` to `config.json` to set your own universe and cost
assumptions. `config.json` is gitignored, as is `data_cache/`.

---

## Where data comes from

Bars arrive through a **source**, selected with `--source`. Whichever you use,
the same validation, the same repair counting, and the same cleaning report
apply — a broker export gets exactly the scrutiny a download gets.

### `yfinance` (default)

```bash
python backtest.py --tickers UAMY SMR --signals mean_reversion
```

Free and convenient, with the caveats that implies: undocumented endpoints, no
uptime guarantee, and a habit of breaking on Yahoo's schedule rather than yours.
Downloads are cached per ticker under `data_cache/`, so repeat runs don't re-hit
the API.

### `csv` — files you exported yourself

```bash
python backtest.py --source csv --csv-dir ~/market-data --signals all
```

For when Yahoo is unreachable, unreliable, or you're paying for better data.
Point it at a directory and it reads what's there:

```
~/market-data/
  UAMY_1d.csv      # preferred: {TICKER}_{interval}.csv
  SMR.csv          # also fine: {TICKER}.csv
```

The reader is deliberately forgiving, because these files come from wherever you
could get them. Headers are matched case-insensitively against the usual
spellings (`Adj Close`, `adj_close`, `Vol`, `volume`, …), the date column is
auto-detected (`Date`, `datetime`, `timestamp`, or an unnamed first column), and
rows are sorted and trimmed to your window. **Only `Close` is required** —
Open/High/Low are backfilled from it when absent or zero, which close-only vendor
files and many broker exports need. Anything it can't parse produces an error
naming the columns it actually found.

Fetched bars are cached like any others, so after one run you can drop the
exports and use `--offline`.

### Adding your own source

`qsb/sources.py` defines a three-method interface. A new vendor is a small class
plus a `register()` call, not surgery on the loader:

```python
class MySource:
    name = "mine"
    def available(self) -> tuple[bool, str]: ...
    def fetch(self, ticker, start, end, interval) -> pd.DataFrame: ...

sources.register(MySource())
```

Return raw bars and stop there — cleaning is not your job, and doing it in the
source would bypass the cleaning report.

---

## The signals, in plain language

Each signal is a pure function of past prices that outputs a target position of
long (+1), flat (0), or short (−1). None of them use fundamentals, news, or
anything but the price and volume series itself.

### `mean_reversion` — z-score reversion
Assume an unusually large move away from the recent average tends to snap back.
Buy when the stock closes a set number of standard deviations *below* its own
trailing mean, short when it is that far above, and step aside once it returns to
normal. The separate entry and exit thresholds create a band, which stops the
position flickering on and off every time the z-score jitters across a single
line.

### `rsi_reversion` — RSI extremes
The same "it snaps back" premise, measured differently: RSI compares the size of
recent up moves to recent down moves on a 0–100 scale. Buy when RSI says
oversold, short when it says overbought, exit on the return to the midline.
Running this alongside `mean_reversion` is deliberate — if one works and the
other does not on the same ticker, that tells you something about the specific
threshold rather than about mean reversion as an idea.

### `ma_crossover` — trend continuation
Assume trends persist. Hold long while a short-term moving average sits above a
long-term one, and flip short when it crosses below. This is the **opposite** bet
to the two reversion signals. On any given ticker at most one of them should look
good, and often neither does — which is itself useful information.

### `momentum_roc` — rate of change
Look at how much the stock has moved over the last N days. Up more than a set
threshold, go long; down more than that, go short; otherwise stand aside. The
dead band around zero keeps the position from flipping on noise in a sideways
tape.

### `vol_regime` — squeeze breakout
Quiet periods often precede large moves. Wait until Bollinger bandwidth is
unusually narrow *compared with its own trailing history*, then follow the
direction in which price eventually breaks out of the band. The comparison uses a
rolling quantile, never a full-sample one — a full-sample quantile would let the
signal know how volatile the stock was going to become later.

---

## How the backtest works

### Walk-forward, not a single in-sample fit

```
|---- train 1 ----|-- test 1 --|
          |---- train 2 ----|-- test 2 --|
                    |---- train 3 ----|-- test 3 --|
```

Each signal declares a small parameter grid. On every **training** window the
engine searches that grid and picks a winner by net Sharpe. Those parameters are
then frozen and applied to the **test** window that immediately follows. Only the
concatenated test windows are reported as the result — no bar the parameters were
fitted on ever appears in the headline number.

Grids are kept deliberately small. A grid with hundreds of combinations will find
something that "works" on any training window at all.

Selection uses *net* Sharpe rather than gross on purpose. Selecting on gross would
pick the highest-turnover parameters every time and then hand them to the test
window to pay for. A side effect worth knowing: because selection is cost-aware,
running the same signal under the options cost regime can legitimately choose
different parameters than under the stock regime.

### The one-bar execution shift

```
positions[t]  is the decision made on the CLOSE of bar t
held[t]       = positions[t-1]        <- the execution shift
gross[t]      = held[t] * return[t]
turnover[t]   = |held[t] - held[t-1]|
net[t]        = gross[t] - turnover[t] * cost_fraction[t]
```

This shift happens in exactly one place (`metrics.compute_returns`). Signals do
not pre-shift and the engine does not re-shift; doing it twice would hide a real
edge, doing it in neither place would manufacture a fake one. Two tests guard
this: `test_no_lookahead_bias_in_return_construction` feeds the engine an oracle
signal and asserts it is *not* profitable, and `test_signals_are_causal` rewrites
the future of every price series and asserts no past position changes.

---

## The cost model

Retail equity and retail options are two different cost regimes. Blending them
into one average is the fastest way to make a marginal signal look viable, so
they are modelled separately.

| | Stock | Option |
|---|---|---|
| Commission | **$0.00** — enforced; the config refuses a non-zero stock commission | per contract, configurable |
| Half spread | narrow (default 5 bps) | wide (default 150 bps) |
| Slippage | default 2 bps | default 50 bps |
| Cost varies with price? | No — constant fraction | **Yes** |

Option cost varies with price because a flat per-contract commission is a much
larger percentage of a cheap contract than an expensive one: $0.65 is 1.3% of a
$50 contract and 0.065% of a $1,000 one. The model reflects that rather than
assuming a single blended figure.

### Options are a proxy in this phase

There is no options chain data in this build. When you run `--instrument option`,
option-level costs are applied to a backtest whose returns come from the
**underlying's** price series. That ignores delta, gamma, theta, vega, and
assignment. It is a cost-sensitivity check — "could this idea survive
options-level frictions at all?" — and **not** an options backtest. Every report
in that mode carries a `PROXY` warning. Do not draw conclusions about an options
strategy from it until real chain data lands.

### Taxes are modelled as ZERO

Deliberately, for this phase. No tax drag, no wash-sale treatment, no holding
period distinction, no tax-loss harvesting. This is a simplification that makes
signals comparable *to each other*; it is **not** a real tax assumption, and every
figure this tool prints is pre-tax. There is intentionally no tax setting to
configure and forget about. Real after-tax results on a strategy turning over 30
times a year would be materially worse than what you see here.

---

## Reading the report

### `AGGREGATE OUT-OF-SAMPLE PERFORMANCE`
Gross and net columns, plus the difference. Hit rate, average win, and average
loss are computed **per trade** (a maximal run of constant non-zero position),
while Sharpe, drawdown, and volatility are computed per bar. `Turnover
(units/yr)` of 28.5 means roughly 14 round trips a year.

### `COST HURDLE`
The most important block in the report, and the one that answers the brief's
central question directly:

```
  Gross edge earned   :      46.5 bps per unit traded
  Assumed cost        :       7.0 bps per unit traded
  Margin              :      39.5 bps (clears the hurdle)
  Cost is             :      6.65x covered by gross edge (need > 1.0)
```

"Gross edge per unit traded" is total gross P&L divided by total turnover. It is
the honest comparison against cost, because both are expressed per unit of
notional moved. A signal earning 4 bps per unit traded cannot survive a 7 bps
one-way cost no matter how impressive its Sharpe looks, and this block says so
before you get attached to the equity curve.

### `PER-PERIOD WALK-FORWARD BREAKDOWN`
One row per out-of-sample test window, with the parameters chosen for it. Read
this before the aggregate. The summary lines beneath it give the spread of net
Sharpe across windows, how many windows were actually profitable, and the
train-minus-test Sharpe gap — the overfitting tax, made explicit.

### `FLAGS`
Raised automatically:

- **UNSTABLE** — out-of-sample Sharpe varies by more than 1.0 across sub-periods.
- **MAJORITY NEGATIVE** — fewer than half the windows were profitable net of costs.
- **OVERFIT GAP** — training Sharpe exceeds test Sharpe by more than 1.0 on
  average; the search is fitting the training window, not finding a durable effect.
- **parameters changed between windows** — the aggregate blends several different
  strategies rather than measuring one.

### `VERDICT`
One of five codes, each with a sentence explaining it:

| Code | Meaning |
|---|---|
| `SURVIVES_COSTS` | Positive net of costs. A survived test, **not** a validated edge. |
| `COSTS_KILL_IT` | Real gross edge, destroyed by frictions. The ordinary outcome. |
| `NO_GROSS_EDGE` | Loses money even with zero costs — costs are not the problem. |
| `NO_TRADES` | Never triggered; nothing to evaluate. |
| `INSUFFICIENT_HISTORY` | Not enough bars for one train/test window. No result produced. |

---

## Random-walk diagnostics (`--diagnostics`)

This tests the premise the whole project rests on. If a ticker's returns are
indistinguishable from a random walk, no signal computed from that price series
alone can have an edge.

**Autocorrelation** — are returns at lag *k* related to returns at lag 0? Under a
random walk all autocorrelations are zero, and sample estimates fall inside
±1.96/√n.

**Variance ratio (Lo–MacKinlay)** — does variance scale linearly with holding
period? Under a random walk, VR(q) = 1.

- VR > 1 → positive serial correlation (trending)
- VR < 1 → negative serial correlation (mean reverting)

The heteroskedasticity-robust z-statistic is used, because equity returns are
emphatically not homoskedastic and the homoskedastic version rejects far too
often on volatile small caps.

**The multiple-testing guard.** Running fourteen tests at the 5% level and
finding two "significant" results is arithmetic, not a discovery. The verdict
compares the *number* of rejections against a binomial null and reports the
probability that noise alone produced at least that many. A single p < 0.05
across four horizons and ten lags is exactly what you should expect from nothing.

---

## Validation before trusting any signal (`--validate`)

A signal that looks good on UAMY, SMR, BBAI, and AEM has been tested on four
names selected *because they were already being watched*. That is a hypothesis,
not a finding.

`--validate` re-runs each signal on a hold-out universe spanning different
sectors, market caps, and liquidity regimes, none of which were used to develop
anything. The config **refuses to load** if the hold-out universe shares any
ticker with the development universe.

The two sets are reported **separately and never pooled into one statistic** —
pooling would let four lucky small caps carry ten unrelated names, or vice versa,
destroying the only thing the comparison is for. Comparisons use the median
across tickers, so one spectacular hold-out result cannot rescue a failing signal.

| Judgement | Meaning |
|---|---|
| `CONFIRMED` | Positive on both sets. The strongest evidence this tool can produce — still far short of proof. |
| `WEAKENED` | Survives but degrades. Size expectations to the hold-out number, not the development one. |
| `OVERFIT` | Positive on development names, absent on hold-out. Treat as fitted to those four names. |
| `REVERSED` | Positive on development, *negative* on hold-out. Likely a property of those specific tickers. |
| `NO_DEV_EDGE` | Nothing to validate. |

**Multiple historical windows.** The walk-forward already tests each signal
across several distinct six-month periods and reports each separately, which
covers the brief's fourth validation requirement. Use `--start` and `--end` to
carve out further independent stretches of history.

---

## Data handling

`yfinance` is convenient and free, which means it is also occasionally wrong.
Every repair is counted and printed in the `DATA` block — silent repair is how a
backtest ends up measuring a data bug instead of a signal.

Defended against: duplicate timestamps, zero/negative/NaN prices, missing
sessions, NaN volume, out-of-order indices, tz-aware indices, MultiIndex columns,
and split/adjustment artifacts (flagged as `suspect moves`, **kept** — small caps
genuinely do move 60% in a day, and deleting real moves would flatter every
reversion signal here).

**Gaps are never forward-filled.** A hole in the data stays a hole and the return
series spans it. Forward-filling prices would manufacture zero-return days and
inflate every Sharpe ratio in the report.

**Thin liquidity.** Average daily dollar volume is computed and reported per
ticker. Sessions below `min_avg_dollar_volume` are marked untradable and forced
flat — not dropped, since dropping them would splice unrelated prices together
and create phantom returns.

**Caching.** Downloads are cached per ticker as CSV (or parquet, with pyarrow) so
repeated runs do not re-hit the API. `--offline` uses only the cache; the entire
test suite runs offline.

**Per-ticker evaluation.** Signals are always evaluated per ticker and never
pooled. A result on a thinly traded small cap is not expected to generalize to a
large cap or a different liquidity regime, and each new ticker is a fresh test —
not evidence for tickers already tested.

---

## Project structure

```
quant-backtester/
├── backtest.py            runnable entry point (works straight from a clone)
├── pyproject.toml         packaging; provides the `qsb-backtest` command
├── Makefile               setup / test / run shortcuts
├── config.example.json    copy to config.json and edit
├── requirements.txt       pinned dependencies
├── qsb/
│   ├── cli.py             argument parsing and the run pipeline
│   ├── config.py          config resolution; cost and walk-forward specs
│   ├── sources.py         pluggable data sources (yfinance, csv)
│   ├── data.py            validation, cleaning, caching, liquidity profiling
│   ├── signals.py         the signal library + parameter grids
│   ├── costs.py           split stock/option cost model
│   ├── metrics.py         return construction, trades, performance stats
│   ├── engine.py          walk-forward driver and stability analysis
│   ├── diagnostics.py     autocorrelation and variance-ratio tests
│   ├── validation.py      broader-universe comparison and overfit judgements
│   ├── reporting.py       report formatting
│   └── doctor.py          environment preflight check
└── tests/                 253 tests, no network required
```

Nothing about the ticker universe, cost numbers, or window sizes is hardcoded in
`signals.py`, `engine.py`, or `metrics.py`. That is a hard rule.

---

## Tests

```bash
pytest                    # 253 tests, ~40s, no network
pytest tests/test_signals.py -v
make test                 # same thing via the venv
```

The suite covers signal calculations against known and synthetic inputs (RSI of a
strictly rising series is 100; a rolling z-score matches hand calculation), the
edge cases the brief names (missing data, a signal that never triggers, zero
trades, insufficient history), the cost model's stock/option split, and the
reporting invariants.

Three tests matter more than the rest:

- **`test_signals_are_causal`** rewrites the tail of every price series and
  asserts no position at or before the cut changes. This catches the one bug
  class — a centred window, a full-sample quantile, a negative shift — that would
  silently invalidate every result the framework produces.
- **`test_no_lookahead_bias_in_return_construction`** feeds the engine a signal
  that knows each bar's own return and asserts it is *not* profitable.
- **`test_a_random_walk_does_not_yield_a_surviving_edge`** runs all five signals
  and the full parameter search over structureless data and asserts nothing comes
  out looking profitable. If the walk-forward split ever leaks, this fails.

The variance-ratio implementation is checked against theory: for AR(1) returns
with coefficient φ, VR(2) = 1 + φ exactly, and the tests assert that for
φ = −0.4 and φ = +0.3.

---

## Troubleshooting

**Start with `python backtest.py --doctor`.** It diagnoses most of what follows
and names the fix.

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'qsb'` | Running from the wrong directory. `cd quant-backtester`, or `pip install -e .` to run from anywhere. |
| pandas/numpy fail to install | Python older than 3.11. Check `python3 --version`, install a newer one, delete `.venv`, and redo setup. |
| `yfinance returned no rows` | Several causes — `--doctor` tells them apart: a delisted or mistyped symbol, a range with no sessions, an interval Yahoo won't serve that far back (intraday history is short), rate limiting, or no route to Yahoo. |
| Yahoo unreachable but the internet works | A corporate network, VPN, DNS filter, or egress policy is blocking it. Allow `query1`/`query2`/`fc.yahoo.com`, or export bars elsewhere and use `--source csv`. |
| `INSUFFICIENT_HISTORY` | Fewer bars than one train+test window. Use an earlier `--start`, or shrink `--train-days` / `--test-days`. |
| `NO_TRADES` everywhere | Usually the liquidity filter. Check the `forced flat` percentage in the LIQUIDITY block and lower `--min-dollar-volume` if it's screening out the whole sample. |
| Results changed between runs | The cache went stale or was refreshed. `make clean-data` then re-run for a clean comparison. |

---

## What this cannot tell you

- **Whether a signal will work going forward.** It measures history. A signal
  that survives every check here has cleared a low bar, not a high one.
- **Real transaction costs.** Spread and slippage are *modelled assumptions*, not
  observed fills. Replace the defaults with numbers from your own executions
  before trusting any net figure. Slippage on a thin small cap is worse than any
  constant-bps model implies, and worst exactly when a signal most wants to trade.
- **Anything about options strategies.** See the proxy warning above.
- **After-tax results.** Taxes are zero here. See above.
- **Capacity.** The model assumes you can trade one unit of notional at the close
  without moving the price. On UAMY or BBAI in a thin stretch, that assumption
  degrades quickly.

The framework also cannot protect you from the deepest problem in backtesting:
every signal you try on the same data spends a little more of its statistical
budget. The walk-forward split, the hold-out universe, and the multiple-testing
guard narrow that leak. They do not close it.

---

## Future enhancements

Not required for the initial build; revisit once the core pipeline has been run
against real data.

1. Real options chain data — bid/ask, implied volatility, open interest, skew —
   replacing the proxy cost model.
2. Earnings and corporate-news event tagging, to separate performance around a
   known catalyst from performance without one.
3. Sector and peer correlation, to see whether a signal's performance on one
   ticker is idiosyncratic or just riding a sector move.
4. Macro overlay data (Treasury yields, relevant commodity futures) to test
   whether signal performance shifts with the macro backdrop.
5. Short interest and float data for squeeze-prone small caps, since squeeze
   dynamics break normal mean-reversion assumptions.
6. Intraday or tick data, for signals that may only exist at shorter timeframes.
7. Broader universe screening — scan a large list to find which names a signal
   currently applies to, rather than only testing names already tracked.
8. A live paper-trading mode tracking a validated signal forward in real time
   with no money at risk, as a further check beyond historical backtesting.

No live trading or brokerage execution is in scope for this phase. If live signal
monitoring is wanted later, it should be scoped explicitly and separately.
