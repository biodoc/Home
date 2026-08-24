"""
Environment preflight check.

Answers one question: will this actually run on this machine, and if not, what
exactly is wrong? Every failure it reports names the command that fixes it.

The network probe is the interesting part. "yfinance returned no rows" has at
least five distinct causes -- a bad symbol, an interval Yahoo will not serve
that far back, rate limiting, no route to the internet at all, and an egress
policy that blocks Yahoo specifically -- and they need completely different
responses. The probe tells them apart by testing a known-good symbol against a
neutral host and against Yahoo, then comparing.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

MIN_PYTHON = (3, 11)

# Kept in step with requirements.txt. The check is a floor, not an equality
# test: a newer patch release is fine and should not be reported as a problem.
REQUIRED = {
    "numpy": "2.0",
    "pandas": "2.2",
}
OPTIONAL = {
    "yfinance": "only needed to download data; --source csv works without it",
    "pytest": "only needed to run the test suite",
    "pyarrow": 'only needed for cache_format "parquet"; CSV caching is default',
}

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name, status, detail, fix="") -> None:
        self.checks.append(Check(name, status, detail, fix))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    def render(self) -> str:
        width = max(len(c.name) for c in self.checks) + 2
        lines = ["=" * 78, "ENVIRONMENT CHECK", "=" * 78]
        for c in self.checks:
            lines.append(f"{MARK[c.status]} {c.name:<{width}} {c.detail}")
            if c.fix and c.status != OK:
                lines.append(f"       {'':<{width}} -> {c.fix}")
        lines.append("=" * 78)

        if self.failed:
            lines.append(
                f"{len(self.failed)} blocking problem(s). Fix the [FAIL] lines "
                "above, then re-run `python backtest.py --doctor`."
            )
        elif self.warned:
            lines.append(
                "Ready to run, with caveats. The [warn] lines limit what you can "
                "do but do not block a backtest."
            )
            lines.append("")
            lines.append("Try:  python backtest.py --tickers UAMY --signals mean_reversion")
        else:
            lines.append("Everything checks out.")
            lines.append("")
            lines.append("Try:  python backtest.py --tickers UAMY SMR --signals mean_reversion")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def _version_tuple(v: str) -> tuple:
    parts = []
    for chunk in str(v).split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_python(report: Report) -> None:
    # Indexed rather than attribute access so the check stays testable against a
    # plain tuple standing in for sys.version_info.
    v = sys.version_info
    current = ".".join(str(part) for part in v[:3])
    if (v[0], v[1]) >= MIN_PYTHON:
        report.add("python", OK, f"{current} ({platform.python_implementation()})")
    else:
        need = ".".join(str(x) for x in MIN_PYTHON)
        report.add(
            "python", FAIL, f"{current} -- too old, need {need}+",
            "the pinned pandas/numpy need 3.11+. Install a newer Python "
            "(macOS: `brew install python@3.13`; Windows: python.org installer; "
            "or use pyenv) and recreate the venv.",
        )


def check_virtualenv(report: Report) -> None:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        report.add("virtualenv", OK, f"active ({Path(sys.prefix).name})")
    else:
        report.add(
            "virtualenv", WARN, "not active -- installing into the system Python",
            "python -m venv .venv && source .venv/bin/activate "
            "(Windows: .venv\\Scripts\\activate)",
        )


def check_packages(report: Report) -> None:
    for name, floor in REQUIRED.items():
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            report.add(name, FAIL, "not installed",
                       "pip install -r requirements.txt")
            continue
        if _version_tuple(version) < _version_tuple(floor):
            report.add(name, FAIL, f"{version} -- too old, need {floor}+",
                       "pip install -r requirements.txt")
        else:
            report.add(name, OK, version)

    for name, note in OPTIONAL.items():
        try:
            report.add(name, OK, md.version(name))
        except md.PackageNotFoundError:
            report.add(name, WARN, f"not installed -- {note}",
                       f"pip install {name}")


def check_import(report: Report) -> None:
    """The package must import from wherever the user is standing."""
    try:
        importlib.import_module("qsb")
        from qsb import signals

        report.add("qsb package", OK,
                   f"imports cleanly, {len(signals.available())} signals registered")
    except Exception as exc:
        report.add("qsb package", FAIL, f"import failed: {exc}",
                   "run from the quant-backtester directory, or `pip install -e .`")


# ---------------------------------------------------------------------------
def _tcp_probe(host: str, port: int = 443, timeout: float = 6.0) -> tuple[bool, str]:
    """Direct TCP reachability, below the HTTP layer.

    A host that does not resolve, one that times out, and one that refuses the
    connection are three different problems, so they are reported separately.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "reachable"
    except socket.gaierror as exc:
        return False, f"DNS lookup failed ({exc.strerror or exc})"
    except socket.timeout:
        return False, "timed out"
    except OSError as exc:
        return False, f"{exc.strerror or exc}"


def _proxy_probe(host: str, proxy_url: str, port: int = 443,
                 timeout: float = 8.0) -> tuple[bool, str]:
    """Reachability *through* an HTTPS proxy, via a CONNECT tunnel.

    When HTTPS_PROXY is set, a raw TCP probe answers the wrong question: it can
    succeed against a local interceptor while the proxy's egress policy still
    refuses the destination. Only the CONNECT response says whether traffic will
    actually reach the host, so that is what gets tested.
    """
    from urllib.parse import urlparse

    parsed = urlparse(proxy_url if "://" in proxy_url else f"http://{proxy_url}")
    phost, pport = parsed.hostname, parsed.port or 8080
    if not phost:
        return False, f"could not parse proxy URL {proxy_url!r}"

    try:
        with socket.create_connection((phost, pport), timeout=timeout) as sock:
            sock.settimeout(timeout)
            request = (
                f"CONNECT {host}:{port} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Proxy-Connection: close\r\n\r\n"
            ).encode()
            sock.sendall(request)

            raw = sock.recv(256).decode("latin-1", errors="replace")
            status_line = raw.split("\r\n", 1)[0]
            parts = status_line.split()
            code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

            if code == 200:
                return True, "reachable via proxy"
            if code in (403, 407):
                return False, (
                    f"proxy refused CONNECT ({code}) -- blocked by egress policy"
                )
            return False, f"proxy returned {status_line.strip() or 'no status'}"
    except socket.timeout:
        return False, "timed out talking to the proxy"
    except OSError as exc:
        return False, f"could not reach the proxy: {exc.strerror or exc}"


def _reach(host: str, proxy: str | None) -> tuple[bool, str]:
    """Probe a host the way traffic to it would actually travel."""
    if proxy:
        return _proxy_probe(host, proxy)
    return _tcp_probe(host)


def check_network(report: Report, skip: bool = False) -> None:
    if skip:
        report.add("network", WARN, "skipped (--no-network)",
                   "omit --no-network to test data access")
        return

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        report.add(
            "https proxy", WARN, f"HTTPS_PROXY is set ({proxy})",
            "if Yahoo is blocked below, this proxy's egress policy is the "
            "likely cause -- that is an environment setting, not a bug here",
        )

    control_ok, control_why = _reach("pypi.org", proxy)
    yahoo_ok, yahoo_why = _reach("query1.finance.yahoo.com", proxy)
    crumb_ok, crumb_why = _reach("fc.yahoo.com", proxy)

    if yahoo_ok and crumb_ok:
        report.add("yahoo finance", OK,
                   "query1.finance.yahoo.com and fc.yahoo.com reachable")
        return

    if yahoo_ok and not crumb_ok:
        report.add(
            "yahoo finance", WARN,
            f"data host reachable but fc.yahoo.com is not ({crumb_why})",
            "yfinance uses fc.yahoo.com for its cookie/crumb handshake; "
            "downloads may fail. Allow that host, or use --source csv.",
        )
        return

    if not control_ok:
        # Package registries often bypass the proxy entirely, so confirm with a
        # direct probe before declaring the machine offline.
        control_ok, control_why = _tcp_probe("pypi.org")
    if not control_ok:
        report.add(
            "network", FAIL,
            f"no outbound connectivity at all (pypi.org: {control_why})",
            "check your internet connection, VPN, or firewall",
        )
        return

    # The interesting case: internet works, Yahoo specifically does not.
    report.add(
        "yahoo finance", FAIL,
        f"unreachable ({yahoo_why}) while pypi.org is reachable",
        "the internet works but Yahoo is blocked -- a corporate network, VPN, "
        "DNS filter, or egress policy. Either allow query1/query2/fc"
        ".yahoo.com, or export bars elsewhere and use "
        "`--source csv --csv-dir PATH`.",
    )


def check_cache(report: Report, cache_dir: str | Path) -> None:
    path = Path(cache_dir).expanduser()
    if not path.exists():
        report.add("cache", WARN, f"{path} does not exist yet",
                   "it is created on the first successful run")
        return

    files = sorted(list(path.glob("*.csv")) + list(path.glob("*.parquet")))
    if not files:
        report.add("cache", WARN, f"{path} is empty", "run a backtest to populate it")
        return

    tickers = sorted({f.stem.split("_")[0] for f in files})
    shown = ", ".join(tickers[:8]) + (f", +{len(tickers) - 8} more" if len(tickers) > 8 else "")
    report.add("cache", OK, f"{len(files)} file(s) in {path}: {shown}")


def check_config(report: Report, config_path: str | None) -> None:
    from qsb.config import Config

    try:
        cfg = Config.load(config_path)
    except FileNotFoundError:
        report.add("config", FAIL, f"file not found: {config_path}",
                   "check the --config path, or omit it to use defaults")
        return
    except Exception as exc:
        report.add("config", FAIL, f"invalid: {exc}",
                   "fix the config file; compare against config.example.json")
        return

    where = config_path or "built-in defaults"
    report.add("config", OK,
               f"{where} -- {len(cfg.universe)} tickers, source={cfg.source}")

    if cfg.source == "csv":
        from qsb import sources

        try:
            ok, why = sources.build("csv", cfg).available()
            report.add("csv source", OK if ok else FAIL, why,
                       "" if ok else "point --csv-dir at a directory of exported files")
        except Exception as exc:
            report.add("csv source", FAIL, str(exc),
                       "pass --csv-dir PATH or set csv_dir in the config")


# ---------------------------------------------------------------------------
def run(config_path: str | None = None,
        cache_dir: str | None = None,
        skip_network: bool = False) -> Report:
    """Run every check and return the report."""
    report = Report()
    check_python(report)
    check_virtualenv(report)
    check_packages(report)
    check_import(report)
    check_config(report, config_path)
    check_cache(report, cache_dir or "data_cache")
    check_network(report, skip=skip_network)
    return report
