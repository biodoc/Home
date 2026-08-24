"""The preflight check.

Its whole value is distinguishing failure modes that look alike from the
outside, so the tests drive it into each one and assert it says the right thing.
"""

from __future__ import annotations

import sys

import pytest

from qsb import doctor


def test_current_environment_reports_a_python_version():
    report = doctor.run(skip_network=True)
    names = [c.name for c in report.checks]
    assert "python" in names
    assert "qsb package" in names


def test_running_python_passes_its_own_version_check():
    report = doctor.Report()
    doctor.check_python(report)
    assert report.checks[0].status == doctor.OK


def test_an_old_python_is_a_blocking_failure(monkeypatch):
    monkeypatch.setattr(doctor.sys, "version_info", (3, 9, 0, "final", 0))
    report = doctor.Report()
    doctor.check_python(report)
    assert report.checks[0].status == doctor.FAIL
    assert "3.11" in report.checks[0].fix


def test_version_comparison_handles_release_suffixes():
    assert doctor._version_tuple("2.4.6") > doctor._version_tuple("2.2.0")
    assert doctor._version_tuple("2.0.0rc1") >= doctor._version_tuple("2.0")
    assert doctor._version_tuple("10.0.0") > doctor._version_tuple("9.9.9")


def test_missing_optional_package_warns_rather_than_fails(monkeypatch):
    import importlib.metadata as md

    real = md.version

    def fake(name):
        if name == "yfinance":
            raise md.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(doctor.md, "version", fake)
    report = doctor.Report()
    doctor.check_packages(report)
    yf = next(c for c in report.checks if c.name == "yfinance")
    assert yf.status == doctor.WARN
    assert "csv" in yf.detail


def test_missing_required_package_is_a_failure(monkeypatch):
    import importlib.metadata as md

    def fake(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(doctor.md, "version", fake)
    report = doctor.Report()
    doctor.check_packages(report)
    pandas_check = next(c for c in report.checks if c.name == "pandas")
    assert pandas_check.status == doctor.FAIL
    assert "requirements.txt" in pandas_check.fix


def test_a_bad_config_path_is_a_clear_failure():
    report = doctor.Report()
    doctor.check_config(report, "/nonexistent/nope.json")
    assert report.checks[0].status == doctor.FAIL
    assert "not found" in report.checks[0].detail


def test_an_invalid_config_is_reported_with_its_reason(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"universe": ["AAA"], "validation_universe": ["AAA"]}')
    report = doctor.Report()
    doctor.check_config(report, str(bad))
    assert report.checks[0].status == doctor.FAIL
    assert "overlap" in report.checks[0].detail


def test_cache_check_counts_what_is_there(tmp_path):
    (tmp_path / "UAMY_1d.csv").write_text("Date,Close\n2024-01-02,10\n")
    (tmp_path / "SMR_1d.csv").write_text("Date,Close\n2024-01-02,10\n")
    report = doctor.Report()
    doctor.check_cache(report, tmp_path)
    assert report.checks[0].status == doctor.OK
    assert "UAMY" in report.checks[0].detail
    assert "SMR" in report.checks[0].detail


def test_an_empty_cache_warns_without_blocking(tmp_path):
    report = doctor.Report()
    doctor.check_cache(report, tmp_path)
    assert report.checks[0].status == doctor.WARN


def test_network_can_be_skipped():
    report = doctor.Report()
    doctor.check_network(report, skip=True)
    assert report.checks[0].status == doctor.WARN
    assert "skipped" in report.checks[0].detail


def test_a_blocked_proxy_connect_is_recognised(monkeypatch):
    """403 from a proxy means egress policy, and must be named as such."""
    monkeypatch.setattr(doctor, "_proxy_probe",
                        lambda host, proxy, **kw: (False, "proxy refused CONNECT (403) -- blocked by egress policy"))
    monkeypatch.setattr(doctor, "_tcp_probe", lambda host, **kw: (True, "reachable"))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")

    report = doctor.Report()
    doctor.check_network(report)

    yahoo = next(c for c in report.checks if c.name == "yahoo finance")
    assert yahoo.status == doctor.FAIL
    assert "blocked" in yahoo.detail or "blocked" in yahoo.fix
    assert "--source csv" in yahoo.fix


def test_total_absence_of_network_is_distinguished_from_a_yahoo_block(monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.setattr(doctor, "_tcp_probe",
                        lambda host, **kw: (False, "DNS lookup failed"))

    report = doctor.Report()
    doctor.check_network(report)

    net = next(c for c in report.checks if c.name == "network")
    assert net.status == doctor.FAIL
    assert "no outbound connectivity" in net.detail


def test_a_reachable_yahoo_passes(monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.setattr(doctor, "_tcp_probe", lambda host, **kw: (True, "reachable"))

    report = doctor.Report()
    doctor.check_network(report)
    assert report.checks[0].status == doctor.OK


def test_a_missing_crumb_host_warns_rather_than_fails(monkeypatch):
    """yfinance needs fc.yahoo.com for its handshake, but this is recoverable."""
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)

    def probe(host, **kw):
        return (False, "timed out") if host == "fc.yahoo.com" else (True, "reachable")

    monkeypatch.setattr(doctor, "_tcp_probe", probe)
    report = doctor.Report()
    doctor.check_network(report)

    yahoo = next(c for c in report.checks if c.name == "yahoo finance")
    assert yahoo.status == doctor.WARN
    assert "fc.yahoo.com" in yahoo.detail


def test_render_names_the_next_command():
    report = doctor.run(skip_network=True)
    text = report.render()
    assert "ENVIRONMENT CHECK" in text
    if not report.failed:
        assert "python backtest.py" in text


def test_render_flags_blocking_problems():
    report = doctor.Report()
    report.add("thing", doctor.FAIL, "broken", "fix it")
    text = report.render()
    assert "[FAIL]" in text
    assert "1 blocking problem" in text
    assert "-> fix it" in text
