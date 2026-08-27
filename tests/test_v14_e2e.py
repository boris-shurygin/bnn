"""V.14: owned server command and real-browser end-to-end smoke."""

from __future__ import annotations

import pytest

from bioplast.viz import __main__ as viz_cli
from bioplast.viz import server, smoke
from bioplast.viz.smoke import BrowserSmokeReport, find_browser, run_browser_smoke


def test_owned_server_command_binds_only_loopback_and_uses_requested_runs_dir(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(server.uvicorn, "run", fake_run)

    result = viz_cli.main(
        ["serve", "--runs-dir", str(tmp_path / "runs"), "--port", "8765"]
    )

    assert result == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["access_log"] is False
    assert captured["app"].state.run_repository.runs_dir == (tmp_path / "runs").resolve()


def test_owned_server_uses_runs_dir_environment_by_default(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("BIOPLAST_RUNS_DIR", str(tmp_path / "from-env"))

    def fake_run(app, **_kwargs):
        captured["app"] = app

    monkeypatch.setattr(server.uvicorn, "run", fake_run)

    result = viz_cli.main(["serve", "--port", "8765"])

    assert result == 0
    assert captured["app"].state.run_repository.runs_dir == (tmp_path / "from-env").resolve()


def test_smoke_subcommand_dispatches_and_reports_result(tmp_path, monkeypatch, capsys):
    report = BrowserSmokeReport(
        browser="test-browser",
        source_run_id="source",
        rerun_id="rerun",
        debug_run_id="debug",
        xor_source_run_id="xor-source",
        xor_debug_run_id="xor-debug",
        pages=(
            "catalog",
            "source",
            "debug",
            "xor-training",
            "xor-debug",
            "compare",
        ),
    )
    captured = {}

    def fake_smoke(work_dir, *, browser, timeout):
        captured.update(work_dir=work_dir, browser=browser, timeout=timeout)
        return report

    monkeypatch.setattr(smoke, "run_browser_smoke", fake_smoke)

    result = viz_cli.main(
        ["smoke", "--work-dir", str(tmp_path / "smoke"), "--timeout", "7"]
    )

    assert result == 0
    assert captured == {
        "work_dir": tmp_path / "smoke",
        "browser": None,
        "timeout": 7.0,
    }
    assert "V.14 smoke passed" in capsys.readouterr().out


@pytest.mark.skipif(find_browser() is None, reason="Chrome/Chromium/Edge не установлен")
def test_real_browser_smoke_covers_catalog_rerun_debug_and_comparison(tmp_path):
    report = run_browser_smoke(tmp_path / "browser-smoke", timeout=40)

    assert report.pages == (
        "catalog",
        "source",
        "debug",
        "xor-training",
        "xor-debug",
        "compare",
    )
    assert report.source_run_id != report.rerun_id
    assert report.debug_run_id not in {report.source_run_id, report.rerun_id}
    assert report.xor_source_run_id != report.xor_debug_run_id
    for page in report.pages:
        html = (tmp_path / "browser-smoke" / f"{page}.html").read_text(
            encoding="utf-8"
        )
        assert "<!DOCTYPE html>" in html
