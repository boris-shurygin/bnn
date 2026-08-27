"""One-command end-to-end smoke for the local run visualizer.

The scenario uses tiny synthetic MNIST and XOR runs, the real HTTP server and
RunSupervisor, then asks a locally installed Chromium browser to execute the UI
JavaScript and dump the resulting DOM. No browser driver or network download is
required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import uvicorn

from bioplast.runner import RunStatus, load_run_manifest, run_config
from bioplast.viz.api import create_app


@dataclass(frozen=True)
class BrowserSmokeReport:
    browser: str
    source_run_id: str
    rerun_id: str
    debug_run_id: str
    xor_source_run_id: str
    xor_debug_run_id: str
    pages: tuple[str, ...]


def find_browser(explicit: Path | str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    configured = os.environ.get("BIOPLAST_BROWSER")
    if configured:
        candidates.append(Path(configured))
    for command in ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge"):
        resolved = shutil.which(command)
        if resolved:
            candidates.append(Path(resolved))
    if os.name == "nt":
        candidates.extend(
            Path(value)
            for value in (
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until(
    predicate: Callable[[], bool],
    message: str,
    *,
    timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as exc:  # transient files/HTTP during atomic publication
            last_error = exc
        time.sleep(0.05)
    suffix = f"; последняя ошибка: {last_error}" if last_error is not None else ""
    raise RuntimeError(message + suffix)


def _json_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(base_url + path, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=5) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path}: HTTP {exc.code}: {detail}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path}: ожидался JSON-объект")
    return value


def _write_fake_mnist(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    train_x = np.zeros((4, 28, 28), dtype=np.uint8)
    train_x[1, 8:20, 13:15] = 255
    train_x[2, 7:21, 7:21] = np.eye(14, dtype=np.uint8) * 255
    train_x[3, 5:23, 5:23] = 180
    test_x = np.zeros((3, 28, 28), dtype=np.uint8)
    test_x[0, 6:22, 12:16] = 255
    test_x[1, 7:21, 7:21] = 200
    test_x[2, 9:19, 9:19] = 120
    np.savez_compressed(
        root / "mnist.npz",
        train_x=train_x,
        train_y=np.array([0, 1, 2, 9], dtype=np.uint8),
        test_x=test_x,
        test_y=np.array([1, 9, 2], dtype=np.uint8),
    )


def _dump_dom(browser: Path, url: str, output: Path) -> str:
    profile = output.parent / f"browser-profile-{output.stem}"
    command = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-background-networking",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile}",
        "--virtual-time-budget=5000",
        "--dump-dom",
        url,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"браузер завершился с кодом {completed.returncode}: {completed.stderr[-2000:]}"
        )
    output.write_text(completed.stdout, encoding="utf-8")
    return completed.stdout


def _require_markers(page: str, html: str, markers: tuple[str, ...]) -> None:
    missing = [marker for marker in markers if marker not in html]
    if missing:
        raise RuntimeError(f"{page}: динамический DOM не содержит {missing}")


def run_browser_smoke(
    work_dir: Path,
    *,
    browser: Path | str | None = None,
    timeout: float = 30,
) -> BrowserSmokeReport:
    browser_path = find_browser(browser)
    if browser_path is None:
        raise RuntimeError(
            "Chrome/Chromium/Edge не найден; задайте путь переменной BIOPLAST_BROWSER"
        )
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = work_dir / "runs"
    data_root = work_dir / "data"
    _write_fake_mnist(data_root)
    source = run_config(
        {
            "session": "V.14-smoke",
            "dataset": "mnist",
            "model": "mlp-784-4-10",
            "experiment": "mnist_mlp_backprop",
            "device": "cpu",
            "seed": 0,
            "hidden": [4],
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "data_root": str(data_root),
            "recovery_every_steps": 1,
        },
        runs_dir=runs_dir,
    )
    xor_source = run_config(
        {
            "session": "V.14-smoke",
            "dataset": "xor",
            "model": "mlp-2-4-1",
            "experiment": "xor_backprop",
            "device": "cpu",
            "seed": 0,
            "hidden": 4,
            "steps": 3,
            "lr": 0.05,
            "log_every": 1,
            "snapshot_every_steps": 1,
            "recovery_every_steps": 1,
        },
        runs_dir=runs_dir,
    )

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    app = create_app(runs_dir)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
    server_thread = threading.Thread(target=server.run, name="v14-smoke-server")
    server_thread.start()
    pages: list[str] = []
    try:
        _wait_until(
            lambda: _json_request(base_url, "/api/health").get("status") == "ok",
            "локальный API не запустился",
            timeout=timeout,
        )

        source_detail = _json_request(base_url, f"/api/runs/{source.name}")
        rerun = _json_request(
            base_url,
            f"/api/runs/{source.name}/rerun",
            method="POST",
            payload={"config": source_detail["config"]},
        )
        rerun_id = str(rerun["run_id"])
        _wait_until(
            lambda: load_run_manifest(runs_dir / rerun_id).status is RunStatus.COMPLETED,
            "повторный MNIST-прогон не завершился",
            timeout=timeout,
        )

        debug = _json_request(
            base_url, f"/api/runs/{source.name}/debug", method="POST"
        )
        debug_run_id = str(debug["run_id"])
        debug_dir = runs_dir / debug_run_id
        _wait_until(
            lambda: load_run_manifest(debug_dir).status is RunStatus.RUNNING,
            "MNIST debug worker не запустился",
            timeout=timeout,
        )
        _json_request(
            base_url,
            f"/api/runs/{debug_run_id}/control",
            method="POST",
            payload={"command": "set_input", "input_values": [1]},
        )
        _wait_until(
            lambda: len(
                _json_request(base_url, f"/api/runs/{debug_run_id}/events").get(
                    "items", []
                )
            )
            >= 3,
            "MNIST debug не опубликовал полный послойный forward",
            timeout=timeout,
        )
        _json_request(
            base_url,
            f"/api/runs/{debug_run_id}/control",
            method="POST",
            payload={"command": "cancel"},
        )
        _wait_until(
            lambda: load_run_manifest(debug_dir).status is RunStatus.CANCELLED,
            "MNIST debug worker не отменился",
            timeout=timeout,
        )

        xor_debug = _json_request(
            base_url, f"/api/runs/{xor_source.name}/debug", method="POST"
        )
        xor_debug_run_id = str(xor_debug["run_id"])
        xor_debug_dir = runs_dir / xor_debug_run_id
        _wait_until(
            lambda: load_run_manifest(xor_debug_dir).status is RunStatus.RUNNING,
            "XOR debug worker не запустился",
            timeout=timeout,
        )
        _json_request(
            base_url,
            f"/api/runs/{xor_debug_run_id}/control",
            method="POST",
            payload={"command": "set_input", "input_values": [0, 1]},
        )
        _wait_until(
            lambda: len(
                _json_request(base_url, f"/api/runs/{xor_debug_run_id}/events").get(
                    "items", []
                )
            )
            >= 3,
            "XOR debug не опубликовал полный послойный forward",
            timeout=timeout,
        )
        _json_request(
            base_url,
            f"/api/runs/{xor_debug_run_id}/control",
            method="POST",
            payload={"command": "cancel"},
        )
        _wait_until(
            lambda: load_run_manifest(xor_debug_dir).status is RunStatus.CANCELLED,
            "XOR debug worker не отменился",
            timeout=timeout,
        )

        compare_query = urlencode(
            [("baseline", source.name), ("candidate", rerun_id)]
        )
        page_specs = (
            (
                "catalog",
                "/runs",
                (
                    source.name,
                    rerun_id,
                    debug_run_id,
                    xor_source.name,
                    xor_debug_run_id,
                ),
            ),
            (
                "source",
                f"/runs/{source.name}",
                ("Инспекция большой модели", "Открыть инспекцию модели"),
            ),
            (
                "debug",
                f"/runs/{debug_run_id}",
                (
                    "test[1]",
                    "layers",
                    "Top classes",
                    "данные без параметров",
                    "Нейроны слоя",
                    "Нейрон 0",
                    "Веса",
                    "Вклад input × weight",
                    "без bias",
                ),
            ),
            (
                "xor-training",
                f"/runs/{xor_source.name}",
                ("Кадр #", "a−baseline/e: нет у backprop"),
            ),
            (
                "xor-debug",
                f"/runs/{xor_debug_run_id}",
                ("Forward завершён", "вход [0, 1]", "P(1)="),
            ),
            (
                "compare",
                f"/compare?{compare_query}",
                ("Сравнение запусков", source.name, rerun_id),
            ),
        )
        for name, path, markers in page_specs:
            html = _dump_dom(browser_path, base_url + path, work_dir / f"{name}.html")
            _require_markers(name, html, markers)
            pages.append(name)

        return BrowserSmokeReport(
            browser=str(browser_path),
            source_run_id=source.name,
            rerun_id=rerun_id,
            debug_run_id=debug_run_id,
            xor_source_run_id=xor_source.name,
            xor_debug_run_id=xor_debug_run_id,
            pages=tuple(pages),
        )
    finally:
        server.should_exit = True
        server_thread.join(timeout=15)
        if server_thread.is_alive():
            server.force_exit = True
            server_thread.join(timeout=5)
        if server_thread.is_alive():
            raise RuntimeError("локальный smoke-сервер не остановился")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bioplast.viz smoke",
        description="Запустить сквозной API/recovery/browser smoke блока V.",
    )
    parser.add_argument("--browser", type=Path, help="явный путь к Chrome/Chromium/Edge")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="сохранить runs и DOM-снимки в указанном каталоге",
    )
    parser.add_argument("--timeout", type=float, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout должен быть положительным")
    try:
        if args.work_dir is not None:
            report = run_browser_smoke(
                args.work_dir, browser=args.browser, timeout=args.timeout
            )
        else:
            with tempfile.TemporaryDirectory(prefix="bioplast-v14-smoke-") as temporary:
                report = run_browser_smoke(
                    Path(temporary), browser=args.browser, timeout=args.timeout
                )
    except (RuntimeError, OSError, subprocess.SubprocessError, URLError) as exc:
        print(f"V.14 smoke failed: {exc}")
        return 1
    print(f"V.14 smoke passed in {report.browser}")
    print(f"  source: {report.source_run_id}")
    print(f"  rerun:  {report.rerun_id}")
    print(f"  debug:  {report.debug_run_id}")
    print(f"  xor:    {report.xor_source_run_id}")
    print(f"  xor db: {report.xor_debug_run_id}")
    print(f"  pages:  {', '.join(report.pages)}")
    return 0
