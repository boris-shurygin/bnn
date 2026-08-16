"""Очередь прогонов на `ProcessPoolExecutor`.

Не shell-скрипт: очередь должна одинаково работать на Windows, в WSL и в облаке.
Число параллельных процессов — параметр с самого начала: модели блоков 0–3
загружают 3060 процентов на десять, и последовательный прогон свипа впустую
тратит вечернее окно.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


def collect_configs(paths: Iterable[Path | str]) -> list[Path]:
    """Разворачивает файлы и папки в отсортированный список конфигов."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"конфиг не найден: {path}")
    return found


def _worker(config_path: str, runs_dir: str | None) -> dict[str, Any]:
    """Выполняется в дочернем процессе. Импортируемая функция верхнего уровня —
    обязательное требование `spawn` на Windows."""
    # Каждый процесс дербанит все ядра под BLAS, если не ограничить; при 3–4
    # параллельных прогонах это оверсабскрипшн и замедление.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from bioplast.runner.run import run_config

    run_dir = run_config(Path(config_path), runs_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "config": config_path,
        "run_dir": str(run_dir),
        "status": metrics.get("status"),
        "duration_sec": metrics.get("duration_sec"),
    }


def _prepared_worker(run_dir: str) -> dict[str, Any]:
    """Выполняет заранее зарезервированный queued-прогон."""
    from bioplast.runner.run import run_prepared

    path = run_prepared(Path(run_dir))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return {
        "run_dir": str(path),
        "status": metrics.get("status"),
        "duration_sec": metrics.get("duration_sec"),
    }


class RunScheduler:
    """Долгоживущая очередь веб-приложения на том же ProcessPoolExecutor."""

    def __init__(self, *, workers: int = 1) -> None:
        if workers < 1:
            raise ValueError("workers должен быть положительным")
        self.workers = workers
        self._executor: ProcessPoolExecutor | None = None
        self._futures: dict[Future[dict[str, Any]], Path] = {}
        self._lock = Lock()

    def submit(self, run_dir: Path | str) -> None:
        path = str(Path(run_dir).resolve())
        with self._lock:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(max_workers=self.workers)
            future = self._executor.submit(_prepared_worker, path)
            self._futures[future] = Path(path)
        future.add_done_callback(self._discard)

    def _discard(self, future: Future[dict[str, Any]]) -> None:
        with self._lock:
            run_dir = self._futures.pop(future, None)
        if run_dir is None or future.cancelled():
            return
        error = future.exception()
        if error is not None:
            from bioplast.runner.run import fail_prepared_run

            fail_prepared_run(run_dir, f"worker process crashed: {error!r}")

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._futures)

    def cancel(self, run_dir: Path | str) -> bool:
        """Снять queued-задачу с executor; running worker отменяется командой."""
        target = Path(run_dir).resolve()
        with self._lock:
            future = next(
                (item for item, path in self._futures.items() if path == target),
                None,
            )
        return bool(future is not None and future.cancel())

    def shutdown(self) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=False)


def run_queue(
    paths: Iterable[Path | str],
    workers: int = 1,
    runs_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Прогнать очередь конфигов. Возвращает сводку по каждому прогону."""
    configs = collect_configs(paths)
    if not configs:
        return []

    runs_dir_arg = str(runs_dir) if runs_dir else None

    if workers <= 1:
        return [_worker(str(cfg), runs_dir_arg) for cfg in configs]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, str(cfg), runs_dir_arg): cfg for cfg in configs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # процесс умер целиком, а не эксперимент упал
                results.append(
                    {"config": str(futures[future]), "status": "crashed", "error": repr(exc)}
                )
    return results
