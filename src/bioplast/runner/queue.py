"""Очередь прогонов на `ProcessPoolExecutor`.

Не shell-скрипт: очередь должна одинаково работать на Windows, в WSL и в облаке.
Число параллельных процессов — параметр с самого начала: модели блоков 0–3
загружают 3060 процентов на десять, и последовательный прогон свипа впустую
тратит вечернее окно.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
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
