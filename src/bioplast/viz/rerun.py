"""Клонирование конфига и постановка повторного запуска в очередь."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from bioplast.runner import (
    RunStatus,
    fail_prepared_run,
    prepare_run,
    validate_run_config,
)
from bioplast.viz.repository import RunRepository


class Scheduler(Protocol):
    def submit(self, run_dir: Path | str) -> None: ...


class RerunValidationError(ValueError):
    pass


class RunNotRerunnable(ValueError):
    pass


class QueueSubmissionError(RuntimeError):
    pass


_LOCKED_FIELDS = {
    "id": "идентификатор нового запуска назначает сервер",
    "parent_run_id": "родителя нового запуска назначает сервер",
    "source_run_id": "исходный прогон интерактивной сессии назначает сервер",
    "experiment": "модуль эксперимента нельзя менять из браузера",
    "session": "этап исследования сохраняется от исходного запуска",
    "dataset": "набор данных сохраняется от исходного запуска",
    "model": "модель сохраняется от исходного запуска",
    "tag": "метка сохраняется от исходного запуска",
}
_PATH_SUFFIXES = ("_path", "_root", "_dir", "_file")


class RerunService:
    def __init__(self, repository: RunRepository, scheduler: Scheduler) -> None:
        self.repository = repository
        self.scheduler = scheduler

    def preview(self, run_id: str) -> dict[str, Any]:
        source = self.repository.get_run(run_id)
        manifest = source["manifest"]
        status = RunStatus(manifest["status"])
        if not status.terminal:
            raise RunNotRerunnable(
                f"повторный запуск доступен только для завершённого прогона, сейчас {status.value}"
            )
        config = source["config"]
        fields = []
        for key, value in config.items():
            reason = _locked_reason(key, value)
            fields.append(
                {
                    "key": key,
                    "value": value,
                    "editable": reason is None,
                    "reason": reason,
                }
            )
        return {
            "source_run_id": run_id,
            "config": deepcopy(config),
            "fields": fields,
        }

    def enqueue(self, run_id: str, proposed: dict[str, Any]) -> dict[str, Any]:
        preview = self.preview(run_id)
        source = preview["config"]
        if set(proposed) != set(source):
            missing = sorted(set(source) - set(proposed))
            extra = sorted(set(proposed) - set(source))
            parts = []
            if missing:
                parts.append(f"удалены поля: {', '.join(missing)}")
            if extra:
                parts.append(f"добавлены поля: {', '.join(extra)}")
            raise RerunValidationError("структуру конфига менять нельзя; " + "; ".join(parts))

        for field in preview["fields"]:
            key = field["key"]
            if not field["editable"] and proposed[key] != source[key]:
                raise RerunValidationError(f"поле {key!r} заблокировано: {field['reason']}")
            if field["editable"] and not _compatible_edit(source[key], proposed[key]):
                raise RerunValidationError(
                    f"поле {key!r} должно сохранять тип JSON-значения исходного конфига"
                )
        device = proposed.get("device")
        if device is not None and not re.fullmatch(r"(?:auto|cpu|cuda(?::\d+)?)", device):
            raise RerunValidationError("device должен быть auto, cpu, cuda или cuda:N")

        config = deepcopy(proposed)
        config.pop("id", None)
        config["parent_run_id"] = run_id
        try:
            validate_run_config(config)
        except (TypeError, ValueError) as exc:
            raise RerunValidationError(str(exc)) from exc

        changes = diff_configs(source, proposed)
        run_dir = prepare_run(config, self.repository.runs_dir, parent_run_id=run_id)
        try:
            self.scheduler.submit(run_dir)
        except Exception as exc:
            fail_prepared_run(run_dir, f"очередь не приняла подготовленный прогон: {exc}")
            raise QueueSubmissionError(f"не удалось поставить прогон в очередь: {exc}") from exc
        return {
            "run_id": run_dir.name,
            "status": RunStatus.QUEUED.value,
            "parent_run_id": run_id,
            "changes": changes,
            "location": f"/runs/{run_dir.name}",
        }


def diff_configs(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"key": key, "before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key) or (key in before) != (key in after)
    ]


def _locked_reason(key: str, value: Any) -> str | None:
    if key in _LOCKED_FIELDS:
        return _LOCKED_FIELDS[key]
    lowered = key.lower()
    if lowered.endswith(_PATH_SUFFIXES) or lowered in {"path", "command", "module"}:
        return "пути и исполняемые модули нельзя менять из браузера"
    if value is None:
        return "тип null-поля нельзя надёжно вывести автоматически"
    if not _editable_value(value):
        return "сложная структура не редактируется в V.4"
    return None


def _editable_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_editable_value(item) and not isinstance(item, list) for item in value)
    return False


def _compatible_edit(original: Any, proposed: Any) -> bool:
    if isinstance(original, bool):
        return isinstance(proposed, bool)
    if isinstance(original, int):
        return isinstance(proposed, int) and not isinstance(proposed, bool)
    if isinstance(original, float):
        return (
            isinstance(proposed, (int, float))
            and not isinstance(proposed, bool)
            and math.isfinite(proposed)
        )
    if isinstance(original, str):
        return isinstance(proposed, str)
    if isinstance(original, list):
        if not isinstance(proposed, list) or not _editable_value(proposed):
            return False
        if not original:
            return True
        exemplar = original[0]
        return all(_compatible_edit(exemplar, item) for item in proposed)
    return False
