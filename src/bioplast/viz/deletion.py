"""Безопасное одиночное и пакетное удаление терминальных запусков."""

from __future__ import annotations

from typing import Iterable

from bioplast.viz.repository import RunRepository


class RunDeletionValidationError(ValueError):
    pass


class RunDeletionConflict(RuntimeError):
    pass


class RunDeletionService:
    def __init__(self, repository: RunRepository) -> None:
        self.repository = repository

    def delete(self, run_ids: Iterable[str]) -> dict[str, object]:
        requested = list(dict.fromkeys(run_ids))
        if not requested:
            raise RunDeletionValidationError("нужно выбрать хотя бы один запуск")
        if len(requested) > 500:
            raise RunDeletionValidationError(
                "за один запрос можно удалить не более 500 запусков"
            )

        selected = set(requested)
        manifests = {
            run_id: self.repository.get_manifest(run_id) for run_id in requested
        }
        active = [
            run_id
            for run_id, manifest in manifests.items()
            if not manifest.status.terminal
        ]
        if active:
            raise RunDeletionConflict(
                "активные запуски нельзя удалять; сначала завершите их: "
                + ", ".join(active)
            )

        items, _errors = self.repository.list_runs()
        remaining_children = [
            item["run_id"]
            for item in items
            if item.get("parent_run_id") in selected and item["run_id"] not in selected
        ]
        if remaining_children:
            raise RunDeletionConflict(
                "вместе с родителем нужно выбрать его дочерние запуски: "
                + ", ".join(remaining_children)
            )

        # Дочерние каталоги удаляются первыми: так при ошибке файловой системы
        # у оставшегося ребёнка не пропадёт родительская ссылка.
        children_first = sorted(
            requested,
            key=lambda run_id: manifests[run_id].parent_run_id is None,
        )
        for run_id in children_first:
            self.repository.delete_run(run_id)
        return {"deleted": requested, "count": len(requested)}
