"""Read-only доступ к каталогу `runs/` для API визуализатора."""

from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from bioplast.runner.contracts import ContractError, RunManifest, RunStatus, load_run_manifest


class RunNotFound(LookupError):
    pass


class ArtifactNotFound(LookupError):
    pass


class UnsafeRunPath(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactInfo:
    path: str
    size_bytes: int
    modified_at: str
    media_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "media_type": self.media_type,
        }


class RunRepository:
    """Нормализует новые и legacy-прогоны, не изменяя файлов на диске."""

    def __init__(self, runs_dir: Path | str) -> None:
        self.runs_dir = Path(runs_dir).resolve()

    def list_runs(
        self,
        *,
        statuses: Iterable[RunStatus] | None = None,
        experiment: str | None = None,
        seed: int | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        allowed_statuses = set(statuses or ())
        items: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for run_dir in self._run_dirs():
            try:
                manifest = self._load_manifest(run_dir)
                config = self._read_config(run_dir, manifest)
                metrics = self._read_metrics(run_dir, manifest, required=False)
                summary = self._summary(manifest, config, metrics)
                if allowed_statuses and manifest.status not in allowed_statuses:
                    continue
                if experiment is not None and manifest.experiment != experiment:
                    continue
                if seed is not None and _optional_int(config.get("seed")) != seed:
                    continue
                started = _parse_timestamp(manifest.started_at)
                if started_after is not None and (
                    started is None or started < _aware(started_after)
                ):
                    continue
                if started_before is not None and (
                    started is None or started > _aware(started_before)
                ):
                    continue
                items.append(summary)
            except (ContractError, OSError, TypeError, ValueError) as exc:
                errors.append({"run_id": run_dir.name, "error": str(exc)})

        items.sort(key=_summary_sort_key, reverse=True)
        errors.sort(key=lambda item: item["run_id"], reverse=True)
        return items, errors

    def get_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.resolve_run(run_id)
        manifest = self._load_manifest(run_dir)
        config = self._read_config(run_dir, manifest)
        metrics = self._read_metrics(run_dir, manifest, required=False)
        return {
            "manifest": manifest.to_dict(),
            "config": config,
            "metrics": metrics,
            "artifacts": [item.to_dict() for item in self.list_artifacts(run_id)],
        }

    def get_metrics(self, run_id: str) -> dict[str, Any]:
        run_dir = self.resolve_run(run_id)
        manifest = self._load_manifest(run_dir)
        return self._read_metrics(run_dir, manifest, required=True)

    def read_log(self, run_id: str, *, offset: int = 0, limit: int = 65_536) -> dict[str, Any]:
        if offset < 0:
            raise ValueError("offset не может быть отрицательным")
        if limit < 1 or limit > 1_048_576:
            raise ValueError("limit должен быть от 1 до 1048576 байт")

        run_dir = self.resolve_run(run_id)
        manifest = self._load_manifest(run_dir)
        relative = manifest.artifacts.get("log", "run.log")
        path = self.resolve_artifact(run_id, relative)
        size = path.stat().st_size
        start = min(offset, size)
        with path.open("rb") as stream:
            stream.seek(start)
            data = stream.read(limit)
            text, data = _decode_utf8_chunk(stream, data)
        next_offset = start + len(data)
        return {
            "text": text,
            "offset": start,
            "next_offset": next_offset,
            "size_bytes": size,
            "eof": next_offset >= size,
        }

    def list_artifacts(self, run_id: str) -> list[ArtifactInfo]:
        run_dir = self.resolve_run(run_id)
        self._load_manifest(run_dir)
        artifacts: list[ArtifactInfo] = []
        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(run_dir):
                continue  # симлинк наружу каталога никогда не становится артефактом
            stat = resolved.stat()
            relative = path.relative_to(run_dir).as_posix()
            artifacts.append(
                ArtifactInfo(
                    path=relative,
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(
                        timespec="seconds"
                    ),
                    media_type=mimetypes.guess_type(path.name)[0],
                )
            )
        return sorted(artifacts, key=lambda item: item.path)

    def resolve_run(self, run_id: str) -> Path:
        if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise UnsafeRunPath("run_id должен быть именем одного каталога")
        candidate = (self.runs_dir / run_id).resolve()
        if candidate.parent != self.runs_dir:
            raise UnsafeRunPath("run_id выходит за пределы runs/")
        if not candidate.is_dir():
            raise RunNotFound(f"прогон не найден: {run_id}")
        return candidate

    def resolve_artifact(self, run_id: str, artifact_path: str) -> Path:
        run_dir = self.resolve_run(run_id)
        self._load_manifest(run_dir)
        if not artifact_path:
            raise UnsafeRunPath("путь артефакта не может быть пустым")
        candidate = (run_dir / artifact_path).resolve()
        if not candidate.is_relative_to(run_dir) or candidate == run_dir:
            raise UnsafeRunPath("путь артефакта выходит за пределы прогона")
        if not candidate.is_file():
            raise ArtifactNotFound(f"артефакт не найден: {artifact_path}")
        return candidate

    def _run_dirs(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            return []
        result: list[Path] = []
        for path in self.runs_dir.iterdir():
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved.parent == self.runs_dir:
                result.append(resolved)
        return result

    @staticmethod
    def _load_manifest(run_dir: Path) -> RunManifest:
        manifest = load_run_manifest(run_dir)
        if manifest.run_id != run_dir.name:
            raise ContractError(
                f"run_id {manifest.run_id!r} не совпадает с именем каталога {run_dir.name!r}"
            )
        return manifest

    def _read_config(self, run_dir: Path, manifest: RunManifest) -> dict[str, Any]:
        relative = manifest.artifacts.get("config", "config.json")
        path = self._artifact_if_exists(run_dir, relative)
        if path is not None:
            return _read_json_object(path)
        metrics = self._read_metrics(run_dir, manifest, required=False)
        config = metrics.get("config", {})
        if not isinstance(config, dict):
            raise ContractError(f"metrics.config должен быть объектом: {run_dir}")
        return config

    def _read_metrics(
        self, run_dir: Path, manifest: RunManifest, *, required: bool
    ) -> dict[str, Any]:
        relative = manifest.artifacts.get("metrics", "metrics.json")
        path = self._artifact_if_exists(run_dir, relative)
        if path is None:
            if required:
                raise ArtifactNotFound(f"у прогона {run_dir.name} ещё нет metrics.json")
            return {}
        return _read_json_object(path)

    @staticmethod
    def _artifact_if_exists(run_dir: Path, relative: str) -> Path | None:
        candidate = (run_dir / relative).resolve()
        if not candidate.is_relative_to(run_dir) or candidate == run_dir:
            raise UnsafeRunPath("манифест ссылается за пределы каталога прогона")
        return candidate if candidate.is_file() else None

    @staticmethod
    def _summary(
        manifest: RunManifest, config: dict[str, Any], metrics: dict[str, Any]
    ) -> dict[str, Any]:
        final = metrics.get("final", {})
        if not isinstance(final, dict):
            final = {}
        git = metrics.get("git", {})
        if not isinstance(git, dict):
            git = {}
        return {
            "run_id": manifest.run_id,
            "status": manifest.status.value,
            "experiment": manifest.experiment,
            "dataset": config.get("dataset"),
            "model": config.get("model") or config.get("name"),
            "seed": _optional_int(config.get("seed")),
            "tag": config.get("tag"),
            "started_at": manifest.started_at,
            "finished_at": manifest.finished_at,
            "duration_sec": manifest.duration_sec,
            "parent_run_id": manifest.parent_run_id,
            "adapted_from_legacy": manifest.adapted_from_legacy,
            "dirty": bool(git.get("dirty", False)),
            "final": final,
        }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"не удалось прочитать {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"ожидался JSON-объект: {path}")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return _aware(datetime.fromisoformat(value))
    except ValueError:
        return None


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.astimezone()


def _summary_sort_key(item: dict[str, Any]) -> tuple[float, str]:
    started = _parse_timestamp(item.get("started_at"))
    return (started.timestamp() if started is not None else float("-inf"), item["run_id"])


def _decode_utf8_chunk(stream, data: bytes) -> tuple[str, bytes]:
    """Дочитать до границы UTF-8; последовательный tail не теряет символы."""
    for _ in range(3):  # после первого байта UTF-8 нужно не больше трёх
        try:
            return data.decode("utf-8"), data
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                return data.decode("utf-8", errors="replace"), data
            extra = stream.read(1)
            if not extra:
                return data.decode("utf-8", errors="replace"), data
            data += extra
    return data.decode("utf-8", errors="replace"), data
