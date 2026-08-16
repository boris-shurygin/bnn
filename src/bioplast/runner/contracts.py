"""Версионированные файловые контракты наблюдаемости прогонов.

`runs/<id>/` остаётся источником истины. Модуль не хранит живые PyTorch-объекты
и не мигрирует старые каталоги: если `run.json` отсутствует, нормализованный
манифест синтезируется из прежних `config.json` и `metrics.json` только в памяти.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

CONTRACT_VERSION = 1
RUN_MANIFEST = "run.json"
MODEL_MANIFEST = "model.json"
EVENTS_FILE = "events.jsonl"
COMMANDS_FILE = "commands.jsonl"
CHECKPOINT_FILE = "checkpoint.pt"


class ContractError(ValueError):
    """Артефакт не соответствует поддерживаемой версии контракта."""


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


def utc_offset_iso(now: datetime | None = None) -> str:
    """Локальное ISO-время с UTC offset, пригодное для сортировки и показа."""
    value = now or datetime.now().astimezone()
    if value.tzinfo is None:
        value = value.astimezone()
    return value.isoformat(timespec="seconds")


def default_artifacts() -> dict[str, str]:
    """Канонические относительные пути, даже если будущий файл ещё не создан."""
    return {
        "config": "config.json",
        "metrics": "metrics.json",
        "log": "run.log",
        "model": MODEL_MANIFEST,
        "events": EVENTS_FILE,
        "commands": COMMANDS_FILE,
        "snapshots": "snapshots",
        "checkpoint": CHECKPOINT_FILE,
    }


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    status: RunStatus
    experiment: str | None
    started_at: str | None
    updated_at: str | None
    finished_at: str | None = None
    duration_sec: float | None = None
    parent_run_id: str | None = None
    artifacts: dict[str, str] = field(default_factory=default_artifacts)
    adapted_from_legacy: bool = False
    schema_version: int = CONTRACT_VERSION
    kind: str = "run"

    def __post_init__(self) -> None:
        _validate_header(self.schema_version, self.kind, "run")
        if not self.run_id:
            raise ContractError("run_id не может быть пустым")
        if self.duration_sec is not None and self.duration_sec < 0:
            raise ContractError("duration_sec не может быть отрицательной")
        _validate_relative_paths(self.artifacts)

    def finish(
        self,
        status: RunStatus,
        duration_sec: float,
        finished_at: str | None = None,
    ) -> RunManifest:
        if not status.terminal:
            raise ContractError(f"финальный статус должен быть терминальным: {status}")
        timestamp = finished_at or utc_offset_iso()
        return replace(
            self,
            status=status,
            updated_at=timestamp,
            finished_at=timestamp,
            duration_sec=round(float(duration_sec), 3),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "status": self.status.value,
            "experiment": self.experiment,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "parent_run_id": self.parent_run_id,
            "artifacts": dict(self.artifacts),
            "adapted_from_legacy": self.adapted_from_legacy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        _validate_header(value.get("schema_version"), value.get("kind"), "run")
        try:
            status = RunStatus(str(value["status"]))
        except (KeyError, ValueError) as exc:
            raise ContractError(f"неизвестный статус прогона: {value.get('status')!r}") from exc
        return cls(
            run_id=str(value.get("run_id", "")),
            status=status,
            experiment=_optional_str(value.get("experiment")),
            started_at=_optional_str(value.get("started_at")),
            updated_at=_optional_str(value.get("updated_at")),
            finished_at=_optional_str(value.get("finished_at")),
            duration_sec=_optional_float(value.get("duration_sec")),
            parent_run_id=_optional_str(value.get("parent_run_id")),
            artifacts=dict(value.get("artifacts") or default_artifacts()),
            adapted_from_legacy=bool(value.get("adapted_from_legacy", False)),
            schema_version=int(value["schema_version"]),
            kind=str(value["kind"]),
        )


@dataclass(frozen=True)
class TensorSummary:
    element_count: int
    finite_count: int
    non_finite_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    l1_norm: float | None
    l2_norm: float | None
    sparsity: float | None

    def __post_init__(self) -> None:
        if min(self.element_count, self.finite_count, self.non_finite_count) < 0:
            raise ContractError("счётчики tensor summary не могут быть отрицательными")
        if self.finite_count + self.non_finite_count != self.element_count:
            raise ContractError("finite и non-finite должны покрывать весь тензор")
        statistic_names = ("minimum", "maximum", "mean", "std", "l1_norm", "l2_norm")
        for name in statistic_names:
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ContractError(f"{name} в tensor summary должен быть конечным")
        if self.finite_count and any(getattr(self, name) is None for name in statistic_names):
            raise ContractError("для конечных элементов tensor summary требует все статистики")
        if not self.finite_count and any(
            getattr(self, name) is not None for name in statistic_names
        ):
            raise ContractError("без конечных элементов числовые статистики должны быть null")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ContractError("min tensor summary не может быть больше max")
        if self.std is not None and self.std < 0:
            raise ContractError("std tensor summary не может быть отрицательной")
        if self.l1_norm is not None and self.l1_norm < 0:
            raise ContractError("l1_norm tensor summary не может быть отрицательной")
        if self.l2_norm is not None and self.l2_norm < 0:
            raise ContractError("l2_norm tensor summary не может быть отрицательной")
        if self.element_count and self.sparsity is None:
            raise ContractError("непустой tensor summary требует sparsity")
        if not self.element_count and self.sparsity is not None:
            raise ContractError("пустой tensor summary должен иметь sparsity=null")
        if self.sparsity is not None and not 0.0 <= self.sparsity <= 1.0:
            raise ContractError("sparsity должна лежать в диапазоне [0, 1]")

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "element_count": self.element_count,
            "finite_count": self.finite_count,
            "non_finite_count": self.non_finite_count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "l1_norm": self.l1_norm,
            "l2_norm": self.l2_norm,
            "sparsity": self.sparsity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorSummary:
        return cls(
            element_count=int(value.get("element_count", -1)),
            finite_count=int(value.get("finite_count", -1)),
            non_finite_count=int(value.get("non_finite_count", -1)),
            minimum=_optional_float(value.get("min")),
            maximum=_optional_float(value.get("max")),
            mean=_optional_float(value.get("mean")),
            std=_optional_float(value.get("std")),
            l1_norm=_optional_float(value.get("l1_norm")),
            l2_norm=_optional_float(value.get("l2_norm")),
            sparsity=_optional_float(value.get("sparsity")),
        )


@dataclass(frozen=True)
class TensorSpec:
    name: str
    role: str
    shape: tuple[int | None, ...]
    dtype: str
    requires_grad: bool | None = None
    value_mode: str = "metadata"
    summary: TensorSummary | None = None
    values: Any = None
    values_omitted_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.role or not self.dtype:
            raise ContractError("tensor name, role и dtype не могут быть пустыми")
        if self.requires_grad is not None and not isinstance(self.requires_grad, bool):
            raise ContractError("requires_grad должен быть bool или null")
        if self.value_mode not in {"metadata", "summary", "full"}:
            raise ContractError(f"неизвестный value_mode тензора: {self.value_mode!r}")
        if self.value_mode == "metadata" and (self.summary is not None or self.values is not None):
            raise ContractError("metadata-тензор не должен содержать summary или values")
        if self.value_mode == "summary" and (self.summary is None or self.values is not None):
            raise ContractError("summary-тензор должен содержать summary без values")
        if self.value_mode == "full" and (self.summary is None or self.values is None):
            raise ContractError("full-тензор должен содержать summary и values")
        if self.value_mode == "full" and self.values_omitted_reason is not None:
            raise ContractError("full-тензор не может иметь values_omitted_reason")
        if self.value_mode == "metadata" and self.values_omitted_reason is not None:
            raise ContractError("metadata-тензор не может иметь values_omitted_reason")
        if self.values is not None:
            value_shape = _json_tensor_shape(self.values)
            if value_shape != self.shape:
                raise ContractError(
                    f"форма values {value_shape} не совпадает с tensor shape {self.shape}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
            "value_mode": self.value_mode,
            "summary": self.summary.to_dict() if self.summary is not None else None,
            "values": self.values,
            "values_omitted_reason": self.values_omitted_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorSpec:
        shape = tuple(_shape_dim(item) for item in value.get("shape", []))
        return cls(
            name=str(value.get("name", "")),
            role=str(value.get("role", "")),
            shape=shape,
            dtype=str(value.get("dtype", "")),
            requires_grad=_optional_bool(value.get("requires_grad")),
            value_mode=str(value.get("value_mode", "metadata")),
            summary=(
                TensorSummary.from_dict(value["summary"])
                if value.get("summary") is not None
                else None
            ),
            values=value.get("values"),
            values_omitted_reason=_optional_str(value.get("values_omitted_reason")),
        )


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    layer_type: str
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    activation: str | None = None
    parameter_count: int = 0
    tensors: tuple[TensorSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise ContractError("layer_id не может быть пустым")
        if not self.layer_type:
            raise ContractError("layer_type не может быть пустым")
        if self.parameter_count < 0:
            raise ContractError("parameter_count не может быть отрицательным")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.layer_id,
            "type": self.layer_type,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "activation": self.activation,
            "parameter_count": self.parameter_count,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LayerSpec:
        return cls(
            layer_id=str(value.get("id", "")),
            layer_type=str(value.get("type", "")),
            input_shape=tuple(_shape_dim(item) for item in value.get("input_shape", [])),
            output_shape=tuple(_shape_dim(item) for item in value.get("output_shape", [])),
            activation=_optional_str(value.get("activation")),
            parameter_count=int(value.get("parameter_count", 0)),
            tensors=tuple(TensorSpec.from_dict(item) for item in value.get("tensors", [])),
        )


@dataclass(frozen=True)
class ConnectionSpec:
    source: str
    target: str
    kind: str = "forward"

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ContractError("source и target связи не могут быть пустыми")
        if self.kind not in {"forward", "learning"}:
            raise ContractError(f"неизвестный kind связи: {self.kind!r}")

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target, "kind": self.kind}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConnectionSpec:
        return cls(
            source=str(value.get("source", "")),
            target=str(value.get("target", "")),
            kind=str(value.get("kind", "forward")),
        )


@dataclass(frozen=True)
class ModelManifest:
    run_id: str
    model_name: str
    layers: tuple[LayerSpec, ...]
    connections: tuple[ConnectionSpec, ...]
    captured_at: str | None = None
    capture_phase: str | None = None
    step: int | None = None
    capture_batch_size: int | None = None
    schema_version: int = CONTRACT_VERSION
    kind: str = "model"

    def __post_init__(self) -> None:
        _validate_header(self.schema_version, self.kind, "model")
        if not self.run_id or not self.model_name:
            raise ContractError("run_id и model_name не могут быть пустыми")
        if not self.layers:
            raise ContractError("модель должна содержать хотя бы один слой")
        if self.step is not None and self.step < 0:
            raise ContractError("step снимка модели не может быть отрицательным")
        if self.capture_batch_size is not None and self.capture_batch_size < 1:
            raise ContractError("capture_batch_size должен быть положительным")
        layer_ids = [layer.layer_id for layer in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ContractError("идентификаторы слоёв должны быть уникальными")
        known = set(layer_ids)
        for connection in self.connections:
            if connection.source not in known or connection.target not in known:
                raise ContractError(
                    f"связь ссылается на неизвестный слой: {connection.source} → {connection.target}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "captured_at": self.captured_at,
            "capture_phase": self.capture_phase,
            "step": self.step,
            "capture_batch_size": self.capture_batch_size,
            "layers": [layer.to_dict() for layer in self.layers],
            "connections": [connection.to_dict() for connection in self.connections],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelManifest:
        _validate_header(value.get("schema_version"), value.get("kind"), "model")
        return cls(
            run_id=str(value.get("run_id", "")),
            model_name=str(value.get("model_name", "")),
            captured_at=_optional_str(value.get("captured_at")),
            capture_phase=_optional_str(value.get("capture_phase")),
            step=int(value["step"]) if value.get("step") is not None else None,
            capture_batch_size=(
                int(value["capture_batch_size"])
                if value.get("capture_batch_size") is not None
                else None
            ),
            layers=tuple(LayerSpec.from_dict(item) for item in value.get("layers", [])),
            connections=tuple(
                ConnectionSpec.from_dict(item) for item in value.get("connections", [])
            ),
            schema_version=int(value["schema_version"]),
            kind=str(value["kind"]),
        )


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    seq: int
    occurred_at: str
    event_type: str
    step: int | None = None
    phase: str | None = None
    layer_id: str | None = None
    scalars: dict[str, int | float | bool | None] = field(default_factory=dict)
    snapshot: str | None = None
    message: str | None = None
    schema_version: int = CONTRACT_VERSION
    kind: str = "event"

    def __post_init__(self) -> None:
        _validate_header(self.schema_version, self.kind, "event")
        if not self.run_id or not self.occurred_at:
            raise ContractError("run_id и occurred_at события не могут быть пустыми")
        if self.seq < 0:
            raise ContractError("seq не может быть отрицательным")
        if self.step is not None and self.step < 0:
            raise ContractError("step не может быть отрицательным")
        if not self.event_type:
            raise ContractError("type события не может быть пустым")
        if self.snapshot is not None:
            _validate_relative_paths({"snapshot": self.snapshot})
        for key, value in self.scalars.items():
            if not isinstance(key, str) or value is not None and not isinstance(
                value, (bool, int, float)
            ):
                raise ContractError("scalars допускает только JSON-числа, bool и null")
            if isinstance(value, float) and not math.isfinite(value):
                raise ContractError("scalars не допускает NaN и бесконечность")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "seq": self.seq,
            "occurred_at": self.occurred_at,
            "type": self.event_type,
            "step": self.step,
            "phase": self.phase,
            "layer_id": self.layer_id,
            "scalars": dict(self.scalars),
            "snapshot": self.snapshot,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunEvent:
        _validate_header(value.get("schema_version"), value.get("kind"), "event")
        return cls(
            run_id=str(value.get("run_id", "")),
            seq=int(value.get("seq", -1)),
            occurred_at=str(value.get("occurred_at", "")),
            event_type=str(value.get("type", "")),
            step=int(value["step"]) if value.get("step") is not None else None,
            phase=_optional_str(value.get("phase")),
            layer_id=_optional_str(value.get("layer_id")),
            scalars=dict(value.get("scalars") or {}),
            snapshot=_optional_str(value.get("snapshot")),
            message=_optional_str(value.get("message")),
            schema_version=int(value["schema_version"]),
            kind=str(value["kind"]),
        )


def write_run_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    path = run_dir / RUN_MANIFEST
    _write_json_atomic(path, manifest.to_dict())
    return path


def load_run_manifest(run_dir: Path) -> RunManifest:
    """Загрузить v1-манифест или адаптировать прежний каталог без записи."""
    path = run_dir / RUN_MANIFEST
    if path.is_file():
        return RunManifest.from_dict(_read_json(path))
    return _adapt_legacy_run(run_dir)


def write_model_manifest(run_dir: Path, manifest: ModelManifest) -> Path:
    path = run_dir / MODEL_MANIFEST
    _write_json_atomic(path, manifest.to_dict())
    return path


def load_model_manifest(run_dir: Path) -> ModelManifest:
    return ModelManifest.from_dict(_read_json(run_dir / MODEL_MANIFEST))


def append_event(run_dir: Path, event: RunEvent) -> Path:
    """Добавить одно целое JSON-событие; flush делает live-tail предсказуемым."""
    if event.run_id != run_dir.name:
        raise ContractError(f"run_id события {event.run_id!r} не совпадает с {run_dir.name!r}")
    path = run_dir / EVENTS_FILE
    line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
    return path


def iter_events(run_dir: Path) -> Iterator[RunEvent]:
    path = run_dir / EVENTS_FILE
    if not path.exists():
        return
    previous_seq = -1
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = RunEvent.from_dict(json.loads(line))
                if event.run_id != run_dir.name:
                    raise ContractError(
                        f"run_id события {event.run_id!r} не совпадает с {run_dir.name!r}"
                    )
                if event.seq <= previous_seq:
                    raise ContractError(
                        f"seq должен возрастать: {event.seq} после {previous_seq}"
                    )
                previous_seq = event.seq
                yield event
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ContractError(f"некорректное событие {path}:{line_number}: {exc}") from exc


def _adapt_legacy_run(run_dir: Path) -> RunManifest:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    if not config_path.is_file() and not metrics_path.is_file():
        raise ContractError(f"в {run_dir} нет ни run.json, ни старых артефактов")

    metrics = _read_json(metrics_path) if metrics_path.is_file() else {}
    config = _read_json(config_path) if config_path.is_file() else metrics.get("config", {})
    status = _legacy_status(metrics)
    started_at = _optional_str(metrics.get("started_at")) or _timestamp_from_run_id(run_dir.name)
    duration = _optional_float(metrics.get("duration_sec"))
    finished_at = _finish_from_duration(started_at, duration) if status.terminal else None
    updated_at = finished_at or started_at
    return RunManifest(
        run_id=str(metrics.get("run_id") or run_dir.name),
        status=status,
        experiment=_optional_str(config.get("experiment")),
        started_at=started_at,
        updated_at=updated_at,
        finished_at=finished_at,
        duration_sec=duration,
        parent_run_id=_optional_str(config.get("parent_run_id")),
        adapted_from_legacy=True,
    )


def _legacy_status(metrics: Mapping[str, Any]) -> RunStatus:
    raw = metrics.get("status")
    if raw is None:
        return RunStatus.RUNNING
    mapping = {
        "ok": RunStatus.COMPLETED,
        "completed": RunStatus.COMPLETED,
        "failed": RunStatus.FAILED,
        "crashed": RunStatus.FAILED,
        "cancelled": RunStatus.CANCELLED,
        "paused": RunStatus.PAUSED,
        "running": RunStatus.RUNNING,
        "queued": RunStatus.QUEUED,
    }
    try:
        return mapping[str(raw)]
    except KeyError as exc:
        raise ContractError(f"неизвестный legacy-статус: {raw!r}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"не удалось прочитать {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"ожидался JSON-объект: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_header(version: Any, kind: Any, expected_kind: str) -> None:
    if version != CONTRACT_VERSION:
        raise ContractError(
            f"неподдерживаемая версия {version!r}; ожидается {CONTRACT_VERSION}"
        )
    if kind != expected_kind:
        raise ContractError(f"ожидался kind={expected_kind!r}, получен {kind!r}")


def _validate_relative_paths(paths: Mapping[str, str]) -> None:
    for name, raw in paths.items():
        if not isinstance(raw, str) or not raw:
            raise ContractError(f"путь артефакта {name!r} должен быть непустой строкой")
        normalized = raw.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or ":" in path.parts[0]
        ):
            raise ContractError(f"путь артефакта должен быть относительным: {raw!r}")


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ContractError("ожидался bool или null")
    return value


def _shape_dim(value: Any) -> int | None:
    if value is None:
        return None
    dimension = int(value)
    if dimension < 0:
        raise ContractError("размерность тензора не может быть отрицательной")
    return dimension


def _json_tensor_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        child_shapes = [_json_tensor_shape(item) for item in value]
        if child_shapes and any(shape != child_shapes[0] for shape in child_shapes[1:]):
            raise ContractError("values тензора не могут быть рваным массивом")
        return (len(value), *(child_shapes[0] if child_shapes else ()))
    if not isinstance(value, (bool, int, float)):
        raise ContractError("values тензора допускает только конечные числа и bool")
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("values тензора не допускает NaN и бесконечность")
    return ()


def _timestamp_from_run_id(run_id: str) -> str | None:
    try:
        parsed = datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    return parsed.astimezone().isoformat(timespec="seconds")


def _finish_from_duration(started_at: str | None, duration_sec: float | None) -> str | None:
    if started_at is None or duration_sec is None:
        return started_at
    try:
        return (datetime.fromisoformat(started_at) + timedelta(seconds=duration_sec)).isoformat(
            timespec="seconds"
        )
    except ValueError:
        return started_at
