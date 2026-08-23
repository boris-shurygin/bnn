"""Versioned aggregate snapshots for debug renderers of larger models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bioplast.runner.contracts import CONTRACT_VERSION, ContractError, TensorSpec


@dataclass(frozen=True)
class ModelDebugClassScore:
    class_index: int
    probability: float

    def __post_init__(self) -> None:
        if self.class_index < 0 or not math.isfinite(self.probability):
            raise ContractError("class score требует неотрицательный класс и конечную вероятность")
        if not 0.0 <= self.probability <= 1.0:
            raise ContractError("вероятность class score должна лежать в [0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        return {"class_index": self.class_index, "probability": self.probability}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelDebugClassScore:
        try:
            return cls(
                class_index=int(value.get("class_index", -1)),
                probability=float(value.get("probability", float("nan"))),
            )
        except (TypeError, ValueError) as exc:
            raise ContractError("некорректный class score model debug") from exc


@dataclass(frozen=True)
class ModelDebugLayerSnapshot:
    layer_id: str
    module_path: str
    layer_type: str
    activation: str
    parameter_count: int
    input_tensor: TensorSpec
    preactivation_tensor: TensorSpec
    output_tensor: TensorSpec

    def __post_init__(self) -> None:
        if not self.layer_id or not self.module_path or not self.layer_type or not self.activation:
            raise ContractError("слой model debug требует id, module path, type и activation")
        if self.parameter_count < 0:
            raise ContractError("parameter_count model debug не может быть отрицательным")
        if self.input_tensor.role != "activation_input":
            raise ContractError("input_tensor model debug должен иметь role=activation_input")
        if self.preactivation_tensor.role != "preactivation":
            raise ContractError("preactivation_tensor model debug должен иметь role=preactivation")
        if self.output_tensor.role != "activation_output":
            raise ContractError("output_tensor model debug должен иметь role=activation_output")

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "module_path": self.module_path,
            "parent_path": self.module_path.rpartition(".")[0] or None,
            "layer_type": self.layer_type,
            "activation": self.activation,
            "parameter_count": self.parameter_count,
            "input": self.input_tensor.to_dict(),
            "preactivation": self.preactivation_tensor.to_dict(),
            "output": self.output_tensor.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelDebugLayerSnapshot:
        return cls(
            layer_id=str(value.get("layer_id", "")),
            module_path=str(value.get("module_path", "")),
            layer_type=str(value.get("layer_type", "")),
            activation=str(value.get("activation", "")),
            parameter_count=int(value.get("parameter_count", -1)),
            input_tensor=TensorSpec.from_dict(value.get("input", {})),
            preactivation_tensor=TensorSpec.from_dict(value.get("preactivation", {})),
            output_tensor=TensorSpec.from_dict(value.get("output", {})),
        )


def _preview_values(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 28:
        raise ContractError("MNIST preview должен содержать 28 строк")
    rows: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 28:
            raise ContractError("каждая строка MNIST preview должна содержать 28 значений")
        converted = tuple(float(item) for item in row)
        if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in converted):
            raise ContractError("MNIST preview допускает только значения [0, 1]")
        rows.append(converted)
    return tuple(rows)


@dataclass(frozen=True)
class ModelDebugSnapshot:
    run_id: str
    seq: int
    input_command_seq: int
    phase: str
    layer_id: str | None
    input_mode: str
    input_index: int
    input_label: int | None
    input_preview: tuple[tuple[float, ...], ...]
    layers: tuple[ModelDebugLayerSnapshot, ...] = ()
    prediction: int | None = None
    top_classes: tuple[ModelDebugClassScore, ...] = ()
    schema_version: int = CONTRACT_VERSION
    kind: str = "model_debug_snapshot"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION or self.kind != "model_debug_snapshot":
            raise ContractError("неподдерживаемый заголовок model debug snapshot")
        if not self.run_id or self.seq < 1 or self.input_command_seq < 1:
            raise ContractError("model debug snapshot требует run_id и положительные seq")
        if self.phase not in {"input", "forward_layer"}:
            raise ContractError(f"неизвестная фаза model debug snapshot: {self.phase!r}")
        if self.input_mode != "dataset_index" or self.input_index < 0:
            raise ContractError("model debug snapshot требует неотрицательный dataset index")
        if self.input_label is not None and self.input_label < 0:
            raise ContractError("input_label model debug не может быть отрицательным")
        _preview_values(self.input_preview)
        layer_ids = [layer.layer_id for layer in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ContractError("слои model debug snapshot должны быть уникальными")
        if self.phase == "input":
            if self.layer_id is not None or self.layers or self.prediction is not None or self.top_classes:
                raise ContractError("input snapshot не содержит вычисленные слои или prediction")
        else:
            if not self.layers or self.layer_id != self.layers[-1].layer_id:
                raise ContractError("forward_layer snapshot должен завершаться указанным layer_id")
        if self.prediction is None and self.top_classes:
            raise ContractError("top_classes допустимы только вместе с prediction")
        if self.prediction is not None:
            if not self.top_classes or self.top_classes[0].class_index != self.prediction:
                raise ContractError("prediction должен совпадать с первым top class")

    @property
    def relative_path(self) -> str:
        return f"snapshots/{self.seq:06d}.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "seq": self.seq,
            "input_command_seq": self.input_command_seq,
            "phase": self.phase,
            "layer_id": self.layer_id,
            "input": {
                "mode": self.input_mode,
                "index": self.input_index,
                "label": self.input_label,
                "shape": [1, 28, 28],
                "preview": [list(row) for row in self.input_preview],
            },
            "layers": [layer.to_dict() for layer in self.layers],
            "prediction": self.prediction,
            "top_classes": [item.to_dict() for item in self.top_classes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelDebugSnapshot:
        raw_input = value.get("input")
        if not isinstance(raw_input, Mapping):
            raise ContractError("model debug snapshot не содержит input")
        try:
            return cls(
                run_id=str(value.get("run_id", "")),
                seq=int(value.get("seq", -1)),
                input_command_seq=int(value.get("input_command_seq", -1)),
                phase=str(value.get("phase", "")),
                layer_id=(str(value["layer_id"]) if value.get("layer_id") is not None else None),
                input_mode=str(raw_input.get("mode", "")),
                input_index=int(raw_input.get("index", -1)),
                input_label=(
                    int(raw_input["label"]) if raw_input.get("label") is not None else None
                ),
                input_preview=_preview_values(raw_input.get("preview")),
                layers=tuple(
                    ModelDebugLayerSnapshot.from_dict(item)
                    for item in value.get("layers", ())
                ),
                prediction=(
                    int(value["prediction"]) if value.get("prediction") is not None else None
                ),
                top_classes=tuple(
                    ModelDebugClassScore.from_dict(item)
                    for item in value.get("top_classes", ())
                ),
                schema_version=int(value.get("schema_version", -1)),
                kind=str(value.get("kind", "")),
            )
        except (TypeError, ValueError) as exc:
            raise ContractError("некорректные скаляры model debug snapshot") from exc


def write_model_debug_snapshot(run_dir: Path, snapshot: ModelDebugSnapshot) -> Path:
    run_dir = run_dir.resolve()
    if snapshot.run_id != run_dir.name:
        raise ContractError("model debug snapshot принадлежит другому запуску")
    path = (run_dir / snapshot.relative_path).resolve()
    if not path.is_relative_to(run_dir):
        raise ContractError("путь model debug snapshot выходит за пределы прогона")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_model_debug_snapshot(path: Path) -> ModelDebugSnapshot:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"не удалось прочитать model debug snapshot {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("model debug snapshot должен быть JSON-объектом")
    return ModelDebugSnapshot.from_dict(value)
