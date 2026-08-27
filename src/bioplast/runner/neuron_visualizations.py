"""Versioned image-space visualizations for neurons of an MNIST MLP."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bioplast.runner.contracts import CONTRACT_VERSION, ContractError


NEURON_VISUALIZATIONS_FILE = "neuron-visualizations.json"


def _image(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ContractError("визуализация нейрона требует непустое двумерное изображение")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for raw_row in value:
        if not isinstance(raw_row, (list, tuple)) or not raw_row:
            raise ContractError("строка визуализации нейрона должна быть непустым списком")
        row = tuple(float(item) for item in raw_row)
        if width is None:
            width = len(row)
        if len(row) != width or any(not math.isfinite(item) for item in row):
            raise ContractError("визуализация нейрона должна быть прямоугольной и конечной")
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class NeuronVisualizationLayer:
    layer_id: str
    module_path: str
    mode: str
    images: tuple[tuple[tuple[float, ...], ...], ...]
    source_indices: tuple[int, ...] = ()
    activation_values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.layer_id or not self.module_path:
            raise ContractError("визуализация нейронов требует layer_id и module_path")
        if self.mode not in {"input_filter", "max_dataset_example"}:
            raise ContractError(f"неизвестный режим визуализации нейронов: {self.mode!r}")
        if not self.images:
            raise ContractError("визуализация слоя требует хотя бы один нейрон")
        image_shape = (len(self.images[0]), len(self.images[0][0]))
        if any((len(image), len(image[0])) != image_shape for image in self.images):
            raise ContractError("изображения нейронов слоя должны иметь одинаковую форму")
        if self.mode == "input_filter":
            if self.source_indices or self.activation_values:
                raise ContractError("input_filter не содержит dataset source или activation")
        elif (
            len(self.source_indices) != len(self.images)
            or len(self.activation_values) != len(self.images)
        ):
            raise ContractError("max_dataset_example требует source и activation для каждого нейрона")
        if any(index < 0 for index in self.source_indices):
            raise ContractError("индекс dataset-примера не может быть отрицательным")
        if any(not math.isfinite(value) for value in self.activation_values):
            raise ContractError("максимальные активации нейронов должны быть конечными")

    @property
    def neuron_count(self) -> int:
        return len(self.images)

    @property
    def image_shape(self) -> tuple[int, int]:
        return len(self.images[0]), len(self.images[0][0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "module_path": self.module_path,
            "mode": self.mode,
            "neuron_count": self.neuron_count,
            "images": [[list(row) for row in image] for image in self.images],
            "source_indices": list(self.source_indices) if self.source_indices else None,
            "activation_values": (
                list(self.activation_values) if self.activation_values else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NeuronVisualizationLayer:
        try:
            images = tuple(_image(item) for item in value.get("images", ()))
            layer = cls(
                layer_id=str(value.get("layer_id", "")),
                module_path=str(value.get("module_path", "")),
                mode=str(value.get("mode", "")),
                images=images,
                source_indices=tuple(int(item) for item in (value.get("source_indices") or ())),
                activation_values=tuple(
                    float(item) for item in (value.get("activation_values") or ())
                ),
            )
            if int(value.get("neuron_count", -1)) != layer.neuron_count:
                raise ContractError("neuron_count не совпадает с числом изображений")
            return layer
        except (TypeError, ValueError) as exc:
            raise ContractError("некорректная визуализация слоя") from exc


@dataclass(frozen=True)
class NeuronVisualizations:
    run_id: str
    input_shape: tuple[int, int]
    layers: tuple[NeuronVisualizationLayer, ...]
    dataset_split: str = "test"
    schema_version: int = CONTRACT_VERSION
    kind: str = "neuron_visualizations"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION or self.kind != "neuron_visualizations":
            raise ContractError("неподдерживаемый заголовок neuron visualizations")
        if not self.run_id or len(self.input_shape) != 2 or min(self.input_shape) < 1:
            raise ContractError("neuron visualizations требует run_id и двумерный input_shape")
        if self.dataset_split not in {"train", "test"}:
            raise ContractError("dataset_split визуализаций должен быть train или test")
        layer_ids = [layer.layer_id for layer in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ContractError("слои neuron visualizations должны быть уникальными")
        if any(layer.image_shape != self.input_shape for layer in self.layers):
            raise ContractError("форма изображения нейрона не совпадает с input_shape")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "input_shape": list(self.input_shape),
            "dataset_split": self.dataset_split,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NeuronVisualizations:
        try:
            return cls(
                run_id=str(value.get("run_id", "")),
                input_shape=tuple(int(item) for item in value.get("input_shape", ())),
                dataset_split=str(value.get("dataset_split", "")),
                layers=tuple(
                    NeuronVisualizationLayer.from_dict(item)
                    for item in value.get("layers", ())
                ),
                schema_version=int(value.get("schema_version", -1)),
                kind=str(value.get("kind", "")),
            )
        except (TypeError, ValueError) as exc:
            raise ContractError("некорректные neuron visualizations") from exc


def write_neuron_visualizations(run_dir: Path, value: NeuronVisualizations) -> Path:
    run_dir = run_dir.resolve()
    if value.run_id != run_dir.name:
        raise ContractError("neuron visualizations принадлежит другому запуску")
    path = run_dir / NEURON_VISUALIZATIONS_FILE
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                value.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_neuron_visualizations(path: Path) -> NeuronVisualizations:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"не удалось прочитать neuron visualizations {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("neuron visualizations должен быть JSON-объектом")
    return NeuronVisualizations.from_dict(value)
