"""Малые версионированные snapshots для интерактивного XOR."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bioplast.runner.contracts import CONTRACT_VERSION, ContractError


def _numeric_tensor(value: Any, *, field_name: str) -> tuple[tuple[int, ...], list[float]]:
    if isinstance(value, list):
        children = [_numeric_tensor(item, field_name=field_name) for item in value]
        if not children:
            raise ContractError(f"{field_name} не может быть пустым")
        child_shape = children[0][0]
        if any(shape != child_shape for shape, _values in children[1:]):
            raise ContractError(f"{field_name} не может быть рваным массивом")
        return (len(value), *child_shape), [
            item for _shape, values in children for item in values
        ]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field_name} допускает только числа")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{field_name} допускает только конечные числа")
    return (), [number]


@dataclass(frozen=True)
class XorParameterUpdate:
    name: str
    before: Any
    delta: Any
    after: Any

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError("имя параметра XOR update не может быть пустым")
        before_shape, before = _numeric_tensor(self.before, field_name=f"{self.name}.before")
        delta_shape, delta = _numeric_tensor(self.delta, field_name=f"{self.name}.delta")
        after_shape, after = _numeric_tensor(self.after, field_name=f"{self.name}.after")
        if before_shape != delta_shape or before_shape != after_shape:
            raise ContractError(f"before/delta/after параметра {self.name!r} имеют разные формы")
        if any(
            not math.isclose(old + change, new, rel_tol=1e-6, abs_tol=1e-7)
            for old, change, new in zip(before, delta, after)
        ):
            raise ContractError(f"after параметра {self.name!r} не совпадает с before + delta")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "before": self.before,
            "delta": self.delta,
            "after": self.after,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> XorParameterUpdate:
        return cls(
            name=str(value.get("name", "")),
            before=value.get("before"),
            delta=value.get("delta"),
            after=value.get("after"),
        )


@dataclass(frozen=True)
class XorTrainingLayer:
    layer_id: str
    parameters: tuple[XorParameterUpdate, ...]
    apical_deviation: tuple[float, ...] | None = None
    local_error: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.layer_id or not self.parameters:
            raise ContractError("XOR training layer требует id и параметры")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ContractError(f"параметры слоя {self.layer_id!r} должны иметь уникальные имена")
        if (self.apical_deviation is None) != (self.local_error is None):
            raise ContractError("a − baseline и e должны присутствовать или отсутствовать вместе")
        learning_values = (
            (*self.apical_deviation, *self.local_error)
            if self.apical_deviation is not None and self.local_error is not None
            else ()
        )
        if any(not math.isfinite(float(value)) for value in learning_values):
            raise ContractError("learning-сигналы XOR должны быть конечными")

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "apical_deviation": (
                list(self.apical_deviation) if self.apical_deviation is not None else None
            ),
            "local_error": list(self.local_error) if self.local_error is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> XorTrainingLayer:
        apical = value.get("apical_deviation")
        local_error = value.get("local_error")
        try:
            return cls(
                layer_id=str(value.get("layer_id", "")),
                parameters=tuple(
                    XorParameterUpdate.from_dict(item)
                    for item in value.get("parameters", ())
                ),
                apical_deviation=(
                    tuple(float(item) for item in apical) if apical is not None else None
                ),
                local_error=(
                    tuple(float(item) for item in local_error)
                    if local_error is not None
                    else None
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ContractError("learning-сигналы XOR должны быть числовыми списками") from exc


@dataclass(frozen=True)
class XorDecisionSurface:
    x0: tuple[float, ...]
    x1: tuple[float, ...]
    probabilities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.x0) < 2 or len(self.x1) < 2:
            raise ContractError("граница решений XOR требует минимум две точки на ось")
        if len(self.probabilities) != len(self.x1) or any(
            len(row) != len(self.x0) for row in self.probabilities
        ):
            raise ContractError("форма probability grid не совпадает с осями XOR")
        values = (*self.x0, *self.x1, *(item for row in self.probabilities for item in row))
        if any(not math.isfinite(float(value)) for value in values):
            raise ContractError("граница решений XOR допускает только конечные числа")
        if any(not 0.0 <= value <= 1.0 for row in self.probabilities for value in row):
            raise ContractError("вероятности границы решений должны лежать в [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "x0": list(self.x0),
            "x1": list(self.x1),
            "probabilities": [list(row) for row in self.probabilities],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> XorDecisionSurface:
        try:
            return cls(
                x0=tuple(float(item) for item in value.get("x0", ())),
                x1=tuple(float(item) for item in value.get("x1", ())),
                probabilities=tuple(
                    tuple(float(item) for item in row)
                    for row in value.get("probabilities", ())
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ContractError("граница решений XOR должна содержать числовые списки") from exc


@dataclass(frozen=True)
class XorTrainStepSnapshot:
    run_id: str
    seq: int
    step: int
    loss: float
    accuracy: float
    updated: bool
    layers: tuple[XorTrainingLayer, ...]
    decision_surface: XorDecisionSurface
    schema_version: int = CONTRACT_VERSION
    kind: str = "xor_train_step_snapshot"
    phase: str = "train_step"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION or self.kind != "xor_train_step_snapshot":
            raise ContractError("неподдерживаемый заголовок XOR train-step snapshot")
        if not self.run_id or self.seq < 1 or self.step < 0:
            raise ContractError("XOR train-step snapshot требует run_id и неотрицательные seq/step")
        if self.phase != "train_step":
            raise ContractError("фаза XOR train-step snapshot должна быть train_step")
        if not math.isfinite(self.loss) or not 0.0 <= self.accuracy <= 1.0:
            raise ContractError("loss/accuracy XOR train-step snapshot некорректны")
        layer_ids = [layer.layer_id for layer in self.layers]
        if not layer_ids or len(layer_ids) != len(set(layer_ids)):
            raise ContractError("XOR train-step snapshot требует уникальные слои")

    @property
    def relative_path(self) -> str:
        return f"snapshots/{self.seq:06d}.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "seq": self.seq,
            "step": self.step,
            "phase": self.phase,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "updated": self.updated,
            "layers": [layer.to_dict() for layer in self.layers],
            "decision_surface": self.decision_surface.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> XorTrainStepSnapshot:
        updated = value.get("updated")
        if not isinstance(updated, bool):
            raise ContractError("updated XOR train-step snapshot должен быть bool")
        try:
            return cls(
                run_id=str(value.get("run_id", "")),
                seq=int(value.get("seq", -1)),
                step=int(value.get("step", -1)),
                phase=str(value.get("phase", "")),
                loss=float(value.get("loss", float("nan"))),
                accuracy=float(value.get("accuracy", float("nan"))),
                updated=updated,
                layers=tuple(
                    XorTrainingLayer.from_dict(item) for item in value.get("layers", ())
                ),
                decision_surface=XorDecisionSurface.from_dict(
                    value.get("decision_surface", {})
                ),
                schema_version=int(value.get("schema_version", -1)),
                kind=str(value.get("kind", "")),
            )
        except (TypeError, ValueError) as exc:
            raise ContractError("некорректные скаляры XOR train-step snapshot") from exc


@dataclass(frozen=True)
class XorForwardSnapshot:
    run_id: str
    seq: int
    input_command_seq: int
    phase: str
    layer_id: str | None
    input_values: tuple[float, float]
    z: tuple[float, ...] = ()
    post: tuple[float, ...] = ()
    hidden: tuple[float, ...] = ()
    probability: float | None = None
    prediction: int | None = None
    schema_version: int = CONTRACT_VERSION
    kind: str = "xor_forward_snapshot"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION or self.kind != "xor_forward_snapshot":
            raise ContractError("неподдерживаемый заголовок XOR snapshot")
        if not self.run_id:
            raise ContractError("run_id XOR snapshot не может быть пустым")
        if self.seq < 1 or self.input_command_seq < 1:
            raise ContractError("seq XOR snapshot должны быть положительными")
        if self.phase not in {"input", "forward_hidden", "forward_output"}:
            raise ContractError(f"неизвестная фаза XOR snapshot: {self.phase!r}")
        expected_layer = {
            "input": None,
            "forward_hidden": "hidden",
            "forward_output": "output",
        }[self.phase]
        if self.layer_id != expected_layer:
            raise ContractError(
                f"фаза {self.phase!r} требует layer_id={expected_layer!r}"
            )
        values = (*self.input_values, *self.z, *self.post, *self.hidden)
        if any(not math.isfinite(float(value)) for value in values):
            raise ContractError("XOR snapshot допускает только конечные значения")
        if self.phase == "input" and (self.z or self.post or self.hidden):
            raise ContractError("input snapshot не содержит z/post/hidden")
        if self.phase != "input" and (not self.z or len(self.z) != len(self.post)):
            raise ContractError("forward snapshot требует одинаковые непустые z/post")
        if self.phase == "forward_hidden" and self.hidden != self.post:
            raise ContractError("hidden snapshot должен дублировать post в hidden")
        if self.phase == "forward_output":
            if (
                len(self.z) != 1
                or not self.hidden
                or self.probability is None
                or self.prediction not in {0, 1}
            ):
                raise ContractError("output snapshot требует logit, probability и prediction")
            if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
                raise ContractError("probability должна лежать в диапазоне [0, 1]")
        elif self.probability is not None or self.prediction is not None:
            raise ContractError("probability/prediction допустимы только для output snapshot")

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
            "input": list(self.input_values),
            "z": list(self.z),
            "post": list(self.post),
            "hidden": list(self.hidden),
            "probability": self.probability,
            "prediction": self.prediction,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> XorForwardSnapshot:
        try:
            input_values = tuple(float(item) for item in value.get("input", ()))
            z = tuple(float(item) for item in value.get("z", ()))
            post = tuple(float(item) for item in value.get("post", ()))
            hidden = tuple(float(item) for item in value.get("hidden", ()))
        except (TypeError, ValueError) as exc:
            raise ContractError("тензоры XOR snapshot должны быть числовыми списками") from exc
        if len(input_values) != 2:
            raise ContractError("XOR snapshot требует ровно два входа")
        return cls(
            run_id=str(value.get("run_id", "")),
            seq=int(value.get("seq", -1)),
            input_command_seq=int(value.get("input_command_seq", -1)),
            phase=str(value.get("phase", "")),
            layer_id=(str(value["layer_id"]) if value.get("layer_id") is not None else None),
            input_values=(input_values[0], input_values[1]),
            z=z,
            post=post,
            hidden=hidden,
            probability=(
                float(value["probability"])
                if value.get("probability") is not None
                else None
            ),
            prediction=(
                int(value["prediction"])
                if value.get("prediction") is not None
                else None
            ),
            schema_version=int(value.get("schema_version", -1)),
            kind=str(value.get("kind", "")),
        )


def write_xor_forward_snapshot(run_dir: Path, snapshot: XorForwardSnapshot) -> Path:
    """Атомарно записать JSON snapshot до публикации ссылающегося события."""
    run_dir = run_dir.resolve()
    if snapshot.run_id != run_dir.name:
        raise ContractError(
            f"run_id snapshot {snapshot.run_id!r} не совпадает с {run_dir.name!r}"
        )
    path = (run_dir / snapshot.relative_path).resolve()
    if not path.is_relative_to(run_dir):
        raise ContractError("путь XOR snapshot выходит за пределы прогона")
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


def load_xor_forward_snapshot(path: Path) -> XorForwardSnapshot:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"не удалось прочитать XOR snapshot {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("XOR snapshot должен быть JSON-объектом")
    return XorForwardSnapshot.from_dict(value)


def write_xor_train_step_snapshot(run_dir: Path, snapshot: XorTrainStepSnapshot) -> Path:
    """Атомарно записать полный малый снимок одного обучающего шага XOR."""
    run_dir = run_dir.resolve()
    if snapshot.run_id != run_dir.name:
        raise ContractError(
            f"run_id snapshot {snapshot.run_id!r} не совпадает с {run_dir.name!r}"
        )
    path = (run_dir / snapshot.relative_path).resolve()
    if not path.is_relative_to(run_dir):
        raise ContractError("путь XOR train-step snapshot выходит за пределы прогона")
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


def load_xor_train_step_snapshot(path: Path) -> XorTrainStepSnapshot:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"не удалось прочитать XOR train-step snapshot {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("XOR train-step snapshot должен быть JSON-объектом")
    return XorTrainStepSnapshot.from_dict(value)
