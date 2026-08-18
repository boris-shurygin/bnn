"""Малые версионированные snapshots для интерактивного XOR-forward."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bioplast.runner.contracts import CONTRACT_VERSION, ContractError


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
