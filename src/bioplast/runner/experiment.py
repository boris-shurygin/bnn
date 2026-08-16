"""Единый интерфейс эксперимента и централизованный экспорт модели.

Эксперимент выполняет вычисления и возвращает ``ExperimentResult``. Он не
выбирает имена канонических файлов и не пишет ``model.json``/``checkpoint.pt``
самостоятельно: раннер создаёт оба артефакта одной политикой.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioplast.runner.checkpoints import write_training_checkpoint
from bioplast.runner.contracts import (
    CHECKPOINT_FILE,
    MODEL_MANIFEST,
    ContractError,
    write_model_manifest,
)
from bioplast.runner.inspection import inspect_model


@dataclass(frozen=True)
class ModelArtifacts:
    """Данные worker-процесса, необходимые раннеру для двух форматов модели."""

    model: Any
    example_args: tuple[Any, ...]
    optimizer: Any | None = None
    example_kwargs: Mapping[str, Any] = field(default_factory=dict)
    layer_ids: Mapping[str, str] = field(default_factory=dict)
    activations: Mapping[str, str | None] = field(default_factory=dict)
    step: int | None = None
    capture_phase: str = "completed"
    full_values_max_elements: int = 256

    def __post_init__(self) -> None:
        if not self.example_args:
            raise ContractError("ModelArtifacts требует example_args для инспекции")
        if self.step is not None and self.step < 0:
            raise ContractError("step ModelArtifacts не может быть отрицательным")
        if not self.capture_phase:
            raise ContractError("capture_phase ModelArtifacts не может быть пустым")
        if self.full_values_max_elements < 0:
            raise ContractError("full_values_max_elements не может быть отрицательным")


@dataclass(frozen=True)
class ExperimentResult:
    """Обязательный результат ``run(config, ctx)`` для любого эксперимента."""

    final: Mapping[str, Any]
    model_artifacts: ModelArtifacts | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.final, Mapping):
            raise ContractError("ExperimentResult.final должен быть словарём")
        if any(not isinstance(key, str) or not key for key in self.final):
            raise ContractError("ключи ExperimentResult.final должны быть непустыми строками")
        try:
            json.dumps(dict(self.final), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"ExperimentResult.final должен содержать конечные JSON-значения: {exc}"
            ) from exc


def finalize_experiment(
    result: ExperimentResult,
    *,
    config: Mapping[str, Any],
    run_id: str,
    run_dir: Path,
    logger: Any,
) -> dict[str, Any]:
    """Проверить результат и атомарно создать общий набор модельных файлов."""
    if not isinstance(result, ExperimentResult):
        raise ContractError(
            "run(config, ctx) должен возвращать ExperimentResult, "
            f"получен {type(result).__name__}"
        )
    artifacts = result.model_artifacts
    if artifacts is None:
        return dict(result.final)

    experiment = str(config.get("experiment", ""))
    model_name = str(config.get("model") or config.get("name") or experiment)
    manifest = inspect_model(
        artifacts.model,
        run_id=run_id,
        model_name=model_name,
        example_args=artifacts.example_args,
        example_kwargs=artifacts.example_kwargs,
        layer_ids=artifacts.layer_ids,
        activations=artifacts.activations,
        full_values_max_elements=artifacts.full_values_max_elements,
        capture_phase=artifacts.capture_phase,
        step=artifacts.step,
    )

    checkpoint_path = run_dir / CHECKPOINT_FILE
    model_path = run_dir / MODEL_MANIFEST
    try:
        write_training_checkpoint(
            run_dir,
            run_id=run_id,
            experiment=experiment,
            model_name=model_name,
            model=artifacts.model,
            optimizer=artifacts.optimizer,
            step=artifacts.step,
        )
        write_model_manifest(run_dir, manifest)
    except Exception:
        # Оба файла принадлежат только текущему новому прогону. Не оставляем
        # половину обещанного общего формата, если второй экспорт не удался.
        checkpoint_path.unlink(missing_ok=True)
        model_path.unlink(missing_ok=True)
        raise

    logger.info(
        "модель экспортирована единым интерфейсом: checkpoint.pt и model.json (%d слоя)",
        len(manifest.layers),
    )
    return dict(result.final)
