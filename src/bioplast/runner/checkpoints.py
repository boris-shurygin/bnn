"""Атомарные бинарные checkpoint для inference и продолжения обучения."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bioplast.runner.contracts import CHECKPOINT_FILE, ContractError, utc_offset_iso

CHECKPOINT_VERSION = 1
CHECKPOINT_KIND = "training_checkpoint"


def write_training_checkpoint(
    run_dir: Path,
    *,
    run_id: str,
    experiment: str,
    model_name: str,
    model: Any,
    optimizer: Any | None = None,
    step: int | None = None,
) -> Path:
    """Сохранить state_dict модели и опциональное состояние оптимизатора.

    Целый ``nn.Module`` намеренно не сериализуется: класс модели и параметры её
    конструктора определяются versioned-кодом эксперимента и ``config.json``.
    """
    import torch
    from torch import nn
    from torch.optim import Optimizer

    if run_id != run_dir.name:
        raise ContractError(f"run_id checkpoint {run_id!r} не совпадает с {run_dir.name!r}")
    if not experiment or not model_name:
        raise ContractError("experiment и model_name checkpoint не могут быть пустыми")
    if not isinstance(model, nn.Module):
        raise ContractError("model checkpoint должен быть torch.nn.Module")
    if optimizer is not None and not isinstance(optimizer, Optimizer):
        raise ContractError("optimizer checkpoint должен быть torch.optim.Optimizer")
    if step is not None and step < 0:
        raise ContractError("step checkpoint не может быть отрицательным")

    payload = {
        "schema_version": CHECKPOINT_VERSION,
        "kind": CHECKPOINT_KIND,
        "run_id": run_id,
        "experiment": experiment,
        "model_name": model_name,
        "created_at": utc_offset_iso(),
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    path = run_dir / CHECKPOINT_FILE
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_training_checkpoint(
    path: Path,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Загрузить checkpoint без разрешения на исполнение произвольного pickle."""
    import torch

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ContractError(f"не удалось прочитать checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("checkpoint должен содержать словарь")
    _validate_checkpoint(payload)
    return payload


def _validate_checkpoint(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CHECKPOINT_VERSION:
        raise ContractError(
            "неподдерживаемая версия checkpoint "
            f"{payload.get('schema_version')!r}; ожидается {CHECKPOINT_VERSION}"
        )
    if payload.get("kind") != CHECKPOINT_KIND:
        raise ContractError(
            f"ожидался kind={CHECKPOINT_KIND!r}, получен {payload.get('kind')!r}"
        )
    for name in ("run_id", "experiment", "model_name", "created_at"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ContractError(f"поле checkpoint {name!r} должно быть непустой строкой")
    step = payload.get("step")
    if step is not None and (not isinstance(step, int) or isinstance(step, bool) or step < 0):
        raise ContractError("step checkpoint должен быть неотрицательным целым или null")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ContractError("checkpoint не содержит model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None and not isinstance(optimizer_state, Mapping):
        raise ContractError("optimizer_state_dict должен быть словарём или null")
