"""Замеры состояния сети: нормы весов и градиентов, статистика активаций.

Всё возвращается плоскими словарями `{"группа/имя": число}`, готовыми к
`MetricsRecorder.update`. Никакого состояния — так же будет работать и с
локальными правилами, где `nn.Module` останется только оболочкой над весами.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def sparsity(tensor: torch.Tensor, eps: float = 1e-6) -> float:
    """Доля (почти) нулевых элементов.

    Для ReLU-активаций это прямая мера разреженности кода; она же связана с
    устойчивостью к забыванию (обновляются только активные нейроны).
    """
    if tensor.numel() == 0:
        return float("nan")
    return (tensor.abs() <= eps).float().mean().item()


def weight_norms(module: nn.Module, prefix: str = "w_norm") -> dict[str, float]:
    """Фробениусова норма каждого весового тензора (bias пропускается)."""
    out: dict[str, float] = {}
    for name, param in module.named_parameters():
        if name.endswith("bias"):
            continue
        out[f"{prefix}/{_short(name)}"] = param.detach().norm().item()
    return out


def grad_norms(module: nn.Module, prefix: str = "grad_norm") -> dict[str, float]:
    """Норма градиента по весам. В блоке 0 это про backprop-эталон; позже —
    эталон, с которым сравнивается локальное обновление при замере alignment."""
    out: dict[str, float] = {}
    for name, param in module.named_parameters():
        if name.endswith("bias") or param.grad is None:
            continue
        out[f"{prefix}/{_short(name)}"] = param.grad.detach().norm().item()
    return out


def activation_stats(
    name: str, tensor: torch.Tensor, eps: float = 1e-6
) -> dict[str, float]:
    """RMS, максимум по модулю и разреженность активаций одного слоя.

    Максимум нужен именно как сторож runaway: взрыв активаций — ожидаемый
    симптом абляции «без вычитания baseline» (§3.5 research-plan).
    """
    detached = tensor.detach()
    return {
        f"act_rms/{name}": detached.pow(2).mean().sqrt().item(),
        f"act_max/{name}": detached.abs().max().item(),
        f"act_sparsity/{name}": sparsity(detached, eps),
    }


def log_module_state(
    module: nn.Module,
    activations: dict[str, torch.Tensor] | None = None,
    include_grads: bool = True,
) -> dict[str, Any]:
    """Полный срез диагностики v0 одним вызовом."""
    state: dict[str, Any] = {}
    state.update(weight_norms(module))
    if include_grads:
        state.update(grad_norms(module))
    for name, tensor in (activations or {}).items():
        state.update(activation_stats(name, tensor))
    return state


def _short(param_name: str) -> str:
    """`net.0.weight` → `net.0`; имя параметра в ключе метрики лишнее."""
    return param_name.removesuffix(".weight") or param_name
