"""Игрушечный эксперимент для тестов раннера: без данных, без обучения.

Живёт в `experiments/`, а не в `tests/`, чтобы тесты проходили тот же путь
разрешения имени эксперимента, что и настоящий прогон.
"""

from __future__ import annotations

from typing import Any

from bioplast.runner import ExperimentResult


def run(config: dict[str, Any], ctx) -> ExperimentResult:
    if config.get("fail"):
        raise RuntimeError("падение по требованию конфига")

    if config.get("touch_device"):
        import torch

        torch.zeros(1024, 1024, device=ctx.device).sum()

    steps = int(config.get("steps", 3))
    for step in range(steps):
        ctx.metrics.update(step, {"loss/train": 1.0 / (step + 1), "w_norm/fc1": 1.0 + step})

    return ExperimentResult(final={"steps": steps, "device": ctx.device})
