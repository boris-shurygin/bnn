"""Backprop-эталон на XOR: 2 → 8 → 1.

Точка отсчёта блока 1: та же архитектура и та же диагностика будут потом
обучаться локальным правилом, и разница должна быть видна на одинаковых
графиках. XOR гоняется на CPU — сеть такого размера на GPU медленнее из-за
накладных расходов на запуск ядер.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from bioplast.diagnostics.probes import log_module_state
from bioplast.runner import (
    ContractError,
    ExperimentResult,
    ModelArtifacts,
    load_training_recovery,
    recovery_interval,
    write_training_recovery,
)

XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = [0.0, 1.0, 1.0, 0.0]
RECOVERY_ADAPTER = "xor_backprop_v1"


class MLP(nn.Module):
    """Простой MLP, отдающий скрытые активации — они нужны диагностике."""

    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x: torch.Tensor, collect: bool = False):
        acts: dict[str, torch.Tensor] = {}
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < last:
                x = torch.relu(x)
                if collect:
                    acts[f"h{i + 1}"] = x
        if collect:
            acts["out"] = x
            return x, acts
        return x


def run(config: dict[str, Any], ctx) -> ExperimentResult:
    hidden = int(config.get("hidden", 8))
    steps = int(config.get("steps", 2000))
    lr = float(config.get("lr", 0.05))
    log_every = int(config.get("log_every", 25))
    device = ctx.device

    x = torch.tensor(XOR_X, device=device)
    y = torch.tensor(XOR_Y, device=device).unsqueeze(1)

    model = MLP([2, hidden, 1]).to(device)
    # autograd здесь — эталон, а не целевой механизм: в блоке 1 обновление
    # станет ручным и локальным, а этот прогон останется для сравнения.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ctx.metrics.step_key = "step"  # на XOR эпох нет, единица времени — шаг
    ctx.log.info("XOR backprop: 2→%d→1, lr=%g, steps=%d", hidden, lr, steps)

    loss_target = float(config.get("loss_target", 0.05))
    solved_at: int | None = None
    loss_value = float("nan")
    accuracy = 0.0
    next_step = 0
    every = recovery_interval(config, default=1)

    recovery_path = ctx.run_dir / "recovery" / "state.json"
    if recovery_path.is_file():
        recovery_state, training = load_training_recovery(
            ctx, config, adapter=RECOVERY_ADAPTER
        )
        try:
            model.load_state_dict(training["model_state_dict"], strict=True)
            optimizer.load_state_dict(training["optimizer_state_dict"])
            next_step = int(training["next_step"])
            solved_at = training.get("solved_at")
            solved_at = int(solved_at) if solved_at is not None else None
            loss_value = float(training["loss_value"])
            accuracy = float(training["accuracy"])
            metrics_rows = training["metrics_rows"]
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ContractError(f"некорректный XOR training recovery: {exc}") from exc
        if not isinstance(metrics_rows, list) or not 0 <= next_step <= steps + 1:
            raise ContractError("XOR training recovery содержит неверный progress cursor")
        ctx.metrics.rows = list(metrics_rows)
        ctx.log.info(
            "XOR training recovery #%s: следующий шаг %d/%d",
            recovery_state.get("generation"),
            next_step,
            steps,
        )
    else:
        write_training_recovery(
            ctx,
            config,
            adapter=RECOVERY_ADAPTER,
            cursor="before_train_step:0",
            progress={"next_step": 0, "global_step": 0},
            training={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "next_step": 0,
                "solved_at": None,
                "loss_value": loss_value,
                "accuracy": accuracy,
                "metrics_rows": [],
            },
        )

    for step in range(next_step, steps + 1):
        ctx.control.checkpoint(step=step, phase="train_step")
        logits, acts = model(x, collect=True)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            correct = ((logits > 0).float() == y).all(dim=1)
            accuracy = correct.float().mean().item()
        loss_value = loss.item()

        # Одной только точности мало: на четырёх точках случайная инициализация
        # угадывает все четыре с вероятностью 1/16, и «решено на шаге 6» — артефакт.
        if solved_at is None and accuracy == 1.0 and loss_value < loss_target:
            solved_at = step
            ctx.log.info("решено на шаге %d (loss=%.4f)", step, loss_value)

        if step % log_every == 0 or step == steps:
            ctx.metrics.update(
                step,
                {
                    "loss/train": loss_value,
                    "acc/train": accuracy,
                    **log_module_state(model, acts),
                },
            )

        if step < steps:
            optimizer.step()

        next_step = step + 1
        if next_step % every == 0 or next_step > steps:
            write_training_recovery(
                ctx,
                config,
                adapter=RECOVERY_ADAPTER,
                cursor=f"after_train_step:{step}",
                progress={"next_step": next_step, "global_step": next_step},
                training={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "next_step": next_step,
                    "solved_at": solved_at,
                    "loss_value": loss_value,
                    "accuracy": accuracy,
                    "metrics_rows": list(ctx.metrics.rows),
                },
            )

    solved = bool(accuracy == 1.0 and loss_value < loss_target)
    ctx.log.info("итог: loss=%.5f, acc=%.2f, solved=%s", loss_value, accuracy, solved)

    return ExperimentResult(
        final={
            "loss": loss_value,
            "acc": accuracy,
            "solved": solved,
            "solved_at_step": solved_at,
        },
        model_artifacts=ModelArtifacts(
            model=model,
            optimizer=optimizer,
            example_args=(x,),
            layer_ids={"layers.0": "hidden", "layers.1": "output"},
            activations={"layers.0": "relu", "layers.1": "identity"},
            step=steps,
        ),
    )
