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
from bioplast.runner import inspect_model, write_model_manifest, write_training_checkpoint

XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = [0.0, 1.0, 1.0, 0.0]


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


def run(config: dict[str, Any], ctx) -> dict[str, Any]:
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

    for step in range(steps + 1):
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

    solved = bool(accuracy == 1.0 and loss_value < loss_target)
    ctx.log.info("итог: loss=%.5f, acc=%.2f, solved=%s", loss_value, accuracy, solved)

    write_training_checkpoint(
        ctx.run_dir,
        run_id=ctx.run_id,
        experiment=str(config["experiment"]),
        model_name=str(config.get("model", "xor-backprop")),
        model=model,
        optimizer=optimizer,
        step=steps,
    )
    model_manifest = inspect_model(
        model,
        run_id=ctx.run_id,
        model_name=str(config.get("model", "xor-backprop")),
        example_args=(x,),
        layer_ids={"layers.0": "hidden", "layers.1": "output"},
        activations={"layers.0": "relu", "layers.1": "identity"},
        capture_phase="completed",
        step=steps,
    )
    write_model_manifest(ctx.run_dir, model_manifest)
    ctx.log.info(
        "модель экспортирована: checkpoint.pt и model.json (%d слоя)",
        len(model_manifest.layers),
    )

    return {
        "loss": loss_value,
        "acc": accuracy,
        "solved": solved,
        "solved_at_step": solved_at,
    }
