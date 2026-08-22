"""Backprop-эталон на XOR: 2 → 8 → 1.

Точка отсчёта блока 1: та же архитектура и та же диагностика будут потом
обучаться локальным правилом, и разница должна быть видна на одинаковых
графиках. XOR гоняется на CPU — сеть такого размера на GPU медленнее из-за
накладных расходов на запуск ядер.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from bioplast.diagnostics.probes import log_module_state
from bioplast.runner import (
    ContractError,
    ExperimentResult,
    ModelArtifacts,
    RunEvent,
    RunStatus,
    XorDecisionSurface,
    XorParameterUpdate,
    XorTrainingLayer,
    XorTrainStepSnapshot,
    append_event,
    iter_events,
    load_training_recovery,
    load_xor_train_step_snapshot,
    recovery_interval,
    utc_offset_iso,
    write_training_recovery,
    write_xor_train_step_snapshot,
)

XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = [0.0, 1.0, 1.0, 0.0]
RECOVERY_ADAPTER = "xor_backprop_v1"
TRAINING_EVENT_TYPE = "xor_train_step"
MAX_AUTOMATIC_SNAPSHOT_FRAMES = 200


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


def _snapshot_interval(config: dict[str, Any], steps: int) -> int:
    raw = config.get("snapshot_every_steps")
    if raw is None:
        return max(1, math.ceil((steps + 1) / MAX_AUTOMATIC_SNAPSHOT_FRAMES))
    if isinstance(raw, bool):
        raise ContractError("snapshot_every_steps должен быть положительным целым")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ContractError("snapshot_every_steps должен быть положительным целым") from exc
    if value < 1:
        raise ContractError("snapshot_every_steps должен быть положительным целым")
    return value


def _parameter_state(model: MLP) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "hidden": {
            "weight": model.layers[0].weight.detach().clone(),
            "bias": model.layers[0].bias.detach().clone(),
        },
        "output": {
            "weight": model.layers[1].weight.detach().clone(),
            "bias": model.layers[1].bias.detach().clone(),
        },
    }


def _training_layers(
    before: dict[str, dict[str, torch.Tensor]], model: MLP
) -> tuple[XorTrainingLayer, ...]:
    after = _parameter_state(model)
    layers: list[XorTrainingLayer] = []
    for layer_id in ("hidden", "output"):
        parameters = []
        for name in ("weight", "bias"):
            old = before[layer_id][name]
            new = after[layer_id][name]
            parameters.append(
                XorParameterUpdate(
                    name=name,
                    before=old.cpu().tolist(),
                    delta=(new - old).cpu().tolist(),
                    after=new.cpu().tolist(),
                )
            )
        layers.append(
            XorTrainingLayer(
                layer_id=layer_id,
                parameters=tuple(parameters),
                # Backprop не имеет сегрегированной апикали. Поля уже входят в
                # контракт и будут заполнены ThreeFactorLayer в блоке 1.
                apical_deviation=None,
                local_error=None,
            )
        )
    return tuple(layers)


def _decision_surface(model: MLP, device: torch.device | str) -> XorDecisionSurface:
    axis = torch.linspace(-0.25, 1.25, 25, device=device)
    x1_grid, x0_grid = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((x0_grid.reshape(-1), x1_grid.reshape(-1)), dim=1)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(points)).reshape(axis.numel(), axis.numel())
    return XorDecisionSurface(
        x0=tuple(float(value) for value in axis.cpu().tolist()),
        x1=tuple(float(value) for value in axis.cpu().tolist()),
        probabilities=tuple(
            tuple(float(value) for value in row)
            for row in probabilities.cpu().tolist()
        ),
    )


def _publish_training_snapshot(
    ctx,
    snapshot: XorTrainStepSnapshot,
    existing_event: RunEvent | None,
) -> int:
    if existing_event is not None:
        if existing_event.snapshot is None:
            raise ContractError("событие XOR train-step не содержит snapshot")
        existing = load_xor_train_step_snapshot(ctx.run_dir / existing_event.snapshot)
        if existing.to_dict() != snapshot.to_dict():
            raise ContractError(
                f"повторно вычисленный XOR snapshot шага {snapshot.step} "
                "не совпал с опубликованным до сбоя"
            )
        return existing_event.seq

    path = write_xor_train_step_snapshot(ctx.run_dir, snapshot)
    append_event(
        ctx.run_dir,
        RunEvent(
            run_id=ctx.run_id,
            seq=snapshot.seq,
            occurred_at=utc_offset_iso(),
            event_type=TRAINING_EVENT_TYPE,
            step=snapshot.step,
            phase=snapshot.phase,
            scalars={
                "loss": snapshot.loss,
                "accuracy": snapshot.accuracy,
                "updated": snapshot.updated,
            },
            snapshot=path.relative_to(ctx.run_dir).as_posix(),
        ),
    )
    return snapshot.seq


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
    snapshot_every = _snapshot_interval(config, steps)
    training_events = [
        event for event in iter_events(ctx.run_dir) if event.event_type == TRAINING_EVENT_TYPE
    ]
    events_by_step = {event.step: event for event in training_events}
    if len(events_by_step) != len(training_events) or None in events_by_step:
        raise ContractError("события XOR train-step должны иметь уникальный step")
    event_seq = max((event.seq for event in iter_events(ctx.run_dir)), default=0)

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
        event_seq = max(event_seq, int(recovery_state.get("last_event_seq", 0)))
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
            event_seq=event_seq,
        )

    for step in range(next_step, steps + 1):
        ctx.control.checkpoint(step=step, phase="train_step")
        capture_snapshot = (
            step < steps
            and (
                step % snapshot_every == 0
                or ctx.control.mode is RunStatus.PAUSED
                or step in events_by_step
            )
        )
        before = _parameter_state(model) if capture_snapshot else None
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

        if capture_snapshot:
            existing_event = events_by_step.get(step)
            snapshot_seq = existing_event.seq if existing_event is not None else event_seq + 1
            snapshot = XorTrainStepSnapshot(
                run_id=ctx.run_id,
                seq=snapshot_seq,
                step=step,
                loss=loss_value,
                accuracy=accuracy,
                updated=step < steps,
                layers=_training_layers(before, model),
                decision_surface=_decision_surface(model, device),
            )
            event_seq = _publish_training_snapshot(ctx, snapshot, existing_event)
            if existing_event is None:
                events_by_step[step] = next(
                    event
                    for event in iter_events(ctx.run_dir)
                    if event.seq == event_seq
                )

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
                event_seq=event_seq,
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
