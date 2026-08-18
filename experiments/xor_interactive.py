"""Интерактивный послойный forward обученного backprop-XOR.

Сессия является отдельным дочерним прогоном. Модель загружается из checkpoint
родителя внутри worker-процесса, а браузер получает только JSON snapshots.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from bioplast.runner import (
    ContractError,
    ExperimentResult,
    ModelArtifacts,
    RunEvent,
    XorForwardSnapshot,
    append_event,
    load_training_checkpoint,
    utc_offset_iso,
    write_xor_forward_snapshot,
)
from experiments.xor_backprop import MLP


def _source_run_dir(config: dict[str, Any], ctx) -> Path:
    source_run_id = config.get("source_run_id")
    if (
        not isinstance(source_run_id, str)
        or not source_run_id
        or source_run_id in {".", ".."}
        or "/" in source_run_id
        or "\\" in source_run_id
    ):
        raise ContractError("source_run_id должен быть именем одного прогона")
    source = (ctx.run_dir.parent / source_run_id).resolve()
    if source.parent != ctx.run_dir.parent.resolve() or not source.is_dir():
        raise ContractError(f"исходный XOR-прогон не найден: {source_run_id}")
    return source


def _publish(ctx, snapshot: XorForwardSnapshot) -> None:
    path = write_xor_forward_snapshot(ctx.run_dir, snapshot)
    scalars: dict[str, int | float | bool | None] = {
        "input_command_seq": snapshot.input_command_seq,
        "x0": snapshot.input_values[0],
        "x1": snapshot.input_values[1],
    }
    if snapshot.probability is not None:
        scalars["probability"] = snapshot.probability
        scalars["prediction"] = snapshot.prediction
    append_event(
        ctx.run_dir,
        RunEvent(
            run_id=ctx.run_id,
            seq=snapshot.seq,
            occurred_at=utc_offset_iso(),
            event_type="xor_forward",
            step=snapshot.input_command_seq,
            phase=snapshot.phase,
            layer_id=snapshot.layer_id,
            scalars=scalars,
            snapshot=path.relative_to(ctx.run_dir).as_posix(),
        ),
    )


def run(config: dict[str, Any], ctx) -> ExperimentResult:
    source_dir = _source_run_dir(config, ctx)
    checkpoint = load_training_checkpoint(source_dir / "checkpoint.pt", map_location=ctx.device)
    if checkpoint["run_id"] != source_dir.name:
        raise ContractError("checkpoint принадлежит другому исходному прогону")
    if checkpoint["experiment"] != "xor_backprop":
        raise ContractError("интерактивный XOR поддерживает checkpoint xor_backprop")

    hidden = int(config.get("hidden", 8))
    model = MLP([2, hidden, 1]).to(ctx.device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        raise ContractError(f"checkpoint несовместим с MLP 2→{hidden}→1: {exc}") from exc
    model.eval()

    max_inputs = config.get("max_inputs")
    if max_inputs is not None:
        max_inputs = int(max_inputs)
        if max_inputs < 1:
            raise ContractError("max_inputs должен быть положительным")

    ctx.log.info(
        "интерактивный XOR: загружен checkpoint %s, ожидаем set_input",
        source_dir.name,
    )
    input_command_seq = 0
    event_seq = 0
    completed_inputs = 0
    last_input = torch.zeros((1, 2), dtype=torch.float32, device=ctx.device)

    while max_inputs is None or completed_inputs < max_inputs:
        input_command_seq, values = ctx.control.wait_for_input(after_seq=input_command_seq)
        if len(values) != 2 or any(not math.isfinite(value) for value in values):
            raise ContractError("интерактивный XOR требует два конечных входа")
        last_input = torch.tensor([values], dtype=torch.float32, device=ctx.device)

        event_seq += 1
        _publish(
            ctx,
            XorForwardSnapshot(
                run_id=ctx.run_id,
                seq=event_seq,
                input_command_seq=input_command_seq,
                phase="input",
                layer_id=None,
                input_values=(values[0], values[1]),
            ),
        )

        ctx.control.checkpoint(step=input_command_seq, phase="forward_hidden")
        with torch.no_grad():
            hidden_z = model.layers[0](last_input)
            hidden_post = torch.relu(hidden_z)
        event_seq += 1
        _publish(
            ctx,
            XorForwardSnapshot(
                run_id=ctx.run_id,
                seq=event_seq,
                input_command_seq=input_command_seq,
                phase="forward_hidden",
                layer_id="hidden",
                input_values=(values[0], values[1]),
                z=tuple(float(value) for value in hidden_z[0].tolist()),
                post=tuple(float(value) for value in hidden_post[0].tolist()),
                hidden=tuple(float(value) for value in hidden_post[0].tolist()),
            ),
        )

        ctx.control.checkpoint(step=input_command_seq, phase="forward_output")
        with torch.no_grad():
            output_z = model.layers[1](hidden_post)
            probability = torch.sigmoid(output_z)
        probability_value = float(probability.item())
        prediction = int(probability_value >= 0.5)
        event_seq += 1
        _publish(
            ctx,
            XorForwardSnapshot(
                run_id=ctx.run_id,
                seq=event_seq,
                input_command_seq=input_command_seq,
                phase="forward_output",
                layer_id="output",
                input_values=(values[0], values[1]),
                z=(float(output_z.item()),),
                post=(probability_value,),
                hidden=tuple(float(value) for value in hidden_post[0].tolist()),
                probability=probability_value,
                prediction=prediction,
            ),
        )
        completed_inputs += 1
        ctx.log.info(
            "XOR forward #%d: input=[%g, %g], p=%.6f, prediction=%d",
            completed_inputs,
            values[0],
            values[1],
            probability_value,
            prediction,
        )

    return ExperimentResult(
        final={"inputs_processed": completed_inputs},
        model_artifacts=ModelArtifacts(
            model=model,
            example_args=(last_input,),
            layer_ids={"layers.0": "hidden", "layers.1": "output"},
            activations={"layers.0": "relu", "layers.1": "identity"},
            step=checkpoint.get("step"),
            capture_phase="interactive_completed",
        ),
    )
