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
    load_recovery,
    iter_events,
    load_xor_forward_snapshot,
    utc_offset_iso,
    write_recovery,
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


def _save_recovery(ctx, state: dict[str, Any]) -> None:
    control = ctx.control
    payload = {
        "adapter": "xor_interactive_v1",
        "session": state,
        "control": {
            "last_seq": control.last_seq,
            "mode": control.mode.value,
            "delay_ms": control.delay_ms,
            "input_seq": control.input_seq,
            "input_values": list(control.input_values) if control.input_values is not None else None,
        },
    }
    write_recovery(
        ctx.run_dir,
        payload,
        adapter="xor_interactive_v1",
        attempt=ctx.attempt,
        cursor=str(state["phase"]),
        command_seq=control.last_seq,
        event_seq=int(state["event_seq"]),
    )


def _reconcile_event_tail(ctx, state: dict[str, Any]) -> bool:
    """Recover the narrow append-event -> publish-recovery crash window."""
    changed = False
    for event in iter_events(ctx.run_dir):
        if event.seq <= int(state["event_seq"]):
            continue
        if event.snapshot is None:
            continue
        snapshot = load_xor_forward_snapshot(ctx.run_dir / event.snapshot)
        previous_phase = str(state["phase"])
        previous_input_seq = int(state["input_command_seq"])
        state.update(
            phase=snapshot.phase,
            input_command_seq=snapshot.input_command_seq,
            event_seq=snapshot.seq,
            input_values=list(snapshot.input_values),
        )
        if snapshot.hidden is not None:
            state["hidden"] = list(snapshot.hidden)
        if (
            snapshot.phase == "forward_output"
            and (previous_phase != "forward_output" or previous_input_seq != snapshot.input_command_seq)
        ):
            state["completed_inputs"] = int(state["completed_inputs"]) + 1
        changed = True
    return changed


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
    recovery_path = ctx.run_dir / "recovery" / "state.json"
    if recovery_path.is_file():
        _recovery_meta, recovery_payload = load_recovery(
            ctx.run_dir, expected_adapter="xor_interactive_v1"
        )
        state = recovery_payload.get("session")
        if not isinstance(state, dict):
            raise ContractError("recovery XOR не содержит session state")
        ctx.log.info(
            "XOR recovery: phase=%s, event_seq=%s",
            state.get("phase"),
            state.get("event_seq"),
        )
        if _reconcile_event_tail(ctx, state):
            ctx.log.warning("recovery продвинут по уже опубликованному хвосту events.jsonl")
            _save_recovery(ctx, state)
    else:
        state = {
            "phase": "waiting_input",
            "input_command_seq": 0,
            "event_seq": 0,
            "completed_inputs": 0,
            "input_values": [0.0, 0.0],
            "hidden": None,
        }
        _save_recovery(ctx, state)

    initial_values = tuple(float(value) for value in state.get("input_values", [0.0, 0.0]))
    last_input = torch.tensor([initial_values], dtype=torch.float32, device=ctx.device)

    while max_inputs is None or int(state["completed_inputs"]) < max_inputs:
        phase = str(state["phase"])
        if phase in {"waiting_input", "forward_output"}:
            input_command_seq, values = ctx.control.wait_for_input(
                after_seq=int(state["input_command_seq"])
            )
            if len(values) != 2 or any(not math.isfinite(value) for value in values):
                raise ContractError("интерактивный XOR требует два конечных входа")
            state.update(
                phase="input",
                input_command_seq=input_command_seq,
                input_values=[values[0], values[1]],
                hidden=None,
            )
            state["event_seq"] = int(state["event_seq"]) + 1
            _publish(
                ctx,
                XorForwardSnapshot(
                    run_id=ctx.run_id,
                    seq=int(state["event_seq"]),
                    input_command_seq=input_command_seq,
                    phase="input",
                    layer_id=None,
                    input_values=(values[0], values[1]),
                ),
            )
            _save_recovery(ctx, state)

        values = tuple(float(value) for value in state["input_values"])
        input_command_seq = int(state["input_command_seq"])
        last_input = torch.tensor([values], dtype=torch.float32, device=ctx.device)

        if state["phase"] == "input":
            ctx.control.checkpoint(step=input_command_seq, phase="forward_hidden")
            with torch.no_grad():
                hidden_z = model.layers[0](last_input)
                hidden_post = torch.relu(hidden_z)
            state["phase"] = "forward_hidden"
            state["hidden"] = [float(value) for value in hidden_post[0].tolist()]
            state["event_seq"] = int(state["event_seq"]) + 1
            _publish(
                ctx,
                XorForwardSnapshot(
                    run_id=ctx.run_id,
                    seq=int(state["event_seq"]),
                    input_command_seq=input_command_seq,
                    phase="forward_hidden",
                    layer_id="hidden",
                    input_values=(values[0], values[1]),
                    z=tuple(float(value) for value in hidden_z[0].tolist()),
                    post=tuple(float(value) for value in hidden_post[0].tolist()),
                    hidden=tuple(float(value) for value in hidden_post[0].tolist()),
                ),
            )
            _save_recovery(ctx, state)

        if state["phase"] == "forward_hidden":
            hidden_values = tuple(float(value) for value in state["hidden"])
            hidden_post = torch.tensor([hidden_values], dtype=torch.float32, device=ctx.device)
            ctx.control.checkpoint(step=input_command_seq, phase="forward_output")
            with torch.no_grad():
                output_z = model.layers[1](hidden_post)
                probability = torch.sigmoid(output_z)
            probability_value = float(probability.item())
            prediction = int(probability_value >= 0.5)
            state["phase"] = "forward_output"
            state["completed_inputs"] = int(state["completed_inputs"]) + 1
            state["event_seq"] = int(state["event_seq"]) + 1
            _publish(
                ctx,
                XorForwardSnapshot(
                    run_id=ctx.run_id,
                    seq=int(state["event_seq"]),
                    input_command_seq=input_command_seq,
                    phase="forward_output",
                    layer_id="output",
                    input_values=(values[0], values[1]),
                    z=(float(output_z.item()),),
                    post=(probability_value,),
                    hidden=hidden_values,
                    probability=probability_value,
                    prediction=prediction,
                ),
            )
            _save_recovery(ctx, state)
            ctx.log.info(
                "XOR forward #%d: input=[%g, %g], p=%.6f, prediction=%d",
                state["completed_inputs"],
                values[0],
                values[1],
                probability_value,
                prediction,
            )

    return ExperimentResult(
        final={"inputs_processed": int(state["completed_inputs"])},
        model_artifacts=ModelArtifacts(
            model=model,
            example_args=(last_input,),
            layer_ids={"layers.0": "hidden", "layers.1": "output"},
            activations={"layers.0": "relu", "layers.1": "identity"},
            step=checkpoint.get("step"),
            capture_phase="interactive_completed",
        ),
    )
