"""Layer-by-layer inspection of a trained MNIST MLP.

The browser submits one test-set index. The worker loads the parent checkpoint
and dataset, then publishes only the selected 28x28 preview, top-k prediction
and aggregate tensor summaries for completed layers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from bioplast.data import load_mnist
from bioplast.runner import (
    ContractError,
    ExperimentResult,
    ModelArtifacts,
    ModelDebugClassScore,
    ModelDebugLayerSnapshot,
    ModelDebugSnapshot,
    RunEvent,
    TensorSpec,
    append_event,
    inspect_tensor,
    iter_events,
    load_model_debug_snapshot,
    load_recovery,
    load_training_checkpoint,
    utc_offset_iso,
    write_model_debug_snapshot,
    write_recovery,
)
from experiments.mnist_mlp_backprop import MLP


RECOVERY_ADAPTER = "mnist_mlp_interactive_v1"


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
        raise ContractError(f"исходный MNIST-прогон не найден: {source_run_id}")
    return source


def _summary(name: str, role: str, tensor: torch.Tensor) -> TensorSpec:
    return inspect_tensor(name, role, tensor, full_values_max_elements=0)


def _layer_from_dict(value: dict[str, Any]) -> ModelDebugLayerSnapshot:
    return ModelDebugLayerSnapshot.from_dict(value)


def _snapshot_from_state(ctx, state: dict[str, Any]) -> ModelDebugSnapshot:
    layers = tuple(_layer_from_dict(item) for item in state.get("layers", ()))
    top_classes = tuple(
        ModelDebugClassScore.from_dict(item) for item in state.get("top_classes", ())
    )
    phase = "input" if not layers else "forward_layer"
    return ModelDebugSnapshot(
        run_id=ctx.run_id,
        seq=int(state["event_seq"]),
        input_command_seq=int(state["input_command_seq"]),
        phase=phase,
        layer_id=layers[-1].layer_id if layers else None,
        input_mode="dataset_index",
        input_index=int(state["input_index"]),
        input_label=int(state["input_label"]),
        input_preview=tuple(tuple(float(item) for item in row) for row in state["input_preview"]),
        layers=layers,
        prediction=(int(state["prediction"]) if state.get("prediction") is not None else None),
        top_classes=top_classes,
    )


def _publish(ctx, state: dict[str, Any]) -> None:
    snapshot = _snapshot_from_state(ctx, state)
    path = write_model_debug_snapshot(ctx.run_dir, snapshot)
    scalars: dict[str, int | float | bool | None] = {
        "input_command_seq": snapshot.input_command_seq,
        "input_index": snapshot.input_index,
        "input_label": snapshot.input_label,
        "completed_layers": len(snapshot.layers),
    }
    if snapshot.prediction is not None:
        scalars["prediction"] = snapshot.prediction
        scalars["correct"] = snapshot.prediction == snapshot.input_label
    append_event(
        ctx.run_dir,
        RunEvent(
            run_id=ctx.run_id,
            seq=snapshot.seq,
            occurred_at=utc_offset_iso(),
            event_type="model_debug",
            step=snapshot.input_command_seq,
            phase=snapshot.phase,
            layer_id=snapshot.layer_id,
            scalars=scalars,
            snapshot=path.relative_to(ctx.run_dir).as_posix(),
        ),
    )


def _save_recovery(ctx, state: dict[str, Any]) -> None:
    control = ctx.control
    write_recovery(
        ctx.run_dir,
        {
            "adapter": RECOVERY_ADAPTER,
            "session": state,
            "control": {
                "last_seq": control.last_seq,
                "mode": control.mode.value,
                "delay_ms": control.delay_ms,
                "input_seq": control.input_seq,
                "input_values": (
                    list(control.input_values) if control.input_values is not None else None
                ),
            },
        },
        adapter=RECOVERY_ADAPTER,
        attempt=ctx.attempt,
        cursor=str(state["phase"]),
        command_seq=control.last_seq,
        event_seq=int(state["event_seq"]),
        progress={
            "input_index": state.get("input_index"),
            "completed_layers": len(state.get("layers", ())),
            "inputs_processed": int(state["completed_inputs"]),
        },
    )


def _reconcile_event_tail(ctx, state: dict[str, Any]) -> bool:
    changed = False
    for event in iter_events(ctx.run_dir):
        if event.seq <= int(state["event_seq"]) or event.event_type != "model_debug":
            continue
        if event.snapshot is None:
            continue
        snapshot = load_model_debug_snapshot(ctx.run_dir / event.snapshot)
        previously_complete = state.get("phase") == "forward_complete"
        state.update(
            phase=("forward_complete" if snapshot.prediction is not None else snapshot.phase),
            input_command_seq=snapshot.input_command_seq,
            event_seq=snapshot.seq,
            input_index=snapshot.input_index,
            input_label=snapshot.input_label,
            input_preview=[list(row) for row in snapshot.input_preview],
            layers=[layer.to_dict() for layer in snapshot.layers],
            prediction=snapshot.prediction,
            top_classes=[item.to_dict() for item in snapshot.top_classes],
        )
        if snapshot.prediction is not None and not previously_complete:
            state["completed_inputs"] = int(state["completed_inputs"]) + 1
        changed = True
    return changed


def _activation_after_layers(model: MLP, x: torch.Tensor, completed: int) -> torch.Tensor:
    value = x
    with torch.no_grad():
        for index, layer in enumerate(model.layers[:completed]):
            value = layer(value)
            if index < len(model.layers) - 1:
                value = torch.relu(value)
    return value


def run(config: dict[str, Any], ctx) -> ExperimentResult:
    source_dir = _source_run_dir(config, ctx)
    checkpoint = load_training_checkpoint(source_dir / "checkpoint.pt", map_location=ctx.device)
    if checkpoint["run_id"] != source_dir.name:
        raise ContractError("checkpoint принадлежит другому исходному прогону")
    if checkpoint["experiment"] != "mnist_mlp_backprop":
        raise ContractError("MNIST debug поддерживает checkpoint mnist_mlp_backprop")

    hidden = [int(value) for value in config.get("hidden", [256])]
    if not hidden or any(value < 1 for value in hidden):
        raise ContractError("MNIST debug требует положительные hidden dimensions")
    model = MLP([784, *hidden, 10]).to(ctx.device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        raise ContractError(f"checkpoint несовместим с MNIST MLP: {exc}") from exc
    model.eval()

    data = load_mnist(
        root=Path(config.get("data_root", "data/mnist")),
        device=ctx.device,
        flatten=True,
        train_size=config.get("train_size"),
    )
    if data.test_x.ndim != 2 or data.test_x.shape[1] != 784:
        raise ContractError("MNIST debug требует test tensor формы N×784")

    max_inputs = config.get("max_inputs")
    if max_inputs is not None:
        max_inputs = int(max_inputs)
        if max_inputs < 1:
            raise ContractError("max_inputs должен быть положительным")

    ctx.log.info(
        "MNIST debug: checkpoint %s, test examples=%d, ожидаем индекс через set_input",
        source_dir.name,
        data.test_x.shape[0],
    )
    recovery_path = ctx.run_dir / "recovery" / "state.json"
    if recovery_path.is_file():
        _meta, payload = load_recovery(ctx.run_dir, expected_adapter=RECOVERY_ADAPTER)
        state = payload.get("session")
        if not isinstance(state, dict):
            raise ContractError("MNIST debug recovery не содержит session state")
        if _reconcile_event_tail(ctx, state):
            ctx.log.warning("MNIST debug recovery продвинут по events.jsonl")
            _save_recovery(ctx, state)
    else:
        state = {
            "phase": "waiting_input",
            "input_command_seq": 0,
            "event_seq": 0,
            "completed_inputs": 0,
            "input_index": 0,
            "input_label": 0,
            "input_preview": [[0.0] * 28 for _ in range(28)],
            "layers": [],
            "prediction": None,
            "top_classes": [],
        }
        _save_recovery(ctx, state)

    last_input = data.test_x[:1]
    while max_inputs is None or int(state["completed_inputs"]) < max_inputs:
        if state["phase"] in {"waiting_input", "forward_complete"}:
            input_command_seq, values = ctx.control.wait_for_input(
                after_seq=int(state["input_command_seq"])
            )
            if len(values) != 1 or not math.isfinite(values[0]) or not values[0].is_integer():
                raise ContractError("MNIST debug требует один целый индекс примера")
            input_index = int(values[0])
            if not 0 <= input_index < data.test_x.shape[0]:
                raise ContractError(
                    f"индекс MNIST должен лежать в [0, {data.test_x.shape[0] - 1}]"
                )
            last_input = data.test_x[input_index : input_index + 1]
            state.update(
                phase="input",
                input_command_seq=input_command_seq,
                input_index=input_index,
                input_label=int(data.test_y[input_index].item()),
                input_preview=last_input.reshape(28, 28).detach().cpu().tolist(),
                layers=[],
                prediction=None,
                top_classes=[],
            )
            state["event_seq"] = int(state["event_seq"]) + 1
            _publish(ctx, state)
            _save_recovery(ctx, state)

        last_input = data.test_x[int(state["input_index"]) : int(state["input_index"]) + 1]
        completed = len(state["layers"])
        value = _activation_after_layers(model, last_input, completed)
        while completed < len(model.layers):
            layer = model.layers[completed]
            layer_id = (
                "output"
                if completed == len(model.layers) - 1
                else "hidden" if len(model.layers) == 2 else f"hidden_{completed + 1}"
            )
            ctx.control.checkpoint(
                step=int(state["input_command_seq"]),
                phase=f"forward_layer:{layer_id}",
            )
            with torch.no_grad():
                preactivation = layer(value)
                output = (
                    preactivation
                    if completed == len(model.layers) - 1
                    else torch.relu(preactivation)
                )
            layer_snapshot = ModelDebugLayerSnapshot(
                layer_id=layer_id,
                module_path=f"layers.{completed}",
                layer_type=type(layer).__name__,
                activation=("identity" if completed == len(model.layers) - 1 else "relu"),
                parameter_count=sum(parameter.numel() for parameter in layer.parameters()),
                input_tensor=_summary("input", "activation_input", value),
                preactivation_tensor=_summary("z", "preactivation", preactivation),
                output_tensor=_summary("post", "activation_output", output),
            )
            state["layers"].append(layer_snapshot.to_dict())
            completed += 1
            value = output
            if completed == len(model.layers):
                probabilities = torch.softmax(output[0], dim=0)
                top_probability, top_index = torch.topk(probabilities, k=min(3, probabilities.numel()))
                state["prediction"] = int(top_index[0].item())
                state["top_classes"] = [
                    {"class_index": int(index.item()), "probability": float(probability.item())}
                    for probability, index in zip(top_probability, top_index)
                ]
                state["phase"] = "forward_complete"
                state["completed_inputs"] = int(state["completed_inputs"]) + 1
            else:
                state["phase"] = "forward_layer"
            state["event_seq"] = int(state["event_seq"]) + 1
            _publish(ctx, state)
            _save_recovery(ctx, state)

        ctx.log.info(
            "MNIST forward #%d: test[%d], label=%d, prediction=%d",
            state["completed_inputs"],
            state["input_index"],
            state["input_label"],
            state["prediction"],
        )

    last_layer = len(model.layers) - 1
    layer_ids = {
        f"layers.{index}": (
            "output"
            if index == last_layer
            else "hidden" if last_layer == 1 else f"hidden_{index + 1}"
        )
        for index in range(len(model.layers))
    }
    activations = {
        f"layers.{index}": "identity" if index == last_layer else "relu"
        for index in range(len(model.layers))
    }
    return ExperimentResult(
        final={"inputs_processed": int(state["completed_inputs"])},
        model_artifacts=ModelArtifacts(
            model=model,
            example_args=(last_input,),
            layer_ids=layer_ids,
            activations=activations,
            step=checkpoint.get("step"),
            capture_phase="interactive_completed",
        ),
    )
