"""Allowlisted registry for child model-debug sessions.

The API process validates only persisted manifests and builds a typed child
configuration. Checkpoints are still loaded exclusively by the child worker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from bioplast.runner import (
    ModelManifest,
    RunStatus,
    fail_prepared_run,
    load_model_manifest,
    prepare_run,
    utc_offset_iso,
    write_model_manifest,
)
from bioplast.viz.repository import RunRepository


class Scheduler(Protocol):
    def submit(self, run_dir: Path | str) -> None: ...


class DebugSessionConflict(RuntimeError):
    pass


class DebugSessionSubmissionError(RuntimeError):
    pass


ConfigBuilder = Callable[[dict[str, Any], ModelManifest, str], dict[str, Any]]


@dataclass(frozen=True)
class DebugAdapter:
    """One explicit source-model to worker/renderer mapping."""

    adapter_id: str
    source_experiment: str
    source_dataset: str
    debug_experiment: str
    renderer: str
    build_config: ConfigBuilder

    @property
    def key(self) -> tuple[str, str]:
        return self.source_experiment, self.source_dataset


def _xor_config(
    source_config: dict[str, Any],
    source_model: ModelManifest,
    source_run_id: str,
) -> dict[str, Any]:
    layer_by_id = {layer.layer_id: layer for layer in source_model.layers}
    hidden_layer = layer_by_id.get("hidden")
    output_layer = layer_by_id.get("output")
    if (
        hidden_layer is None
        or output_layer is None
        or hidden_layer.input_shape[-1:] != (2,)
        or output_layer.output_shape[-1:] != (1,)
    ):
        raise DebugSessionConflict("model.json не описывает поддерживаемую сеть XOR 2→H→1")
    hidden = hidden_layer.output_shape[-1]
    if not isinstance(hidden, int) or hidden < 1:
        raise DebugSessionConflict("не удалось определить размер скрытого слоя XOR")
    return {
        "session": "V.9",
        "dataset": "xor",
        "model": source_config.get("model") or source_model.model_name,
        "experiment": "xor_interactive",
        "device": "cpu",
        "seed": int(source_config.get("seed", 0)),
        "hidden": hidden,
        "source_run_id": source_run_id,
        "tag": "interactive-forward",
        "debug": {
            "protocol": "model_debug_v1",
            "adapter": "xor_interactive_v1",
            "renderer": "xor_neurons_v1",
            "accepts_input": True,
            "input_mode": "manual_vector",
            "input_size": 2,
            "supports_step": True,
            "step_scope": "layer",
            "views": ["neurons", "tensor_summary"],
        },
    }


def _mnist_config(
    source_config: dict[str, Any],
    source_model: ModelManifest,
    source_run_id: str,
) -> dict[str, Any]:
    layers = source_model.layers
    if (
        not layers
        or layers[0].input_shape[-1:] != (784,)
        or layers[-1].output_shape[-1:] != (10,)
    ):
        raise DebugSessionConflict(
            "model.json не описывает поддерживаемый MNIST MLP с входом 784 и выходом 10"
        )
    hidden: list[int] = []
    for layer in layers[:-1]:
        width = layer.output_shape[-1] if layer.output_shape else None
        if not isinstance(width, int) or width < 1:
            raise DebugSessionConflict("не удалось определить размеры скрытых слоёв MNIST")
        hidden.append(width)
    if not hidden:
        raise DebugSessionConflict("MNIST debug требует хотя бы один скрытый слой")
    config: dict[str, Any] = {
        "session": "V.13",
        "dataset": "mnist",
        "model": source_config.get("model") or source_model.model_name,
        "experiment": "mnist_interactive",
        "device": "cpu",
        "seed": int(source_config.get("seed", 0)),
        "hidden": hidden,
        "source_run_id": source_run_id,
        "data_root": source_config.get("data_root", "data/mnist"),
        "tag": "interactive-inspection",
        "debug": {
            "protocol": "model_debug_v1",
            "adapter": "mnist_mlp_interactive_v1",
            "renderer": "tensor_flow_v1",
            "accepts_input": True,
            "input_mode": "dataset_index",
            "input_size": 1,
            "input_min": 0,
            "input_max": 9999,
            "input_integer": True,
            "dataset_split": "test",
            "supports_step": True,
            "step_scope": "layer",
            "views": [
                "module_hierarchy",
                "tensor_flow",
                "activation_summary",
                "neuron_heatmap",
                "neuron_images",
            ],
        },
    }
    if source_config.get("train_size") is not None:
        config["train_size"] = source_config["train_size"]
    return config


DEBUG_ADAPTERS = (
    DebugAdapter(
        adapter_id="xor_interactive_v1",
        source_experiment="xor_backprop",
        source_dataset="xor",
        debug_experiment="xor_interactive",
        renderer="xor_neurons_v1",
        build_config=_xor_config,
    ),
    DebugAdapter(
        adapter_id="mnist_mlp_interactive_v1",
        source_experiment="mnist_mlp_backprop",
        source_dataset="mnist",
        debug_experiment="mnist_interactive",
        renderer="tensor_flow_v1",
        build_config=_mnist_config,
    ),
)
_ADAPTER_BY_SOURCE = {adapter.key: adapter for adapter in DEBUG_ADAPTERS}


def registered_debug_adapters() -> tuple[DebugAdapter, ...]:
    return DEBUG_ADAPTERS


def debug_adapter_metadata(config: dict[str, Any]) -> dict[str, str] | None:
    key = (str(config.get("experiment", "")), str(config.get("dataset", "")))
    adapter = _ADAPTER_BY_SOURCE.get(key)
    if adapter is None:
        return None
    return {
        "adapter": adapter.adapter_id,
        "debug_experiment": adapter.debug_experiment,
        "renderer": adapter.renderer,
    }


class DebugSessionService:
    def __init__(self, repository: RunRepository, scheduler: Scheduler) -> None:
        self.repository = repository
        self.scheduler = scheduler

    def start(self, source_run_id: str) -> dict[str, Any]:
        source = self.repository.get_run(source_run_id)
        manifest = source["manifest"]
        config = source["config"]
        if manifest["status"] != RunStatus.COMPLETED.value:
            raise DebugSessionConflict("отладочная сессия требует завершённый прогон")

        key = (str(config.get("experiment", "")), str(config.get("dataset", "")))
        adapter = _ADAPTER_BY_SOURCE.get(key)
        if adapter is None:
            raise DebugSessionConflict(
                f"для experiment={key[0]!r}, dataset={key[1]!r} нет debug-адаптера"
            )

        source_dir = self.repository.resolve_run(source_run_id)
        for artifact in ("model.json", "checkpoint.pt"):
            if not (source_dir / artifact).is_file():
                raise DebugSessionConflict(f"исходный прогон не содержит {artifact}")
        source_model = load_model_manifest(source_dir)
        debug_config = adapter.build_config(config, source_model, source_run_id)
        if debug_config.get("experiment") != adapter.debug_experiment:
            raise DebugSessionConflict("debug adapter сформировал несовместимый experiment")
        debug = debug_config.get("debug")
        if not isinstance(debug, dict) or debug.get("renderer") != adapter.renderer:
            raise DebugSessionConflict("debug adapter сформировал несовместимый renderer")

        run_dir = prepare_run(
            debug_config,
            self.repository.runs_dir,
            parent_run_id=source_run_id,
        )
        write_model_manifest(
            run_dir,
            replace(
                source_model,
                run_id=run_dir.name,
                captured_at=utc_offset_iso(),
                capture_phase="loaded_checkpoint",
                capture_batch_size=1,
            ),
        )
        try:
            self.scheduler.submit(run_dir)
        except Exception as exc:
            fail_prepared_run(run_dir, f"очередь не приняла debug-сессию: {exc}")
            raise DebugSessionSubmissionError(
                f"не удалось запустить отладочную сессию: {exc}"
            ) from exc
        return {
            "run_id": run_dir.name,
            "status": RunStatus.QUEUED.value,
            "parent_run_id": source_run_id,
            "adapter": adapter.adapter_id,
            "renderer": adapter.renderer,
            "location": f"/runs/{run_dir.name}",
        }


# Compatibility names for callers from V.9–V.12. New code uses generic names.
XorDebugConflict = DebugSessionConflict
XorDebugSubmissionError = DebugSessionSubmissionError
XorDebugService = DebugSessionService
