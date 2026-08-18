"""Создание из завершённого XOR-прогона отдельной live-debug сессии."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from bioplast.runner import (
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


class XorDebugConflict(RuntimeError):
    pass


class XorDebugSubmissionError(RuntimeError):
    pass


class XorDebugService:
    def __init__(self, repository: RunRepository, scheduler: Scheduler) -> None:
        self.repository = repository
        self.scheduler = scheduler

    def start(self, source_run_id: str) -> dict[str, Any]:
        source = self.repository.get_run(source_run_id)
        manifest = source["manifest"]
        config = source["config"]
        if manifest["status"] != RunStatus.COMPLETED.value:
            raise XorDebugConflict("интерактивная сессия требует завершённый прогон")
        if config.get("experiment") != "xor_backprop" or config.get("dataset") != "xor":
            raise XorDebugConflict("интерактивная сессия V.9 доступна только для xor_backprop")

        source_dir = self.repository.resolve_run(source_run_id)
        for artifact in ("model.json", "checkpoint.pt"):
            if not (source_dir / artifact).is_file():
                raise XorDebugConflict(f"исходный прогон не содержит {artifact}")
        source_model = load_model_manifest(source_dir)
        layer_by_id = {layer.layer_id: layer for layer in source_model.layers}
        hidden_layer = layer_by_id.get("hidden")
        output_layer = layer_by_id.get("output")
        if (
            hidden_layer is None
            or output_layer is None
            or hidden_layer.input_shape[-1:] != (2,)
            or output_layer.output_shape[-1:] != (1,)
        ):
            raise XorDebugConflict("model.json не описывает поддерживаемую сеть XOR 2→H→1")
        hidden = hidden_layer.output_shape[-1]
        if not isinstance(hidden, int) or hidden < 1:
            raise XorDebugConflict("не удалось определить размер скрытого слоя XOR")

        debug_config = {
            "session": "V.9",
            "dataset": "xor",
            "model": config.get("model") or source_model.model_name,
            "experiment": "xor_interactive",
            "device": "cpu",
            "seed": int(config.get("seed", 0)),
            "hidden": hidden,
            "source_run_id": source_run_id,
            "tag": "interactive-forward",
            "debug": {
                "protocol": "model_debug_v1",
                "renderer": "xor_neurons_v1",
                "accepts_input": True,
                "input_size": 2,
                "supports_step": True,
                "step_scope": "layer",
            },
        }
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
            fail_prepared_run(run_dir, f"очередь не приняла XOR debug-сессию: {exc}")
            raise XorDebugSubmissionError(
                f"не удалось запустить интерактивный XOR: {exc}"
            ) from exc
        return {
            "run_id": run_dir.name,
            "status": RunStatus.QUEUED.value,
            "parent_run_id": source_run_id,
            "location": f"/runs/{run_dir.name}",
        }
