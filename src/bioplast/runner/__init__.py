"""Раннер: конфиг-файл → прогон → `runs/<id>/`.

Прогон полностью описывается JSON-конфигом; ручных шагов нет ни одного —
это условие того, что вечерние свипы и (позже) аренда облака вообще окупятся.
"""

from bioplast.runner.run import (
    RunContext,
    config_slug,
    fail_prepared_run,
    git_provenance,
    make_run_id,
    prepare_run,
    run_config,
    run_prepared,
    validate_run_config,
)
from bioplast.runner.contracts import (
    CONTRACT_VERSION,
    ConnectionSpec,
    ContractError,
    LayerSpec,
    ModelManifest,
    RunEvent,
    RunManifest,
    RunStatus,
    TensorSpec,
    append_event,
    iter_events,
    load_model_manifest,
    load_run_manifest,
    write_model_manifest,
    write_run_manifest,
)
from bioplast.runner.queue import RunScheduler, run_queue

__all__ = [
    "CONTRACT_VERSION",
    "ConnectionSpec",
    "ContractError",
    "LayerSpec",
    "ModelManifest",
    "RunEvent",
    "RunContext",
    "RunManifest",
    "RunScheduler",
    "RunStatus",
    "TensorSpec",
    "append_event",
    "config_slug",
    "fail_prepared_run",
    "git_provenance",
    "iter_events",
    "load_model_manifest",
    "load_run_manifest",
    "make_run_id",
    "prepare_run",
    "run_config",
    "run_prepared",
    "run_queue",
    "validate_run_config",
    "write_model_manifest",
    "write_run_manifest",
]
