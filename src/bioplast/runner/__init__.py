"""Раннер: конфиг-файл → прогон → `runs/<id>/`.

Прогон полностью описывается JSON-конфигом; ручных шагов нет ни одного —
это условие того, что вечерние свипы и (позже) аренда облака вообще окупятся.
"""

from bioplast.runner.run import (
    RunContext,
    config_slug,
    git_provenance,
    make_run_id,
    run_config,
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
from bioplast.runner.queue import run_queue

__all__ = [
    "CONTRACT_VERSION",
    "ConnectionSpec",
    "ContractError",
    "LayerSpec",
    "ModelManifest",
    "RunEvent",
    "RunContext",
    "RunManifest",
    "RunStatus",
    "TensorSpec",
    "append_event",
    "config_slug",
    "git_provenance",
    "iter_events",
    "load_model_manifest",
    "load_run_manifest",
    "make_run_id",
    "run_config",
    "run_queue",
    "write_model_manifest",
    "write_run_manifest",
]
