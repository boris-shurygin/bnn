"""Раннер: конфиг-файл → прогон → `runs/<id>/`.

Прогон полностью описывается JSON-конфигом; ручных шагов нет ни одного —
это условие того, что вечерние свипы и (позже) аренда облака вообще окупятся.
"""

from bioplast.runner.run import (
    RunContext,
    cancel_prepared_run,
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
    CHECKPOINT_FILE,
    COMMANDS_FILE,
    CONTRACT_VERSION,
    ConnectionSpec,
    ContractError,
    LayerSpec,
    ModelManifest,
    RunEvent,
    RunManifest,
    RunStatus,
    TensorSpec,
    TensorSummary,
    append_event,
    iter_events,
    load_model_manifest,
    load_run_manifest,
    utc_offset_iso,
    write_model_manifest,
    write_run_manifest,
)
from bioplast.runner.control import (
    CooperativeRunControl,
    RunCancelled,
    RunCommand,
    RunCommandType,
    append_run_command,
    read_run_commands,
)
from bioplast.runner.checkpoints import (
    CHECKPOINT_VERSION,
    load_training_checkpoint,
    write_training_checkpoint,
)
from bioplast.runner.inspection import InspectionError, inspect_model, inspect_tensor
from bioplast.runner.experiment import ExperimentResult, ModelArtifacts
from bioplast.runner.queue import RunScheduler, run_queue
from bioplast.runner.snapshots import (
    XorForwardSnapshot,
    load_xor_forward_snapshot,
    write_xor_forward_snapshot,
)

__all__ = [
    "CONTRACT_VERSION",
    "CHECKPOINT_FILE",
    "COMMANDS_FILE",
    "CHECKPOINT_VERSION",
    "ConnectionSpec",
    "ContractError",
    "CooperativeRunControl",
    "ExperimentResult",
    "InspectionError",
    "LayerSpec",
    "ModelManifest",
    "ModelArtifacts",
    "RunEvent",
    "RunCancelled",
    "RunCommand",
    "RunCommandType",
    "RunContext",
    "RunManifest",
    "RunScheduler",
    "RunStatus",
    "TensorSpec",
    "TensorSummary",
    "XorForwardSnapshot",
    "append_event",
    "append_run_command",
    "cancel_prepared_run",
    "config_slug",
    "fail_prepared_run",
    "git_provenance",
    "inspect_model",
    "inspect_tensor",
    "iter_events",
    "load_model_manifest",
    "load_run_manifest",
    "load_training_checkpoint",
    "load_xor_forward_snapshot",
    "make_run_id",
    "prepare_run",
    "read_run_commands",
    "run_config",
    "run_prepared",
    "run_queue",
    "utc_offset_iso",
    "validate_run_config",
    "write_model_manifest",
    "write_run_manifest",
    "write_training_checkpoint",
    "write_xor_forward_snapshot",
]
