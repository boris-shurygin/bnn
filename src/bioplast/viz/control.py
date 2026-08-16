"""API-сервис типизированных команд для активного прогона."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from bioplast.runner import (
    RunCommandType,
    RunStatus,
    append_run_command,
    cancel_prepared_run,
    load_run_manifest,
    read_run_commands,
)
from bioplast.viz.repository import RunRepository


class RunControlConflict(RuntimeError):
    pass


class RunControlValidationError(ValueError):
    pass


class RunControlService:
    def __init__(self, repository: RunRepository, scheduler: Any) -> None:
        self.repository = repository
        self.scheduler = scheduler
        self._lock = Lock()

    def state(self, run_id: str) -> dict[str, Any]:
        run_dir = self.repository.resolve_run(run_id)
        manifest = load_run_manifest(run_dir)
        commands = read_run_commands(run_dir)
        desired = (
            RunStatus.PAUSED
            if manifest.status is RunStatus.PAUSED
            else RunStatus.RUNNING
        )
        delay_ms = 0
        cancel_requested = False
        for item in commands:
            if item.command is RunCommandType.PAUSE:
                desired = RunStatus.PAUSED
            elif item.command is RunCommandType.RESUME:
                desired = RunStatus.RUNNING
            elif item.command is RunCommandType.STEP:
                desired = RunStatus.PAUSED
            elif item.command is RunCommandType.SET_DELAY:
                delay_ms = int(item.delay_ms or 0)
            elif item.command is RunCommandType.CANCEL:
                cancel_requested = True

        available: list[str] = []
        if not manifest.status.terminal and not cancel_requested:
            if desired is RunStatus.PAUSED:
                available.extend([RunCommandType.RESUME.value, RunCommandType.STEP.value])
            else:
                available.append(RunCommandType.PAUSE.value)
            available.extend([RunCommandType.SET_DELAY.value, RunCommandType.CANCEL.value])

        if manifest.status.terminal:
            requested_status = manifest.status
        elif cancel_requested:
            requested_status = RunStatus.CANCELLED
        else:
            requested_status = desired
        return {
            "run_id": manifest.run_id,
            "status": manifest.status.value,
            "requested_status": requested_status.value,
            "delay_ms": delay_ms,
            "last_command_seq": commands[-1].seq if commands else 0,
            "available_commands": available,
        }

    def issue(
        self,
        run_id: str,
        command: RunCommandType,
        *,
        delay_ms: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            before = self.state(run_id)
            if command.value not in before["available_commands"]:
                raise RunControlConflict(
                    f"команда {command.value!r} недоступна при status={before['status']!r} "
                    f"и requested_status={before['requested_status']!r}"
                )
            if command is RunCommandType.SET_DELAY:
                if isinstance(delay_ms, bool) or not isinstance(delay_ms, int):
                    raise RunControlValidationError(
                        "set_delay требует целое поле delay_ms"
                    )
            elif delay_ms is not None:
                raise RunControlValidationError(
                    "delay_ms допустим только для команды set_delay"
                )

            run_dir = self.repository.resolve_run(run_id)
            item = append_run_command(run_dir, command, delay_ms=delay_ms)
            if command is RunCommandType.CANCEL and before["status"] == RunStatus.QUEUED.value:
                cancel = getattr(self.scheduler, "cancel", None)
                if cancel is not None:
                    cancel(Path(run_dir))
                cancel_prepared_run(run_dir)

            return {"command": item.to_dict(), "control": self.state(run_id)}
