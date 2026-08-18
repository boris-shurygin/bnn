"""API-сервис типизированных команд для активного прогона."""

from __future__ import annotations

import math
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
    recovery_availability,
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
        config = self.repository.get_run(run_id)["config"]
        debug = config.get("debug")
        if not isinstance(debug, dict):
            debug = None
        accepts_input = bool(debug and debug.get("accepts_input")) or (
            config.get("experiment") == "xor_interactive"
        )
        desired = (
            RunStatus.PAUSED
            if manifest.status is RunStatus.PAUSED
            else RunStatus.RUNNING
        )
        recovery = recovery_availability(run_dir)
        recovery_seq = int((recovery.state or {}).get("last_command_seq", 0))
        pending_wake = False
        delay_ms = 0
        cancel_requested = False
        input_seq = 0
        input_values: list[float] | None = None
        for item in commands:
            if item.command is RunCommandType.PAUSE:
                desired = RunStatus.PAUSED
            elif item.command is RunCommandType.RESUME:
                desired = RunStatus.RUNNING
            elif item.command is RunCommandType.STEP:
                desired = RunStatus.PAUSED
            elif item.command is RunCommandType.SET_DELAY:
                delay_ms = int(item.delay_ms or 0)
            elif item.command is RunCommandType.SET_INPUT:
                input_seq = item.seq
                input_values = list(item.input_values or ())
                if item.seq > recovery_seq:
                    pending_wake = True
            elif item.command is RunCommandType.CANCEL:
                cancel_requested = True
            if item.seq > recovery_seq and item.command in {
                RunCommandType.RESUME,
                RunCommandType.STEP,
            }:
                pending_wake = True

        available: list[str] = []
        if not manifest.status.terminal and not cancel_requested:
            if manifest.status in {RunStatus.SUSPENDED, RunStatus.INTERRUPTED}:
                if recovery.available:
                    available.append(RunCommandType.RESUME.value)
                    if debug and debug.get("supports_step"):
                        available.append(RunCommandType.STEP.value)
            elif desired is RunStatus.PAUSED:
                available.extend([RunCommandType.RESUME.value, RunCommandType.STEP.value])
            else:
                available.append(RunCommandType.PAUSE.value)
            available.append(RunCommandType.SET_DELAY.value)
            if accepts_input and (
                manifest.status not in {RunStatus.SUSPENDED, RunStatus.INTERRUPTED}
                or recovery.available
            ):
                available.append(RunCommandType.SET_INPUT.value)
            available.append(RunCommandType.CANCEL.value)

        if manifest.status.terminal:
            requested_status = manifest.status
        elif cancel_requested:
            requested_status = RunStatus.CANCELLED
        elif manifest.status in {RunStatus.SUSPENDED, RunStatus.INTERRUPTED} and not pending_wake:
            requested_status = manifest.status
        else:
            requested_status = desired
        return {
            "run_id": manifest.run_id,
            "status": manifest.status.value,
            "requested_status": requested_status.value,
            "delay_ms": delay_ms,
            "last_command_seq": commands[-1].seq if commands else 0,
            "debug": debug,
            "accepts_input": accepts_input,
            "input_seq": input_seq,
            "input_values": input_values,
            "available_commands": available,
            "lifecycle": (
                self.scheduler.describe(run_dir)
                if hasattr(self.scheduler, "describe")
                else {
                    "pool": "debug" if debug else "main",
                    "worker": None,
                    "activity": None,
                    "recovery": recovery.state,
                    "resume_available": recovery.available,
                    "resume_unavailable_reason": recovery.reason,
                }
            ),
        }

    def issue(
        self,
        run_id: str,
        command: RunCommandType,
        *,
        delay_ms: int | None = None,
        input_values: list[float] | tuple[float, ...] | None = None,
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
                if input_values is not None:
                    raise RunControlValidationError(
                        "input_values допустимы только для команды set_input"
                    )
            elif command is RunCommandType.SET_INPUT:
                if delay_ms is not None:
                    raise RunControlValidationError(
                        "delay_ms допустим только для команды set_delay"
                    )
                debug = before.get("debug")
                expected_input_size = (
                    debug.get("input_size") if isinstance(debug, dict) else 2
                )
                if (
                    isinstance(expected_input_size, bool)
                    or not isinstance(expected_input_size, int)
                    or not 1 <= expected_input_size <= 4096
                ):
                    raise RunControlValidationError(
                        "debug capability input_size должен быть целым от 1 до 4096"
                    )
                if (
                    not isinstance(input_values, (list, tuple))
                    or len(input_values) != expected_input_size
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in input_values
                    )
                ):
                    raise RunControlValidationError(
                        f"debug-сессия требует {expected_input_size} конечных input_values"
                    )
            elif delay_ms is not None:
                raise RunControlValidationError(
                    "delay_ms допустим только для команды set_delay"
                )
            elif input_values is not None:
                raise RunControlValidationError(
                    "input_values допустимы только для команды set_input"
                )

            run_dir = self.repository.resolve_run(run_id)
            touch = getattr(self.scheduler, "touch", None)
            if touch is not None:
                touch(run_dir)
            item = append_run_command(
                run_dir,
                command,
                delay_ms=delay_ms,
                input_values=(
                    tuple(float(value) for value in input_values)
                    if input_values is not None
                    else None
                ),
            )
            if command is RunCommandType.CANCEL and before["status"] in {
                RunStatus.QUEUED.value,
                RunStatus.SUSPENDED.value,
                RunStatus.INTERRUPTED.value,
            }:
                cancel = getattr(self.scheduler, "cancel", None)
                if cancel is not None:
                    cancel(Path(run_dir))
                cancel_prepared_run(run_dir)
            elif command in {
                RunCommandType.RESUME,
                RunCommandType.STEP,
                RunCommandType.SET_INPUT,
            } and before["status"] in {
                RunStatus.SUSPENDED.value,
                RunStatus.INTERRUPTED.value,
            }:
                wake = getattr(self.scheduler, "wake", None)
                if wake is None:
                    raise RunControlConflict("scheduler не поддерживает пробуждение сессии")
                try:
                    wake(run_dir)
                except RuntimeError as exc:
                    raise RunControlConflict(str(exc)) from exc

            return {"command": item.to_dict(), "control": self.state(run_id)}
