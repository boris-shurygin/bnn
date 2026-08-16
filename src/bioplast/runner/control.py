"""Типизированные команды и кооперативное управление worker-процессом.

API дописывает команды в ``commands.jsonl``. Worker перечитывает журнал только
между атомарными операциями эксперимента, поэтому пауза и отмена не оставляют
наполовину обновлённый слой. Живые Python-объекты через границу процессов не
передаются.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping

from bioplast.runner.contracts import (
    COMMANDS_FILE,
    CONTRACT_VERSION,
    ContractError,
    RunStatus,
    load_run_manifest,
    utc_offset_iso,
    write_run_manifest,
)

MAX_DELAY_MS = 60_000
_APPEND_LOCK = Lock()


class RunCommandType(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"
    SET_DELAY = "set_delay"
    CANCEL = "cancel"


@dataclass(frozen=True)
class RunCommand:
    run_id: str
    seq: int
    command: RunCommandType
    issued_at: str
    delay_ms: int | None = None
    schema_version: int = CONTRACT_VERSION
    kind: str = "run_command"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION or self.kind != "run_command":
            raise ContractError("неподдерживаемый заголовок команды запуска")
        if not self.run_id:
            raise ContractError("run_id команды не может быть пустым")
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 1:
            raise ContractError("seq команды должен быть положительным целым")
        if not self.issued_at:
            raise ContractError("issued_at команды не может быть пустым")
        if self.command is RunCommandType.SET_DELAY:
            if (
                isinstance(self.delay_ms, bool)
                or not isinstance(self.delay_ms, int)
                or not 0 <= self.delay_ms <= MAX_DELAY_MS
            ):
                raise ContractError(
                    f"delay_ms должен быть целым от 0 до {MAX_DELAY_MS}"
                )
        elif self.delay_ms is not None:
            raise ContractError("delay_ms допустим только для команды set_delay")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "seq": self.seq,
            "issued_at": self.issued_at,
            "command": self.command.value,
            "delay_ms": self.delay_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunCommand:
        try:
            command = RunCommandType(str(value["command"]))
            seq = value["seq"]
        except (KeyError, ValueError) as exc:
            raise ContractError(f"неизвестная команда запуска: {value.get('command')!r}") from exc
        return cls(
            run_id=str(value.get("run_id", "")),
            seq=seq,
            command=command,
            issued_at=str(value.get("issued_at", "")),
            delay_ms=value.get("delay_ms"),
            schema_version=value.get("schema_version"),
            kind=value.get("kind"),
        )


class RunCancelled(RuntimeError):
    """Кооперативная отмена, замеченная в безопасной точке эксперимента."""


def read_run_commands(run_dir: Path | str, *, after_seq: int = 0) -> list[RunCommand]:
    """Прочитать только полные строки журнала и проверить монотонность seq."""
    run_dir = Path(run_dir).resolve()
    manifest = load_run_manifest(run_dir)
    relative = manifest.artifacts.get("commands", COMMANDS_FILE)
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir) or path == run_dir:
        raise ContractError("путь журнала команд выходит за пределы прогона")
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines = lines[:-1]

    commands: list[RunCommand] = []
    expected_seq = 1
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ContractError(f"пустая строка {line_number} в {path.name}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"некорректная JSON-команда в строке {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ContractError(f"команда в строке {line_number} должна быть объектом")
        command = RunCommand.from_dict(value)
        if command.run_id != manifest.run_id:
            raise ContractError(
                f"run_id команды {command.run_id!r} не совпадает с {manifest.run_id!r}"
            )
        if command.seq != expected_seq:
            raise ContractError(
                f"ожидался seq команды {expected_seq}, получен {command.seq}"
            )
        expected_seq += 1
        if command.seq > after_seq:
            commands.append(command)
    return commands


def append_run_command(
    run_dir: Path | str,
    command: RunCommandType,
    *,
    delay_ms: int | None = None,
) -> RunCommand:
    """Атомарно для одного API-процесса дописать следующую команду."""
    run_dir = Path(run_dir).resolve()
    with _APPEND_LOCK:
        manifest = load_run_manifest(run_dir)
        previous = read_run_commands(run_dir)
        item = RunCommand(
            run_id=manifest.run_id,
            seq=previous[-1].seq + 1 if previous else 1,
            command=command,
            issued_at=utc_offset_iso(),
            delay_ms=delay_ms,
        )
        relative = manifest.artifacts.get("commands", COMMANDS_FILE)
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir) or path == run_dir:
            raise ContractError("путь журнала команд выходит за пределы прогона")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(item.to_dict(), ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return item


class CooperativeRunControl:
    """Применяет команды в явно отмеченных безопасных точках эксперимента."""

    def __init__(
        self,
        run_dir: Path | str,
        *,
        poll_interval: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        logger: Any | None = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval должен быть положительным")
        self.run_dir = Path(run_dir).resolve()
        self.poll_interval = float(poll_interval)
        self._sleep = sleep
        self._monotonic = monotonic
        self._logger = logger
        self._last_seq = 0
        manifest = load_run_manifest(self.run_dir)
        self._mode = (
            RunStatus.PAUSED
            if manifest.status is RunStatus.PAUSED
            else RunStatus.RUNNING
        )
        self._step_budget = 0
        self._delay_ms = 0
        self._cancel_requested = False

    @property
    def delay_ms(self) -> int:
        return self._delay_ms

    @property
    def last_seq(self) -> int:
        return self._last_seq

    def checkpoint(
        self,
        *,
        step: int | None = None,
        phase: str | None = None,
        apply_delay: bool = True,
    ) -> None:
        """Дождаться разрешения выполнить одну следующую атомарную операцию."""
        while True:
            self._apply_new_commands()
            if self._cancel_requested:
                raise RunCancelled("прогон отменён пользователем")

            if self._mode is RunStatus.RUNNING:
                self._transition(RunStatus.RUNNING)
                if not apply_delay or self._apply_delay(pause_interrupts=True):
                    return
                continue

            self._transition(RunStatus.PAUSED)
            if self._step_budget > 0:
                self._step_budget -= 1
                if self._logger is not None:
                    location = ", ".join(
                        value
                        for value in (
                            f"phase={phase}" if phase else None,
                            f"step={step}" if step is not None else None,
                        )
                        if value is not None
                    )
                    self._logger.info(
                        "пошаговая команда: разрешена одна атомарная операция%s",
                        f" ({location})" if location else "",
                    )
                if apply_delay:
                    self._apply_delay(pause_interrupts=False)
                return
            self._sleep(self.poll_interval)

    def _apply_new_commands(self) -> None:
        for command in read_run_commands(self.run_dir, after_seq=self._last_seq):
            self._last_seq = command.seq
            if command.command is RunCommandType.PAUSE:
                self._mode = RunStatus.PAUSED
                self._step_budget = 0
            elif command.command is RunCommandType.RESUME:
                self._mode = RunStatus.RUNNING
                self._step_budget = 0
            elif command.command is RunCommandType.STEP:
                self._mode = RunStatus.PAUSED
                self._step_budget += 1
            elif command.command is RunCommandType.SET_DELAY:
                self._delay_ms = int(command.delay_ms or 0)
            elif command.command is RunCommandType.CANCEL:
                self._cancel_requested = True

    def _apply_delay(self, *, pause_interrupts: bool) -> bool:
        if not self._delay_ms:
            return True
        deadline = self._monotonic() + self._delay_ms / 1000
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return True
            self._sleep(min(self.poll_interval, remaining))
            self._apply_new_commands()
            if self._cancel_requested:
                raise RunCancelled("прогон отменён пользователем")
            if pause_interrupts and self._mode is RunStatus.PAUSED:
                return False

    def _transition(self, status: RunStatus) -> None:
        manifest = load_run_manifest(self.run_dir)
        if manifest.status.terminal or manifest.status is status:
            return
        if status not in {RunStatus.RUNNING, RunStatus.PAUSED}:
            raise ContractError(f"недопустимый кооперативный статус: {status.value}")
        write_run_manifest(
            self.run_dir,
            replace(manifest, status=status, updated_at=utc_offset_iso()),
        )
        if self._logger is not None:
            self._logger.info("управление прогоном: статус %s", status.value)
