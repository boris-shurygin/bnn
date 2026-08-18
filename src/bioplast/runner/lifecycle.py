"""Durable worker leases, activity leases and recovery generations.

Files under ``runs/<id>`` are the source of truth.  The supervisor may vanish;
another instance can validate these files and continue the same debug session.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import get_ident
from typing import Any, Mapping

from bioplast.runner.contracts import ContractError, utc_offset_iso

LIFECYCLE_VERSION = 1
RECOVERY_ADAPTER_XOR = "xor_interactive_v1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{get_ident()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for attempt in range(20):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"не удалось прочитать {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"ожидался JSON-объект: {path}")
    return value


def touch_activity(
    run_dir: Path | str,
    *,
    timeout_sec: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    if timeout_sec <= 0:
        raise ValueError("timeout_sec activity lease должен быть положительным")
    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.astimezone()
    value = {
        "schema_version": LIFECYCLE_VERSION,
        "kind": "activity_lease",
        "run_id": Path(run_dir).resolve().name,
        "last_activity_at": utc_offset_iso(current),
        "expires_at": utc_offset_iso(current + timedelta(seconds=float(timeout_sec))),
        "timeout_sec": float(timeout_sec),
    }
    _atomic_json(Path(run_dir).resolve() / "activity.json", value)
    return value


def load_activity(run_dir: Path | str) -> dict[str, Any] | None:
    path = Path(run_dir).resolve() / "activity.json"
    if not path.is_file():
        return None
    value = _read_object(path)
    if value.get("schema_version") != LIFECYCLE_VERSION or value.get("kind") != "activity_lease":
        raise ContractError("неподдерживаемый activity lease")
    if value.get("run_id") != path.parent.name:
        raise ContractError("activity lease принадлежит другому запуску")
    return value


def activity_expired(run_dir: Path | str, *, now: datetime | None = None) -> bool:
    lease = load_activity(run_dir)
    if lease is None:
        return False
    try:
        expires = datetime.fromisoformat(str(lease["expires_at"]))
    except (KeyError, ValueError) as exc:
        raise ContractError("activity lease не содержит корректный expires_at") from exc
    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.astimezone()
    return current >= expires


def claim_worker(
    run_dir: Path | str,
    *,
    supervisor_id: str,
    attempt: int,
    pool_kind: str,
    pid: int | None = None,
    exclusive: bool = False,
) -> dict[str, Any]:
    if pool_kind not in {"main", "debug"}:
        raise ValueError("pool_kind должен быть main или debug")
    timestamp = utc_offset_iso()
    value = {
        "schema_version": LIFECYCLE_VERSION,
        "kind": "worker_lease",
        "run_id": Path(run_dir).resolve().name,
        "supervisor_id": supervisor_id,
        "attempt": int(attempt),
        "pool_kind": pool_kind,
        "pid": pid,
        "claimed_at": timestamp,
        "heartbeat_at": timestamp,
    }
    path = Path(run_dir).resolve() / "worker.json"
    if exclusive:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise ContractError("запуск уже имеет worker lease") from exc
    else:
        _atomic_json(path, value)
    return value


def heartbeat_worker(run_dir: Path | str, lease: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(lease)
    value["pid"] = os.getpid()
    value["heartbeat_at"] = utc_offset_iso()
    _atomic_json(Path(run_dir).resolve() / "worker.json", value)
    return value


def load_worker_lease(run_dir: Path | str) -> dict[str, Any] | None:
    path = Path(run_dir).resolve() / "worker.json"
    if not path.is_file():
        return None
    value = _read_object(path)
    if value.get("schema_version") != LIFECYCLE_VERSION or value.get("kind") != "worker_lease":
        raise ContractError("неподдерживаемый worker lease")
    if value.get("run_id") != path.parent.name:
        raise ContractError("worker lease принадлежит другому запуску")
    return value


def clear_worker_lease(run_dir: Path | str, *, attempt: int | None = None) -> None:
    path = Path(run_dir).resolve() / "worker.json"
    if not path.exists():
        return
    if attempt is not None:
        try:
            if int(load_worker_lease(run_dir)["attempt"]) != int(attempt):  # type: ignore[index]
                return
        except (ContractError, KeyError, TypeError, ValueError):
            return
    path.unlink(missing_ok=True)


def worker_lease_stale(
    run_dir: Path | str,
    *,
    stale_sec: float,
    now: datetime | None = None,
) -> bool:
    lease = load_worker_lease(run_dir)
    if lease is None:
        return True
    try:
        heartbeat = datetime.fromisoformat(str(lease["heartbeat_at"]))
    except (KeyError, ValueError) as exc:
        raise ContractError("worker lease не содержит корректный heartbeat_at") from exc
    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.astimezone()
    return (current - heartbeat).total_seconds() > stale_sec


def write_recovery(
    run_dir: Path | str,
    payload: Mapping[str, Any],
    *,
    adapter: str,
    attempt: int,
    cursor: str,
    command_seq: int,
    event_seq: int,
) -> dict[str, Any]:
    """Publish a binary payload first and ``state.json`` last."""
    import torch

    root = Path(run_dir).resolve() / "recovery"
    root.mkdir(parents=True, exist_ok=True)
    previous = load_recovery_state(run_dir, required=False)
    generation = int(previous.get("generation", 0)) + 1 if previous else 1
    checkpoint_name = f"checkpoint-{generation:08d}.pt"
    checkpoint = root / checkpoint_name
    temporary = root / f"{checkpoint_name}.tmp"
    torch.save(dict(payload), temporary)
    checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(checkpoint)
    state = {
        "schema_version": LIFECYCLE_VERSION,
        "kind": "recovery_state",
        "run_id": Path(run_dir).resolve().name,
        "adapter": adapter,
        "generation": generation,
        "attempt": int(attempt),
        "safe_point_cursor": cursor,
        "last_command_seq": int(command_seq),
        "last_event_seq": int(event_seq),
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checksum,
        "written_at": utc_offset_iso(),
    }
    _atomic_json(root / "state.json", state)
    return state


def load_recovery_state(
    run_dir: Path | str, *, required: bool = True
) -> dict[str, Any] | None:
    path = Path(run_dir).resolve() / "recovery" / "state.json"
    if not path.is_file():
        if required:
            raise ContractError("recovery state отсутствует")
        return None
    value = _read_object(path)
    if value.get("schema_version") != LIFECYCLE_VERSION or value.get("kind") != "recovery_state":
        raise ContractError("неподдерживаемый recovery state")
    if value.get("run_id") != Path(run_dir).resolve().name:
        raise ContractError("recovery state принадлежит другому запуску")
    return value


def load_recovery(
    run_dir: Path | str,
    *,
    expected_adapter: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    state = load_recovery_state(run_dir)
    assert state is not None
    if expected_adapter is not None and state.get("adapter") != expected_adapter:
        raise ContractError(
            f"recovery adapter {state.get('adapter')!r} несовместим с {expected_adapter!r}"
        )
    checkpoint = Path(run_dir).resolve() / "recovery" / str(state.get("checkpoint", ""))
    if checkpoint.parent != Path(run_dir).resolve() / "recovery" or not checkpoint.is_file():
        raise ContractError("recovery checkpoint отсутствует или имеет небезопасный путь")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checksum != state.get("checkpoint_sha256"):
        raise ContractError("checksum recovery checkpoint не совпадает")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ContractError(f"не удалось загрузить recovery checkpoint: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("recovery checkpoint должен содержать mapping")
    return state, payload


@dataclass(frozen=True)
class RecoveryAvailability:
    available: bool
    reason: str | None
    state: dict[str, Any] | None


def recovery_availability(run_dir: Path | str) -> RecoveryAvailability:
    try:
        state, _payload = load_recovery(run_dir)
    except ContractError as exc:
        try:
            state = load_recovery_state(run_dir, required=False)
        except ContractError:
            state = None
        return RecoveryAvailability(False, str(exc), state)
    return RecoveryAvailability(True, None, state)
