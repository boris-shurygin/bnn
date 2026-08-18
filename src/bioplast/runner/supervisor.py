"""Process supervision for observable runs and long-lived debug sessions."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
import uuid
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from bioplast.runner.contracts import (
    RunStatus,
    load_run_manifest,
    utc_offset_iso,
    write_run_manifest,
)
from bioplast.runner.lifecycle import (
    claim_worker,
    clear_worker_lease,
    heartbeat_worker,
    load_activity,
    load_recovery_state,
    load_worker_lease,
    recovery_availability,
    touch_activity,
    worker_lease_stale,
)


def _is_debug_run(run_dir: Path) -> bool:
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    debug = config.get("debug")
    return isinstance(debug, dict) and debug.get("protocol") == "model_debug_v1"


class _WorkerHeartbeat:
    def __init__(self, run_dir: Path, lease: dict[str, Any], interval: float) -> None:
        self.run_dir = run_dir
        self.lease = lease
        self.interval = interval
        self.stop_event = Event()
        self.thread = Thread(target=self._run, name=f"heartbeat-{run_dir.name}", daemon=True)

    def __enter__(self) -> _WorkerHeartbeat:
        self.lease = heartbeat_worker(self.run_dir, self.lease)
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.interval * 2))

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            self.lease = heartbeat_worker(self.run_dir, self.lease)


def _supervised_worker(
    run_dir_raw: str,
    supervisor_id: str,
    attempt: int,
    pool_kind: str,
    heartbeat_sec: float,
    resume: bool,
    activity_timeout_sec: float | None,
) -> dict[str, Any]:
    """Top-level spawn target. PyTorch and experiment imports stay in child."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from bioplast.runner.run import run_prepared

    run_dir = Path(run_dir_raw).resolve()
    if activity_timeout_sec is not None:
        # Spawn/import PyTorch can take longer than a deliberately short test
        # lease. Activity starts when the claimed worker is actually alive.
        touch_activity(run_dir, timeout_sec=activity_timeout_sec)
    lease = load_worker_lease(run_dir)
    if (
        lease is None
        or lease.get("supervisor_id") != supervisor_id
        or int(lease.get("attempt", -1)) != attempt
    ):
        raise RuntimeError("worker не получил действительную lease supervisor")
    with _WorkerHeartbeat(run_dir, lease, heartbeat_sec):
        run_prepared(
            run_dir,
            resume=resume,
            attempt=attempt,
            pool_kind=pool_kind,
            debug_inactive_timeout_sec=activity_timeout_sec,
        )
    return {"run_dir": str(run_dir), "status": load_run_manifest(run_dir).status.value}


def _debug_worker_entry(*args: Any) -> None:
    _supervised_worker(*args)


class RunSupervisor:
    """Owns separate main futures and disposable per-session debug processes."""

    def __init__(
        self,
        runs_dir: Path | str,
        *,
        main_workers: int = 1,
        debug_workers: int = 2,
        debug_inactive_timeout_sec: float = 1800,
        heartbeat_sec: float = 5,
        stale_sec: float = 30,
        shutdown_grace_sec: float = 15,
        monitor_interval: float = 0.2,
    ) -> None:
        if min(main_workers, debug_workers) < 1:
            raise ValueError("число main/debug workers должно быть положительным")
        if min(debug_inactive_timeout_sec, heartbeat_sec, stale_sec, shutdown_grace_sec) <= 0:
            raise ValueError("lifecycle timeouts должны быть положительными")
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.main_workers = int(main_workers)
        self.debug_workers = int(debug_workers)
        self.debug_inactive_timeout_sec = float(debug_inactive_timeout_sec)
        self.heartbeat_sec = float(heartbeat_sec)
        self.stale_sec = float(stale_sec)
        self.shutdown_grace_sec = float(shutdown_grace_sec)
        self.monitor_interval = float(monitor_interval)
        self.instance_id = uuid.uuid4().hex
        self._main_executor: ProcessPoolExecutor | None = None
        self._main_futures: dict[Future[dict[str, Any]], tuple[Path, int]] = {}
        self._debug_processes: dict[Path, tuple[multiprocessing.Process, int]] = {}
        self._debug_queue: deque[tuple[Path, bool]] = deque()
        self._queued: set[Path] = set()
        self._lock = Lock()
        self._stop = Event()
        self._monitor: Thread | None = None
        self._accepting = True

    def start(self) -> None:
        with self._lock:
            if self._monitor is not None:
                return
            self._reconcile_locked()
            self._monitor = Thread(target=self._monitor_loop, name="run-supervisor", daemon=True)
            self._monitor.start()

    def submit(self, run_dir: Path | str) -> None:
        path = self._resolve_run(run_dir)
        with self._lock:
            self._ensure_accepting()
            self._ensure_monitor_locked()
            if path in self._queued or path in self._debug_processes or any(
                item == path for item, _attempt in self._main_futures.values()
            ):
                return
            if _is_debug_run(path):
                touch_activity(path, timeout_sec=self.debug_inactive_timeout_sec)
                self._debug_queue.append((path, False))
                self._queued.add(path)
                self._dispatch_debug_locked()
            else:
                self._submit_main_locked(path, resume=False)

    def touch(self, run_dir: Path | str) -> dict[str, Any] | None:
        path = self._resolve_run(run_dir)
        if not _is_debug_run(path) or load_run_manifest(path).status.terminal:
            return None
        return touch_activity(path, timeout_sec=self.debug_inactive_timeout_sec)

    def wake(self, run_dir: Path | str) -> bool:
        path = self._resolve_run(run_dir)
        with self._lock:
            self._ensure_accepting()
            self._ensure_monitor_locked()
            manifest = load_run_manifest(path)
            if manifest.status in {RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.QUEUED}:
                touch_activity(path, timeout_sec=self.debug_inactive_timeout_sec)
                return False
            if manifest.status not in {RunStatus.SUSPENDED, RunStatus.INTERRUPTED}:
                return False
            availability = recovery_availability(path)
            if not availability.available:
                raise RuntimeError(f"resume недоступен: {availability.reason}")
            touch_activity(path, timeout_sec=self.debug_inactive_timeout_sec)
            if path not in self._queued and path not in self._debug_processes:
                self._debug_queue.append((path, True))
                self._queued.add(path)
                self._dispatch_debug_locked()
            return True

    def cancel(self, run_dir: Path | str) -> bool:
        path = self._resolve_run(run_dir)
        with self._lock:
            removed = False
            kept: deque[tuple[Path, bool]] = deque()
            while self._debug_queue:
                item = self._debug_queue.popleft()
                if item[0] == path:
                    removed = True
                    self._queued.discard(path)
                else:
                    kept.append(item)
            self._debug_queue = kept
            if removed:
                return True
            future = next(
                (future for future, (candidate, _attempt) in self._main_futures.items() if candidate == path),
                None,
            )
            return bool(future is not None and future.cancel())

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._main_futures) + len(self._debug_processes) + len(self._debug_queue)

    def describe(self, run_dir: Path | str) -> dict[str, Any]:
        path = self._resolve_run(run_dir)
        availability = recovery_availability(path)
        return {
            "pool": "debug" if _is_debug_run(path) else "main",
            "worker": load_worker_lease(path),
            "activity": load_activity(path),
            "recovery": availability.state,
            "resume_available": availability.available,
            "resume_unavailable_reason": availability.reason,
        }

    def shutdown(self) -> None:
        with self._lock:
            if not self._accepting and self._stop.is_set():
                return
            self._accepting = False
            for path in self._debug_processes:
                touch_activity(path, timeout_sec=0.001)
            executor = self._main_executor
            self._main_executor = None
        deadline = time.monotonic() + self.shutdown_grace_sec
        while time.monotonic() < deadline:
            with self._lock:
                self._reap_debug_locked()
                if not self._debug_processes:
                    break
            time.sleep(min(self.monitor_interval, 0.05))
        with self._lock:
            remaining = list(self._debug_processes.items())
        for path, (process, attempt) in remaining:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            self._mark_interrupted(path, "supervisor shutdown timeout")
            clear_worker_lease(path, attempt=attempt)
        self._stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=2)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _ensure_accepting(self) -> None:
        if not self._accepting:
            raise RuntimeError("RunSupervisor остановлен")

    def _resolve_run(self, run_dir: Path | str) -> Path:
        path = Path(run_dir).resolve()
        if path.parent != self.runs_dir or not path.is_dir():
            raise ValueError("RunSupervisor принимает только непосредственный каталог runs/<id>")
        return path

    def _ensure_monitor_locked(self) -> None:
        if self._monitor is None:
            self._monitor = Thread(target=self._monitor_loop, name="run-supervisor", daemon=True)
            self._monitor.start()

    def _next_attempt(self, path: Path) -> int:
        values = [0]
        lease = load_worker_lease(path)
        recovery = load_recovery_state(path, required=False)
        for value in (lease, recovery):
            if value is not None:
                try:
                    values.append(int(value.get("attempt", 0)))
                except (TypeError, ValueError):
                    pass
        return max(values) + 1

    def _submit_main_locked(self, path: Path, *, resume: bool) -> None:
        if self._main_executor is None:
            self._main_executor = ProcessPoolExecutor(max_workers=self.main_workers)
        attempt = self._next_attempt(path)
        claim_worker(
            path,
            supervisor_id=self.instance_id,
            attempt=attempt,
            pool_kind="main",
            exclusive=True,
        )
        try:
            future = self._main_executor.submit(
                _supervised_worker,
                str(path),
                self.instance_id,
                attempt,
                "main",
                self.heartbeat_sec,
                resume,
                None,
            )
        except Exception:
            clear_worker_lease(path, attempt=attempt)
            raise
        self._main_futures[future] = (path, attempt)
        future.add_done_callback(self._main_done)

    def _main_done(self, future: Future[dict[str, Any]]) -> None:
        with self._lock:
            item = self._main_futures.pop(future, None)
        if item is None:
            return
        path, attempt = item
        try:
            error = None if future.cancelled() else future.exception()
        except BaseException as exc:
            error = exc
        if error is not None:
            self._mark_interrupted(path, f"main worker process crashed: {error!r}")
        clear_worker_lease(path, attempt=attempt)

    def _dispatch_debug_locked(self) -> None:
        if not self._accepting:
            return
        context = multiprocessing.get_context("spawn")
        while self._debug_queue and len(self._debug_processes) < self.debug_workers:
            path, resume = self._debug_queue.popleft()
            self._queued.discard(path)
            manifest = load_run_manifest(path)
            expected = {RunStatus.SUSPENDED, RunStatus.INTERRUPTED} if resume else {RunStatus.QUEUED}
            if manifest.status not in expected:
                continue
            attempt = self._next_attempt(path)
            claim_worker(
                path,
                supervisor_id=self.instance_id,
                attempt=attempt,
                pool_kind="debug",
                exclusive=True,
            )
            process = context.Process(
                target=_debug_worker_entry,
                args=(
                    str(path), self.instance_id, attempt, "debug",
                    self.heartbeat_sec, resume, self.debug_inactive_timeout_sec,
                ),
                name=f"bioplast-debug-{path.name[-24:]}",
            )
            try:
                process.start()
            except Exception:
                clear_worker_lease(path, attempt=attempt)
                raise
            self._debug_processes[path] = (process, attempt)

    def _reap_debug_locked(self) -> None:
        for path, (process, attempt) in list(self._debug_processes.items()):
            if process.is_alive():
                continue
            process.join(timeout=0)
            self._debug_processes.pop(path, None)
            try:
                status = load_run_manifest(path).status
            except Exception:
                status = None
            if status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.PAUSED}:
                self._mark_interrupted(
                    path,
                    f"debug worker exited unexpectedly with code {process.exitcode}",
                )
            clear_worker_lease(path, attempt=attempt)
        self._dispatch_debug_locked()

    def _monitor_loop(self) -> None:
        while not self._stop.wait(self.monitor_interval):
            with self._lock:
                self._reap_debug_locked()

    def _reconcile_locked(self) -> None:
        for path in sorted(self.runs_dir.iterdir() if self.runs_dir.exists() else ()):
            if not path.is_dir() or not (path / "run.json").is_file():
                continue
            try:
                manifest = load_run_manifest(path)
            except Exception:
                continue
            if manifest.status is RunStatus.QUEUED:
                lease = load_worker_lease(path)
                if lease is not None and not worker_lease_stale(path, stale_sec=self.stale_sec):
                    continue
                if lease is not None:
                    clear_worker_lease(path)
                if _is_debug_run(path):
                    touch_activity(path, timeout_sec=self.debug_inactive_timeout_sec)
                    self._debug_queue.append((path, False))
                    self._queued.add(path)
                else:
                    self._submit_main_locked(path, resume=False)
            elif manifest.status in {RunStatus.RUNNING, RunStatus.PAUSED}:
                stale = worker_lease_stale(path, stale_sec=self.stale_sec)
                if stale:
                    self._mark_interrupted(path, "stale worker lease при старте supervisor")
                    clear_worker_lease(path)
        self._dispatch_debug_locked()

    @staticmethod
    def _mark_interrupted(path: Path, reason: str) -> None:
        try:
            manifest = load_run_manifest(path)
        except Exception:
            return
        if manifest.status.terminal or manifest.status is RunStatus.SUSPENDED:
            return
        write_run_manifest(
            path,
            replace(manifest, status=RunStatus.INTERRUPTED, updated_at=utc_offset_iso()),
        )
        interruption = {
            "schema_version": 1,
            "kind": "interruption",
            "run_id": path.name,
            "occurred_at": utc_offset_iso(),
            "reason": reason,
        }
        temporary = path / "interruption.json.tmp"
        temporary.write_text(
            json.dumps(interruption, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path / "interruption.json")
