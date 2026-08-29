"""V.10: separate debug workers, hibernation and durable resume."""

from __future__ import annotations

import json
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from bioplast.runner import (
    ContractError,
    RunStatus,
    RunSupervisor,
    iter_events,
    load_recovery,
    load_run_manifest,
    prepare_run,
    run_config,
    write_run_manifest,
)
from bioplast.runner.lifecycle import claim_worker, recovery_availability
from bioplast.viz.api import create_app


def _wait_until(predicate, message: str, *, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail(message)


def _source_xor(runs_dir):
    return run_config(
        {
            "session": "test",
            "dataset": "xor",
            "model": "mlp-2-3-1",
            "experiment": "xor_backprop",
            "device": "cpu",
            "seed": 0,
            "hidden": 3,
            "steps": 1,
            "log_every": 1,
        },
        runs_dir=runs_dir,
    )


def _supervisor(runs_dir, *, timeout: float = 1.5) -> RunSupervisor:
    return RunSupervisor(
        runs_dir,
        main_workers=1,
        debug_workers=1,
        debug_inactive_timeout_sec=timeout,
        heartbeat_sec=0.05,
        stale_sec=0.5,
        shutdown_grace_sec=2,
        monitor_interval=0.02,
    )


def test_debug_session_hibernates_and_resumes_same_run_without_duplicate_events(tmp_path):
    runs_dir = tmp_path / "runs"
    source = _source_xor(runs_dir)
    supervisor = _supervisor(runs_dir)

    with TestClient(create_app(runs_dir, scheduler=supervisor)) as client:
        created = client.post(f"/api/runs/{source.name}/debug").json()
        run_id = created["run_id"]
        child = runs_dir / run_id

        _wait_until(
            lambda: load_run_manifest(child).status is RunStatus.SUSPENDED,
            "debug-сессия не освободила worker после inactivity timeout",
        )
        first_state, first_payload = load_recovery(
            child, expected_adapter="xor_interactive_v1"
        )
        assert first_state["safe_point_cursor"] == "waiting_input"
        assert first_payload["session"]["event_seq"] == 0
        _wait_until(lambda: supervisor.pending == 0, "supervisor не освободил debug process")
        # Reproduce the restart window where suspended is already durable but
        # the exiting worker's fresh lease has not disappeared yet.
        claim_worker(
            child,
            supervisor_id="exiting-supervisor",
            attempt=1,
            pool_kind="debug",
            exclusive=True,
        )

        response = client.post(
            f"/api/runs/{run_id}/control",
            json={"command": "set_input", "input_values": [1, 0]},
        )
        assert response.status_code == 202
        _wait_until(
            lambda: len(list(iter_events(child))) == 3,
            "возобновлённый worker не завершил послойный forward",
        )
        events = list(iter_events(child))
        assert [event.seq for event in events] == [1, 2, 3]
        assert [event.phase for event in events] == [
            "input",
            "forward_hidden",
            "forward_output",
        ]
        _wait_until(
            lambda: load_recovery(child)[0]["last_event_seq"] == 3,
            "recovery не догнал опубликованный output event",
        )
        second_state, second_payload = load_recovery(
            child, expected_adapter="xor_interactive_v1"
        )
        assert second_state["generation"] > first_state["generation"]
        assert second_state["attempt"] == 2
        assert second_payload["session"]["completed_inputs"] == 1
        assert load_run_manifest(child).status in {RunStatus.RUNNING, RunStatus.PAUSED}

        assert client.post(
            f"/api/runs/{run_id}/control", json={"command": "cancel"}
        ).status_code == 202
        _wait_until(
            lambda: load_run_manifest(child).status is RunStatus.CANCELLED,
            "возобновлённая debug-сессия не завершилась",
        )


def test_waiting_debug_session_does_not_block_main_pool(tmp_path):
    runs_dir = tmp_path / "runs"
    source = _source_xor(runs_dir)
    supervisor = _supervisor(runs_dir, timeout=10)

    with TestClient(create_app(runs_dir, scheduler=supervisor)) as client:
        child_id = client.post(f"/api/runs/{source.name}/debug").json()["run_id"]
        child = runs_dir / child_id
        _wait_until(
            lambda: load_run_manifest(child).status is RunStatus.RUNNING,
            "debug worker не стартовал",
        )

        ordinary = prepare_run(
            {
                "session": "V.10",
                "dataset": "toy",
                "model": "selftest",
                "experiment": "_selftest",
                "device": "cpu",
                "seed": 0,
                "steps": 2,
            },
            runs_dir=runs_dir,
        )
        supervisor.submit(ordinary)
        _wait_until(
            lambda: load_run_manifest(ordinary).status is RunStatus.COMPLETED,
            "main pool заблокирован ожидающей debug-сессией",
        )
        assert load_run_manifest(child).status is RunStatus.RUNNING

        client.post(f"/api/runs/{child_id}/control", json={"command": "cancel"})
        _wait_until(
            lambda: load_run_manifest(child).status is RunStatus.CANCELLED,
            "debug worker не отменился",
        )


def test_corrupt_recovery_disables_resume_with_explicit_reason(tmp_path):
    runs_dir = tmp_path / "runs"
    source = _source_xor(runs_dir)
    supervisor = _supervisor(runs_dir)

    with TestClient(create_app(runs_dir, scheduler=supervisor)) as client:
        run_id = client.post(f"/api/runs/{source.name}/debug").json()["run_id"]
        child = runs_dir / run_id
        _wait_until(
            lambda: load_run_manifest(child).status is RunStatus.SUSPENDED,
            "debug-сессия не перешла в suspended",
        )
        recovery_state, _payload = load_recovery(child)
        checkpoint = child / "recovery" / recovery_state["checkpoint"]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")

        availability = recovery_availability(child)
        state = client.get(f"/api/runs/{run_id}/control").json()
        assert availability.available is False
        assert "checksum" in str(availability.reason)
        assert state["lifecycle"]["resume_available"] is False
        assert "resume" not in state["available_commands"]
        assert "set_input" not in state["available_commands"]
        failed_wake = client.post(
            f"/api/runs/{run_id}/control",
            json={"command": "set_input", "input_values": [0, 1]},
        )
        assert failed_wake.status_code == 409

        # Corruption is inspectable; cancellation remains possible without a worker.
        cancelled = client.post(
            f"/api/runs/{run_id}/control", json={"command": "cancel"}
        )
        assert cancelled.status_code == 202
        assert load_run_manifest(child).status is RunStatus.CANCELLED


def test_startup_reconciliation_marks_only_stale_worker_lease_interrupted(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = prepare_run(
        {
            "session": "V.10",
            "dataset": "xor",
            "model": "debug-fixture",
            "experiment": "xor_interactive",
            "device": "cpu",
            "seed": 0,
            "debug": {"protocol": "model_debug_v1"},
        },
        runs_dir=runs_dir,
    )
    manifest = load_run_manifest(run_dir)
    write_run_manifest(
        run_dir,
        replace(manifest, status=RunStatus.RUNNING, started_at=manifest.updated_at),
    )
    claim_worker(
        run_dir,
        supervisor_id="dead-supervisor",
        attempt=7,
        pool_kind="debug",
    )
    with pytest.raises(ContractError, match="worker lease"):
        claim_worker(
            run_dir,
            supervisor_id="competing-supervisor",
            attempt=8,
            pool_kind="debug",
            exclusive=True,
        )
    lease_path = run_dir / "worker.json"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["heartbeat_at"] = "2000-01-01T00:00:00+00:00"
    lease_path.write_text(json.dumps(lease), encoding="utf-8")

    supervisor = _supervisor(runs_dir)
    try:
        supervisor.start()
        assert load_run_manifest(run_dir).status is RunStatus.INTERRUPTED
        interruption = json.loads(
            (run_dir / "interruption.json").read_text(encoding="utf-8")
        )
        assert "stale worker lease" in interruption["reason"]
        assert not (run_dir / "worker.json").exists()
    finally:
        supervisor.shutdown()


def test_late_broken_pool_callback_does_not_discard_replacement_executor(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = prepare_run(
        {
            "session": "V.11",
            "dataset": "xor",
            "model": "mlp-2-3-1",
            "experiment": "xor_backprop",
            "device": "cpu",
            "seed": 0,
        },
        runs_dir=runs_dir,
    )
    supervisor = _supervisor(runs_dir)
    old_executor = object()
    replacement_executor = object()
    future: Future = Future()
    future.set_exception(BrokenProcessPool("old pool crashed"))
    supervisor._main_futures[future] = (  # type: ignore[assignment]
        run_dir,
        1,
        old_executor,
    )
    supervisor._main_executor = replacement_executor  # type: ignore[assignment]

    supervisor._main_done(future)

    assert supervisor._main_executor is replacement_executor
    assert load_run_manifest(run_dir).status is RunStatus.INTERRUPTED
