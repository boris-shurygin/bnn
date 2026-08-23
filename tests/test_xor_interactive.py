"""V.9: дочерняя debug-сессия и послойные snapshots XOR."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from bioplast.runner import (
    RunCommandType,
    RunScheduler,
    RunStatus,
    append_run_command,
    iter_events,
    load_model_manifest,
    load_run_manifest,
    load_xor_forward_snapshot,
    prepare_run,
    read_run_commands,
    run_config,
    run_prepared,
)
from bioplast.viz.api import create_app


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


class _Scheduler:
    def __init__(self) -> None:
        self.submitted = []

    def submit(self, run_dir) -> None:
        self.submitted.append(run_dir)

    def shutdown(self) -> None:
        pass


def test_debug_api_creates_running_child_and_accepts_manual_input(tmp_path):
    runs_dir = tmp_path / "runs"
    source = _source_xor(runs_dir)
    scheduler = _Scheduler()
    client = TestClient(create_app(runs_dir, scheduler=scheduler))

    created = client.post(f"/api/runs/{source.name}/debug")

    assert created.status_code == 202
    child = runs_dir / created.json()["run_id"]
    assert scheduler.submitted == [child]
    assert load_run_manifest(child).parent_run_id == source.name
    config = json.loads((child / "config.json").read_text(encoding="utf-8"))
    assert config["experiment"] == "xor_interactive"
    assert config["debug"] == {
        "protocol": "model_debug_v1",
        "adapter": "xor_interactive_v1",
        "renderer": "xor_neurons_v1",
        "accepts_input": True,
        "input_mode": "manual_vector",
        "input_size": 2,
        "supports_step": True,
        "step_scope": "layer",
        "views": ["neurons", "tensor_summary"],
    }
    assert load_model_manifest(child).run_id == child.name
    assert read_run_commands(child) == []

    state = client.get(f"/api/runs/{child.name}/control").json()
    invalid = client.post(
        f"/api/runs/{child.name}/control",
        json={"command": "set_input", "input_values": [0]},
    )
    submitted = client.post(
        f"/api/runs/{child.name}/control",
        json={"command": "set_input", "input_values": [0, 1]},
    )

    assert state["accepts_input"] is True
    assert state["debug"] == config["debug"]
    assert state["requested_status"] == RunStatus.RUNNING.value
    assert "pause" in state["available_commands"]
    assert "set_input" in state["available_commands"]
    assert invalid.status_code == 422
    assert submitted.status_code == 202
    assert submitted.json()["control"]["input_values"] == [0.0, 1.0]


def test_interactive_worker_steps_hidden_then_output_and_writes_snapshots(tmp_path):
    runs_dir = tmp_path / "runs"
    source = _source_xor(runs_dir)
    child = prepare_run(
        {
            "session": "V.9",
            "dataset": "xor",
            "model": "mlp-2-3-1",
            "experiment": "xor_interactive",
            "device": "cpu",
            "seed": 0,
            "hidden": 3,
            "source_run_id": source.name,
            "max_inputs": 1,
        },
        runs_dir=runs_dir,
        parent_run_id=source.name,
    )
    append_run_command(child, RunCommandType.PAUSE)
    append_run_command(child, RunCommandType.SET_INPUT, input_values=(0.0, 1.0))
    append_run_command(child, RunCommandType.STEP)
    append_run_command(child, RunCommandType.STEP)
    # Тестовый max_inputs завершает бесконечную debug-сессию; третий пропуск
    # нужен существующей safe point финального экспорта раннера.
    append_run_command(child, RunCommandType.STEP)

    run_prepared(child)

    assert load_run_manifest(child).status is RunStatus.COMPLETED
    events = list(iter_events(child))
    snapshots = [load_xor_forward_snapshot(child / event.snapshot) for event in events]
    assert [snapshot.phase for snapshot in snapshots] == [
        "input",
        "forward_hidden",
        "forward_output",
    ]
    assert snapshots[1].hidden == snapshots[1].post
    assert len(snapshots[1].hidden) == 3
    assert len(snapshots[2].hidden) == 3
    assert snapshots[2].prediction in {0, 1}
    assert 0.0 <= snapshots[2].probability <= 1.0
    assert (child / "model.json").is_file()
    assert (child / "checkpoint.pt").is_file()

    client = TestClient(create_app(runs_dir, scheduler=_Scheduler()))
    payload = client.get(f"/api/runs/{child.name}/events", params={"after_seq": 1}).json()
    assert [item["phase"] for item in payload["items"]] == [
        "forward_hidden",
        "forward_output",
    ]


def test_real_process_pool_interactive_xor_smoke(tmp_path):
    """Проверяет Windows spawn, mailbox и live snapshots одним сценарием."""
    runs_dir = tmp_path / "runs"
    source = _source_xor(runs_dir)
    scheduler = RunScheduler()
    client = TestClient(create_app(runs_dir, scheduler=scheduler))
    created = client.post(f"/api/runs/{source.name}/debug").json()
    child = runs_dir / created["run_id"]
    deadline = time.monotonic() + 20

    def wait_until(predicate, message):
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        pytest.fail(message)

    try:
        wait_until(
            lambda: load_run_manifest(child).status is RunStatus.RUNNING,
            "интерактивный worker не перешёл в running",
        )
        control_url = f"/api/runs/{child.name}/control"
        assert client.post(
            control_url,
            json={"command": "set_input", "input_values": [1, 0]},
        ).status_code == 202
        wait_until(
            lambda: len(list(iter_events(child))) >= 3,
            "worker не завершил непрерывный forward",
        )
        assert list(iter_events(child))[-1].phase == "forward_output"

        assert client.post(control_url, json={"command": "pause"}).status_code == 202
        wait_until(
            lambda: load_run_manifest(child).status is RunStatus.PAUSED,
            "интерактивный worker не перешёл в paused",
        )
        assert client.post(
            control_url,
            json={"command": "set_input", "input_values": [0, 1]},
        ).status_code == 202
        wait_until(
            lambda: len(list(iter_events(child))) >= 4,
            "worker не записал snapshot нового входа",
        )

        assert client.post(control_url, json={"command": "step"}).status_code == 202
        wait_until(
            lambda: len(list(iter_events(child))) >= 5,
            "worker не записал snapshot скрытого слоя",
        )
        assert list(iter_events(child))[-1].phase == "forward_hidden"
        assert load_run_manifest(child).status is RunStatus.PAUSED

        assert client.post(control_url, json={"command": "step"}).status_code == 202
        wait_until(
            lambda: len(list(iter_events(child))) >= 6,
            "worker не записал snapshot выходного слоя",
        )
        assert list(iter_events(child))[-1].phase == "forward_output"

        assert client.post(control_url, json={"command": "cancel"}).status_code == 202
        wait_until(
            lambda: load_run_manifest(child).status is RunStatus.CANCELLED,
            "worker не завершил debug-сессию кооперативно",
        )
    finally:
        scheduler.shutdown()
