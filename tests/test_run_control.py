"""Кооперативные команды V.8 применяются только в безопасных точках worker."""

from dataclasses import replace
import time

import pytest
from fastapi.testclient import TestClient

from bioplast.runner import (
    CooperativeRunControl,
    RunCancelled,
    RunCommandType,
    RunScheduler,
    RunStatus,
    append_run_command,
    load_run_manifest,
    prepare_run,
    read_run_commands,
    run_prepared,
    write_run_manifest,
)
from bioplast.viz.api import create_app

BASE = {
    "session": "V.8",
    "dataset": "toy",
    "model": "selftest",
    "experiment": "_selftest",
    "device": "cpu",
    "seed": 0,
    "steps": 3,
}


def _running_run(tmp_path):
    run_dir = prepare_run(BASE, runs_dir=tmp_path)
    manifest = load_run_manifest(run_dir)
    write_run_manifest(
        run_dir,
        replace(
            manifest,
            status=RunStatus.RUNNING,
            started_at=manifest.updated_at,
        ),
    )
    return run_dir


def test_command_log_round_trip_is_monotonic_and_ignores_partial_tail(tmp_path):
    run_dir = _running_run(tmp_path)

    pause = append_run_command(run_dir, RunCommandType.PAUSE)
    delay = append_run_command(run_dir, RunCommandType.SET_DELAY, delay_ms=250)
    with (run_dir / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"incomplete":')

    commands = read_run_commands(run_dir)

    assert [item.seq for item in commands] == [1, 2]
    assert pause.command is RunCommandType.PAUSE
    assert delay.delay_ms == 250


def test_controller_pauses_steps_once_resumes_delays_and_cancels(tmp_path):
    run_dir = _running_run(tmp_path)
    sleeps: list[float] = []
    clock = 0.0
    step_sent = False

    def fake_sleep(seconds: float) -> None:
        nonlocal clock, step_sent
        sleeps.append(seconds)
        clock += seconds
        if seconds == 0.01 and not step_sent:
            step_sent = True
            append_run_command(run_dir, RunCommandType.STEP)

    control = CooperativeRunControl(
        run_dir,
        poll_interval=0.01,
        sleep=fake_sleep,
        monotonic=lambda: clock,
    )
    append_run_command(run_dir, RunCommandType.PAUSE)

    control.checkpoint(step=4, phase="train_step")

    assert load_run_manifest(run_dir).status is RunStatus.PAUSED
    assert control.last_seq == 2

    append_run_command(run_dir, RunCommandType.SET_DELAY, delay_ms=250)
    append_run_command(run_dir, RunCommandType.RESUME)
    control.checkpoint(step=5, phase="train_step")

    assert load_run_manifest(run_dir).status is RunStatus.RUNNING
    assert control.delay_ms == 250
    assert sum(sleeps) >= 0.25

    append_run_command(run_dir, RunCommandType.CANCEL)
    with pytest.raises(RunCancelled):
        control.checkpoint(step=6, phase="train_step")


def test_pause_interrupts_delay_before_next_operation(tmp_path):
    run_dir = _running_run(tmp_path)
    append_run_command(run_dir, RunCommandType.SET_DELAY, delay_ms=1000)
    clock = 0.0
    calls = 0
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        nonlocal calls, clock
        calls += 1
        clock += seconds
        sleeps.append(seconds)
        if calls == 1:
            append_run_command(run_dir, RunCommandType.PAUSE)
        elif calls == 2:
            append_run_command(run_dir, RunCommandType.STEP)

    control = CooperativeRunControl(
        run_dir,
        poll_interval=0.05,
        sleep=fake_sleep,
        monotonic=lambda: clock,
    )

    control.checkpoint(step=1, phase="train_step")

    assert sleeps[0] == pytest.approx(0.05)
    assert clock >= 1.0  # ручной step всё равно уважает настроенную задержку
    assert load_run_manifest(run_dir).status is RunStatus.PAUSED


def test_controller_receives_typed_input_without_consuming_step(tmp_path):
    run_dir = _running_run(tmp_path)
    control = CooperativeRunControl(run_dir, poll_interval=0.01)
    append_run_command(run_dir, RunCommandType.PAUSE)
    input_command = append_run_command(
        run_dir,
        RunCommandType.SET_INPUT,
        input_values=(0.0, 1.0),
    )
    append_run_command(run_dir, RunCommandType.STEP)

    input_seq, values = control.wait_for_input()
    control.checkpoint(step=input_seq, phase="forward_hidden")

    assert input_seq == input_command.seq
    assert values == (0.0, 1.0)
    assert control.input_values == values
    assert load_run_manifest(run_dir).status is RunStatus.PAUSED


def test_run_prepared_records_cooperative_cancellation(tmp_path):
    run_dir = prepare_run(BASE, runs_dir=tmp_path)
    append_run_command(run_dir, RunCommandType.CANCEL)

    run_prepared(run_dir)

    assert load_run_manifest(run_dir).status is RunStatus.CANCELLED
    metrics = (run_dir / "metrics.json").read_text(encoding="utf-8")
    assert '"status": "cancelled"' in metrics
    assert '"error"' not in metrics


class _Scheduler:
    def __init__(self) -> None:
        self.cancelled = []

    def cancel(self, run_dir) -> bool:
        self.cancelled.append(run_dir)
        return True

    def shutdown(self) -> None:
        pass


def test_control_api_validates_transitions_and_cancels_queued_run(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = prepare_run(BASE, runs_dir=runs_dir)
    scheduler = _Scheduler()
    client = TestClient(create_app(runs_dir, scheduler=scheduler))
    url = f"/api/runs/{run_dir.name}/control"

    initial = client.get(url)
    paused = client.post(url, json={"command": "pause"})
    stepped = client.post(url, json={"command": "step"})
    resumed = client.post(url, json={"command": "resume"})
    missing_delay = client.post(url, json={"command": "set_delay"})
    delayed = client.post(url, json={"command": "set_delay", "delay_ms": 125})
    cancelled = client.post(url, json={"command": "cancel"})
    after_terminal = client.post(url, json={"command": "pause"})

    assert initial.status_code == 200
    assert initial.json()["available_commands"] == ["pause", "set_delay", "cancel"]
    assert paused.status_code == 202
    assert paused.json()["control"]["requested_status"] == "paused"
    assert stepped.status_code == 202
    assert resumed.json()["control"]["requested_status"] == "running"
    assert missing_delay.status_code == 422
    assert delayed.status_code == 202
    assert delayed.json()["control"]["delay_ms"] == 125
    assert cancelled.status_code == 202
    assert cancelled.json()["control"]["status"] == "cancelled"
    assert cancelled.json()["control"]["requested_status"] == "cancelled"
    assert scheduler.cancelled == [run_dir]
    assert load_run_manifest(run_dir).status is RunStatus.CANCELLED
    assert after_terminal.status_code == 409


def test_control_api_uses_debug_input_capability_not_experiment_name(tmp_path):
    runs_dir = tmp_path / "runs"
    config = {
        **BASE,
        "experiment": "generic_model_debug",
        "debug": {
            "protocol": "model_debug_v1",
            "renderer": "tensor_summary_v1",
            "accepts_input": True,
            "input_size": 3,
            "supports_step": True,
            "step_scope": "module",
        },
    }
    run_dir = prepare_run(config, runs_dir=runs_dir)
    client = TestClient(create_app(runs_dir, scheduler=_Scheduler()))
    url = f"/api/runs/{run_dir.name}/control"

    state = client.get(url)
    wrong_size = client.post(
        url,
        json={"command": "set_input", "input_values": [1, 2]},
    )
    accepted = client.post(
        url,
        json={"command": "set_input", "input_values": [1, 2, 3]},
    )

    assert state.json()["debug"] == config["debug"]
    assert "set_input" in state.json()["available_commands"]
    assert wrong_size.status_code == 422
    assert accepted.status_code == 202
    assert accepted.json()["control"]["input_values"] == [1.0, 2.0, 3.0]


def test_real_process_pool_pause_step_resume_smoke(tmp_path):
    """Мок API не проверяет Windows spawn и чтение команд дочерним процессом."""
    run_dir = prepare_run(BASE, runs_dir=tmp_path / "runs")
    append_run_command(run_dir, RunCommandType.PAUSE)
    scheduler = RunScheduler()
    scheduler.submit(run_dir)
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            if load_run_manifest(run_dir).status is RunStatus.PAUSED:
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker не перешёл в paused через настоящий process pool")

        append_run_command(run_dir, RunCommandType.STEP)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            log_path = run_dir / "run.log"
            if log_path.exists() and "разрешена одна атомарная операция" in log_path.read_text(
                encoding="utf-8"
            ):
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker не применил step через настоящий process pool")

        assert load_run_manifest(run_dir).status is RunStatus.PAUSED
        append_run_command(run_dir, RunCommandType.RESUME)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if load_run_manifest(run_dir).status is RunStatus.COMPLETED:
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker не завершился после resume")
        assert scheduler.pending == 0
    finally:
        scheduler.shutdown()
