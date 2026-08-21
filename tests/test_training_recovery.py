"""V.11: exact training resume for XOR and MNIST."""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import time
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import experiments.mnist_mlp_backprop as mnist_experiment
from bioplast.data import Dataset
from bioplast.diagnostics.metrics import MetricsRecorder
from bioplast.runner import (
    RunCommandType,
    RunContext,
    RunStatus,
    RunSupervisor,
    append_run_command,
    capture_rng_state,
    load_recovery,
    load_run_manifest,
    load_training_checkpoint,
    prepare_run,
    restore_rng_state,
    run_config,
)
from bioplast.viz.api import create_app


def _wait_until(predicate, message: str, *, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail(message)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_rng_round_trip_restores_python_numpy_and_torch() -> None:
    _seed(17)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(4))

    _seed(99)
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(4))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


class _SyntheticInterruption(BaseException):
    pass


class _Control:
    def __init__(self, *, interrupt_before_call: int | None = None) -> None:
        self.last_seq = 0
        self.mode = RunStatus.RUNNING
        self.delay_ms = 0
        self.input_seq = 0
        self.input_values = None
        self.calls = 0
        self.interrupt_before_call = interrupt_before_call

    def checkpoint(self, **_kwargs: Any) -> None:
        self.calls += 1
        if self.calls == self.interrupt_before_call:
            raise _SyntheticInterruption


def _context(run_dir, *, attempt: int, control: _Control) -> RunContext:
    return RunContext(
        run_id=run_dir.name,
        run_dir=run_dir,
        device="cpu",
        seed=7,
        log=logging.getLogger(f"test.training-recovery.{run_dir.name}.{attempt}"),
        control=control,  # type: ignore[arg-type]
        attempt=attempt,
        pool_kind="main",
        metrics=MetricsRecorder(),
    )


def test_mnist_resumes_mid_epoch_without_repeating_or_skipping_batch(tmp_path, monkeypatch):
    dataset = Dataset(
        train_x=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.2, 0.2, 0.8, 0.8],
                [0.8, 0.8, 0.2, 0.2],
            ]
        ),
        train_y=torch.tensor([0, 0, 1, 1, 0, 1, 0, 1]),
        test_x=torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        test_y=torch.tensor([0, 1]),
    )
    monkeypatch.setattr(mnist_experiment, "load_mnist", lambda **_kwargs: dataset)
    config = {
        "session": "V.11",
        "dataset": "mnist",
        "model": "mlp-4-3-2",
        "experiment": "mnist_mlp_backprop",
        "device": "cpu",
        "seed": 7,
        "hidden": [3],
        "epochs": 2,
        "batch_size": 2,
        "lr": 0.01,
        "recovery_every_steps": 1,
    }

    resumed_dir = prepare_run(config, runs_dir=tmp_path / "resumed")
    _seed(7)
    with pytest.raises(_SyntheticInterruption):
        mnist_experiment.run(
            config,
            _context(resumed_dir, attempt=1, control=_Control(interrupt_before_call=2)),
        )
    interrupted_state, interrupted_payload = load_recovery(
        resumed_dir, expected_adapter="mnist_mlp_backprop_v1"
    )
    assert interrupted_state["progress"] == {"epoch": 1, "batch": 1, "global_step": 1}
    assert interrupted_payload["training"]["batch_start"] == 2
    assert interrupted_payload["training"]["permutation"].numel() == 8

    _seed(7)  # run_prepared seeds a fresh worker before adapter restore
    resumed_ctx = _context(resumed_dir, attempt=2, control=_Control())
    resumed = mnist_experiment.run(config, resumed_ctx)

    reference_dir = prepare_run(config, runs_dir=tmp_path / "reference")
    _seed(7)
    reference_ctx = _context(reference_dir, attempt=1, control=_Control())
    reference = mnist_experiment.run(config, reference_ctx)

    assert resumed.final == reference.final
    assert resumed_ctx.metrics.rows == reference_ctx.metrics.rows
    assert resumed.model_artifacts is not None
    assert reference.model_artifacts is not None
    _assert_nested_equal(
        resumed.model_artifacts.model.state_dict(),
        reference.model_artifacts.model.state_dict(),
    )
    _assert_nested_equal(
        resumed.model_artifacts.optimizer.state_dict(),
        reference.model_artifacts.optimizer.state_dict(),
    )
    completed_state, completed_payload = load_recovery(
        resumed_dir, expected_adapter="mnist_mlp_backprop_v1"
    )
    assert completed_state["attempt"] == 2
    assert completed_state["progress"] == {"epoch": 3, "batch": 0, "global_step": 8}
    assert completed_payload["training"]["permutation"] is None
    assert len(list((resumed_dir / "recovery").glob("checkpoint-*.pt"))) == 3


def test_main_training_recovers_after_real_spawn_worker_crash(tmp_path):
    """Hard-kill a real Windows spawn worker, then continue the same run_id."""
    runs_dir = tmp_path / "runs"
    config = {
        "session": "V.11",
        "dataset": "xor",
        "model": "mlp-2-4-1",
        "experiment": "xor_backprop",
        "device": "cpu",
        "seed": 5,
        "hidden": 4,
        "steps": 20,
        "log_every": 2,
        "lr": 0.03,
        "recovery_every_steps": 1,
    }
    run_dir = prepare_run(config, runs_dir=runs_dir)
    append_run_command(run_dir, RunCommandType.SET_DELAY, delay_ms=40)
    supervisor = RunSupervisor(
        runs_dir,
        main_workers=1,
        debug_workers=1,
        heartbeat_sec=0.05,
        stale_sec=0.5,
        shutdown_grace_sec=2,
        monitor_interval=0.02,
    )
    supervisor.submit(run_dir)
    try:
        _wait_until(
            lambda: (run_dir / "recovery" / "state.json").is_file()
            and int(load_recovery(run_dir)[0]["generation"]) >= 4,
            "main worker не опубликовал несколько recovery generations",
        )
        recovery_before_crash, _payload = load_recovery(
            run_dir, expected_adapter="xor_backprop_v1"
        )
        lease = supervisor.describe(run_dir)["worker"]
        assert lease is not None and isinstance(lease["pid"], int)
        os.kill(int(lease["pid"]), signal.SIGTERM)

        _wait_until(
            lambda: load_run_manifest(run_dir).status is RunStatus.INTERRUPTED,
            "hard crash main worker не перевёл запуск в interrupted",
        )
        with TestClient(create_app(runs_dir, scheduler=supervisor)) as client:
            resumed = client.post(
                f"/api/runs/{run_dir.name}/control", json={"command": "resume"}
            )
            assert resumed.status_code == 202
            assert resumed.json()["control"]["lifecycle"]["pool"] == "main"
            _wait_until(
                lambda: load_run_manifest(run_dir).status is RunStatus.COMPLETED,
                "возобновлённый main worker не завершил XOR",
            )
    finally:
        supervisor.shutdown()

    recovered_state, recovered_payload = load_recovery(
        run_dir, expected_adapter="xor_backprop_v1"
    )
    assert recovered_state["generation"] > recovery_before_crash["generation"]
    assert recovered_state["attempt"] == 2
    assert recovered_payload["training"]["next_step"] == config["steps"] + 1

    reference_dir = run_config(config, runs_dir=tmp_path / "reference")
    recovered_metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    reference_metrics = json.loads(
        (reference_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert recovered_metrics["epochs"] == reference_metrics["epochs"]
    assert recovered_metrics["final"] == reference_metrics["final"]
    recovered_checkpoint = load_training_checkpoint(run_dir / "checkpoint.pt")
    reference_checkpoint = load_training_checkpoint(reference_dir / "checkpoint.pt")
    _assert_nested_equal(
        recovered_checkpoint["model_state_dict"], reference_checkpoint["model_state_dict"]
    )
    _assert_nested_equal(
        recovered_checkpoint["optimizer_state_dict"],
        reference_checkpoint["optimizer_state_dict"],
    )
