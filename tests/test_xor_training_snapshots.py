"""V.12: точные снимки атомарного обучающего шага XOR."""

from __future__ import annotations

import logging
import random

import numpy as np
import pytest
import torch

import experiments.xor_backprop as xor_experiment
from bioplast.diagnostics.metrics import MetricsRecorder

from bioplast.runner import (
    ContractError,
    RunContext,
    RunStatus,
    XorDecisionSurface,
    XorParameterUpdate,
    XorTrainingLayer,
    XorTrainStepSnapshot,
    iter_events,
    load_xor_train_step_snapshot,
    load_recovery,
    prepare_run,
    run_config,
    write_xor_train_step_snapshot,
)


class _Control:
    last_seq = 0
    mode = RunStatus.RUNNING
    delay_ms = 0
    input_seq = 0
    input_values = None

    def checkpoint(self, **_kwargs) -> None:
        pass


def _context(run_dir, attempt: int) -> RunContext:
    return RunContext(
        run_id=run_dir.name,
        run_dir=run_dir,
        device="cpu",
        seed=5,
        log=logging.getLogger(f"test.xor-training-snapshot.{attempt}"),
        control=_Control(),  # type: ignore[arg-type]
        attempt=attempt,
        pool_kind="main",
        metrics=MetricsRecorder(),
    )


def _seed() -> None:
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)


def _snapshot(run_id: str = "run") -> XorTrainStepSnapshot:
    return XorTrainStepSnapshot(
        run_id=run_id,
        seq=1,
        step=0,
        loss=0.75,
        accuracy=0.5,
        updated=True,
        layers=(
            XorTrainingLayer(
                layer_id="hidden",
                parameters=(
                    XorParameterUpdate(
                        name="weight",
                        before=[[1.0, -1.0]],
                        delta=[[0.25, -0.5]],
                        after=[[1.25, -1.5]],
                    ),
                ),
            ),
        ),
        decision_surface=XorDecisionSurface(
            x0=(0.0, 1.0),
            x1=(0.0, 1.0),
            probabilities=((0.1, 0.9), (0.8, 0.2)),
        ),
    )


def test_xor_train_step_snapshot_round_trip_and_exact_delta(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    path = write_xor_train_step_snapshot(run_dir, _snapshot())
    loaded = load_xor_train_step_snapshot(path)

    assert loaded == _snapshot()
    assert loaded.layers[0].apical_deviation is None
    assert loaded.layers[0].local_error is None

    with pytest.raises(ContractError, match=r"before \+ delta"):
        XorParameterUpdate(
            name="weight",
            before=[[1.0]],
            delta=[[0.25]],
            after=[[2.0]],
        )


def test_xor_backprop_publishes_weight_update_loss_and_decision_surface(tmp_path):
    run_dir = run_config(
        {
            "session": "V.12",
            "dataset": "xor",
            "model": "mlp-2-3-1",
            "experiment": "xor_backprop",
            "device": "cpu",
            "seed": 3,
            "hidden": 3,
            "steps": 3,
            "log_every": 1,
            "snapshot_every_steps": 1,
        },
        runs_dir=tmp_path / "runs",
    )

    events = list(iter_events(run_dir))
    snapshots = [
        load_xor_train_step_snapshot(run_dir / event.snapshot) for event in events
    ]

    assert [event.event_type for event in events] == ["xor_train_step"] * 3
    assert [snapshot.step for snapshot in snapshots] == [0, 1, 2]
    assert all(snapshot.updated is True for snapshot in snapshots)
    assert all([layer.layer_id for layer in snapshot.layers] == ["hidden", "output"] for snapshot in snapshots)
    assert all(
        [parameter.name for parameter in layer.parameters] == ["weight", "bias"]
        for snapshot in snapshots
        for layer in snapshot.layers
    )
    assert any(
        abs(value) > 0
        for parameter in snapshots[0].layers[0].parameters
        for row in parameter.delta
        for value in (row if isinstance(row, list) else [row])
    )
    surface = snapshots[0].decision_surface
    assert len(surface.x0) == len(surface.x1) == 25
    assert len(surface.probabilities) == 25
    assert all(len(row) == 25 for row in surface.probabilities)
    assert all(event.scalars["loss"] == snapshot.loss for event, snapshot in zip(events, snapshots))


def test_resume_reuses_event_published_before_recovery_pointer(tmp_path, monkeypatch):
    config = {
        "session": "V.12",
        "dataset": "xor",
        "model": "mlp-2-3-1",
        "experiment": "xor_backprop",
        "device": "cpu",
        "seed": 5,
        "hidden": 3,
        "steps": 1,
        "log_every": 1,
        "snapshot_every_steps": 1,
    }
    run_dir = prepare_run(config, runs_dir=tmp_path / "runs")
    original_write = xor_experiment.write_training_recovery
    calls = 0

    class _Crash(BaseException):
        pass

    def crash_before_second_recovery(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _Crash
        return original_write(*args, **kwargs)

    _seed()
    monkeypatch.setattr(xor_experiment, "write_training_recovery", crash_before_second_recovery)
    with pytest.raises(_Crash):
        xor_experiment.run(config, _context(run_dir, 1))
    assert [event.step for event in iter_events(run_dir)] == [0]
    recovery_before, _payload = load_recovery(run_dir, expected_adapter="xor_backprop_v1")
    assert recovery_before["last_event_seq"] == 0

    monkeypatch.setattr(xor_experiment, "write_training_recovery", original_write)
    _seed()
    xor_experiment.run(config, _context(run_dir, 2))

    events = list(iter_events(run_dir))
    assert [event.step for event in events] == [0]
    assert [event.seq for event in events] == [1]
    recovery_after, _payload = load_recovery(run_dir, expected_adapter="xor_backprop_v1")
    assert recovery_after["last_event_seq"] == 1
