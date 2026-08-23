"""V.13: debug adapter registry and aggregate MNIST tensor flow."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bioplast.runner import (
    RunCommandType,
    RunSupervisor,
    RunStatus,
    append_run_command,
    iter_events,
    load_model_debug_snapshot,
    load_run_manifest,
    prepare_run,
    run_config,
    run_prepared,
)
from bioplast.viz.api import create_app
from bioplast.viz.debug import debug_adapter_metadata, registered_debug_adapters


class _Scheduler:
    def __init__(self) -> None:
        self.submitted = []

    def submit(self, run_dir) -> None:
        self.submitted.append(run_dir)

    def shutdown(self) -> None:
        pass


def _fake_mnist(root) -> None:
    root.mkdir(parents=True)
    train_x = np.zeros((4, 28, 28), dtype=np.uint8)
    train_x[1, 8:20, 13:15] = 255
    train_x[2, 7:21, 7:21] = np.eye(14, dtype=np.uint8) * 255
    train_x[3, 5:23, 5:23] = 180
    test_x = np.zeros((3, 28, 28), dtype=np.uint8)
    test_x[0, 6:22, 12:16] = 255
    test_x[1, 7:21, 7:21] = 200
    test_x[2, 9:19, 9:19] = 120
    np.savez_compressed(
        root / "mnist.npz",
        train_x=train_x,
        train_y=np.array([0, 1, 2, 9], dtype=np.uint8),
        test_x=test_x,
        test_y=np.array([1, 9, 2], dtype=np.uint8),
    )


def _source_mnist(runs_dir, data_root):
    _fake_mnist(data_root)
    return run_config(
        {
            "session": "test",
            "dataset": "mnist",
            "model": "mlp-784-4-10",
            "experiment": "mnist_mlp_backprop",
            "device": "cpu",
            "seed": 0,
            "hidden": [4],
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "data_root": str(data_root),
            "recovery_every_steps": 1,
        },
        runs_dir=runs_dir,
    )


def test_registry_exposes_xor_and_mnist_without_dynamic_experiment_names():
    adapters = registered_debug_adapters()

    assert {(item.source_experiment, item.debug_experiment) for item in adapters} == {
        ("xor_backprop", "xor_interactive"),
        ("mnist_mlp_backprop", "mnist_interactive"),
    }
    assert debug_adapter_metadata(
        {"experiment": "mnist_mlp_backprop", "dataset": "mnist"}
    ) == {
        "adapter": "mnist_mlp_interactive_v1",
        "debug_experiment": "mnist_interactive",
        "renderer": "tensor_flow_v1",
    }
    assert debug_adapter_metadata({"experiment": "unknown", "dataset": "mnist"}) is None


def test_debug_api_creates_mnist_child_from_registered_adapter(tmp_path):
    runs_dir = tmp_path / "runs"
    source = _source_mnist(runs_dir, tmp_path / "data")
    scheduler = _Scheduler()
    client = TestClient(create_app(runs_dir, scheduler=scheduler))

    detail = client.get(f"/api/runs/{source.name}").json()
    created = client.post(f"/api/runs/{source.name}/debug")

    assert detail["debug_adapter"]["renderer"] == "tensor_flow_v1"
    assert created.status_code == 202
    assert created.json()["adapter"] == "mnist_mlp_interactive_v1"
    child = runs_dir / created.json()["run_id"]
    assert scheduler.submitted == [child]
    config = json.loads((child / "config.json").read_text(encoding="utf-8"))
    assert config["experiment"] == "mnist_interactive"
    assert config["hidden"] == [4]
    assert config["debug"] == {
        "protocol": "model_debug_v1",
        "adapter": "mnist_mlp_interactive_v1",
        "renderer": "tensor_flow_v1",
        "accepts_input": True,
        "input_mode": "dataset_index",
        "input_size": 1,
        "input_min": 0,
        "input_max": 9999,
        "input_integer": True,
        "dataset_split": "test",
        "supports_step": True,
        "step_scope": "layer",
        "views": ["module_hierarchy", "tensor_flow", "activation_summary"],
    }

    invalid = client.post(
        f"/api/runs/{child.name}/control",
        json={"command": "set_input", "input_values": [0, 1]},
    )
    fractional = client.post(
        f"/api/runs/{child.name}/control",
        json={"command": "set_input", "input_values": [1.5]},
    )
    outside = client.post(
        f"/api/runs/{child.name}/control",
        json={"command": "set_input", "input_values": [10000]},
    )
    selected = client.post(
        f"/api/runs/{child.name}/control",
        json={"command": "set_input", "input_values": [2]},
    )
    assert invalid.status_code == 422
    assert fractional.status_code == 422
    assert outside.status_code == 422
    assert selected.status_code == 202


def test_mnist_worker_steps_layers_and_publishes_only_aggregate_activations(tmp_path):
    runs_dir = tmp_path / "runs"
    data_root = tmp_path / "data"
    source = _source_mnist(runs_dir, data_root)
    child = prepare_run(
        {
            "session": "V.13",
            "dataset": "mnist",
            "model": "mlp-784-4-10",
            "experiment": "mnist_interactive",
            "device": "cpu",
            "seed": 0,
            "hidden": [4],
            "source_run_id": source.name,
            "data_root": str(data_root),
            "max_inputs": 1,
            "debug": {
                "protocol": "model_debug_v1",
                "adapter": "mnist_mlp_interactive_v1",
                "renderer": "tensor_flow_v1",
                "accepts_input": True,
                "input_mode": "dataset_index",
                "input_size": 1,
                "supports_step": True,
                "step_scope": "layer",
            },
        },
        runs_dir=runs_dir,
        parent_run_id=source.name,
    )
    append_run_command(child, RunCommandType.PAUSE)
    append_run_command(child, RunCommandType.SET_INPUT, input_values=(1.0,))
    append_run_command(child, RunCommandType.STEP)
    append_run_command(child, RunCommandType.STEP)
    append_run_command(child, RunCommandType.STEP)

    run_prepared(child)

    assert load_run_manifest(child).status is RunStatus.COMPLETED
    events = list(iter_events(child))
    assert [item.event_type for item in events] == ["model_debug"] * 3
    snapshots = [load_model_debug_snapshot(child / item.snapshot) for item in events]
    assert [item.phase for item in snapshots] == ["input", "forward_layer", "forward_layer"]
    final = snapshots[-1]
    assert final.input_index == 1
    assert final.input_label == 9
    assert len(final.input_preview) == 28
    assert [layer.module_path for layer in final.layers] == ["layers.0", "layers.1"]
    assert final.prediction in range(10)
    assert len(final.top_classes) == 3
    assert final.top_classes[0].class_index == final.prediction
    for layer in final.layers:
        for tensor in (layer.input_tensor, layer.preactivation_tensor, layer.output_tensor):
            assert tensor.value_mode == "summary"
            assert tensor.values is None
            assert tensor.values_omitted_reason == "size_limit"
            assert tensor.summary is not None


def test_real_windows_spawn_runs_registered_mnist_debug_adapter(tmp_path):
    runs_dir = tmp_path / "runs"
    source = _source_mnist(runs_dir, tmp_path / "data")
    supervisor = RunSupervisor(
        runs_dir,
        main_workers=1,
        debug_workers=1,
        debug_inactive_timeout_sec=30,
        heartbeat_sec=0.1,
        stale_sec=2,
        shutdown_grace_sec=2,
        monitor_interval=0.05,
    )
    client = TestClient(create_app(runs_dir, scheduler=supervisor))
    created = client.post(f"/api/runs/{source.name}/debug")
    assert created.status_code == 202
    child = runs_dir / created.json()["run_id"]

    def wait_until(predicate, message, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        pytest.fail(message)

    try:
        wait_until(
            lambda: load_run_manifest(child).status is RunStatus.RUNNING,
            "MNIST debug worker не перешёл в running",
        )
        control_url = f"/api/runs/{child.name}/control"
        assert client.post(
            control_url,
            json={"command": "set_input", "input_values": [1]},
        ).status_code == 202
        wait_until(
            lambda: len(list(iter_events(child))) >= 3,
            "MNIST debug worker не завершил послойный forward",
        )
        final_event = list(iter_events(child))[-1]
        final = load_model_debug_snapshot(child / final_event.snapshot)
        assert final.prediction is not None
        assert len(final.layers) == 2

        assert client.post(control_url, json={"command": "cancel"}).status_code == 202
        wait_until(
            lambda: load_run_manifest(child).status is RunStatus.CANCELLED,
            "MNIST debug worker не завершился кооперативно",
        )
    finally:
        supervisor.shutdown()
