"""Контракт наблюдаемости V.1 и совместимость с артефактами блока 0."""

import json

import pytest

from bioplast.runner import (
    CONTRACT_VERSION,
    ConnectionSpec,
    ContractError,
    LayerSpec,
    ModelManifest,
    RunEvent,
    RunStatus,
    TensorSpec,
    append_event,
    iter_events,
    load_model_manifest,
    load_run_manifest,
    run_config,
    write_model_manifest,
)

BASE = {
    "session": "V.1",
    "dataset": "toy",
    "model": "selftest",
    "experiment": "_selftest",
    "device": "cpu",
    "seed": 0,
    "steps": 1,
}


def test_new_run_writes_versioned_terminal_manifest(tmp_path):
    run_dir = run_config(dict(BASE), runs_dir=tmp_path)

    raw = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest = load_run_manifest(run_dir)

    assert raw["schema_version"] == CONTRACT_VERSION
    assert raw["kind"] == "run"
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.started_at and manifest.finished_at
    assert manifest.duration_sec is not None
    assert manifest.artifacts["events"] == "events.jsonl"
    assert manifest.artifacts["checkpoint"] == "checkpoint.pt"
    assert manifest.adapted_from_legacy is False


def test_failed_run_has_failed_manifest_and_legacy_metrics_status(tmp_path):
    run_dir = run_config({**BASE, "fail": True}, runs_dir=tmp_path)

    assert load_run_manifest(run_dir).status is RunStatus.FAILED
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"


def test_legacy_run_is_adapted_without_migration(tmp_path):
    run_dir = tmp_path / "20260808-143012-ses0.3-toy-xor-s0"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"experiment": "xor_backprop", "parent_run_id": "old"}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": "ok",
                "started_at": "2026-08-08T14:30:12",
                "duration_sec": 2.5,
            }
        ),
        encoding="utf-8",
    )

    manifest = load_run_manifest(run_dir)

    assert manifest.status is RunStatus.COMPLETED
    assert manifest.experiment == "xor_backprop"
    assert manifest.parent_run_id == "old"
    assert manifest.adapted_from_legacy is True
    assert not (run_dir / "run.json").exists()


def test_legacy_run_without_metrics_is_visible_as_running(tmp_path):
    run_dir = tmp_path / "custom-id"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"experiment": "_selftest"}), encoding="utf-8"
    )

    manifest = load_run_manifest(run_dir)

    assert manifest.status is RunStatus.RUNNING
    assert manifest.started_at is None


def test_model_manifest_round_trip(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = ModelManifest(
        run_id="run",
        model_name="xor-mlp",
        layers=(
            LayerSpec(
                layer_id="hidden",
                layer_type="Linear",
                input_shape=(None, 2),
                output_shape=(None, 8),
                activation="relu",
                parameter_count=24,
                tensors=(TensorSpec("weight", "parameter", (8, 2), "float32"),),
            ),
            LayerSpec("output", "Linear", (None, 8), (None, 1), parameter_count=9),
        ),
        connections=(ConnectionSpec("hidden", "output"),),
    )

    write_model_manifest(run_dir, model)

    assert load_model_manifest(run_dir) == model


def test_model_rejects_connection_to_unknown_layer():
    with pytest.raises(ContractError, match="неизвестный слой"):
        ModelManifest(
            run_id="run",
            model_name="bad",
            layers=(LayerSpec("a", "Linear", (1,), (1,)),),
            connections=(ConnectionSpec("a", "missing"),),
        )


def test_events_round_trip_as_json_lines(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = [
        RunEvent("run", 0, "2026-08-15T12:00:00+03:00", "run_started"),
        RunEvent(
            "run",
            1,
            "2026-08-15T12:00:01+03:00",
            "layer_forward",
            step=0,
            phase="forward",
            layer_id="hidden",
            scalars={"act/mean": 0.25},
            snapshot="snapshots/1.npz",
        ),
    ]
    for event in events:
        append_event(run_dir, event)

    assert list(iter_events(run_dir)) == events
    assert (run_dir / "events.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_contract_rejects_unknown_version(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"schema_version": 99, "kind": "run"}), encoding="utf-8"
    )

    with pytest.raises(ContractError, match="неподдерживаемая версия"):
        load_run_manifest(run_dir)


def test_event_rejects_path_traversal():
    with pytest.raises(ContractError, match="относительным"):
        RunEvent("run", 0, "now", "snapshot", snapshot="../secret.npz")


def test_event_reader_rejects_non_monotonic_sequence(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = RunEvent("run", 2, "2026-08-15T12:00:00+03:00", "first")
    second = RunEvent("run", 1, "2026-08-15T12:00:01+03:00", "second")
    path = run_dir / "events.jsonl"
    path.write_text(
        "\n".join(json.dumps(event.to_dict()) for event in (first, second)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="seq должен возрастать"):
        list(iter_events(run_dir))
