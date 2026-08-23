"""V.4: безопасное клонирование конфига и постановка повторного запуска."""

import json

from fastapi.testclient import TestClient

from bioplast.runner import RunManifest, RunStatus, load_run_manifest, write_run_manifest
from bioplast.viz.api import create_app


class RecordingScheduler:
    def __init__(self, error: Exception | None = None) -> None:
        self.submitted = []
        self.error = error

    def submit(self, run_dir) -> None:
        if self.error is not None:
            raise self.error
        self.submitted.append(run_dir)


def _source_run(tmp_path, *, status: RunStatus = RunStatus.COMPLETED):
    runs_dir = tmp_path / "runs"
    run_id = "20260815-120000-sesv.3-xor-mlp-s0"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    config = {
        "session": "V.3",
        "dataset": "xor",
        "model": "mlp-2-8-1",
        "tag": "bp",
        "experiment": "_selftest",
        "device": "cpu",
        "seed": 0,
        "hidden": 8,
        "steps": 3,
        "lr": 0.05,
        "data_root": "data/xor",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"status": "ok", "config": config, "epochs": [], "final": {}}),
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text("done\n", encoding="utf-8")
    write_run_manifest(
        run_dir,
        RunManifest(
            run_id=run_id,
            status=status,
            experiment="_selftest",
            started_at="2026-08-15T12:00:00+03:00",
            updated_at="2026-08-15T12:00:01+03:00",
            finished_at=(
                "2026-08-15T12:00:01+03:00" if status.terminal else None
            ),
        ),
    )
    return runs_dir, run_id, run_dir, config


def test_rerun_preview_marks_parameters_and_paths_explicitly(tmp_path):
    runs_dir, run_id, _, _ = _source_run(tmp_path)
    client = TestClient(create_app(runs_dir, scheduler=RecordingScheduler()))

    response = client.get(f"/api/runs/{run_id}/rerun")

    assert response.status_code == 200
    fields = {item["key"]: item for item in response.json()["fields"]}
    assert fields["lr"]["editable"] is True
    assert fields["seed"]["editable"] is True
    assert fields["experiment"]["editable"] is False
    assert fields["data_root"]["editable"] is False


def test_rerun_locks_debug_session_source_run(tmp_path):
    runs_dir, run_id, run_dir, config = _source_run(tmp_path)
    debug_config = {**config, "source_run_id": "trusted-source-run"}
    (run_dir / "config.json").write_text(
        json.dumps(debug_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    client = TestClient(create_app(runs_dir, scheduler=RecordingScheduler()))

    preview = client.get(f"/api/runs/{run_id}/rerun")
    changed_source = client.post(
        f"/api/runs/{run_id}/rerun",
        json={"config": {**debug_config, "source_run_id": "other-run"}},
    )

    fields = {item["key"]: item for item in preview.json()["fields"]}
    assert preview.status_code == 200
    assert fields["source_run_id"]["editable"] is False
    assert changed_source.status_code == 422


def test_rerun_reserves_queued_child_and_preserves_source(tmp_path):
    runs_dir, run_id, source_dir, config = _source_run(tmp_path)
    original_config = (source_dir / "config.json").read_bytes()
    scheduler = RecordingScheduler()
    client = TestClient(create_app(runs_dir, scheduler=scheduler))
    proposed = {**config, "seed": 7, "lr": 0.01}

    response = client.post(f"/api/runs/{run_id}/rerun", json={"config": proposed})

    assert response.status_code == 202
    payload = response.json()
    child_dir = runs_dir / payload["run_id"]
    assert scheduler.submitted == [child_dir.resolve()]
    assert payload["status"] == "queued"
    assert payload["parent_run_id"] == run_id
    assert payload["location"] == f"/runs/{child_dir.name}"
    assert payload["changes"] == [
        {"key": "lr", "before": 0.05, "after": 0.01},
        {"key": "seed", "before": 0, "after": 7},
    ]

    child_config = json.loads((child_dir / "config.json").read_text(encoding="utf-8"))
    manifest = load_run_manifest(child_dir)
    assert child_config["seed"] == 7 and child_config["lr"] == 0.01
    assert child_config["parent_run_id"] == run_id
    assert manifest.status is RunStatus.QUEUED
    assert manifest.parent_run_id == run_id
    assert (source_dir / "config.json").read_bytes() == original_config


def test_rerun_rejects_changed_structure_and_locked_fields(tmp_path):
    runs_dir, run_id, _, config = _source_run(tmp_path)
    client = TestClient(create_app(runs_dir, scheduler=RecordingScheduler()))

    changed_experiment = client.post(
        f"/api/runs/{run_id}/rerun",
        json={"config": {**config, "experiment": "other"}},
    )
    changed_path = client.post(
        f"/api/runs/{run_id}/rerun",
        json={"config": {**config, "data_root": "C:/secret"}},
    )
    added_key = client.post(
        f"/api/runs/{run_id}/rerun",
        json={"config": {**config, "new_parameter": 1}},
    )
    changed_type = client.post(
        f"/api/runs/{run_id}/rerun",
        json={"config": {**config, "seed": False}},
    )
    invalid_device = client.post(
        f"/api/runs/{run_id}/rerun",
        json={"config": {**config, "device": "cuda:everything"}},
    )

    assert changed_experiment.status_code == 422
    assert changed_path.status_code == 422
    assert added_key.status_code == 422
    assert changed_type.status_code == 422
    assert invalid_device.status_code == 422
    assert len(list(runs_dir.iterdir())) == 1


def test_rerun_is_unavailable_until_source_is_terminal(tmp_path):
    runs_dir, run_id, _, config = _source_run(tmp_path, status=RunStatus.RUNNING)
    client = TestClient(create_app(runs_dir, scheduler=RecordingScheduler()))

    preview = client.get(f"/api/runs/{run_id}/rerun")
    submit = client.post(f"/api/runs/{run_id}/rerun", json={"config": config})

    assert preview.status_code == submit.status_code == 409
    assert "только для завершённого" in preview.json()["detail"]


def test_scheduler_submission_failure_becomes_failed_run(tmp_path):
    runs_dir, run_id, _, config = _source_run(tmp_path)
    client = TestClient(
        create_app(runs_dir, scheduler=RecordingScheduler(RuntimeError("pool closed")))
    )

    response = client.post(f"/api/runs/{run_id}/rerun", json={"config": config})

    assert response.status_code == 503
    children = [path for path in runs_dir.iterdir() if path.name != run_id]
    assert len(children) == 1
    assert load_run_manifest(children[0]).status is RunStatus.FAILED
    metrics = json.loads((children[0] / "metrics.json").read_text(encoding="utf-8"))
    assert "pool closed" in metrics["error"]
