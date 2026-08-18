"""HTTP API V.2: каталог, карточка, данные и границы файлового доступа."""

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from bioplast.runner import RunManifest, RunStatus, write_run_manifest
from bioplast.viz.api import create_app
from bioplast.viz.repository import RunRepository, UnsafeRunPath


def _legacy_run(
    runs_dir,
    run_id: str,
    *,
    experiment: str = "xor_backprop",
    seed: int = 0,
    status: str = "ok",
    started_at: str = "2026-08-08T14:30:12+03:00",
):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    config = {
        "experiment": experiment,
        "dataset": "xor",
        "model": "mlp-2-8-1",
        "seed": seed,
    }
    metrics = {
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "duration_sec": 1.5,
        "config": config,
        "epochs": [{"step": 0, "loss/train": 0.7}],
        "final": {"loss": 0.7, "acc": 0.5},
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "run.log").write_text("first line\nsecond line\n", encoding="utf-8")
    return run_dir


def _contract_run(
    runs_dir,
    run_id: str,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    parent_run_id: str | None = None,
    debug: bool = False,
    started_at: str = "2026-08-18T12:00:00+03:00",
):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    config = {
        "experiment": "xor_interactive" if debug else "xor_backprop",
        "dataset": "xor",
        "model": "mlp-2-8-1",
        "seed": 0,
    }
    if debug:
        config["debug"] = {
            "protocol": "model_debug_v1",
            "renderer": "xor_neurons_v1",
        }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_run_manifest(
        run_dir,
        RunManifest(
            run_id=run_id,
            status=status,
            experiment=config["experiment"],
            started_at=started_at,
            updated_at=started_at,
            finished_at=started_at if status.terminal else None,
            duration_sec=1.0 if status.terminal else None,
            parent_run_id=parent_run_id,
        ),
    )
    return run_dir


@pytest.fixture
def runs_dir(tmp_path):
    root = tmp_path / "runs"
    _legacy_run(root, "20260808-143012-xor-s0", seed=0)
    _legacy_run(
        root,
        "20260809-100000-xor-s1",
        experiment="other",
        seed=1,
        status="failed",
        started_at="2026-08-09T10:00:00+03:00",
    )
    return root


@pytest.fixture
def client(runs_dir):
    return TestClient(create_app(runs_dir))


def test_list_runs_normalizes_legacy_and_sorts_newest_first(client):
    response = client.get("/api/runs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert [item["seed"] for item in payload["items"]] == [1, 0]
    assert payload["items"][0]["status"] == "failed"
    assert all(item["adapted_from_legacy"] for item in payload["items"])


def test_list_filters_and_paginates(client):
    response = client.get(
        "/api/runs",
        params={
            "status": "completed",
            "experiment": "xor_backprop",
            "seed": 0,
            "started_after": "2026-08-08T00:00:00+03:00",
            "limit": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["run_id"] == "20260808-143012-xor-s0"


def test_catalog_marks_filters_and_paginates_debug_sessions(tmp_path):
    runs_dir = tmp_path / "runs"
    _contract_run(runs_dir, "parent", started_at="2026-08-18T10:00:00+03:00")
    _contract_run(
        runs_dir,
        "debug-child",
        parent_run_id="parent",
        debug=True,
        started_at="2026-08-18T12:00:00+03:00",
    )
    _contract_run(runs_dir, "newest", started_at="2026-08-18T13:00:00+03:00")
    client = TestClient(create_app(runs_dir))

    first = client.get("/api/runs", params={"offset": 0, "limit": 2}).json()
    hidden = client.get("/api/runs", params={"include_debug": "false"}).json()

    assert first["total"] == 3
    assert [item["run_id"] for item in first["items"]] == ["newest", "debug-child"]
    assert first["items"][1]["is_debug"] is True
    assert first["debug_count"] == 1
    assert first["counts"]["completed"] == 3
    assert first["experiments"] == ["xor_backprop", "xor_interactive"]
    assert hidden["total"] == 2
    assert all(not item["is_debug"] for item in hidden["items"])


def test_run_detail_lists_only_debug_children_and_preserves_parent_link(tmp_path):
    runs_dir = tmp_path / "runs"
    _contract_run(runs_dir, "parent")
    _contract_run(runs_dir, "debug-child", parent_run_id="parent", debug=True)
    _contract_run(runs_dir, "rerun-child", parent_run_id="parent")
    client = TestClient(create_app(runs_dir))

    parent = client.get("/api/runs/parent").json()
    child = client.get("/api/runs/debug-child").json()

    assert [item["run_id"] for item in parent["debug_sessions"]] == ["debug-child"]
    assert child["manifest"]["parent_run_id"] == "parent"


def test_run_card_contains_config_metrics_and_artifacts(client):
    run_id = "20260808-143012-xor-s0"
    response = client.get(f"/api/runs/{run_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["manifest"]["status"] == "completed"
    assert payload["config"]["model"] == "mlp-2-8-1"
    assert payload["metrics"]["final"]["acc"] == 0.5
    assert {item["path"] for item in payload["artifacts"]} == {
        "config.json",
        "metrics.json",
        "run.log",
    }


def test_metrics_and_paginated_log(client):
    run_id = "20260808-143012-xor-s0"
    metrics = client.get(f"/api/runs/{run_id}/metrics")
    first = client.get(f"/api/runs/{run_id}/log", params={"limit": 6})
    second = client.get(
        f"/api/runs/{run_id}/log", params={"offset": first.json()["next_offset"]}
    )

    assert metrics.status_code == 200
    assert metrics.json()["epochs"][0]["loss/train"] == 0.7
    assert first.json()["text"] == "first "
    assert second.json()["text"].startswith("line")
    assert "second line" in second.json()["text"]
    assert second.json()["eof"] is True


def test_log_pagination_does_not_split_utf8_character(runs_dir):
    run_id = "20260808-143012-xor-s0"
    (runs_dir / run_id / "run.log").write_bytes("яz".encode("utf-8"))
    client = TestClient(create_app(runs_dir))

    first = client.get(f"/api/runs/{run_id}/log", params={"limit": 1}).json()
    second = client.get(
        f"/api/runs/{run_id}/log", params={"offset": first["next_offset"], "limit": 1}
    ).json()

    assert first["text"] == "я"
    assert first["next_offset"] == 2
    assert second["text"] == "z"


def test_artifact_download_and_missing_run(client):
    run_id = "20260808-143012-xor-s0"

    artifact = client.get(f"/api/runs/{run_id}/artifacts/config.json")
    missing = client.get("/api/runs/no-such-run")

    assert artifact.status_code == 200
    assert artifact.json()["experiment"] == "xor_backprop"
    assert missing.status_code == 404


def test_single_and_batch_deletion_require_terminal_runs_and_complete_family(tmp_path):
    runs_dir = tmp_path / "runs"
    _contract_run(runs_dir, "parent")
    _contract_run(runs_dir, "debug-child", parent_run_id="parent", debug=True)
    _contract_run(runs_dir, "active", status=RunStatus.RUNNING)
    _contract_run(runs_dir, "standalone")
    client = TestClient(create_app(runs_dir))

    active = client.delete("/api/runs/active")
    parent_only = client.delete("/api/runs/parent")
    standalone = client.delete("/api/runs/standalone")
    family = client.request(
        "DELETE",
        "/api/runs",
        json={"run_ids": ["parent", "debug-child"]},
    )

    assert active.status_code == 409
    assert "сначала завершите" in active.json()["detail"]
    assert parent_only.status_code == 409
    assert "debug-child" in parent_only.json()["detail"]
    assert standalone.status_code == 200
    assert standalone.json()["deleted"] == ["standalone"]
    assert family.status_code == 200
    assert family.json()["count"] == 2
    assert not (runs_dir / "parent").exists()
    assert not (runs_dir / "debug-child").exists()
    assert (runs_dir / "active").is_dir()


def test_repository_rejects_run_and_artifact_traversal(runs_dir, tmp_path):
    repository = RunRepository(runs_dir)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    with pytest.raises(UnsafeRunPath):
        repository.resolve_run("../secret")
    with pytest.raises(UnsafeRunPath):
        repository.resolve_artifact("20260808-143012-xor-s0", "../../secret.txt")


def test_corrupt_run_is_reported_without_breaking_catalog(client, runs_dir):
    broken = runs_dir / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{not json", encoding="utf-8")

    response = client.get("/api/runs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["errors"][0]["run_id"] == "broken"


def test_mismatched_manifest_id_is_reported_as_corrupt(runs_dir):
    run_dir = runs_dir / "renamed"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    write_run_manifest(
        run_dir,
        RunManifest(
            run_id="different-id",
            status=RunStatus.RUNNING,
            experiment=None,
            started_at=None,
            updated_at=None,
        ),
    )

    response = TestClient(create_app(runs_dir)).get("/api/runs").json()

    assert response["total"] == 2
    assert response["errors"][0]["run_id"] == "renamed"
    assert "не совпадает" in response["errors"][0]["error"]


def test_new_manifest_is_returned_without_legacy_flag(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "new-run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"experiment": "_selftest", "seed": 2}), encoding="utf-8"
    )
    write_run_manifest(
        run_dir,
        RunManifest(
            run_id="new-run",
            status=RunStatus.RUNNING,
            experiment="_selftest",
            started_at="2026-08-15T12:00:00+03:00",
            updated_at="2026-08-15T12:00:00+03:00",
        ),
    )

    payload = TestClient(create_app(runs_dir)).get("/api/runs").json()

    assert payload["items"][0]["adapted_from_legacy"] is False
    assert payload["items"][0]["status"] == "running"


def test_repository_date_filter_accepts_naive_legacy_timestamp(runs_dir):
    repository = RunRepository(runs_dir)

    items, errors = repository.list_runs(started_after=datetime(2026, 8, 8, 15, 0, 0))

    assert not errors
    assert [item["seed"] for item in items] == [1]
