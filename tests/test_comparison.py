"""V.5: семантика сравнения конфигов, итогов и сохранённых рядов."""

import json

import pytest
from fastapi.testclient import TestClient

from bioplast.runner import RunManifest, RunStatus, write_run_manifest
from bioplast.viz.api import create_app
from bioplast.viz.comparison import RunComparisonError, compare_runs
from bioplast.viz.repository import RunRepository


def _run(
    runs_dir,
    run_id: str,
    *,
    config: dict,
    final: dict,
    rows: list[dict],
    parent_run_id: str | None = None,
):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    metrics = {
        "run_id": run_id,
        "status": "ok",
        "config": config,
        "epochs": rows,
        "final": final,
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "run.log").write_text("done\n", encoding="utf-8")
    write_run_manifest(
        run_dir,
        RunManifest(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            experiment=config.get("experiment"),
            started_at="2026-08-16T10:00:00+03:00",
            updated_at="2026-08-16T10:00:01+03:00",
            finished_at="2026-08-16T10:00:01+03:00",
            parent_run_id=parent_run_id,
        ),
    )


@pytest.fixture
def comparison_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    baseline = "baseline"
    candidate = "candidate"
    incompatible = "incompatible"
    _run(
        runs_dir,
        baseline,
        config={
            "experiment": "xor_backprop",
            "dataset": "xor",
            "model": "mlp",
            "seed": 0,
            "lr": 0.05,
            "flag": 0,
        },
        final={"loss": 0.5, "acc": 0.8, "zero": 0, "solved": True},
        rows=[
            {"step": 0, "loss/train": 0.7, "acc/train": 0.5},
            {"step": 2, "loss/train": 0.5, "acc/train": 0.8},
        ],
    )
    _run(
        runs_dir,
        candidate,
        config={
            "experiment": "xor_backprop",
            "dataset": "xor",
            "model": "mlp",
            "seed": 1,
            "lr": 0.01,
            "flag": False,
            "parent_run_id": baseline,
        },
        final={"loss": 0.25, "acc": 0.9, "zero": 1, "solved": True},
        rows=[
            {"step": 0, "loss/train": 0.65},
            {"step": 1, "loss/train": 0.25},
        ],
        parent_run_id=baseline,
    )
    _run(
        runs_dir,
        incompatible,
        config={
            "experiment": "mnist_mlp_backprop",
            "dataset": "mnist",
            "model": "other",
            "seed": 2,
            "lr": 0.001,
            "flag": 0,
        },
        final={"loss": 0.4},
        rows=[{"epoch": 1, "loss/train": 0.4}],
    )
    return runs_dir, baseline, candidate, incompatible


def test_comparison_preserves_actual_points_and_reports_missing_series(comparison_runs):
    runs_dir, baseline, candidate, _ = comparison_runs

    payload = compare_runs(RunRepository(runs_dir), baseline, [candidate])

    loss = next(item for item in payload["metric_series"] if item["key"] == "loss/train")
    baseline_points, candidate_points = loss["runs"]
    assert baseline_points["x"] == [0, 2]
    assert candidate_points["x"] == [0, 1]
    assert baseline_points["y"] == [0.7, 0.5]
    assert candidate_points["y"] == [0.65, 0.25]

    accuracy = next(item for item in payload["metric_series"] if item["key"] == "acc/train")
    assert accuracy["missing_run_ids"] == [candidate]
    assert accuracy["runs"][0]["run_id"] == baseline


def test_config_diff_distinguishes_json_types_and_final_deltas(comparison_runs):
    runs_dir, baseline, candidate, _ = comparison_runs

    payload = compare_runs(RunRepository(runs_dir), baseline, [candidate])

    diffs = {item["key"]: item for item in payload["config_diff"]}
    assert set(diffs) == {"flag", "lr", "seed"}
    assert "parent_run_id" not in diffs
    assert diffs["flag"]["baseline"]["value"] == 0
    assert diffs["flag"]["candidates"][0]["value"] is False

    finals = {item["key"]: item for item in payload["final_metrics"]}
    loss = finals["loss"]["candidates"][0]
    assert loss["delta"] == -0.25
    assert loss["relative_delta"] == -0.5
    assert finals["zero"]["candidates"][0]["delta"] == 1
    assert finals["zero"]["candidates"][0]["relative_delta"] is None
    assert "solved" not in finals


def test_comparison_marks_identity_and_step_axis_mismatches(comparison_runs):
    runs_dir, baseline, _, incompatible = comparison_runs

    payload = compare_runs(RunRepository(runs_dir), baseline, [incompatible])

    warning_types = [warning["type"] for warning in payload["warnings"]]
    assert warning_types.count("identity_mismatch") == 3
    assert warning_types.count("step_key_mismatch") == 1
    assert payload["candidates"][0]["step_key"] == "epoch"


def test_comparison_does_not_invent_an_axis_from_metric_name(comparison_runs):
    runs_dir, baseline, candidate, _ = comparison_runs
    metrics_path = runs_dir / candidate / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["epochs"] = [{"loss/train": 0.25}]
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    payload = compare_runs(RunRepository(runs_dir), baseline, [candidate])

    assert payload["candidates"][0]["step_key"] is None
    assert any(
        warning["type"] == "step_key_missing"
        and warning["run_id"] == candidate
        for warning in payload["warnings"]
    )


@pytest.mark.parametrize(
    "baseline,candidates,message",
    [
        ("baseline", [], "хотя бы один"),
        ("baseline", ["candidate"] * 5, "не больше 4"),
        ("baseline", ["candidate", "candidate"], "не должны повторяться"),
        ("baseline", ["baseline"], "не может одновременно"),
    ],
)
def test_comparison_rejects_invalid_selection(
    comparison_runs, baseline, candidates, message
):
    repository = RunRepository(comparison_runs[0])

    with pytest.raises(RunComparisonError, match=message):
        compare_runs(repository, baseline, candidates)


def test_compare_api_accepts_repeated_candidate_query(comparison_runs):
    runs_dir, baseline, candidate, incompatible = comparison_runs
    client = TestClient(create_app(runs_dir))

    response = client.get(
        "/api/compare",
        params=[("baseline", baseline), ("candidate", candidate), ("candidate", incompatible)],
    )

    assert response.status_code == 200
    assert response.json()["baseline"]["run_id"] == baseline
    assert [run["run_id"] for run in response.json()["candidates"]] == [
        candidate,
        incompatible,
    ]


def test_compare_api_validation_and_missing_run(comparison_runs):
    runs_dir, baseline, _, _ = comparison_runs
    client = TestClient(create_app(runs_dir))

    invalid = client.get("/api/compare", params={"baseline": baseline})
    missing = client.get(
        "/api/compare", params={"baseline": baseline, "candidate": "missing"}
    )

    assert invalid.status_code == 422
    assert missing.status_code == 404
