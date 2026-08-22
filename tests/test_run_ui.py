"""Серверная часть UI V.3 и локальные assets без CDN."""

import json

from fastapi.testclient import TestClient

from bioplast.viz.api import create_app


def _client(tmp_path) -> tuple[TestClient, str]:
    run_id = "20260815-120000-sesv.3-xor-mlp-s0"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    config = {"experiment": "xor_backprop", "dataset": "xor", "model": "mlp", "seed": 0}
    metrics = {
        "run_id": run_id,
        "status": "ok",
        "started_at": "2026-08-15T12:00:00+03:00",
        "duration_sec": 1.0,
        "config": config,
        "git": {"commit": "abc", "branch": "test", "dirty": True},
        "env": {"device": "cpu"},
        "epochs": [{"step": 0, "loss/train": 0.7}],
        "final": {"loss": 0.7},
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "run.log").write_text("ready\n", encoding="utf-8")
    return TestClient(create_app(tmp_path / "runs")), run_id


def test_root_redirects_to_run_catalog(tmp_path):
    client, _ = _client(tmp_path)

    response = client.get("/", follow_redirects=False)

    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/runs"


def test_catalog_page_loads_local_styles_and_script(tmp_path):
    client, _ = _client(tmp_path)

    page = client.get("/runs")
    css = client.get("/static/app.css")
    script = client.get("/static/runs.js")

    assert page.status_code == css.status_code == script.status_code == 200
    assert "Экспериментальный журнал" in page.text
    assert "/static/app.css" in page.text and "/static/runs.js" in page.text
    assert "fetch(`/api/runs?" in script.text
    assert 'id="show-debug-runs"' in page.text
    assert 'id="experiment-filter"' in page.text
    assert "datalist" not in page.text
    assert 'id="select-page-runs"' in page.text
    assert 'id="page-indicator"' in page.text
    assert "debug-badge" in script.text
    assert 'method: "DELETE"' in script.text
    assert 'params.set("offset"' in script.text
    assert "knownExperiments" in script.text


def test_detail_page_uses_local_plotly_and_live_log_script(tmp_path):
    client, run_id = _client(tmp_path)

    page = client.get(f"/runs/{run_id}")
    script = client.get("/static/run-detail.js")

    assert page.status_code == script.status_code == 200
    assert run_id in page.text
    assert 'src="/assets/plotly.min.js"' in page.text
    assert "cdn.plot.ly" not in page.text
    assert "pollLog" in script.text and "Plotly.newPlot" in script.text
    assert "gate.title =" in script.text
    assert 'gate.append(node("span"' not in script.text
    assert "Сбросить масштаб" in script.text
    assert 'dragmode: "pan"' in script.text
    assert "Скролл двигает страницу" not in script.text
    assert "Zoom / pan активны" not in script.text
    assert 'id="debug-sessions-panel"' in page.text
    assert 'id="delete-run"' in page.text
    assert "renderDebugSessions" in script.text
    assert "Отладочная сессия модели из" in script.text


def test_detail_page_contains_rerun_form_and_live_diff(tmp_path):
    client, run_id = _client(tmp_path)

    page = client.get(f"/runs/{run_id}")
    script = client.get("/static/run-detail.js")

    assert "Повторный запуск" in page.text
    assert "Поставить в очередь" in page.text
    assert f"/api/runs/${{encodedRunId}}/rerun" in script.text
    assert "renderRerunDiff" in script.text


def test_detail_page_contains_cooperative_run_controls(tmp_path):
    client, run_id = _client(tmp_path)

    page = client.get(f"/runs/{run_id}")
    script = client.get("/static/run-detail.js")
    css = client.get("/static/app.css")

    assert "Управление запуском" in page.text
    assert 'id="control-panel"' in page.text
    assert 'data-run-command="pause"' in page.text
    assert 'data-run-command="step" data-debug-step' in page.text
    assert 'id="delay-ms"' in page.text
    assert f"/api/runs/${{encodedRunId}}/control" in script.text
    assert "issueControl" in script.text
    assert "setupRunControl" in script.text
    assert "debugCapabilities" in script.text
    assert 'renderer !== "xor_neurons_v1"' in script.text
    assert ".control-actions" in css.text


def test_detail_page_contains_model_graph_layer_and_tensor_views(tmp_path):
    client, run_id = _client(tmp_path)

    page = client.get(f"/runs/{run_id}")
    script = client.get("/static/run-detail.js")
    css = client.get("/static/app.css")

    assert "Модель" in page.text
    assert 'id="model-graph"' in page.text
    assert 'id="layer-inspector"' in page.text
    assert "validateModelPayload" in script.text
    assert "renderSelectedLayer" in script.text
    assert "renderTensorTable" in script.text
    assert "renderTensorHeatmap" in script.text
    assert "Батч снимка" in script.text
    assert "Матрица weight" in script.text
    assert "weight ${formatShape(weight.shape)}" in script.text
    assert "/artifacts/${encodedPath}" in script.text
    assert "Это нормальное состояние для старых прогонов" in script.text
    assert "torch.load" not in script.text
    assert ".model-node-selected" in css.text
    assert ".tensor-values-table" in css.text
    assert ".tensor-heatmap-cell" in css.text


def test_detail_page_contains_interactive_xor_controls_and_neuron_graph(tmp_path):
    client, run_id = _client(tmp_path)

    page = client.get(f"/runs/{run_id}")
    script = client.get("/static/run-detail.js")
    css = client.get("/static/app.css")

    assert "Интерактивный XOR" in page.text
    assert 'id="start-xor-debug"' in page.text
    assert 'data-xor-input="0,1"' in page.text
    assert 'id="xor-network"' in page.text
    assert f"/api/runs/${{encodedRunId}}/debug" in script.text
    assert 'issueControl("set_input"' in script.text
    assert 'submit.textContent = xorInputLoading ? "Подаём вход…" : "Подать вход"' in script.text
    assert "controlRevision" in script.text
    assert "xorForwardPending" in script.text
    assert "finally" in script.text
    assert "renderXorNetwork" in script.text
    assert "/events?after_seq=" in script.text
    assert ".xor-neuron-active" in css.text


def test_detail_page_contains_xor_training_step_animation(tmp_path):
    client, run_id = _client(tmp_path)

    page = client.get(f"/runs/{run_id}")
    script = client.get("/static/run-detail.js")
    css = client.get("/static/app.css")

    assert "Обучающий шаг XOR" in page.text
    assert 'id="xor-training-play"' in page.text
    assert 'id="xor-weight-before"' in page.text
    assert 'id="xor-weight-delta"' in page.text
    assert 'id="xor-weight-after"' in page.text
    assert 'id="xor-decision-boundary"' in page.text
    assert "validateXorTrainingSnapshot" in script.text
    assert 'event.type === "xor_train_step"' in script.text
    assert "renderDecisionBoundary" in script.text
    assert "toggleXorTrainingPlayback" in script.text
    assert 'payload.config?.experiment === "xor_backprop"' in script.text
    assert ".xor-update-grid" in css.text
    assert ".xor-decision-boundary" in css.text


def test_comparison_page_uses_local_plotly_and_comparison_api(tmp_path):
    client, _ = _client(tmp_path)

    page = client.get("/compare")
    script = client.get("/static/compare.js")
    css = client.get("/static/app.css")

    assert page.status_code == script.status_code == css.status_code == 200
    assert "Сравнение запусков" in page.text
    assert 'src="/assets/plotly.min.js"' in page.text
    assert "/static/compare.js" in page.text
    assert "cdn.plot.ly" not in page.text
    assert "/api/compare?" in script.text
    assert "connectgaps: false" in script.text
    assert "missing_run_ids" in script.text
    assert 'width: isBaseline ? 5 : 2' in script.text
    assert 'symbol: isBaseline ? "circle-open" : "circle"' in script.text
    assert "gate.title =" in script.text
    assert 'gate.append(node("span"' not in script.text
    assert "Сбросить масштаб" in script.text
    assert 'dragmode: "pan"' in script.text
    assert "Скролл двигает страницу" not in script.text
    assert "Zoom / pan активны" not in script.text
    assert "grid-template-columns: minmax(0, .8fr) minmax(0, 1.4fr) auto" in css.text
    assert ".compare-picker select { width: 100%; min-width: 0" in css.text


def test_plotly_bundle_is_served_locally_and_cached(tmp_path):
    client, _ = _client(tmp_path)

    response = client.get("/assets/plotly.min.js")

    assert response.status_code == 200
    assert len(response.content) > 1_000_000
    assert b"Plotly" in response.content[:10_000]
    assert "immutable" in response.headers["cache-control"]


def test_missing_run_page_is_404(tmp_path):
    client, _ = _client(tmp_path)

    response = client.get("/runs/missing")

    assert response.status_code == 404


def test_catalog_api_exposes_dirty_for_explicit_warning(tmp_path):
    client, _ = _client(tmp_path)

    item = client.get("/api/runs").json()["items"][0]

    assert item["dirty"] is True
