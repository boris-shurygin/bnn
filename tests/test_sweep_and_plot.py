"""Разворачивание свипа и графики уровня 0."""

import json

from bioplast.runner.sweep import expand, write_queue, _parse_set
from bioplast.viz import plot_run

BASE = {
    "session": "0.0",
    "dataset": "toy",
    "model": "sweep",
    "experiment": "_selftest",
    "seed": 0,
    "K": 1,
}


def test_expand_is_cartesian_product():
    configs = expand(BASE, {"seed": [0, 1, 2], "K": [1, 10]})

    assert len(configs) == 6
    assert {(c["seed"], c["K"]) for c in configs} == {
        (s, k) for s in (0, 1, 2) for k in (1, 10)
    }
    assert BASE["seed"] == 0  # базовый конфиг не мутирован


def test_expand_without_grid_returns_base():
    assert expand(BASE, {}) == [BASE]


def test_parse_set_reads_json_values():
    assert _parse_set("K=1,2,4") == ("K", [1, 2, 4])
    assert _parse_set("device=cpu,cuda") == ("device", ["cpu", "cuda"])


def test_write_queue_gives_distinct_names(tmp_path):
    written = write_queue(expand(BASE, {"seed": [0, 1, 2]}), tmp_path)

    assert len({p.name for p in written}) == 3
    assert json.loads(written[0].read_text(encoding="utf-8"))["experiment"] == "_selftest"


def test_plot_run_writes_png_per_group(tmp_path):
    metrics = {
        "run_id": "test",
        "epochs": [
            {"epoch": 0, "loss/train": 1.0, "w_norm/fc1": 2.0, "acc/test": 0.1},
            {"epoch": 1, "loss/train": 0.5, "w_norm/fc1": 2.5, "acc/test": 0.8},
        ],
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    written = plot_run(tmp_path)

    assert {p.name for p in written} == {"loss.png", "w_norm.png", "acc.png"}
    assert all(p.stat().st_size > 0 for p in written)


def test_plot_run_survives_empty_metrics(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"epochs": []}), encoding="utf-8")

    assert plot_run(tmp_path) == []
