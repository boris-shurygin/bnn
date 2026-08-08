"""Раннер: контракт `конфиг → runs/<id>/`."""

import json
import re
from datetime import datetime

import pytest

from bioplast.runner import config_slug, make_run_id, run_config, run_queue
from bioplast.runner.queue import collect_configs

BASE = {
    "session": "0.0",
    "dataset": "toy",
    "model": "selftest",
    "experiment": "_selftest",
    "device": "cpu",
    "seed": 0,
    "steps": 3,
}
STAMP = datetime(2026, 8, 8, 14, 30, 12)


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def test_run_writes_expected_files(tmp_path):
    run_dir = run_config(dict(BASE), runs_dir=tmp_path)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "run.log").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "ok"
    assert metrics["final"]["steps"] == 3
    assert len(metrics["epochs"]) == 3
    assert metrics["config"]["experiment"] == "_selftest"


def test_run_logs_device_availability(tmp_path):
    """Проверка cuda пишется в каждый прогон: на Windows легко получить CPU-сборку."""
    run_dir = run_config(dict(BASE), runs_dir=tmp_path)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "cuda_available" in metrics["env"]
    assert metrics["env"]["device"] == "cpu"

    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" in log
    # доступность CUDA и выбор устройства — независимые факты, каждый своей
    # строкой: XOR намеренно идёт на CPU при исправной карте
    assert "устройство прогона: CPU — задано в конфиге (device=cpu)" in log
    assert metrics["env"]["device_spec"] == "cpu"
    # в логе полная дата, а не только время: разбирать прогон будут не сегодня
    assert re.search(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", log, re.MULTILINE)


def test_cpu_run_reports_no_gpu_memory(tmp_path):
    """На CPU-прогоне поля VRAM быть не должно — иначе оно ничего не доказывает."""
    run_dir = run_config(dict(BASE), runs_dir=tmp_path)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "vram_peak_mb" not in metrics["env"]
    assert metrics["env"]["device_name"] is None  # имя карты не пишется на CPU-прогоне


@pytest.mark.skipif(not _cuda_available(), reason="нет CUDA")
def test_gpu_run_records_nonzero_vram(tmp_path):
    """Ненулевая пиковая VRAM — проверяемое доказательство работы на карте."""
    run_dir = run_config({**BASE, "device": "cuda", "touch_device": True}, runs_dir=tmp_path)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "ok"
    assert metrics["env"]["vram_peak_mb"] > 0
    assert "3060" in metrics["env"]["device_name"] or metrics["env"]["device_name"]


@pytest.mark.skipif(not _cuda_available(), reason="нет CUDA")
def test_gpu_run_warns_when_nothing_touched_card(tmp_path):
    """Эксперимент, проигнорировавший ctx.device, должен себя выдать."""
    run_dir = run_config({**BASE, "device": "cuda"}, runs_dir=tmp_path)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["env"]["vram_peak_mb"] == 0
    assert "на карте не считалось ничего" in (run_dir / "run.log").read_text(encoding="utf-8")


def test_failed_run_does_not_raise(tmp_path):
    """Упавшая конфигурация не должна ронять очередь целиком."""
    run_dir = run_config({**BASE, "fail": True}, runs_dir=tmp_path)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert "падение по требованию" in metrics["error"]


def test_run_id_is_readable_and_sorts_by_time():
    run_id = make_run_id(dict(BASE), STAMP)

    assert run_id == "20260808-143012-ses0.0-toy-selftest-s0"
    assert len(run_id) < 80  # лимит пути в 260 символов на Windows
    # лексикографическая сортировка обязана совпадать с хронологической
    later = make_run_id(dict(BASE), datetime(2026, 8, 8, 14, 30, 13))
    next_day = make_run_id(dict(BASE), datetime(2026, 8, 9, 9, 0, 0))
    assert sorted([next_day, later, run_id]) == [run_id, later, next_day]


def test_run_id_carries_tag_and_seed():
    run_id = make_run_id({**BASE, "seed": 7, "tag": "без baseline"}, STAMP)

    assert run_id.endswith("-s7-baseline")  # кириллица выброшена слагификацией
    assert "ses0.0" in run_id and "toy" in run_id


def test_run_dirs_do_not_collide_within_one_second(tmp_path):
    """Очередь на 4 воркерах стартует прогоны в одну и ту же секунду."""
    first = run_config(dict(BASE), runs_dir=tmp_path)
    second = run_config(dict(BASE), runs_dir=tmp_path)

    assert first != second
    assert second.name.startswith(first.name)


def test_config_slug_is_deterministic_and_timeless():
    """Имя файла в очереди не должно зависеть от времени: иначе перегенерация
    свипа плодит дубли вместо перезаписи."""
    slug = config_slug(dict(BASE))

    assert slug == config_slug(dict(BASE))
    assert slug != config_slug({**BASE, "steps": 4})
    assert not slug[0].isdigit()  # без метки времени
    assert slug.startswith("ses0.0-toy-selftest-s0-")


def test_explicit_id_wins():
    assert make_run_id({**BASE, "id": "custom"}, STAMP) == "custom"


def test_collect_configs_expands_directories(tmp_path):
    for i in range(2):
        (tmp_path / f"cfg{i}.json").write_text(json.dumps(BASE), encoding="utf-8")
    (tmp_path / "note.txt").write_text("не конфиг", encoding="utf-8")

    found = collect_configs([tmp_path])

    assert [p.name for p in found] == ["cfg0.json", "cfg1.json"]


def test_collect_configs_reports_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_configs([tmp_path / "нет-такого.json"])


def test_queue_runs_every_config(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    for steps in (2, 5):
        path = queue_dir / f"cfg{steps}.json"
        path.write_text(json.dumps({**BASE, "steps": steps}), encoding="utf-8")

    results = run_queue([queue_dir], workers=1, runs_dir=tmp_path / "runs")

    assert len(results) == 2
    assert {r["status"] for r in results} == {"ok"}
