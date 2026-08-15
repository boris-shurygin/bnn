"""Один прогон: конфиг → `runs/<id>/{config,metrics}.json` + `run.log`.

Контракт эксперимента: модуль в `experiments/` определяет функцию

    def run(config: dict, ctx: RunContext) -> dict

Возвращённый словарь попадает в `metrics.json` как `final`. Всё, что нужно
записать по эпохам, эксперимент кладёт в `ctx.metrics`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bioplast.diagnostics.metrics import MetricsRecorder
from bioplast.runner.contracts import (
    ContractError,
    RunManifest,
    RunStatus,
    load_run_manifest,
    utc_offset_iso,
    write_run_manifest,
)

LOGGER_NAME = "bioplast.run"
# Метка времени первой компонентой имени: лексикографическая сортировка папок
# в `runs/` совпадает с хронологической.
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"


def project_root() -> Path:
    """Корень проекта — та папка, где лежит `experiments/`."""
    cwd = Path.cwd()
    if (cwd / "experiments").is_dir():
        return cwd
    # src/bioplast/runner/run.py → src/bioplast/runner → src/bioplast → src → корень
    return Path(__file__).resolve().parents[3]


def _canonical(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _slug(value: Any, max_len: int = 20) -> str:
    """Компонента имени папки: только `a-z0-9.`, остальное — дефис."""
    text = re.sub(r"[^a-z0-9.]+", "-", str(value).strip().lower()).strip("-")
    return text[:max_len] or "x"


def describe_config(config: dict[str, Any]) -> list[str]:
    """Читаемые компоненты имени: сессия, данные, модель, сид, комментарий.

    Раннер не может вывести «какая это модель» из кода эксперимента, поэтому
    поля берутся из конфига. Отсутствующие просто пропускаются — старые конфиги
    остаются рабочими.
    """
    parts: list[str] = []
    if config.get("session"):
        parts.append(f"ses{_slug(config['session'], 8)}")
    if config.get("dataset"):
        parts.append(_slug(config["dataset"], 12))
    parts.append(_slug(config.get("model") or config.get("name") or config.get("experiment")))
    parts.append(f"s{_slug(config.get('seed', 0), 4)}")
    if config.get("tag"):
        parts.append(_slug(config["tag"]))
    return parts


def make_run_id(config: dict[str, Any], now: datetime | None = None) -> str:
    """Имя папки прогона: `20260808-143012-ses0.4-mnist-mlp-784-256-10-s0-bp`.

    Метка времени идёт первой, поэтому сортировка по имени = сортировка по
    времени запуска. Длина под контролем: лимит пути в 260 символов на Windows
    реален, а внутри папки ещё `plots/act_sparsity.png`.
    """
    if config.get("id"):
        return str(config["id"])
    stamp = (now or datetime.now()).strftime(RUN_STAMP_FORMAT)
    return "-".join([stamp, *describe_config(config)])


def config_slug(config: dict[str, Any]) -> str:
    """Детерминированное имя для файла конфига в очереди.

    Без метки времени: перегенерация свипа должна перезаписывать те же файлы,
    а не плодить дубли. Хеш отделяет конфиги, различающиеся полями, которых
    нет в читаемой части имени.
    """
    digest = hashlib.sha1(_canonical(config).encode("utf-8")).hexdigest()[:6]
    return "-".join([*describe_config(config), digest])


def resolve_device(spec: str) -> str:
    """`auto` → cuda при наличии, иначе cpu. Остальное — как указано."""
    import torch

    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


@dataclass
class RunContext:
    """Всё, что эксперимент получает от раннера помимо самого конфига."""

    run_id: str
    run_dir: Path
    device: str
    seed: int
    log: logging.Logger
    metrics: MetricsRecorder = field(default_factory=MetricsRecorder)

    def artifact(self, name: str) -> Path:
        """Путь для побочного файла прогона (веса, картинка, дамп)."""
        return self.run_dir / name


class _ConsoleFormatter(logging.Formatter):
    """Не даёт Windows-консоли уронить emit на символе вне её code page."""

    def __init__(self, fmt: str, encoding: str | None) -> None:
        super().__init__(fmt)
        self.encoding = encoding or "utf-8"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record).replace("→", "->").replace("←", "<-")
        return text.encode(self.encoding, errors="replace").decode(self.encoding)


def _setup_logging(run_dir: Path, run_id: str, label: str) -> logging.Logger:
    logger = logging.getLogger(f"{LOGGER_NAME}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    # Полная дата, а не только время: прогоны разбираются через неделю после
    # запуска, и «14:30:12» без дня сам по себе бесполезен.
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # В консоли префиксом идёт короткая метка без даты: при `--workers 4`
    # строки перемежаются, и различать надо конфигурации, а не секунды запуска.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        _ConsoleFormatter(f"[{label}] %(message)s", getattr(sys.stdout, "encoding", None))
    )
    logger.addHandler(stream_handler)

    return logger


def _log_environment(logger: logging.Logger, device: str, spec: str) -> dict[str, Any]:
    """Печать окружения в лог каждого прогона.

    На Windows `pip install torch` молча ставит CPU-сборку, и обнаруживается это
    не по ошибке, а по времени прогона. Поэтому — в каждый лог, а не однократно.

    Две строки, а не одна: доступность CUDA и выбор устройства — независимые
    факты. XOR намеренно считается на CPU при исправной карте, и «cuda_available
    = True → вычисления на CPU» читалось бы как следствие, а не как совпадение.
    """
    import torch

    on_gpu = device.startswith("cuda")
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda,  # None у CPU-сборки — главный признак
        "device": device,
        "device_spec": spec,
        # имя карты пишется только если прогон реально на ней: иначе строка
        # «device = cpu (RTX 3060)» читается как «работаем на 3060»
        "device_name": torch.cuda.get_device_name(device) if on_gpu else None,
        "python": sys.version.split()[0],
    }

    logger.info(
        "torch %s (cuda build: %s), python %s",
        info["torch"],
        info["cuda_build"] or "НЕТ — CPU-сборка",
        info["python"],
    )
    logger.info(
        "CUDA в системе: %s (torch.cuda.is_available() = %s)",
        "доступна" if info["cuda_available"] else "НЕДОСТУПНА",
        info["cuda_available"],
    )
    logger.info(
        "устройство прогона: %s — %s",
        info["device_name"] or "CPU",
        "выбрано автоматически" if spec == "auto" else f"задано в конфиге (device={spec})",
    )
    if on_gpu and not torch.cuda.is_available():
        logger.warning("конфиг просит cuda, но torch её не видит — прогон упадёт")
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)
    return info


def git_provenance(root: Path | None = None) -> dict[str, Any]:
    """Чем посчитан результат: коммит, ветка, признак незакоммиченных правок.

    `runs/` не версионируется, поэтому без этой записи через месяц по строке
    «MNIST дал 94% при K=2» нечем восстановить код. Ветки делают дыру шире:
    после `git branch -D` неудачной гипотезы её коммиты недостижимы, и хеш
    остаётся единственной ниточкой (обычно к тегу `exp/<имя>`).

    Флаг `dirty` не менее важен, чем хеш: при отладке правки не коммитятся, и
    хеш врёт ровно тогда, когда точность нужнее всего.
    """
    root = root or project_root()
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode != 0:
            return {}
        commit, _, branch = head.stdout.strip().partition("\n")

        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}  # git не установлен или репозитория нет — не повод ронять прогон

    changed = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    info: dict[str, Any] = {"commit": commit, "branch": branch, "dirty": bool(changed)}
    if changed:
        info["dirty_files"] = sorted(changed)[:20]
    return info


def _log_provenance(logger: logging.Logger, info: dict[str, Any]) -> None:
    if not info:
        logger.warning("git недоступен — прогон не привязан к состоянию кода")
        return
    logger.info("код: %s @ %s", info["commit"][:10], info["branch"])
    if info["dirty"]:
        logger.warning(
            "в дереве незакоммиченные правки (%d файлов) — прогон точно не воспроизводим",
            len(info.get("dirty_files", [])),
        )


def _gpu_usage(device: str, logger: logging.Logger) -> dict[str, Any]:
    """Пиковая VRAM за прогон — доказательство, что на карте что-то считалось.

    Ноль при `device=cuda` означает, что ни один тензор на карту не попал:
    эксперимент проигнорировал `ctx.device` и всё посчитал на CPU. Молчаливая
    ошибка, которая обнаруживается иначе только по времени прогона.
    """
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {}

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / 1024**2
    if peak_bytes == 0:
        logger.warning("device=%s, но пиковая VRAM = 0 — на карте не считалось ничего", device)
    else:
        logger.info("пиковая VRAM: %.1f МБ", peak_mb)
    return {"vram_peak_mb": round(peak_mb, 1)}


def _apply_backends(config: dict[str, Any], logger: logging.Logger) -> None:
    """TF32 включаем, AMP/fp16 не трогаем принципиально (см. CLAUDE.md)."""
    import torch

    allow_tf32 = bool(config.get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    logger.info("tf32 = %s, dtype = fp32 (fp16/AMP не используется)", allow_tf32)


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_experiment(name: str):
    """`xor_backprop` → `experiments.xor_backprop`; с точкой — импорт как есть."""
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module_name = name if "." in name else f"experiments.{name}"
    module = importlib.import_module(module_name)
    if not hasattr(module, "run"):
        raise AttributeError(f"{module_name} не определяет функцию run(config, ctx)")
    return module


def _unique_dir(runs_dir: Path, run_id: str) -> Path:
    """Развести прогоны, стартовавшие в одну секунду (очередь на 4 воркерах)."""
    candidate = runs_dir / run_id
    for suffix in range(2, 100):
        if not candidate.exists():
            return candidate
        candidate = runs_dir / f"{run_id}-{suffix}"
    raise RuntimeError(f"не удалось подобрать имя папки для {run_id}")


def _reserve_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Атомарно резервирует имя, в том числе при параллельной очереди."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    candidate = runs_dir / run_id
    for suffix in range(2, 100):
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = runs_dir / f"{run_id}-{suffix}"
    raise RuntimeError(f"не удалось подобрать имя папки для {run_id}")


def validate_run_config(config: dict[str, Any]) -> None:
    """Проверяет общий контракт конфига до создания каталога прогона.

    Семантика параметров принадлежит конкретному эксперименту, но раннер может
    заранее отсеять конфиги, которые иначе упадут ещё до его импорта.
    """
    if not isinstance(config, dict):
        raise TypeError("конфиг должен быть JSON-объектом")
    experiment = config.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment должен быть непустой строкой")
    if "id" in config:
        run_id = config["id"]
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in {".", ".."}
            or "/" in run_id
            or "\\" in run_id
        ):
            raise ValueError("id должен быть именем одного каталога")
    try:
        json.dumps(config, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"конфиг должен содержать только конечные JSON-значения: {exc}") from exc
    try:
        int(config.get("seed", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("seed должен быть целым числом") from exc
    if not isinstance(config.get("device", "auto"), str):
        raise ValueError("device должен быть строкой")


def prepare_run(
    config: dict[str, Any] | Path | str,
    runs_dir: Path | str | None = None,
    *,
    parent_run_id: str | None = None,
) -> Path:
    """Резервирует новый прогон и записывает манифест со статусом `queued`."""
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("конфиг должен быть JSON-объектом")
    prepared = deepcopy(config)
    if parent_run_id is not None:
        prepared["parent_run_id"] = parent_run_id
    validate_run_config(prepared)

    root = Path(runs_dir) if runs_dir else project_root() / "runs"
    queued_at = datetime.now().astimezone()
    run_dir = _reserve_run_dir(root, make_run_id(prepared, queued_at))
    run_id = run_dir.name
    (run_dir / "config.json").write_text(
        json.dumps(prepared, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_run_manifest(
        run_dir,
        RunManifest(
            run_id=run_id,
            status=RunStatus.QUEUED,
            experiment=prepared["experiment"],
            started_at=None,
            updated_at=queued_at.isoformat(timespec="seconds"),
            parent_run_id=(
                str(prepared["parent_run_id"])
                if prepared.get("parent_run_id") is not None
                else None
            ),
        ),
    )
    return run_dir


def fail_prepared_run(run_dir: Path | str, error: str) -> None:
    """Фиксирует сбой очереди/воркера, чтобы прогон не завис в active-статусе."""
    run_dir = Path(run_dir).resolve()
    manifest = load_run_manifest(run_dir)
    if manifest.status.terminal:
        return
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        metrics_path.write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "status": "failed",
                    "config": config,
                    "error": error,
                    "epochs": [],
                    "final": {},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    write_run_manifest(run_dir, manifest.finish(RunStatus.FAILED, 0.0))


def run_prepared(run_dir: Path | str) -> Path:
    """Выполняет атомарно подготовленный прогон из существующей очереди."""
    run_dir = Path(run_dir).resolve()
    manifest = load_run_manifest(run_dir)
    if manifest.run_id != run_dir.name:
        raise ContractError("run_id подготовленного прогона не совпадает с каталогом")
    if manifest.status is not RunStatus.QUEUED:
        raise ContractError(f"ожидался статус queued, получен {manifest.status.value}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    validate_run_config(config)

    started_at = datetime.now().astimezone()
    manifest = RunManifest(
        run_id=manifest.run_id,
        status=RunStatus.RUNNING,
        experiment=manifest.experiment,
        started_at=started_at.isoformat(timespec="seconds"),
        updated_at=started_at.isoformat(timespec="seconds"),
        parent_run_id=manifest.parent_run_id,
        artifacts=manifest.artifacts,
    )
    write_run_manifest(run_dir, manifest)
    run_id = run_dir.name

    logger = _setup_logging(run_dir, run_id, "-".join(describe_config(config)))
    device_spec = str(config.get("device", "auto"))
    device = resolve_device(device_spec)
    seed = int(config.get("seed", 0))

    logger.info("старт: %s", config.get("experiment"))
    git = git_provenance()
    _log_provenance(logger, git)
    env = _log_environment(logger, device, device_spec)
    _apply_backends(config, logger)
    _seed_everything(seed)
    logger.info("seed = %d", seed)

    ctx = RunContext(run_id=run_id, run_dir=run_dir, device=device, seed=seed, log=logger)

    started = time.perf_counter()
    status, final, error = "ok", {}, None
    try:
        module = load_experiment(str(config["experiment"]))
        final = module.run(config, ctx) or {}
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        logger.error("прогон упал:\n%s", error)
    duration = time.perf_counter() - started
    env.update(_gpu_usage(device, logger))

    metrics = {
        "run_id": run_id,
        "status": status,
        "started_at": started_at.isoformat(timespec="seconds"),
        "git": git,
        "config": config,
        "env": env,
        "duration_sec": round(duration, 3),
        "epochs": ctx.metrics.rows,
        "final": final,
    }
    if error:
        metrics["error"] = error

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    final_status = RunStatus.COMPLETED if status == "ok" else RunStatus.FAILED
    write_run_manifest(
        run_dir,
        manifest.finish(final_status, duration, finished_at=utc_offset_iso()),
    )
    logger.info("готово: %s за %.2f с → %s", status, duration, run_dir)

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    return run_dir


def run_config(
    config: dict[str, Any] | Path | str,
    runs_dir: Path | str | None = None,
) -> Path:
    """Выполнить один прогон. Возвращает папку прогона.

    Исключение эксперимента не пробрасывается наружу: оно записывается в
    `metrics.json` со `status: failed`, чтобы одна упавшая конфигурация не
    роняла очередь целиком.
    """
    return run_prepared(prepare_run(config, runs_dir))
