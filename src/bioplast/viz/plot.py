"""Стандартный набор графиков по `runs/<id>/metrics.json`.

Скрипт ничего не знает про конкретный эксперимент: ключи метрик вида
`группа/имя` группируются по префиксу, каждая группа — отдельный PNG.
Добавление новой метрики в эксперименте автоматически добавляет её на график.

    uv run python -m bioplast.viz runs/xor-bp-s0-3cae8a
    uv run python -m bioplast.viz runs --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # прогоны идут фоном, интерактивный бэкенд не нужен
import matplotlib.pyplot as plt

from bioplast.diagnostics.metrics import group_series

# Метрики нормы и разреженности бывают на разных порядках; логарифм по умолчанию
# там, где он почти всегда уместен.
LOG_SCALE_GROUPS = {"w_norm", "grad_norm", "act_rms", "act_max"}


def plot_run(run_dir: Path | str, out_dir: Path | str | None = None) -> list[Path]:
    """Нарисовать все группы метрик одного прогона. Возвращает пути к PNG."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"нет {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = metrics.get("epochs") or []
    if not rows:
        return []

    step_key = next((k for k in ("epoch", "step") if k in rows[0]), "epoch")
    grouped = group_series(rows, step_key=step_key)
    out_dir = Path(out_dir) if out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for group, series in sorted(grouped.items()):
        path = out_dir / f"{group}.png"
        _plot_group(group, series, step_key, metrics.get("run_id", run_dir.name), path)
        written.append(path)
    return written


def _plot_group(group, series, step_key, run_id, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    for key, (steps, values) in sorted(series.items()):
        label = key.split("/", 1)[-1] if "/" in key else key
        ax.plot(steps, values, marker="o", markersize=2.5, linewidth=1.4, label=label)

    ax.set_xlabel(step_key)
    ax.set_ylabel(group)
    ax.set_title(f"{group} — {run_id}", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    if group in LOG_SCALE_GROUPS and _all_positive(series):
        ax.set_yscale("log")
    if len(series) > 1:
        ax.legend(fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _all_positive(series) -> bool:
    return all(value > 0 for _, values in series.values() for value in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bioplast.viz.plot")
    parser.add_argument("paths", nargs="+", type=Path, help="папки прогонов")
    parser.add_argument("--all", action="store_true", help="трактовать пути как runs/ целиком")
    args = parser.parse_args(argv)

    targets: list[Path] = []
    for path in args.paths:
        if args.all:
            targets.extend(sorted(p.parent for p in path.glob("*/metrics.json")))
        else:
            targets.append(path)

    for run_dir in targets:
        written = plot_run(run_dir)
        print(f"{run_dir}: {len(written)} png")
        for item in written:
            print(f"  {item}")
    return 0
