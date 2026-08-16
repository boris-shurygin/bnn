"""Сравнение сохранённых прогонов без пересчёта и интерполяции метрик."""

from __future__ import annotations

import json
import math
from typing import Any

from bioplast.viz.repository import RunRepository

MAX_CANDIDATES = 4
_MISSING = object()
_IGNORED_CONFIG_KEYS = {"parent_run_id"}


class RunComparisonError(ValueError):
    pass


def compare_runs(
    repository: RunRepository,
    baseline_id: str,
    candidate_ids: list[str],
) -> dict[str, Any]:
    if not baseline_id:
        raise RunComparisonError("baseline обязателен")
    if not candidate_ids:
        raise RunComparisonError("нужен хотя бы один candidate")
    if len(candidate_ids) > MAX_CANDIDATES:
        raise RunComparisonError(f"можно сравнить не больше {MAX_CANDIDATES} кандидатов")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RunComparisonError("candidate не должны повторяться")
    if baseline_id in candidate_ids:
        raise RunComparisonError("baseline не может одновременно быть candidate")

    baseline = _comparison_run(repository.get_run(baseline_id))
    candidates = [_comparison_run(repository.get_run(run_id)) for run_id in candidate_ids]
    runs = [baseline, *candidates]
    return {
        "baseline": _public_run(baseline),
        "candidates": [_public_run(run) for run in candidates],
        "config_diff": _config_diff(baseline, candidates),
        "final_metrics": _final_metrics(baseline, candidates),
        "metric_series": _metric_series(runs),
        "warnings": _compatibility_warnings(baseline, candidates),
    }


def _comparison_run(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload["manifest"]
    config = payload["config"]
    metrics = payload["metrics"]
    final = metrics.get("final", {})
    if not isinstance(final, dict):
        final = {}
    rows = metrics.get("epochs", [])
    if not isinstance(rows, list):
        rows = []
    step_key, series = _extract_series(rows)
    return {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "experiment": manifest.get("experiment"),
        "dataset": config.get("dataset"),
        "model": config.get("model") or config.get("name"),
        "seed": config.get("seed"),
        "parent_run_id": manifest.get("parent_run_id"),
        "config": config,
        "final": final,
        "step_key": step_key,
        "series": series,
    }


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        key: run[key]
        for key in (
            "run_id",
            "status",
            "experiment",
            "dataset",
            "model",
            "seed",
            "parent_run_id",
            "step_key",
        )
    }


def _config_diff(
    baseline: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    configs = [baseline["config"], *(run["config"] for run in candidates)]
    keys = sorted(
        set().union(*(config.keys() for config in configs)) - _IGNORED_CONFIG_KEYS
    )
    result = []
    for key in keys:
        baseline_value = baseline["config"].get(key, _MISSING)
        candidate_cells = [
            _diff_cell(run["run_id"], baseline_value, run["config"].get(key, _MISSING))
            for run in candidates
        ]
        if any(cell["differs"] for cell in candidate_cells):
            result.append(
                {
                    "key": key,
                    "baseline": _value_cell(baseline_value),
                    "candidates": candidate_cells,
                }
            )
    return result


def _diff_cell(run_id: str, baseline: Any, value: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        **_value_cell(value),
        "differs": not _same_json(baseline, value),
    }


def _final_metrics(
    baseline: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    finals = [baseline["final"], *(run["final"] for run in candidates)]
    keys = sorted(
        key
        for key in set().union(*(final.keys() for final in finals))
        if any(_finite_number(final.get(key, _MISSING)) is not None for final in finals)
    )
    result = []
    for key in keys:
        baseline_value = baseline["final"].get(key, _MISSING)
        candidate_cells = []
        for run in candidates:
            value = run["final"].get(key, _MISSING)
            cell = {"run_id": run["run_id"], **_value_cell(value)}
            before_number = _finite_number(baseline_value)
            after_number = _finite_number(value)
            if before_number is not None and after_number is not None:
                delta = after_number - before_number
                cell["delta"] = delta
                cell["relative_delta"] = (
                    delta / before_number if before_number != 0 else None
                )
            else:
                cell["delta"] = None
                cell["relative_delta"] = None
            candidate_cells.append(cell)
        result.append(
            {
                "key": key,
                "baseline": _value_cell(baseline_value),
                "candidates": candidate_cells,
            }
        )
    return result


def _metric_series(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(set().union(*(run["series"].keys() for run in runs)))
    result = []
    for key in keys:
        available = []
        missing = []
        for run in runs:
            points = run["series"].get(key)
            if points is None:
                missing.append(run["run_id"])
            else:
                available.append(
                    {
                        "run_id": run["run_id"],
                        "step_key": run["step_key"],
                        "x": points["x"],
                        "y": points["y"],
                    }
                )
        result.append(
            {
                "key": key,
                "group": key.split("/", 1)[0] if "/" in key else "scalar",
                "runs": available,
                "missing_run_ids": missing,
            }
        )
    return result


def _extract_series(rows: list[Any]) -> tuple[str | None, dict[str, dict[str, list[Any]]]]:
    objects = [row for row in rows if isinstance(row, dict)]
    if not objects:
        return None, {}
    step_key = next(
        (key for key in ("step", "epoch") if any(key in row for row in objects)),
        None,
    )
    if step_key is None:
        return None, {}

    series: dict[str, dict[str, list[Any]]] = {}
    for row in objects:
        if step_key not in row:
            continue
        x = row[step_key]
        for key, raw in row.items():
            value = _finite_number(raw)
            if key == step_key or value is None:
                continue
            points = series.setdefault(key, {"x": [], "y": []})
            points["x"].append(x)
            points["y"].append(value)
    return step_key, series


def _compatibility_warnings(
    baseline: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    warnings = []
    if baseline["step_key"] is None:
        warnings.append(
            {
                "type": "step_key_missing",
                "run_id": baseline["run_id"],
                "message": f"{baseline['run_id']}: в рядах нет оси step или epoch",
            }
        )
    for candidate in candidates:
        for field in ("experiment", "dataset", "model"):
            if not _same_json(baseline[field], candidate[field]):
                warnings.append(
                    {
                        "type": "identity_mismatch",
                        "run_id": candidate["run_id"],
                        "field": field,
                        "baseline": baseline[field],
                        "candidate": candidate[field],
                        "message": (
                            f"{candidate['run_id']}: {field} отличается от baseline "
                            f"({baseline[field]!r} → {candidate[field]!r})"
                        ),
                    }
                )
        if candidate["step_key"] is None:
            warnings.append(
                {
                    "type": "step_key_missing",
                    "run_id": candidate["run_id"],
                    "message": f"{candidate['run_id']}: в рядах нет оси step или epoch",
                }
            )
        elif (
            baseline["step_key"] is not None
            and baseline["step_key"] != candidate["step_key"]
        ):
            warnings.append(
                {
                    "type": "step_key_mismatch",
                    "run_id": candidate["run_id"],
                    "baseline": baseline["step_key"],
                    "candidate": candidate["step_key"],
                    "message": (
                        f"{candidate['run_id']}: ось {candidate['step_key']!r} "
                        f"не совпадает с baseline {baseline['step_key']!r}"
                    ),
                }
            )
    return warnings


def _value_cell(value: Any) -> dict[str, Any]:
    if value is _MISSING:
        return {"present": False, "value": None}
    return {"present": True, "value": value}


def _same_json(left: Any, right: Any) -> bool:
    if left is _MISSING or right is _MISSING:
        return left is right
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False
    )


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None
