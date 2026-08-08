"""Регистратор скалярных метрик по шагам.

Схема плоская: одна строка на замер, ключи вида `группа/имя`
(`loss/train`, `acc/test`, `w_norm/fc1`). Группировка по префиксу —
единственное, что нужно `plot.py`, чтобы рисовать графики, ничего не зная
про конкретный эксперимент.
"""

from __future__ import annotations

from typing import Any, Iterable


class MetricsRecorder:
    def __init__(self, step_key: str = "epoch") -> None:
        self.step_key = step_key
        self.rows: list[dict[str, Any]] = []

    def log(self, step: int, **scalars: Any) -> dict[str, Any]:
        """Записать замер. Значения приводятся к float/int, тензоры — к числам."""
        row: dict[str, Any] = {self.step_key: int(step)}
        for key, value in scalars.items():
            row[key] = _to_scalar(value)
        self.rows.append(row)
        return row

    def update(self, step: int, scalars: dict[str, Any]) -> dict[str, Any]:
        """То же, но словарём — удобно для ключей с `/`, которые не имена аргументов."""
        return self.log(step, **scalars)

    def last(self) -> dict[str, Any]:
        return self.rows[-1] if self.rows else {}

    def keys(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.rows:
            for key in row:
                seen.setdefault(key, None)
        return list(seen)

    def series(self, key: str) -> tuple[list[int], list[float]]:
        """Пара (шаги, значения) для одного ключа; строки без ключа пропускаются."""
        steps: list[int] = []
        values: list[float] = []
        for row in self.rows:
            if key in row and row[key] is not None:
                steps.append(row[self.step_key])
                values.append(row[key])
        return steps, values

    def __len__(self) -> int:
        return len(self.rows)


def _to_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (ValueError, RuntimeError):  # не-скалярный тензор
            return None
    return float(value)


def group_series(rows: list[dict[str, Any]], step_key: str = "epoch") -> dict[str, dict[str, tuple[list, list]]]:
    """Разложить строки метрик на `{группа: {ключ: (шаги, значения)}}`.

    Группа — часть ключа до первого `/`; ключи без `/` попадают в группу `scalar`.
    """
    grouped: dict[str, dict[str, tuple[list, list]]] = {}
    for row in rows:
        step = row.get(step_key)
        for key, value in row.items():
            if key == step_key or value is None or isinstance(value, str):
                continue
            group, _, _ = key.partition("/")
            group = group if "/" in key else "scalar"
            steps, values = grouped.setdefault(group, {}).setdefault(key, ([], []))
            steps.append(step)
            values.append(value)
    return grouped


def flatten(prefix: str, values: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """`flatten("w_norm", [("fc1", 1.2)])` → `{"w_norm/fc1": 1.2}`."""
    return {f"{prefix}/{name}": _to_scalar(value) for name, value in values}
