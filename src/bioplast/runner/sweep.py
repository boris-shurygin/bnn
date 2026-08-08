"""Разворачивание базового конфига в очередь: декартово произведение по сетке.

Свипы в этом проекте маленькие и частые (первый — `K = 1/2/4/8/10` в блоке 2),
и всегда минимум по трём сидам: у хеббовских и WTA-схем разброс между сидами
заметный, одиночный прогон вводит в заблуждение.

    uv run python -m bioplast.runner.sweep configs/xor_backprop.json \
        --set seed=0,1,2 --out queue/
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from bioplast.runner.run import config_slug


def expand(base: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Базовый конфиг × сетка значений → список конфигов."""
    if not grid:
        return [deepcopy(base)]
    keys = list(grid)
    configs = []
    for combo in product(*(grid[key] for key in keys)):
        config = deepcopy(base)
        config.pop("id", None)  # id пересчитывается под конкретную комбинацию
        for key, value in zip(keys, combo):
            config[key] = value
        configs.append(config)
    return configs


def write_queue(configs: list[dict[str, Any]], out_dir: Path | str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for config in configs:
        path = out_dir / f"{config_slug(config)}.json"
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(path)
    return written


def _parse_set(item: str) -> tuple[str, list[Any]]:
    """`K=1,2,4` → `("K", [1, 2, 4])`. Значения разбираются как JSON, иначе строка."""
    key, _, raw = item.partition("=")
    if not key or not raw:
        raise ValueError(f"ожидался вид ключ=знач1,знач2, получено: {item!r}")
    values = []
    for chunk in raw.split(","):
        try:
            values.append(json.loads(chunk))
        except json.JSONDecodeError:
            values.append(chunk)
    return key, values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bioplast.runner.sweep")
    parser.add_argument("base", type=Path, help="базовый конфиг")
    parser.add_argument("--set", action="append", default=[], metavar="КЛЮЧ=A,B,C")
    parser.add_argument("--out", type=Path, default=Path("queue"), help="куда писать очередь")
    args = parser.parse_args(argv)

    base = json.loads(args.base.read_text(encoding="utf-8"))
    grid = dict(_parse_set(item) for item in args.set)
    written = write_queue(expand(base, grid), args.out)

    print(f"конфигов записано: {len(written)} → {args.out}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
