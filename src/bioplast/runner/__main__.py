"""CLI раннера.

    uv run python -m bioplast.runner configs/xor_backprop.json
    uv run python -m bioplast.runner queue/ --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bioplast.runner.queue import collect_configs, run_queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bioplast.runner", description="прогон конфигов")
    parser.add_argument("paths", nargs="+", type=Path, help="конфиги или папки с конфигами")
    parser.add_argument(
        "--workers", type=int, default=1, help="сколько прогонов держать одновременно"
    )
    parser.add_argument("--runs-dir", type=Path, default=None, help="куда писать (по умолч. runs/)")
    parser.add_argument("--dry-run", action="store_true", help="только показать список конфигов")
    args = parser.parse_args(argv)

    configs = collect_configs(args.paths)
    print(f"конфигов в очереди: {len(configs)}, воркеров: {args.workers}")
    for cfg in configs:
        print(f"  {cfg}")
    if args.dry_run:
        return 0

    results = run_queue(configs, workers=args.workers, runs_dir=args.runs_dir)

    print("\nсводка:")
    failed = 0
    for item in sorted(results, key=lambda r: str(r.get("config"))):
        status = item.get("status")
        failed += status != "ok"
        print(f"  {status:>7}  {item.get('duration_sec', '-'):>8}s  {item.get('run_dir', item['config'])}")
    print(f"\nвсего {len(results)}, неуспешных {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
