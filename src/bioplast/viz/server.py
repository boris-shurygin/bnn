"""Owned local-server command for the run visualizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from bioplast.viz.api import create_app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bioplast.viz serve",
        description="Запустить локальный стенд прогонов с управляемым lifecycle.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="каталог runs (по умолчанию BIOPLAST_RUNS_DIR или ./runs)",
    )
    parser.add_argument("--port", type=int, default=8000, help="локальный TCP-порт")
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port должен лежать в диапазоне 1–65535")
    runs_dir = args.runs_dir.resolve() if args.runs_dir is not None else None
    app = create_app(runs_dir)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0
