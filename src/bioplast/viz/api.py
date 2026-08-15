"""Read-only HTTP API локального визуализатора прогонов.

Запуск из корня проекта:

    uv run uvicorn bioplast.viz.api:app --reload
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from bioplast.runner import ContractError, RunStatus
from bioplast.runner.run import project_root
from bioplast.viz.repository import (
    ArtifactNotFound,
    RunNotFound,
    RunRepository,
    UnsafeRunPath,
)


def create_app(runs_dir: Path | str | None = None) -> FastAPI:
    root = Path(runs_dir) if runs_dir is not None else _default_runs_dir()
    repository = RunRepository(root)
    app = FastAPI(
        title="bioplast run visualizer",
        version="0.1.0",
        description="Read-only API файловых артефактов runs/<id>/.",
    )
    app.state.run_repository = repository

    @app.exception_handler(RunNotFound)
    @app.exception_handler(ArtifactNotFound)
    @app.exception_handler(UnsafeRunPath)
    async def not_found_handler(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ContractError)
    async def contract_error_handler(_request: Request, exc: ContractError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "runs_dir": str(repository.runs_dir)}

    @app.get("/api/runs")
    def list_runs(
        status: list[RunStatus] | None = Query(default=None),
        experiment: str | None = None,
        seed: int | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict:
        items, errors = repository.list_runs(
            statuses=status,
            experiment=experiment,
            seed=seed,
            started_after=started_after,
            started_before=started_before,
        )
        return {
            "items": items[offset : offset + limit],
            "total": len(items),
            "offset": offset,
            "limit": limit,
            "errors": errors,
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return repository.get_run(run_id)

    @app.get("/api/runs/{run_id}/metrics")
    def get_metrics(run_id: str) -> dict:
        return repository.get_metrics(run_id)

    @app.get("/api/runs/{run_id}/log")
    def get_log(
        run_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=65_536, ge=1, le=1_048_576),
    ) -> dict:
        return repository.read_log(run_id, offset=offset, limit=limit)

    @app.get("/api/runs/{run_id}/artifacts")
    def list_artifacts(run_id: str) -> dict[str, object]:
        return {
            "items": [item.to_dict() for item in repository.list_artifacts(run_id)]
        }

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def get_artifact(run_id: str, artifact_path: str) -> FileResponse:
        path = repository.resolve_artifact(run_id, artifact_path)
        return FileResponse(path)

    return app


def _default_runs_dir() -> Path:
    configured = os.environ.get("BIOPLAST_RUNS_DIR")
    return Path(configured) if configured else project_root() / "runs"


app = create_app()
