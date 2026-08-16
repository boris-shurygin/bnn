"""HTTP API локального визуализатора и очереди повторных прогонов.

Запуск из корня проекта:

    uv run uvicorn bioplast.viz.api:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from bioplast.runner import ContractError, RunCommandType, RunScheduler, RunStatus
from bioplast.runner.run import project_root
from bioplast.viz.comparison import RunComparisonError, compare_runs
from bioplast.viz.control import (
    RunControlConflict,
    RunControlService,
    RunControlValidationError,
)
from bioplast.viz.repository import (
    ArtifactNotFound,
    RunNotFound,
    RunRepository,
    UnsafeRunPath,
)
from bioplast.viz.rerun import (
    QueueSubmissionError,
    RerunService,
    RerunValidationError,
    RunNotRerunnable,
)


class RerunRequest(BaseModel):
    config: dict[str, Any]


class RunControlRequest(BaseModel):
    command: RunCommandType
    delay_ms: int | None = None


def create_app(
    runs_dir: Path | str | None = None,
    *,
    scheduler: Any | None = None,
) -> FastAPI:
    root = Path(runs_dir) if runs_dir is not None else _default_runs_dir()
    repository = RunRepository(root)
    scheduler = scheduler or RunScheduler(workers=_default_workers())
    reruns = RerunService(repository, scheduler)
    controls = RunControlService(repository, scheduler)
    viz_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=viz_dir / "templates")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        shutdown = getattr(scheduler, "shutdown", None)
        if shutdown is not None:
            shutdown()

    app = FastAPI(
        title="bioplast run visualizer",
        version="0.1.0",
        description="API файловых артефактов runs/<id>/ и повторных запусков.",
        lifespan=lifespan,
    )
    app.state.run_repository = repository
    app.state.run_scheduler = scheduler
    app.state.rerun_service = reruns
    app.state.run_control_service = controls
    app.mount("/static", StaticFiles(directory=viz_dir / "static"), name="static")

    @app.exception_handler(RunNotFound)
    @app.exception_handler(ArtifactNotFound)
    @app.exception_handler(UnsafeRunPath)
    async def not_found_handler(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ContractError)
    async def contract_error_handler(_request: Request, exc: ContractError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RerunValidationError)
    async def rerun_validation_handler(
        _request: Request, exc: RerunValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RunNotRerunnable)
    async def rerun_conflict_handler(
        _request: Request, exc: RunNotRerunnable
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(QueueSubmissionError)
    async def queue_submission_handler(
        _request: Request, exc: QueueSubmissionError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(RunComparisonError)
    async def comparison_validation_handler(
        _request: Request, exc: RunComparisonError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RunControlValidationError)
    async def control_validation_handler(
        _request: Request, exc: RunControlValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RunControlConflict)
    async def control_conflict_handler(
        _request: Request, exc: RunControlConflict
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "runs_dir": str(repository.runs_dir)}

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/runs")

    @app.get("/runs", response_class=HTMLResponse, include_in_schema=False)
    def runs_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="runs.html", context={})

    @app.get("/compare", response_class=HTMLResponse, include_in_schema=False)
    def compare_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="compare.html", context={})

    @app.get("/runs/{run_id}", response_class=HTMLResponse, include_in_schema=False)
    def run_page(request: Request, run_id: str) -> HTMLResponse:
        repository.resolve_run(run_id)
        return templates.TemplateResponse(
            request=request,
            name="run-detail.html",
            context={"run_id": run_id},
        )

    @app.get("/assets/plotly.min.js", include_in_schema=False)
    def plotly_javascript() -> Response:
        return Response(
            _plotly_javascript(),
            media_type="text/javascript",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

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

    @app.get("/api/compare")
    def compare(
        baseline: str,
        candidate: list[str] = Query(),
    ) -> dict[str, Any]:
        return compare_runs(repository, baseline, candidate)

    @app.get("/api/runs/{run_id}/rerun")
    def preview_rerun(run_id: str) -> dict[str, Any]:
        return reruns.preview(run_id)

    @app.post("/api/runs/{run_id}/rerun", status_code=202)
    def enqueue_rerun(run_id: str, request: RerunRequest) -> dict[str, Any]:
        return reruns.enqueue(run_id, request.config)

    @app.get("/api/runs/{run_id}/control")
    def get_run_control(run_id: str) -> dict[str, Any]:
        return controls.state(run_id)

    @app.post("/api/runs/{run_id}/control", status_code=202)
    def control_run(run_id: str, request: RunControlRequest) -> dict[str, Any]:
        return controls.issue(
            run_id,
            request.command,
            delay_ms=request.delay_ms,
        )

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


def _default_workers() -> int:
    configured = os.environ.get("BIOPLAST_RUN_WORKERS", "1")
    try:
        return max(1, int(configured))
    except ValueError:
        return 1


@lru_cache(maxsize=1)
def _plotly_javascript() -> str:
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


app = create_app()
