"""Read-only HTTP API локального визуализатора прогонов.

Запуск из корня проекта:

    uv run uvicorn bioplast.viz.api:app --reload
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    viz_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=viz_dir / "templates")
    app = FastAPI(
        title="bioplast run visualizer",
        version="0.1.0",
        description="Read-only API файловых артефактов runs/<id>/.",
    )
    app.state.run_repository = repository
    app.mount("/static", StaticFiles(directory=viz_dir / "static"), name="static")

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

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/runs")

    @app.get("/runs", response_class=HTMLResponse, include_in_schema=False)
    def runs_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="runs.html", context={})

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


@lru_cache(maxsize=1)
def _plotly_javascript() -> str:
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


app = create_app()
