from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routers import analysis, jobs, meetings, service
from backend.services import service_runtime
from backend.services.data_import import run_first_run_import
from backend.services.recovery import recover_stuck_meetings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Bundled-only: redirect model cache + ffmpeg into the bundle, then import a
    # prior Terminal installation's meetings. No-ops in a dev checkout (BR-18).
    service_runtime.bootstrap()
    run_first_run_import()
    recover_stuck_meetings()
    yield


app = FastAPI(title="Meeting Transcriber", lifespan=lifespan)

app.include_router(meetings.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(service.router, prefix="/api")

frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
@app.get("/{path:path}")
async def serve_spa(path: str = ""):
    # Serve API routes normally (handled by routers above)
    # Everything else gets the SPA shell
    return FileResponse(str(frontend_dir / "index.html"))
