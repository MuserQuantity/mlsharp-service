from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.core.config import settings
from app.db.repo import Repository
from app.services.predictor import PredictService, PredictorManager
from app.services.renderer import RenderService
from app.tasks.runner import TaskRunner


class AppState:
    def __init__(self) -> None:
        self.repo = Repository(settings.db_path)
        self.runner = TaskRunner(max_workers=4)
        self.predictor_manager = PredictorManager(settings.model_path)
        self.predict_service = PredictService(self.repo, self.predictor_manager)
        self.render_service = RenderService()


app = FastAPI(title="mlsharp-service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
state = AppState()
app.state.repo = state.repo
app.state.runner = state.runner
app.state.predict_service = state.predict_service
app.state.render_service = state.render_service


@app.on_event("startup")
async def on_startup() -> None:
    state.runner.set_loop(asyncio.get_running_loop())


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


app.include_router(routes.router)
