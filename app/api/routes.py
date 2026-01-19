import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import ApiKeyDep
from app.api.models import FileResponse as FileInfo
from app.api.models import PredictResponse, RenderRequest, RenderResponse, TaskResponse
from app.core.config import settings
from app.db.repo import Repository
from app.services.predictor import PredictService
from app.services.renderer import RenderParams, RenderService
from app.storage import files as storage_files
from app.storage import paths as storage_paths
from app.tasks.runner import TaskRunner

router = APIRouter(prefix="/v1")


def _file_response(record) -> FileInfo:
    return FileInfo(
        file_id=record.file_id,
        original_name=record.original_name,
        original_path=record.original_path,
        gaussians_path=record.gaussians_path,
        render_path=record.render_path,
        render_depth_path=record.render_depth_path,
    )


def _ensure_exists(path: str) -> None:
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="file not found")


def _services(request: Request) -> tuple[Repository, TaskRunner, PredictService, RenderService]:
    repo: Repository = request.app.state.repo
    runner: TaskRunner = request.app.state.runner
    predict: PredictService = request.app.state.predict_service
    render: RenderService = request.app.state.render_service
    return repo, runner, predict, render


@router.post("/predict", response_model=PredictResponse, dependencies=[ApiKeyDep])
async def predict(request: Request, upload: UploadFile = File(...)):
    repo, runner, service, _ = _services(request)
    content = await upload.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    filename = upload.filename or "upload.bin"
    file_id = uuid.uuid4().hex
    task_id = uuid.uuid4().hex
    input_path = storage_files.persist_upload(settings.data_dir, file_id, filename, content)
    repo.create_file(file_id=file_id, original_name=filename, original_path=str(input_path))
    repo.create_task(task_id=task_id, task_type="predict", file_id=file_id)

    def _run_predict():
        try:
            repo.update_task(task_id, "running")
            result = service.run(file_id=file_id, input_path=input_path, device_request=None)
            repo.update_file_outputs(file_id, gaussians_path=str(result.gaussians_path))
            repo.update_task(task_id, "completed")
        except Exception as exc:
            repo.update_task(task_id, "failed", str(exc))

    runner.submit(task_id, _run_predict)
    return PredictResponse(task_id=task_id, file_id=file_id)


@router.post("/render", response_model=RenderResponse, dependencies=[ApiKeyDep])
async def render(request: Request, payload: RenderRequest):
    repo, runner, _, service = _services(request)
    try:
        record = repo.get_file(payload.file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="file not found")

    gaussians_path = storage_paths.gaussians_path(settings.data_dir, payload.file_id)
    if not Path(gaussians_path).exists() or record.gaussians_path is None:
        raise HTTPException(status_code=400, detail="gaussians not found")

    task_id = uuid.uuid4().hex
    repo.create_task(task_id=task_id, task_type="render", file_id=payload.file_id)

    params = RenderParams(
        trajectory_type=payload.trajectory_type,
        lookat_mode=payload.lookat_mode,
        max_disparity=payload.max_disparity,
        max_zoom=payload.max_zoom,
        distance_m=payload.distance_m,
        num_steps=payload.num_steps,
        num_repeats=payload.num_repeats,
    )

    def _run_render():
        try:
            repo.update_task(task_id, "running")
            result = service.run(file_id=payload.file_id, params=params)
            repo.update_file_outputs(
                payload.file_id,
                render_path=str(result.render_path),
                render_depth_path=str(result.render_depth_path),
            )
            repo.update_task(task_id, "completed")
        except Exception as exc:
            repo.update_task(task_id, "failed", str(exc))

    runner.submit(task_id, _run_render, gpu=True)
    return RenderResponse(task_id=task_id, file_id=payload.file_id)


@router.get("/tasks/{task_id}", response_model=TaskResponse, dependencies=[ApiKeyDep])
async def get_task(request: Request, task_id: str):
    repo, _, _, _ = _services(request)
    try:
        task = repo.get_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskResponse(
        task_id=task.task_id,
        task_type=task.task_type,
        status=task.status,
        error=task.error,
        file_id=task.file_id,
    )


@router.get("/files/{file_id}", response_model=FileInfo, dependencies=[ApiKeyDep])
async def get_file(request: Request, file_id: str):
    repo, _, _, _ = _services(request)
    try:
        record = repo.get_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="file not found")
    return _file_response(record)


@router.get("/files/{file_id}/original", dependencies=[ApiKeyDep])
async def get_original(request: Request, file_id: str):
    repo, _, _, _ = _services(request)
    try:
        record = repo.get_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="file not found")
    _ensure_exists(record.original_path)
    return FileResponse(record.original_path)


@router.get("/files/{file_id}/gaussians", dependencies=[ApiKeyDep])
async def get_gaussians(request: Request, file_id: str):
    repo, _, _, _ = _services(request)
    try:
        record = repo.get_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="file not found")
    if record.gaussians_path is None:
        raise HTTPException(status_code=404, detail="gaussians not ready")
    _ensure_exists(record.gaussians_path)
    return FileResponse(record.gaussians_path)


@router.get("/files/{file_id}/render", dependencies=[ApiKeyDep])
async def get_render(request: Request, file_id: str):
    repo, _, _, _ = _services(request)
    try:
        record = repo.get_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="file not found")
    if record.render_path is None:
        raise HTTPException(status_code=404, detail="render not ready")
    _ensure_exists(record.render_path)
    return FileResponse(record.render_path)


@router.get("/files/{file_id}/render-depth", dependencies=[ApiKeyDep])
async def get_render_depth(request: Request, file_id: str):
    repo, _, _, _ = _services(request)
    try:
        record = repo.get_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="file not found")
    if record.render_depth_path is None:
        raise HTTPException(status_code=404, detail="render depth not ready")
    _ensure_exists(record.render_depth_path)
    return FileResponse(record.render_depth_path)
