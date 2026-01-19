from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class RenderRequest(BaseModel):
    file_id: str
    trajectory_type: Literal["swipe", "shake", "rotate", "rotate_forward"] | None = None
    lookat_mode: Literal["point", "ahead"] | None = None
    max_disparity: float | None = None
    max_zoom: float | None = None
    distance_m: float | None = None
    num_steps: int | None = None
    num_repeats: int | None = None


class PredictResponse(BaseModel):
    task_id: str
    file_id: str


class RenderResponse(BaseModel):
    task_id: str
    file_id: str


class TaskResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    error: str | None
    file_id: str


class FileResponse(BaseModel):
    file_id: str
    original_name: str
    original_path: str
    gaussians_path: str | None
    render_path: str | None
    render_depth_path: str | None
