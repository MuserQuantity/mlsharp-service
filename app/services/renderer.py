from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from app.core.config import settings
from app.storage import paths as storage_paths


@dataclass(frozen=True)
class RenderParams:
    trajectory_type: str | None = None
    lookat_mode: str | None = None
    max_disparity: float | None = None
    max_zoom: float | None = None
    distance_m: float | None = None
    num_steps: int | None = None
    num_repeats: int | None = None


@dataclass(frozen=True)
class RenderResult:
    render_path: Path
    render_depth_path: Path


class RenderService:
    def run(self, file_id: str, params: RenderParams) -> RenderResult:
        if not torch.cuda.is_available():
            raise RuntimeError("Rendering requires CUDA")

        from sharp.utils.gaussians import load_ply
        from sharp.utils import camera
        from sharp.cli.render import render_gaussians

        gaussians, metadata = load_ply(storage_paths.gaussians_path(settings.data_dir, file_id))
        trajectory = camera.TrajectoryParams()
        if params.trajectory_type is not None:
            trajectory.type = params.trajectory_type
        if params.lookat_mode is not None:
            trajectory.lookat_mode = params.lookat_mode
        if params.max_disparity is not None:
            trajectory.max_disparity = params.max_disparity
        if params.max_zoom is not None:
            trajectory.max_zoom = params.max_zoom
        if params.distance_m is not None:
            trajectory.distance_m = params.distance_m
        if params.num_steps is not None:
            trajectory.num_steps = params.num_steps
        if params.num_repeats is not None:
            trajectory.num_repeats = params.num_repeats

        output_path = storage_paths.render_path(settings.data_dir, file_id)
        render_gaussians(
            gaussians=gaussians,
            metadata=metadata,
            output_path=output_path,
            params=trajectory,
        )
        return RenderResult(
            render_path=output_path,
            render_depth_path=storage_paths.render_depth_path(settings.data_dir, file_id),
        )
