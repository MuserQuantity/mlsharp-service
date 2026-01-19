from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_root(data_dir: str, file_id: str) -> Path:
    return Path(data_dir) / "files" / file_id


def original_path(data_dir: str, file_id: str, ext: str) -> Path:
    return file_root(data_dir, file_id) / f"original{ext}"


def gaussians_path(data_dir: str, file_id: str) -> Path:
    return file_root(data_dir, file_id) / "gaussians.ply"


def render_path(data_dir: str, file_id: str) -> Path:
    return file_root(data_dir, file_id) / "render.mp4"


def render_depth_path(data_dir: str, file_id: str) -> Path:
    return file_root(data_dir, file_id) / "render.depth.mp4"
