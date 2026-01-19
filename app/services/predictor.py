from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from sharp.utils.gaussians import convert_spherical_harmonics_to_rgb

from app.core.config import settings
from app.db.repo import Repository
from app.storage import files as storage_files
from app.storage import paths as storage_paths


@dataclass(frozen=True)
class PredictResult:
    gaussians_path: Path


class PredictorManager:
    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)
        self._cache: dict[str, torch.nn.Module] = {}
        self._lock = threading.Lock()

    def get_predictor(self, device: torch.device) -> torch.nn.Module:
        key = str(device)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if not self._model_path.exists():
                raise FileNotFoundError(f"Model not found at {self._model_path}")
            from sharp.models import PredictorParams, create_predictor

            state_dict = torch.load(self._model_path, map_location="cpu", weights_only=True)
            predictor = create_predictor(PredictorParams())
            predictor.load_state_dict(state_dict)
            predictor.eval()
            predictor.to(device)
            self._cache[key] = predictor
            return predictor


def resolve_device(requested: str | None) -> torch.device:
    device_value = (requested or settings.device_default).lower()
    if device_value in {"auto", "default"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    if device_value == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return torch.device("mps")
    if device_value == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"Unsupported device: {device_value}")


@torch.no_grad()
def predict_image(
    predictor: torch.nn.Module,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
):
    from sharp.utils.gaussians import unproject_gaussians

    internal_shape = (1536, 1536)
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )
    return gaussians


def ensure_ply_has_rgb(path: Path) -> None:
    plydata = PlyData.read(path)
    vertices = next(filter(lambda x: x.name == "vertex", plydata.elements))

    if "red" in vertices and "green" in vertices and "blue" in vertices:
        return

    required_props = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    for prop in required_props:
        if prop not in vertices:
            raise KeyError(f"Incompatible ply file: property {prop} not found in ply elements.")

    sh0 = np.stack(
        (
            np.asarray(vertices["f_dc_0"]),
            np.asarray(vertices["f_dc_1"]),
            np.asarray(vertices["f_dc_2"]),
        ),
        axis=1,
    )
    colors = convert_spherical_harmonics_to_rgb(torch.from_numpy(sh0).float())
    colors = colors.clamp(0, 1).cpu().numpy()
    colors_uint8 = (colors * 255.0).round().astype(np.uint8)

    vertex_count = len(vertices)
    dtype_full = [(name, vertices[name].dtype) for name in vertices.data.dtype.names]
    dtype_full.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

    elements = np.empty(vertex_count, dtype=dtype_full)
    for name in vertices.data.dtype.names:
        elements[name] = vertices[name]
    elements["red"] = colors_uint8[:, 0]
    elements["green"] = colors_uint8[:, 1]
    elements["blue"] = colors_uint8[:, 2]

    vertex_element = PlyElement.describe(elements, "vertex")
    other_elements = [element for element in plydata.elements if element.name != "vertex"]
    PlyData([vertex_element] + other_elements).write(path)


class PredictService:
    def __init__(self, repo: Repository, manager: PredictorManager) -> None:
        self._repo = repo
        self._manager = manager

    def run(self, file_id: str, input_path: Path, device_request: str | None) -> PredictResult:
        device = resolve_device(device_request)
        predictor = self._manager.get_predictor(device)

        from sharp.utils import io
        from sharp.utils.gaussians import save_ply

        image, _, f_px = io.load_rgb(input_path)
        height, width = image.shape[:2]
        gaussians = predict_image(predictor, image, f_px, device)

        output_path = storage_paths.gaussians_path(settings.data_dir, file_id)
        storage_files.ensure_file_dir(settings.data_dir, file_id)
        save_ply(gaussians, f_px, (height, width), output_path)
        ensure_ply_has_rgb(output_path)
        return PredictResult(gaussians_path=output_path)
