from __future__ import annotations

import os
from dataclasses import dataclass


def _get_env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str
    model_path: str
    data_dir: str
    db_path: str
    max_upload_mb: int
    max_gpu_tasks: int
    device_default: str
    port: int


settings = Settings(
    api_key=_get_env("API_KEY"),
    model_path=_get_env("MODEL_PATH", "/app/models/sharp_2572gikvuh.pt"),
    data_dir=_get_env("DATA_DIR", "/app/data"),
    db_path=_get_env("DB_PATH", "/app/data/mlsharp.db"),
    max_upload_mb=int(_get_env("MAX_UPLOAD_MB", "10")),
    max_gpu_tasks=int(_get_env("MAX_GPU_TASKS", "1")),
    device_default=_get_env("DEVICE_DEFAULT", "auto"),
    port=int(_get_env("PORT", "11011")),
)
