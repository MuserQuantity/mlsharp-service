from __future__ import annotations

import shutil
from pathlib import Path

from . import paths


def persist_upload(data_dir: str, file_id: str, filename: str, content: bytes) -> Path:
    ext = Path(filename).suffix or ".bin"
    target = paths.ensure_dir(paths.file_root(data_dir, file_id)) / f"original{ext}"
    with open(target, "wb") as f:
        f.write(content)
    return target


def ensure_file_dir(data_dir: str, file_id: str) -> Path:
    return paths.ensure_dir(paths.file_root(data_dir, file_id))


def ensure_clean_dir(data_dir: str, file_id: str) -> Path:
    root = paths.file_root(data_dir, file_id)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root
