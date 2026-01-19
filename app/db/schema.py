from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FileRecord:
    file_id: str
    original_name: str
    original_path: str
    gaussians_path: str | None
    render_path: str | None
    render_depth_path: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_type: str
    status: str
    error: str | None
    file_id: str
    created_at: str
    updated_at: str


def ensure_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                original_path TEXT NOT NULL,
                gaussians_path TEXT,
                render_path TEXT,
                render_depth_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                file_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(file_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_file_id ON tasks(file_id)")
        conn.commit()
