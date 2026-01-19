from __future__ import annotations

import sqlite3
from typing import Iterable

from .schema import FileRecord, TaskRecord, ensure_db, utc_now


class Repository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        ensure_db(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_file(self, file_id: str, original_name: str, original_path: str) -> FileRecord:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (
                    file_id, original_name, original_path, gaussians_path,
                    render_path, render_depth_path, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (file_id, original_name, original_path, now, now),
            )
        return self.get_file(file_id)

    def update_file_outputs(
        self,
        file_id: str,
        gaussians_path: str | None = None,
        render_path: str | None = None,
        render_depth_path: str | None = None,
    ) -> FileRecord:
        now = utc_now()
        with self._connect() as conn:
            current = self.get_file(file_id)
            conn.execute(
                """
                UPDATE files
                SET gaussians_path = ?, render_path = ?, render_depth_path = ?, updated_at = ?
                WHERE file_id = ?
                """,
                (
                    gaussians_path if gaussians_path is not None else current.gaussians_path,
                    render_path if render_path is not None else current.render_path,
                    render_depth_path if render_depth_path is not None else current.render_depth_path,
                    now,
                    file_id,
                ),
            )
        return self.get_file(file_id)

    def get_file(self, file_id: str) -> FileRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        if row is None:
            raise KeyError("file not found")
        return FileRecord(**dict(row))

    def list_files(self, file_ids: Iterable[str]) -> list[FileRecord]:
        ids = list(file_ids)
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM files WHERE file_id IN ({placeholders})", ids
            ).fetchall()
        return [FileRecord(**dict(row)) for row in rows]

    def create_task(self, task_id: str, task_type: str, file_id: str) -> TaskRecord:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (task_id, task_type, status, error, file_id, created_at, updated_at)
                VALUES (?, ?, ?, NULL, ?, ?, ?)
                """,
                (task_id, task_type, "queued", file_id, now, now),
            )
        return self.get_task(task_id)

    def update_task(self, task_id: str, status: str, error: str | None = None) -> TaskRecord:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks SET status = ?, error = ?, updated_at = ? WHERE task_id = ?
                """,
                (status, error, now, task_id),
            )
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> TaskRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError("task not found")
        return TaskRecord(**dict(row))
