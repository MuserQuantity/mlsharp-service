from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class TaskHandle:
    task_id: str
    future: Future


class TaskRunner:
    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._gpu_semaphore = asyncio.Semaphore(settings.max_gpu_tasks)
        self._loop = None
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def submit(self, task_id: str, fn, *args, gpu: bool = False, **kwargs) -> TaskHandle:
        if gpu:
            future = self._executor.submit(self._run_gpu, fn, *args, **kwargs)
        else:
            future = self._executor.submit(fn, *args, **kwargs)
        return TaskHandle(task_id=task_id, future=future)

    def _run_gpu(self, fn, *args, **kwargs):
        if self._loop is None:
            raise RuntimeError("TaskRunner loop not set")
        return asyncio.run_coroutine_threadsafe(
            self._with_gpu(fn, *args, **kwargs), self._loop
        ).result()

    async def _with_gpu(self, fn, *args, **kwargs):
        async with self._gpu_semaphore:
            return fn(*args, **kwargs)
