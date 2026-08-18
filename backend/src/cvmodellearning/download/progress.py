"""Persistent, pollable progress for the dataset download step."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvmodellearning.paths import download_progress_path


_REPLACE_ATTEMPTS = 6
_REPLACE_RETRY_DELAY = 0.01


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Atomically replace a file, tolerating brief Windows sharing violations."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_RETRY_DELAY * (2**attempt))


class DownloadProgressTracker:
    """Write an atomic progress snapshot after every resolved image."""

    def __init__(
        self,
        job_id: str,
        total: int,
        *,
        resume: bool = False,
        persist_interval: float = 0.5,
    ) -> None:
        if persist_interval < 0:
            raise ValueError("persist_interval must be non-negative.")
        self.path = download_progress_path(job_id)
        self._lock = threading.Lock()
        self._persist_interval = persist_interval
        self._last_write = 0.0
        previous = read_download_progress(self.path) if resume else None
        self._state: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "downloaded": 0,
            "processed": 0,
            "failed": 0,
            "failed_datasets": [],
            "total": total,
            "current_image": None,
            "previous_downloaded": int((previous or {}).get("downloaded", 0)),
        }
        self._write(force=True)

    def record(self, *, successful: bool, image_path: str) -> None:
        with self._lock:
            self._state["processed"] += 1
            self._state["current_image"] = image_path
            if successful:
                self._state["downloaded"] += 1
            else:
                self._state["failed"] += 1
            self._write()

    def finish(self, status: str) -> None:
        with self._lock:
            self._state["status"] = status
            self._state["current_image"] = None
            self._write(force=True)

    def record_failed_datasets(self, dataset_names: list[str]) -> None:
        """Record dataset names that could not satisfy their download allocation."""
        with self._lock:
            existing = set(self._state["failed_datasets"])
            for name in dataset_names:
                if name and name not in existing:
                    self._state["failed_datasets"].append(name)
                    existing.add(name)
            self._write(force=True)

    def _write(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < self._persist_interval:
            return
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
            _replace_with_retry(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self._last_write = now


def read_download_progress(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
