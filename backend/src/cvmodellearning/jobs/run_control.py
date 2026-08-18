"""Small, durable control plane for one linear pipeline run."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvmodellearning.paths import run_dir


class PipelineCancelled(Exception):
    """Raised when a user-requested cancellation reaches a safe boundary."""


_lock = threading.RLock()
_cancelled: set[str] = set()


def state_path(job_id: str) -> Path:
    return run_dir(job_id) / "run_state.json"


def read_run_state(job_id: str) -> dict[str, Any] | None:
    path = state_path(job_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_run_state(
    job_id: str,
    status: str,
    *,
    active_step: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    with _lock:
        previous = read_run_state(job_id) or {}
        attempt = int(previous.get("attempt", 0))
        if status == "running" and previous.get("status") != "running":
            attempt += 1
        state = {
            "job_id": job_id,
            "status": status,
            "active_step": active_step,
            "attempt": attempt,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            state["error"] = error
        path = state_path(job_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(path)
        return state


def request_cancellation(job_id: str) -> dict[str, Any]:
    with _lock:
        _cancelled.add(job_id)
    current = read_run_state(job_id) or {}
    return write_run_state(job_id, "cancelling", active_step=current.get("active_step"))


def clear_cancellation(job_id: str) -> None:
    with _lock:
        _cancelled.discard(job_id)


def cancellation_requested(job_id: str) -> bool:
    with _lock:
        if job_id in _cancelled:
            return True
    state = read_run_state(job_id)
    return bool(state and state.get("status") == "cancelling")


def raise_if_cancelled(job_id: str) -> None:
    if cancellation_requested(job_id):
        raise PipelineCancelled(f"Pipeline run {job_id} was stopped by the user.")


def mark_stopped(job_id: str, step_id: str) -> None:
    write_run_state(job_id, "stopped", active_step=step_id)


def mark_failed(job_id: str, step_id: str, error: str) -> None:
    """Publish a durable terminal failure for status polling and restarts."""
    write_run_state(job_id, "failed", active_step=step_id, error=error)


def finish_or_stop(job_id: str, step_id: str, *, status: str = "waiting") -> None:
    """Atomically publish completion unless cancellation won the race."""
    with _lock:
        state = read_run_state(job_id)
        if job_id in _cancelled or (state and state.get("status") == "cancelling"):
            write_run_state(job_id, "stopped", active_step=step_id)
            raise PipelineCancelled(f"Pipeline run {job_id} was stopped by the user.")
        write_run_state(job_id, status, active_step=None)
