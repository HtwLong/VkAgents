import json
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute

from cvmodellearning.paths import run_dir


_error_file_lock = threading.Lock()


def save_run_error(
    job_id: str,
    endpoint: str,
    error: Any,
    *,
    status_code: int = 500,
    error_type: str | None = None,
) -> None:
    """Append an endpoint or background-task failure to the job's run."""
    path = run_dir(job_id) / "errors.json"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "endpoint": endpoint,
        "status_code": status_code,
        "error_type": error_type,
        "error": error,
    }

    with _error_file_lock:
        try:
            errors = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            if not isinstance(errors, list):
                errors = []
        except (OSError, json.JSONDecodeError):
            errors = []
        errors.append(entry)
        path.write_text(json.dumps(errors, indent=2, default=str), encoding="utf-8")


async def _job_id_from_request(request: Request) -> str | None:
    job_id = request.path_params.get("job_id") or request.query_params.get("job_id")
    if job_id:
        return str(job_id)

    try:
        body = await request.json()
    except Exception:
        return None
    if isinstance(body, dict) and body.get("job_id"):
        return str(body["job_id"])
    return None


class ErrorPersistingRoute(APIRoute):
    """Persist failures for requests that identify a job, then re-raise them."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Any]]:
        route_handler = super().get_route_handler()

        async def handler(request: Request) -> Any:
            try:
                return await route_handler(request)
            except Exception as exc:
                job_id = await _job_id_from_request(request)
                if job_id:
                    if isinstance(exc, HTTPException):
                        error = exc.detail
                        status_code = exc.status_code
                    else:
                        error = traceback.format_exc()
                        status_code = 500
                    try:
                        save_run_error(
                            job_id,
                            request.url.path,
                            error,
                            status_code=status_code,
                            error_type=type(exc).__name__,
                        )
                    except OSError:
                        # Error logging must never replace the original endpoint error.
                        pass
                raise

        return handler
