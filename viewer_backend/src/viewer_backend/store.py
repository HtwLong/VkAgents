from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .settings import RUNS_ROOT


SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")


def run_dir(job_id: str, *, create: bool = False) -> Path:
    if not SAFE_ID.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="Invalid run identifier.")
    path = (RUNS_ROOT / job_id).resolve()
    if RUNS_ROOT not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid run identifier.")
    if create:
        (path / "artifacts" / "planning").mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {job_id}")
    return path


def safe_file(job_id: str, relative: str) -> Path:
    base = run_dir(job_id)
    path = (base / relative).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return path


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, (dict, list)) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def planning_dir(job_id: str) -> Path:
    return run_dir(job_id, create=True) / "artifacts" / "planning"


def latest_context(base: Path) -> dict[str, Any] | None:
    planning = base / "artifacts" / "planning"
    candidates = sorted(planning.glob("STATE_*.json"), reverse=True)
    for path in candidates:
        value = read_json(path)
        if isinstance(value, dict):
            return value
    return None

