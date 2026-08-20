from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path(os.getenv("VIEWER_RUNS_DIR", PACKAGE_ROOT / "runs")).expanduser().resolve()
ONTOLOGY_ROOT = Path(
    os.getenv("VIEWER_ONTOLOGY_DIR", PACKAGE_ROOT / "ontology_data")
).expanduser().resolve()
PLANNING_MODEL = os.getenv("PLANNING_LLM_MODEL", "gpt-5-nano")
ASSESSMENT_MODEL = os.getenv("ASSESSMENT_LLM_MODEL", PLANNING_MODEL)
ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if item.strip()
]
