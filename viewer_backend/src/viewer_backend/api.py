from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import artifacts, disabled, planning, runs
from .settings import ALLOWED_ORIGINS


app = FastAPI(
    title="VKG Planning and Results API",
    description="CPU-only planning and historical run viewer; execution is intentionally disabled.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "viewer", "pytorch": False}


@app.get("/api/v1/capabilities")
def capabilities():
    return {
        "mode": "viewer",
        "planning": True,
        "graphrag": True,
        "planning_revisions": True,
        "assessment_revision": True,
        "post_training_assessment": True,
        "run_viewing": True,
        "artifact_downloads": True,
        "data_download": False,
        "data_preparation": False,
        "training": False,
        "evaluation_execution": False,
        "inference": False,
    }


app.include_router(planning.router, prefix="/api/v1")
app.include_router(runs.router, prefix="/api/v1")
app.include_router(disabled.router, prefix="/api/v1")
app.include_router(artifacts.router)
