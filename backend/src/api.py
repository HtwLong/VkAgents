# api.py (Revised Main File)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the routers
from routers.planning import router as planning_router
from routers.execution import router as execution_router
from routers.artifacts import router as artifacts_router
from routers.runs import router as runs_router

# -----------------------------------------------------------------------------
# App Initialization and Router Inclusion
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Computer Vision Pipeline API",
    description="Orchestrates CV pipeline steps, job management, and model serving.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Include Routers
# -----------------------------------------------------------------------------
# The order here determines the order in the generated API documentation
app.include_router(planning_router, prefix="/api/v1")
app.include_router(execution_router, prefix="/api/v1")
app.include_router(runs_router, prefix="/api/v1")
app.include_router(artifacts_router) # Artifacts doesn't need a specific prefix beyond /artifacts
