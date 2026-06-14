from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# Import shared dependencies
from cvmodellearning.paths import run_dir

router = APIRouter(
    prefix="/artifacts",
    tags=["3 - Artifacts"],
)

# --- Utility Function (Copied from original api.py) ---
def _safe_artifact(job_id: str, relative: str) -> Path:
    base = run_dir(job_id)
    path = (base / relative).resolve()
    # Security check: ensure the path is within the job's run_dir
    if base not in path.parents and path != base:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {relative}")
    return path


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@router.get("/{job_id}/report_pdf", response_class=FileResponse)
async def artifact_report_pdf(job_id: str):
    p = _safe_artifact(job_id, "artifacts/report_summary.pdf")
    return FileResponse(p, media_type="application/pdf", filename=f"report_summary_{job_id}.pdf")

# @router.get("/{job_id}/metrics_csv", response_class=FileResponse)
# async def artifact_metrics_csv(job_id: str):
#     p = _safe_artifact(job_id, "artifacts/metrics_log.csv")
#     return FileResponse(p, media_type="text/csv", filename="metrics_log.csv")

@router.get("/{job_id}/model", response_class=FileResponse)
async def artifact_model(job_id: str):
    """
    Downloads the best model artifact. 
    Checks for 'best_model.pt' (YOLO) first, then 'best_model.pth' (Torchvision).
    """
    # 1. Try finding YOLO model (.pt)
    try:
        p = _safe_artifact(job_id, "artifacts/best_model.pt")
        return FileResponse(p, media_type="application/octet-stream", filename="best_model.pt")
    except HTTPException:
        pass # If not found, continue to check .pth

    # 2. Try finding Torchvision model (.pth)
    try:
        p = _safe_artifact(job_id, "artifacts/best_model.pth")
        return FileResponse(p, media_type="application/octet-stream", filename="best_model.pth")
    except HTTPException:
        pass # If not found, continue to error

    # 3. If neither exists
    raise HTTPException(
        status_code=404, 
        detail=f"Model artifact not found for job {job_id}. Checked for 'best_model.pt' and 'best_model.pth'."
    )

@router.get("/{job_id}/config", response_class=FileResponse)
async def artifact_config(job_id: str):
    p = _safe_artifact(job_id, "config.json")
    return FileResponse(p, media_type="application/json", filename="config.json")

@router.get("/{job_id}/summary", response_class=FileResponse)
async def artifact_summary(job_id: str):
    p = _safe_artifact(job_id, "summary.json")
    return FileResponse(p, media_type="application/json", filename="summary.json")

@router.get("/{job_id}/planning/interpretation", response_class=FileResponse)
async def artifact_planning_interpretation(job_id: str):
    p = _safe_artifact(job_id, "artifacts/planning/RESULT_INTERPRETATION.json")
    return FileResponse(p, media_type="application/json", filename="interpretation.json")

@router.get("/{job_id}/planning/model_selection", response_class=FileResponse)
async def artifact_planning_model_selection(job_id: str):
    p = _safe_artifact(job_id, "artifacts/planning/RESULT_MODEL.json")
    return FileResponse(p, media_type="application/json", filename="model_selection.json")

@router.get("/{job_id}/planning/preprocessing", response_class=FileResponse)
async def artifact_planning_preprocessing(job_id: str):
    p = _safe_artifact(job_id, "artifacts/planning/RESULT_PREPROCESSING.json")
    return FileResponse(p, media_type="application/json", filename="preprocessing.json")

@router.get("/{job_id}/planning/hyperparameters", response_class=FileResponse)
async def artifact_planning_hyperparameters(job_id: str):
    p = _safe_artifact(job_id, "artifacts/planning/RESULT_HYPERPARAMETERS.json")
    return FileResponse(p, media_type="application/json", filename="hyperparameters.json")

