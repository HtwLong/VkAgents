from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..store import read_json, run_dir, safe_file


router = APIRouter(tags=["Artifacts"])

ALLOWED_SUFFIXES = {".json", ".csv", ".txt", ".yaml", ".yml", ".sparql", ".png", ".svg", ".pdf"}
BLOCKED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".zip"}

KNOWN_ARTIFACTS = (
    ("hyperparameters", "configuration", "Hyperparameter configuration", "artifacts/planning/RESULT_HYPERPARAMETERS.json", "planning/hyperparameters"),
    ("data_provenance", "provenance_audit", "Data provenance audit", "artifacts/data_provenance.json", "data-provenance"),
    ("evaluation_report", "evaluation_report", "Evaluation report", "artifacts/evaluation_report.json", "summary"),
    ("metrics", "training_metrics", "Training metrics", "artifacts/metrics_log.csv", "evaluation/metrics/metrics_log.csv"),
    ("classification_report", "evaluation_report", "Classification report", "artifacts/test_classification_report.json", "evaluation/report/test_classification_report.json"),
    ("confusion_matrix", "evaluation_matrix", "Confusion matrix", "artifacts/test_confusion_matrix.csv", "evaluation/matrix/test_confusion_matrix.csv"),
)


def manifest_for(job_id: str) -> list[dict]:
    base = run_dir(job_id)
    result = []
    for artifact_id, kind, label, relative, endpoint in KNOWN_ARTIFACTS:
        path = base / relative
        if path.is_file():
            result.append({
                "id": artifact_id,
                "kind": kind,
                "label": label,
                "filename": path.name,
                "download_url": f"/artifacts/{job_id}/{endpoint}",
                "description": "Persisted lightweight run evidence.",
                "standalone": True,
                "generated_on_download": False,
            })
    return result


@router.get("/artifacts/{job_id}/manifest")
def artifact_manifest(job_id: str):
    return {
        "artifacts": manifest_for(job_id),
        "execution_available": False,
        "note": "Model checkpoints are intentionally not hosted by this service.",
    }


def _serve(job_id: str, relative: str) -> FileResponse:
    path = safe_file(job_id, relative)
    if path.suffix.lower() in BLOCKED_SUFFIXES or path.suffix.lower() not in ALLOWED_SUFFIXES:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="This artifact type is not served in viewer mode.")
    return FileResponse(path, filename=path.name)


@router.get("/artifacts/{job_id}/summary")
def summary(job_id: str):
    return _serve(job_id, "artifacts/evaluation_report.json")


@router.get("/artifacts/{job_id}/data-provenance")
def provenance(job_id: str):
    return _serve(job_id, "artifacts/data_provenance.json")


@router.get("/artifacts/{job_id}/planning/hyperparameters")
def hyperparameters(job_id: str):
    return _serve(job_id, "artifacts/planning/RESULT_HYPERPARAMETERS.json")


@router.get("/artifacts/{job_id}/evaluation/{category}/{filename}")
def evaluation_artifact(job_id: str, category: str, filename: str):
    allowed = {
        "metrics_log.csv", "metrics_log.json", "test_classification_report.json",
        "test_confusion_matrix.csv", "evaluation_report.json",
    }
    if filename not in allowed:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return _serve(job_id, f"artifacts/{filename}")


@router.get("/api/v1/evaluate/{job_id}/report")
def evaluation_report(job_id: str):
    path = safe_file(job_id, "artifacts/evaluation_report.json")
    value = read_json(path)
    return value

