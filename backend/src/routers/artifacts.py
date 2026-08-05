from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# Import shared dependencies
from cvmodellearning.models.classification_artifacts import (
    ensure_lora_adapter_bundle,
    ensure_merged_lora_model,
)
from cvmodellearning.models.classification_lora import LORA_CHECKPOINT_FORMAT
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

def _descriptor(
    job_id: str,
    artifact_id: str,
    kind: str,
    label: str,
    filename: str,
    endpoint: str,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "kind": kind,
        "label": label,
        "filename": filename,
        "download_url": f"/artifacts/{job_id}/{endpoint}",
        **metadata,
    }


def _torchvision_checkpoint(job_id: str) -> tuple[Path, dict] | None:
    try:
        path = _safe_artifact(job_id, "artifacts/best_model.pth")
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        return None
    return path, torch.load(path, map_location="cpu", mmap=True)


@router.get("/{job_id}/manifest")
def artifact_manifest(job_id: str):
    """Describe downloadable artifacts without making the frontend infer their type."""
    artifacts: list[dict[str, Any]] = []
    training_mode = None

    try:
        _safe_artifact(job_id, "artifacts/best_model.pt")
        artifacts.append(_descriptor(
            job_id,
            "model",
            "full_model",
            "Trained model",
            "best_model.pt",
            "model",
            standalone=True,
            description="Standalone trained model.",
        ))
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        checkpoint_result = _torchvision_checkpoint(job_id)
        if checkpoint_result is not None:
            _, checkpoint = checkpoint_result
            config = checkpoint.get("config", {})
            training_mode = config.get("training_mode")
            if checkpoint.get("checkpoint_format") == LORA_CHECKPOINT_FORMAT:
                bundle = ensure_lora_adapter_bundle(job_id)
                base_model = checkpoint["adapter_metadata"]["base_weights_id"]
                artifacts.extend([
                    _descriptor(
                        job_id,
                        "lora_adapter",
                        "lora_adapter_bundle",
                        "LoRA adapter bundle",
                        bundle.name,
                        "lora-adapter",
                        standalone=False,
                        required_base_model=base_model,
                        description=f"Adapter and classifier head; requires {base_model}.",
                    ),
                    _descriptor(
                        job_id,
                        "merged_model",
                        "merged_model",
                        "Merged standalone model",
                        "best_merged_model.pth",
                        "merged-model",
                        standalone=True,
                        generated_on_download=True,
                        description="Full model generated from the pretrained base and adapter on download.",
                    ),
                ])
            else:
                artifacts.append(_descriptor(
                    job_id,
                    "model",
                    "full_model",
                    "Trained model",
                    "best_model.pth",
                    "model",
                    standalone=True,
                    description="Standalone full-model checkpoint.",
                ))

    try:
        config = _safe_artifact(job_id, "artifacts/planning/RESULT_HYPERPARAMETERS.json")
        artifacts.append(_descriptor(
            job_id,
            "hyperparameters",
            "configuration",
            "Hyperparameter configuration",
            config.name,
            "planning/hyperparameters",
            standalone=True,
            description="Validated configuration and decision rationale.",
        ))
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        pass

    try:
        provenance = _safe_artifact(job_id, "artifacts/data_provenance.json")
        artifacts.append(_descriptor(
            job_id,
            "data_provenance",
            "provenance_audit",
            "Data provenance audit",
            provenance.name,
            "data-provenance",
            standalone=True,
            description="Fingerprints the exact dataset splits consumed by training and evaluation.",
        ))
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        pass

    if not artifacts:
        raise HTTPException(status_code=404, detail=f"No artifacts found for job {job_id}.")
    return {"job_id": job_id, "training_mode": training_mode, "artifacts": artifacts}

# @router.get("/{job_id}/metrics_csv", response_class=FileResponse)
# async def artifact_metrics_csv(job_id: str):
#     p = _safe_artifact(job_id, "artifacts/metrics_log.csv")
#     return FileResponse(p, media_type="text/csv", filename="metrics_log.csv")

@router.get("/{job_id}/model", response_class=FileResponse)
def artifact_model(job_id: str):
    """
    Downloads the best model artifact. 
    Checks for 'best_model.pt' (YOLO) first, then 'best_model.pth' (Torchvision).
    """
    # 1. Try finding YOLO model (.pt)
    try:
        p = _safe_artifact(job_id, "artifacts/best_model.pt")
        return FileResponse(p, media_type="application/octet-stream", filename="best_model.pt")
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        pass # If not found, continue to check .pth

    # 2. A LoRA checkpoint is not standalone, so expose its adapter bundle.
    checkpoint_result = _torchvision_checkpoint(job_id)
    if checkpoint_result is not None:
        path, checkpoint = checkpoint_result
        if checkpoint.get("checkpoint_format") == LORA_CHECKPOINT_FORMAT:
            bundle = ensure_lora_adapter_bundle(job_id)
            return FileResponse(
                bundle,
                media_type="application/zip",
                filename="best_lora_adapter.zip",
            )
        return FileResponse(path, media_type="application/octet-stream", filename="best_model.pth")

    # 3. If neither exists
    raise HTTPException(
        status_code=404,
        detail=f"Model artifact not found for job {job_id}. Checked for 'best_model.pt' and 'best_model.pth'."
    )


@router.get("/{job_id}/lora-adapter", response_class=FileResponse)
def artifact_lora_adapter(job_id: str):
    try:
        _safe_artifact(job_id, "artifacts/best_model.pth")
        bundle = ensure_lora_adapter_bundle(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(bundle, media_type="application/zip", filename=bundle.name)


@router.get("/{job_id}/merged-model", response_class=FileResponse)
def artifact_merged_model(job_id: str):
    try:
        _safe_artifact(job_id, "artifacts/best_model.pth")
        merged = ensure_merged_lora_model(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(
        merged,
        media_type="application/octet-stream",
        filename=merged.name,
    )

@router.get("/{job_id}/config", response_class=FileResponse)
async def artifact_config(job_id: str):
    p = _safe_artifact(job_id, "config.json")
    return FileResponse(p, media_type="application/json", filename="config.json")

@router.get("/{job_id}/summary", response_class=FileResponse)
async def artifact_summary(job_id: str):
    p = _safe_artifact(job_id, "summary.json")
    return FileResponse(p, media_type="application/json", filename="summary.json")

@router.get("/{job_id}/data-provenance", response_class=FileResponse)
async def artifact_data_provenance(job_id: str):
    p = _safe_artifact(job_id, "artifacts/data_provenance.json")
    return FileResponse(p, media_type="application/json", filename="data_provenance.json")

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
