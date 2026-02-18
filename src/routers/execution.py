import asyncio
import json
import traceback
import uuid
from io import BytesIO
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel, ValidationError
from PIL import Image

# Import shared dependencies
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from cvmodellearning.pipelines.detection_pipe import DetectionPipeline  
from cvmodellearning.models.model_manager import MODEL_CACHE_MANAGER
from cvmodellearning.jobs.job_manager import JOB_MANAGER
from cvmodellearning.paths import (
    train_json_path,
    val_json_path,
    test_json_path, 
    run_dir,
    csv_labels_path,
    train_csv_path,
    interpretation_path
)

# Define a generic type for the pipelines for clean type hinting
PipelineType = Union[ClassificationPipeline, DetectionPipeline]

# =============================================================================
# DYNAMIC PIPELINE FACTORY
# =============================================================================

def get_pipeline_by_task(job_id: str) -> PipelineType:
    """
    Determines and instantiates the correct CV pipeline based on the 
    'task' field in the RESULT_INTERPRETATION.json for the given job_id.
    """
    ip_path = interpretation_path(job_id)
    
    if not ip_path.exists():
        # Fallback for steps that might run before planning is complete, 
        # or if the task is implicitly classification (default/legacy).
        # We will assume Classification if no plan exists.
        return ClassificationPipeline() 

    try:
        with open(ip_path, 'r') as f:
            data = json.load(f)
            task: str = data.get("task", "classification").lower()
            
            if task == "classification":
                return ClassificationPipeline()
            elif task == "detection":
                # Ensure the detection JSON path exists for detection data steps
                return DetectionPipeline()
            else:
                # Handle other tasks as they are implemented (segmentation, etc.)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Task '{task}' found in interpretation file, but the corresponding pipeline is not yet implemented or supported."
                )

    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error reading or parsing interpretation file for job {job_id}: {e}"
        )

def _validate_config(pipeline: PipelineType, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the input parameters against the correct Pydantic model.
    """
    if isinstance(pipeline, ClassificationPipeline):
        ConfigModel = ClassificationConfigModel
    elif isinstance(pipeline, DetectionPipeline):
        ConfigModel = DetectionConfigModel
    else:
        raise RuntimeError("Unsupported pipeline type encountered.")
        
    try:
        # Use the specific model to validate and dump the final clean config
        return ConfigModel.model_validate(params).model_dump()
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Configuration validation failed for {pipeline.__class__.__name__}: {e.errors()}",
        )

router = APIRouter(
    tags=["2 - Execution & Inference"],
)

# --- Schemas ---
class StepRequest(BaseModel):
    job_id: str
    chosen_parameters: Union[Dict[str, Any], str]

class TrainStartRequest(BaseModel):
    chosen_parameters: Union[Dict[str, Any], str]
    job_id: Optional[str] = None
    
class LoadModelRequest(BaseModel):
    job_id: str



# -----------------------------------------------------------------------------
# Background Task
# -----------------------------------------------------------------------------

async def _bg_train(job_id: str, params: Dict[str, Any]):
    """Background task logic."""
    try:
        # Get the correct pipeline for the job
        pipeline: PipelineType = get_pipeline_by_task(job_id)

        cfg = _validate_config(pipeline, params)
        run = run_dir(job_id)
        
        # Check if basic data files are missing, if so, run download/prepare
        if isinstance(pipeline, ClassificationPipeline):
            has_train_data = train_csv_path(job_id).exists()
        elif isinstance(pipeline, DetectionPipeline):
            has_train_data = train_json_path(job_id).exists()
        else:
            has_train_data = False # Should not happen
        
        if not has_train_data:
            await asyncio.to_thread(pipeline.download_data_step, cfg, job_id)
            await asyncio.to_thread(pipeline.prepare_data_step, cfg, job_id)
        
        
        if isinstance(pipeline, ClassificationPipeline):
            train_out = await asyncio.to_thread(pipeline.train_model_step, cfg, job_id)
        elif isinstance(pipeline, DetectionPipeline):
            train_out = await pipeline.train_model_step(cfg, job_id)
        else:
            raise RuntimeError("Unsupported pipeline type encountered during training.")
        
        summary = {
            "job_id": job_id,
            "summary": {k: v for k, v in train_out.items() if k != "test_batch_size"},
        }
        
        (run / "summary.json").write_text(json.dumps(summary, indent=2))
        (run / "config.json").write_text(json.dumps(cfg, indent=2))
        
        JOB_MANAGER.update_job_status(job_id, "completed", result=summary)
        
    except Exception as e:
        error_detail = f"CRITICAL BACKGROUND ERROR:\n{traceback.format_exc()}"
        print(error_detail)
        JOB_MANAGER.update_job_status(job_id, "error", error=error_detail)

# -----------------------------------------------------------------------------
# Pipeline Steps
# -----------------------------------------------------------------------------

@router.post("/download-data")
async def step_download(req: StepRequest):
    pipeline = get_pipeline_by_task(req.job_id)
    params = req.chosen_parameters if isinstance(req.chosen_parameters, dict) else json.loads(req.chosen_parameters)
    cfg = _validate_config(pipeline, params)
    run = run_dir(req.job_id)
    out = pipeline.download_data_step(cfg, req.job_id)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    return {"status": "ok", "job_id": req.job_id, "output": out}

@router.post("/prepare-data")
async def step_prepare(req: StepRequest):
    pipeline = get_pipeline_by_task(req.job_id)
    params = req.chosen_parameters if isinstance(req.chosen_parameters, dict) else json.loads(req.chosen_parameters)
    cfg = _validate_config(pipeline, params) 
    # Adapt data existence check
    if isinstance(pipeline, ClassificationPipeline) and not csv_labels_path(req.job_id).exists():
        raise HTTPException(status_code=400, detail="Labels CSV missing; call /download-data first (Classification)")
    if isinstance(pipeline, DetectionPipeline) and not train_json_path(req.job_id).exists() and not val_json_path(req.job_id).exists() and not test_json_path(req.job_id).exists():
        raise HTTPException(status_code=400, detail="Annotation JSON missing; call /download-data first (Detection)")
    out = pipeline.prepare_data_step(cfg, req.job_id)
    return {"status": "ok", "job_id": req.job_id, "output": out}

@router.post("/evaluate")
async def step_evaluate(req: StepRequest):
    pipeline = get_pipeline_by_task(req.job_id)
    params = req.chosen_parameters if isinstance(req.chosen_parameters, dict) else json.loads(req.chosen_parameters)
    cfg = _validate_config(pipeline, params)
    out = pipeline.evaluate_model_step(cfg, req.job_id)
    return {"status": "ok", "job_id": req.job_id, "output": out}

# -----------------------------------------------------------------------------
# Training Endpoints
# -----------------------------------------------------------------------------

@router.post("/train/start")
async def train_start(req: TrainStartRequest, bg: BackgroundTasks):
    params = req.chosen_parameters if isinstance(req.chosen_parameters, dict) else json.loads(req.chosen_parameters)
    job_id = req.job_id or str(uuid.uuid4())
    run_dir(job_id)

    try:
        # NOTE: This assumes interpretation_path exists or defaults to Classification
        # If get_pipeline_by_task fails, it raises HTTPException which FastAPI handles.
        temp_pipeline = get_pipeline_by_task(job_id)
        # _validate_config will raise HTTPException(400) on validation error.
        _validate_config(temp_pipeline, params) 
        
    except HTTPException:
        # Let FastAPI handle the HTTPException (400 or 500)
        raise
    except Exception as e:
        # Catch any other immediate setup errors
        raise HTTPException(status_code=500, detail=f"Initial setup failed: {str(e)}")
    
    JOB_MANAGER.create_job(job_id)
    
    bg.add_task(_bg_train, job_id, params)
    
    return {
        "job_id": job_id,
        "status": "running",
        "status_url": f"/train/status/{job_id}",
        "result_url": f"/train/result/{job_id}",
    }

@router.get("/train/status/{job_id}")
async def train_status(job_id: str):
    """
    Returns the current job status, including the training progress (epoch) 
    if the job is running and progress information is available.
    """
    job = JOB_MANAGER.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found.")
        
    status = job.get("status")
    
    response = {"job_id": job_id, "status": status}
    
    if status == "running":
        progress_file = run_dir(job_id) / "progress.json"
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    response.update(progress_data)
            except json.JSONDecodeError:
                response["progress_error"] = "Could not parse training progress file."
        else:
             response["message"] = "Training started, but progress tracking file not yet created."
            
    elif status == "completed":
        response["result"] = job.get("result", {})
    elif status == "error":
        response["error"] = job.get("error")

    return response

@router.get("/train/result/{job_id}")
async def train_result(job_id: str):
    job = JOB_MANAGER.get_job(job_id)
    if not job:
        return {"status": "not_found"}
    if job.get("status") != "completed":
        return {"status": job.get("status")}
    return {"status": "completed", **job.get("result", {})}

# -----------------------------------------------------------------------------
# Model Inference/Loading/Unloading Endpoints
# -----------------------------------------------------------------------------

@router.post("/load-model")
async def load_model(req: LoadModelRequest):
    pipeline = get_pipeline_by_task(req.job_id)
    out = pipeline.load_model_step(req.job_id)
    return {"status": "ok", "details": out}


@router.post("/infer")
async def infer_image(
    job_id: str,
    file: UploadFile = File(...),
):
    pipeline = get_pipeline_by_task(job_id)
    if not MODEL_CACHE_MANAGER.get_model_bundle(job_id):
        raise HTTPException(status_code=400, detail=f"Model for job {job_id} not loaded. Call /load-model first.")
    try:
        img_bytes = await file.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        result = pipeline.infer_step(job_id, image)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload-models")
async def unload_models():
    # Since we need to call a method that delegates to the cache manager, 
    # we can use either pipeline (they implement the same cache unloading)
    pipeline = ClassificationPipeline() # Or DetectionPipeline()
    result = pipeline.unload_all_models() 
    return {"status": "ok", "details": result}


@router.post("/unload-model")
async def unload_single_model(job_id: str):
    # Get the correct pipeline instance to call its unload method
    pipeline = get_pipeline_by_task(job_id) 
    result = pipeline.unload_model(job_id) 
    return {"status": "ok", "details": result}