import asyncio
import json
import traceback
from io import BytesIO
from typing import Any, Dict, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel, ValidationError
from PIL import Image

# Import shared dependencies
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import (
    DetectionConfigModel,
    expand_detection_config_for_validation,
)
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.graphrag.hyperparameter_context import validate_executable_recipe_config
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from cvmodellearning.pipelines.detection_pipe import DetectionPipeline  
from cvmodellearning.models.model_manager import MODEL_CACHE_MANAGER
from cvmodellearning.jobs.job_manager import JOB_MANAGER
from cvmodellearning.jobs.run_control import (
    PipelineCancelled,
    clear_cancellation,
    finish_or_stop,
    mark_stopped,
    raise_if_cancelled,
    write_run_state,
)
from cvmodellearning.jobs.error_persistence import ErrorPersistingRoute, save_run_error
from cvmodellearning.paths import (
    json_labels_path,
    run_dir,
    csv_labels_path,
    dataset_manifest_path,
    hpo_config_path,
    evaluation_report_path,
    interpretation_path,
    download_progress_path,
)
from cvmodellearning.download.progress import read_download_progress

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
            elif task == "visual question answering":
                pass
            else:
                # Handle other tasks as they are implemented.
                raise HTTPException(
                    status_code=400, 
                    detail=f"Task '{task}' found in interpretation file, but the corresponding pipeline is not yet implemented or supported."
                )

    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error reading or parsing interpretation file for job {job_id}: {e}"
        )

def _validate_config(
    pipeline: PipelineType,
    params: Dict[str, Any],
    *,
    job_id: str | None = None,
    require_saved_config: bool = False,
) -> Dict[str, Any]:
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
        # Planning persists optimizer/criterion as nested runtime objects, while
        # the structured-output schemas are intentionally flat. Normalize both
        # representations before applying the same deterministic validation at
        # the execution boundary.
        normalized_params = training_compatible_hpo_config(params)
        normalized_params.setdefault("rationale", "Validated execution configuration.")
        if ConfigModel is DetectionConfigModel:
            normalized_params = expand_detection_config_for_validation(normalized_params)
        # Validate the expanded schema, then project it back to the clean
        # execution shape before comparison and training.
        validated_model = ConfigModel.model_validate(normalized_params)
        validated_full = validated_model.model_dump()
        validate_executable_recipe_config(validated_full)
        validated = training_compatible_hpo_config(validated_model.runtime_config())
        saved_path = hpo_config_path(job_id) if job_id else None
        if require_saved_config and (saved_path is None or not saved_path.exists()):
            raise HTTPException(
                status_code=409,
                detail=(
                    "No graph-validated hyperparameter configuration was saved for this job. "
                    "Complete choose-hyperparameters before execution."
                ),
            )
        if saved_path and saved_path.exists():
            try:
                saved_params = json.loads(saved_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Saved hyperparameter configuration is unreadable: {exc}",
                ) from exc
            if not isinstance(saved_params, dict):
                raise HTTPException(
                    status_code=500,
                    detail="Saved hyperparameter configuration must be a JSON object.",
                )

            try:
                saved_normalized = training_compatible_hpo_config(saved_params)
                saved_normalized.setdefault("rationale", "Validated execution configuration.")
                if ConfigModel is DetectionConfigModel:
                    saved_normalized = expand_detection_config_for_validation(saved_normalized)
                saved_model = ConfigModel.model_validate(saved_normalized)
                saved_full = saved_model.model_dump()
                validate_executable_recipe_config(saved_full)
                saved_validated = training_compatible_hpo_config(saved_model.runtime_config())
            except (ValidationError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Saved hyperparameter configuration is invalid: {exc}",
                ) from exc
            changed_fields = sorted(
                field
                for field in validated_full.keys() | saved_full.keys()
                if field not in {"rationale", "llm_field_rationales"}
                and validated_full.get(field) != saved_full.get(field)
            )
            if changed_fields:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            "Submitted parameters differ from the graph-validated configuration "
                            "saved during planning."
                        ),
                        "changed_fields": changed_fields,
                    },
                )
            return saved_validated
        return validated
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Configuration validation failed for {pipeline.__class__.__name__}: {e.errors()}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Configuration validation failed for {pipeline.__class__.__name__}: {e}",
        )

router = APIRouter(
    tags=["2 - Execution & Inference"],
    route_class=ErrorPersistingRoute,
)

# --- Schemas ---
class StepRequest(BaseModel):
    job_id: str
    chosen_parameters: Union[Dict[str, Any], str]

class TrainStartRequest(BaseModel):
    chosen_parameters: Union[Dict[str, Any], str]
    job_id: str
    
class LoadModelRequest(BaseModel):
    job_id: str



# -----------------------------------------------------------------------------
# Background Task
# -----------------------------------------------------------------------------

async def _bg_train(job_id: str, params: Dict[str, Any]):
    """Background task logic."""
    try:
        write_run_state(job_id, "running", active_step="train-model")
        # The summary is the training commit marker. Remove an older marker so
        # a stopped retry can never be confused with a successful prior attempt.
        (run_dir(job_id) / "summary.json").unlink(missing_ok=True)
        (run_dir(job_id) / "progress.json").unlink(missing_ok=True)
        # Get the correct pipeline for the job
        pipeline: PipelineType = get_pipeline_by_task(job_id)

        cfg = _validate_config(
            pipeline,
            params,
            job_id=job_id,
            require_saved_config=isinstance(pipeline, (ClassificationPipeline, DetectionPipeline)),
        )
        run = run_dir(job_id)
        
        try:
            pipeline._require_prepared_data(cfg, job_id)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
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
        
        summary_path = run / "summary.json"
        temporary_summary = summary_path.with_suffix(".json.tmp")
        temporary_summary.write_text(json.dumps(summary, indent=2))
        temporary_summary.replace(summary_path)
        config_path = run / "config.json"
        temporary_config = config_path.with_suffix(".json.tmp")
        temporary_config.write_text(json.dumps(cfg, indent=2))
        temporary_config.replace(config_path)
        
        JOB_MANAGER.update_job_status(job_id, "completed", result=summary)
        finish_or_stop(job_id, "train-model")
        
    except PipelineCancelled:
        mark_stopped(job_id, "train-model")
        JOB_MANAGER.update_job_status(job_id, "stopped")
    except Exception as e:
        error_detail = f"CRITICAL BACKGROUND ERROR:\n{traceback.format_exc()}"
        print(error_detail)
        try:
            save_run_error(
                job_id,
                "background:train",
                error_detail,
                error_type=type(e).__name__,
            )
        except OSError:
            pass
        JOB_MANAGER.update_job_status(job_id, "error", error=error_detail)

# -----------------------------------------------------------------------------
# Pipeline Steps
# -----------------------------------------------------------------------------

@router.post("/download-data")
async def step_download(req: StepRequest):
    if not JOB_MANAGER.start_step(req.job_id, "download-data"):
        raise HTTPException(status_code=409, detail="Data download is already running for this job.")
    try:
        clear_cancellation(req.job_id)
        write_run_state(req.job_id, "running", active_step="download-data")
        pipeline = get_pipeline_by_task(req.job_id)
        params = req.chosen_parameters if isinstance(req.chosen_parameters, dict) else json.loads(req.chosen_parameters)
        cfg = _validate_config(
            pipeline,
            params,
            job_id=req.job_id,
            require_saved_config=isinstance(pipeline, (ClassificationPipeline, DetectionPipeline)),
        )
        run = run_dir(req.job_id)
    except Exception:
        JOB_MANAGER.finish_step(req.job_id, "download-data")
        raise

    def execute_download():
        try:
            raise_if_cancelled(req.job_id)
            out = pipeline.download_data_step(cfg, req.job_id)
            (run / "config.json").write_text(json.dumps(cfg, indent=2))
            finish_or_stop(req.job_id, "download-data")
            return out
        except PipelineCancelled:
            mark_stopped(req.job_id, "download-data")
            raise
        finally:
            # Cleanup belongs to the worker: cancelling the HTTP request does not
            # stop asyncio.to_thread, and the step remains active until it truly exits.
            JOB_MANAGER.finish_step(req.job_id, "download-data")

    try:
        out = await asyncio.to_thread(execute_download)
    except PipelineCancelled:
        return {"status": "stopped", "job_id": req.job_id}
    return {"status": "ok", "job_id": req.job_id, "output": out}

@router.get("/download-data/status/{job_id}")
async def download_status(job_id: str):
    """Return the latest atomic download-progress snapshot for polling clients."""
    try:
        progress = read_download_progress(download_progress_path(job_id))
    except json.JSONDecodeError:
        raise HTTPException(status_code=503, detail="Download progress is being updated; retry shortly.")
    if progress is None:
        return {"job_id": job_id, "status": "pending", "downloaded": 0, "processed": 0,
                "active": JOB_MANAGER.is_step_active(job_id, "download-data")}
    progress["active"] = JOB_MANAGER.is_step_active(job_id, "download-data")
    return progress

@router.post("/prepare-data")
async def step_prepare(req: StepRequest):
    clear_cancellation(req.job_id)
    write_run_state(req.job_id, "running", active_step="prepare-data")
    raise_if_cancelled(req.job_id)
    pipeline = get_pipeline_by_task(req.job_id)
    params = req.chosen_parameters if isinstance(req.chosen_parameters, dict) else json.loads(req.chosen_parameters)
    cfg = _validate_config(
        pipeline,
        params,
        job_id=req.job_id,
        require_saved_config=isinstance(pipeline, (ClassificationPipeline, DetectionPipeline)),
    )
    if not dataset_manifest_path(req.job_id).exists():
        raise HTTPException(status_code=400, detail="Dataset manifest missing; call /download-data first")
    # Adapt data existence check
    if isinstance(pipeline, ClassificationPipeline) and not csv_labels_path(req.job_id).exists():
        raise HTTPException(status_code=400, detail="Labels CSV missing; call /download-data first (Classification)")
    if isinstance(pipeline, DetectionPipeline) and not json_labels_path(req.job_id).exists():
        raise HTTPException(status_code=400, detail="Annotation JSON missing; call /download-data first (Detection)")
    def execute_prepare():
        try:
            raise_if_cancelled(req.job_id)
            result = pipeline.prepare_data_step(cfg, req.job_id)
            finish_or_stop(req.job_id, "prepare-data")
            return result
        except PipelineCancelled:
            mark_stopped(req.job_id, "prepare-data")
            raise

    try:
        out = await asyncio.to_thread(execute_prepare)
    except PipelineCancelled:
        return {"status": "stopped", "job_id": req.job_id}
    return {"status": "ok", "job_id": req.job_id, "output": out}

@router.post("/evaluate")
async def step_evaluate(req: StepRequest):
    clear_cancellation(req.job_id)
    write_run_state(req.job_id, "running", active_step="running-evaluation")
    raise_if_cancelled(req.job_id)
    pipeline = get_pipeline_by_task(req.job_id)
    params = req.chosen_parameters if isinstance(req.chosen_parameters, dict) else json.loads(req.chosen_parameters)
    cfg = _validate_config(
        pipeline,
        params,
        job_id=req.job_id,
        require_saved_config=isinstance(pipeline, (ClassificationPipeline, DetectionPipeline)),
    )
    def execute_evaluation():
        try:
            raise_if_cancelled(req.job_id)
            result = pipeline.evaluate_model_step(cfg, req.job_id)
            finish_or_stop(req.job_id, "running-evaluation")
            return result
        except PipelineCancelled:
            mark_stopped(req.job_id, "running-evaluation")
            raise

    try:
        out = await asyncio.to_thread(execute_evaluation)
    except PipelineCancelled:
        return {"status": "stopped", "job_id": req.job_id}
    return {"status": "ok", "job_id": req.job_id, "output": out}

@router.get("/evaluate/{job_id}/report")
async def evaluation_report(job_id: str):
    """Return the structured results produced by the completed evaluation."""
    path = evaluation_report_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not found; run /evaluate first")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Evaluation report is invalid") from exc

# -----------------------------------------------------------------------------
# Training Endpoints
# -----------------------------------------------------------------------------

@router.post("/train/start")
async def train_start(req: TrainStartRequest, bg: BackgroundTasks):
    params = req.chosen_parameters if isinstance(req.chosen_parameters, dict) else json.loads(req.chosen_parameters)
    job_id = req.job_id
    run_dir(job_id)

    try:
        # NOTE: This assumes interpretation_path exists or defaults to Classification
        # If get_pipeline_by_task fails, it raises HTTPException which FastAPI handles.
        temp_pipeline = get_pipeline_by_task(job_id)
        # _validate_config will raise HTTPException(400) on validation error.
        _validate_config(
            temp_pipeline,
            params,
            job_id=job_id,
            require_saved_config=isinstance(temp_pipeline, (ClassificationPipeline, DetectionPipeline)),
        )
        
    except HTTPException:
        # Let FastAPI handle the HTTPException (400 or 500)
        raise
    except Exception as e:
        # Catch any other immediate setup errors
        raise HTTPException(status_code=500, detail=f"Initial setup failed: {str(e)}")
    
    existing = JOB_MANAGER.get_job(job_id)
    if existing and existing.get("status") == "running":
        raise HTTPException(status_code=409, detail="Training is already running for this job.")
    clear_cancellation(job_id)
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
    
    progress_file = run_dir(job_id) / "progress.json"
    if progress_file.exists():
        try:
            progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
            # The job manager is authoritative; progress snapshots are written
            # while training and may still contain status="running" afterward.
            response.update({key: value for key, value in progress_data.items() if key != "status"})
        except json.JSONDecodeError:
            response["progress_error"] = "Could not parse training progress file."
    elif status == "running":
        response["message"] = "Training started, but no epoch has completed yet."

    if status == "completed":
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
    if out.get("status") == "load_failed":
        raise HTTPException(status_code=500, detail=out.get("error", "Model loading failed."))
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
        if result.get("status") == "inference_failed":
            raise HTTPException(status_code=500, detail=result.get("error", "Inference failed."))
        return {"status": "ok", "result": result}
    except HTTPException:
        raise
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
