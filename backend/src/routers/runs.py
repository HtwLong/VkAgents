import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cvmodellearning.paths import RUNS_ROOT
from cvmodellearning.jobs.job_manager import JOB_MANAGER
from cvmodellearning.jobs.run_control import mark_stopped, read_run_state, request_cancellation
from cvmodellearning.graphrag.decision_evidence import build_dataset_selection_decision_evidence


router = APIRouter(prefix="/runs", tags=["4 - Runs"])

STEP_FILES = {
    "task-interpretation": ["artifacts/planning/STATE_01_INTERPRETATION.json"],
    "check-data": [
        "artifacts/planning/STATE_02_DATA_CHECK.json",
        "artifacts/planning/DATA_CHECK_QUERY.sparql",
    ],
    "model-selection": ["artifacts/planning/STATE_03_MODEL_SELECTION.json"],
    "dataset-selection": [
        "artifacts/planning/STATE_04_DATASET_SELECTION.json",
        "artifacts/planning/STATE_04_PREPROCESSING.json",
    ],
    "choose-hyperparameters": [
        "artifacts/planning/STATE_05_HYPERPARAMETERS.json",
        "artifacts/planning/RESULT_HYPERPARAMETERS.json",
        "artifacts/planning/planning_rationales.txt",
    ],
    "ask-change-requests": ["artifacts/planning/STATE_USER_CHANGE_REQUEST.json"],
    "download-data": [
        "artifacts/download_report.json",
        "data/dataset_manifest.json",
        "artifacts/data_provenance.json",
    ],
    "prepare-data": [
        "data/preparation_summary.json",
        "data/train_labels.csv",
        "data/val_labels.csv",
        "data/test_labels.csv",
        "data/train_annotations.json",
        "data/val_annotations.json",
        "data/test_annotations.json",
        "data/yolo_data.yaml",
    ],
    "train-model": [
        "progress.json",
        "artifacts/training_log.txt",
        "artifacts/metrics_log.csv",
        "artifacts/metrics_log.json",
        "artifacts/tool_call_args.json",
    ],
    "running-evaluation": [
        "artifacts/test_classification_report.json",
        "artifacts/test_confusion_matrix.csv",
    ],
    "preparing-trained-model": [
        "artifacts/best_model.pt",
        "artifacts/best_model.pth",
        "artifacts/best_lora_adapter.zip",
        "artifacts/best_merged_model.pth",
    ],
    "preparing-results": ["artifacts/evaluation_report.json"],
}

INLINE_SUFFIXES = {".json", ".txt", ".csv", ".yaml", ".yml", ".sparql"}
MAX_INLINE_BYTES = 1_000_000
STEP_ORDER = list(STEP_FILES)
TIMINGS_DIR = "artifacts/timings"
OUTPUT_LABELS = {
    "STATE_01_INTERPRETATION.json": "Task interpretation",
    "STATE_02_DATA_CHECK.json": "Data check",
    "DATA_CHECK_QUERY.sparql": "Data availability query",
    "STATE_03_MODEL_SELECTION.json": "Model selection output",
    "STATE_04_DATASET_SELECTION.json": "Dataset selection output",
    "STATE_04_PREPROCESSING.json": "Preprocessing output",
    "STATE_05_HYPERPARAMETERS.json": "Hyperparameter planning output",
    "RESULT_HYPERPARAMETERS.json": "Chosen hyperparameters",
    "planning_rationales.txt": "Planning rationales",
    "STATE_USER_CHANGE_REQUEST.json": "Updated plan",
    "download_report.json": "Download output",
    "dataset_manifest.json": "Dataset assignment output",
    "data_provenance.json": "Data provenance",
    "preparation_summary.json": "Preparation output",
    "training_log.txt": "Training output",
    "metrics_log.csv": "Training metrics",
    "metrics_log.json": "Training metrics",
    "tool_call_args.json": "Training configuration",
    "test_classification_report.json": "Evaluation output",
    "test_confusion_matrix.csv": "Confusion matrix",
    "evaluation_report.json": "Evaluation report",
}


class StepTimingUpdate(BaseModel):
    duration_ms: int = Field(ge=0)
    status: str


@router.post("/{job_id}/cancel")
def cancel_run(job_id: str):
    """Request cooperative cancellation; repeated requests are harmless."""
    run = _existing_run(job_id)
    state = read_run_state(job_id) or {}
    if state.get("status") == "cancelling":
        active_step = state.get("active_step")
        if active_step in {"download-data", "train-model"} and not _step_running(job_id, active_step):
            mark_stopped(job_id, active_step)
            return read_run_state(job_id)
        return state
    if state.get("status") == "stopped" or _step_complete(run, "preparing-results"):
        return state
    if state.get("status") != "running":
        mark_stopped(job_id, state.get("active_step") or "pipeline")
        return read_run_state(job_id)
    return request_cancellation(job_id)


def _existing_run(job_id: str) -> Path:
    if not job_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID.")
    path = (RUNS_ROOT / job_id).resolve()
    if path.parent != RUNS_ROOT or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run {job_id} was not found.")
    return path


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _json_object(path: Path) -> dict | None:
    value = _read_json(path)
    return value if isinstance(value, dict) and value else None


def _step_complete(run: Path, step_id: str) -> bool:
    """Validate the final contract of a step, not merely an intermediate file."""
    planning = run / "artifacts" / "planning"
    if step_id == "task-interpretation":
        return _json_object(planning / "STATE_01_INTERPRETATION.json") is not None
    if step_id == "check-data":
        return _json_object(planning / "STATE_02_DATA_CHECK.json") is not None
    if step_id == "model-selection":
        return _json_object(planning / "STATE_03_MODEL_SELECTION.json") is not None
    if step_id == "dataset-selection":
        state = _json_object(planning / "STATE_04_DATASET_SELECTION.json")
        return state is not None and bool(state.get("selected_data"))
    if step_id == "choose-hyperparameters":
        state = _json_object(planning / "STATE_05_HYPERPARAMETERS.json")
        config = _json_object(planning / "RESULT_HYPERPARAMETERS.json")
        return state is not None and config is not None and bool(state.get("hpo_config"))
    if step_id == "ask-change-requests":
        # This is a user gate. Reaching it is represented by a completed HPO plan.
        return _step_complete(run, "choose-hyperparameters")
    if step_id == "download-data":
        report = _json_object(run / "artifacts" / "download_report.json")
        manifest = _json_object(run / "data" / "dataset_manifest.json")
        return bool(report and report.get("complete") is True and manifest and manifest.get("samples"))
    if step_id == "prepare-data":
        summary = _json_object(run / "data" / "preparation_summary.json")
        if not summary:
            return False
        task = summary.get("task")
        required = (
            ["train_labels.csv", "val_labels.csv", "test_labels.csv"]
            if task == "classification"
            else ["train_annotations.json", "val_annotations.json", "test_annotations.json"]
        )
        return all(_nonempty(run / "data" / name) for name in required)
    model_ready = any(_nonempty(run / "artifacts" / name) for name in (
        "best_model.pt", "best_model.pth", "best_lora_adapter.zip", "best_merged_model.pth"
    ))
    if step_id == "train-model":
        return model_ready and _json_object(run / "summary.json") is not None
    report_ready = _json_object(run / "artifacts" / "evaluation_report.json") is not None
    if step_id == "running-evaluation":
        return report_ready
    if step_id == "preparing-trained-model":
        return model_ready
    if step_id == "preparing-results":
        return report_ready
    return False


def _step_running(job_id: str, step_id: str) -> bool:
    if step_id == "download-data":
        return JOB_MANAGER.is_step_active(job_id, step_id)
    if step_id == "train-model":
        job = JOB_MANAGER.get_job(job_id)
        return bool(job and job.get("status") == "running")
    return False


def _timing_path(run: Path, step_id: str) -> Path:
    if step_id not in STEP_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline step: {step_id}.")
    return run / TIMINGS_DIR / f"{step_id}.json"


@router.put("/{job_id}/steps/{step_id}/timing")
def save_step_timing(job_id: str, step_id: str, timing: StepTimingUpdate):
    """Persist the frontend-observed wall-clock duration for one pipeline step."""
    if timing.status not in {"done", "failed"}:
        raise HTTPException(status_code=400, detail="Timing status must be done or failed.")
    run = _existing_run(job_id)
    path = _timing_path(run, step_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"step_id": step_id, "duration_ms": timing.duration_ms, "status": timing.status},
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return {"step_id": step_id, "duration_ms": timing.duration_ms, "status": timing.status}


def _latest_state(run: Path):
    planning = run / "artifacts" / "planning"
    candidates = [
        planning / "STATE_USER_CHANGE_REQUEST.json",
        planning / "STATE_05_HYPERPARAMETERS.json",
        planning / "STATE_04_DATASET_SELECTION.json",
        planning / "STATE_04_PREPROCESSING.json",
        planning / "STATE_03_MODEL_SELECTION.json",
        planning / "STATE_02_DATA_CHECK.json",
        planning / "STATE_01_INTERPRETATION.json",
    ]
    for path in candidates:
        if path.is_file() and (state := _read_json(path)) is not None:
            return state
    return None


def _file_output(path: Path):
    if path.suffix.lower() not in INLINE_SUFFIXES or path.stat().st_size > MAX_INLINE_BYTES:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if path.suffix.lower() == ".json":
        value = _read_json(path)
        if value is not None:
            text = json.dumps(value, indent=2)
    return f"{OUTPUT_LABELS.get(path.name, path.stem.replace('_', ' ').title())}:\n{text}"


def _deliverables(job_id: str, run: Path):
    artifacts = []
    model = next(
        (name for name in ("best_model.pt", "best_model.pth") if (run / "artifacts" / name).is_file()),
        None,
    )
    if model:
        artifacts.append({
            "id": "model",
            "kind": "full_model",
            "label": "Trained model",
            "filename": model,
            "download_url": f"/artifacts/{job_id}/model",
            "description": "Persisted model artifact for this run.",
        })
    for artifact_id, kind, label, relative, endpoint in (
        (
            "hyperparameters",
            "configuration",
            "Hyperparameter configuration",
            "artifacts/planning/RESULT_HYPERPARAMETERS.json",
            "planning/hyperparameters",
        ),
        (
            "data_provenance",
            "provenance_audit",
            "Data provenance audit",
            "artifacts/data_provenance.json",
            "data-provenance",
        ),
    ):
        if (run / relative).is_file():
            artifacts.append({
                "id": artifact_id,
                "kind": kind,
                "label": label,
                "filename": Path(relative).name,
                "download_url": f"/artifacts/{job_id}/{endpoint}",
            })
    return artifacts


@router.get("/{job_id}")
def run_snapshot(job_id: str):
    """Reconstruct the persisted UI state for one pipeline run."""
    run = _existing_run(job_id)
    steps = {}
    persisted_steps = set()
    for step_id, relatives in STEP_FILES.items():
        available = [relative for relative in relatives if (run / relative).is_file()]
        if available:
            persisted_steps.add(step_id)
        outputs = [
            output
            for relative in available
            if (output := _file_output(run / relative)) is not None
        ]
        timing = _read_json(_timing_path(run, step_id))
        duration_ms = timing.get("duration_ms") if isinstance(timing, dict) else None
        complete = _step_complete(run, step_id)
        running = not complete and _step_running(job_id, step_id)
        steps[step_id] = {
            "status": "done" if complete else "running" if running else "pending",
            "outputs": outputs,
            "duration_ms": duration_ms if isinstance(duration_ms, int) and duration_ms >= 0 else None,
        }

    state = _latest_state(run)
    saved_config = _json_object(
        run / "artifacts" / "planning" / "RESULT_HYPERPARAMETERS.json"
    )
    evidence = {}
    if isinstance(state, dict):
        for step_id, field in {
            "model-selection": "model_selection_decision_evidence",
            "dataset-selection": "dataset_selection_decision_evidence",
            "choose-hyperparameters": "hyperparameter_decision_evidence",
        }.items():
            if state.get(field):
                evidence[step_id] = state[field]
        if "dataset-selection" not in evidence and state.get("selected_data"):
            rationale = next(
                (
                    entry.removeprefix("Data Selection Rationale: ")
                    for entry in state.get("step_history", [])
                    if entry.startswith("Data Selection Rationale: ")
                ),
                "Persisted dataset selection.",
            )
            evidence["dataset-selection"] = build_dataset_selection_decision_evidence(
                state["selected_data"],
                rationale,
                state.get("dataset_selection_graph_context") or {},
            )

    report = _read_json(run / "artifacts" / "evaluation_report.json")
    errors = _read_json(run / "errors.json")
    artifacts = _deliverables(job_id, run)
    run_state = read_run_state(job_id)
    if run_state and run_state.get("status") == "cancelling":
        active_step = run_state.get("active_step")
        if active_step in {"download-data", "train-model"} and not _step_running(job_id, active_step):
            mark_stopped(job_id, active_step)
            run_state = read_run_state(job_id)
    if run_state and run_state.get("status") in {"running", "cancelling"}:
        active_step = run_state.get("active_step")
        if active_step in steps and steps[active_step]["status"] != "done":
            steps[active_step]["status"] = "running"

    terminal_complete = bool(report) and _step_complete(run, "preparing-trained-model")
    if terminal_complete:
        # A valid final report and model are conclusive evidence that this linear
        # pipeline finished, including for runs created before every checkpoint existed.
        for step in steps.values():
            step["status"] = "done"
    all_complete = all(step["status"] == "done" for step in steps.values())
    any_running = any(step["status"] == "running" for step in steps.values())
    if all_complete:
        status = "done"
    elif run_state and run_state.get("status") in {"running", "cancelling", "stopped"}:
        status = run_state["status"]
    elif any_running:
        status = "running"
    elif errors:
        status = "failed"
    elif state and state.get("hpo_config"):
        status = "waiting"
    else:
        status = "idle"

    return {
        "job_id": job_id,
        "status": status,
        "steps": steps,
        "context": state,
        "chosen_parameters": saved_config,
        "decision_evidence": evidence,
        "evaluation_report": report,
        "artifacts": artifacts,
        "errors": errors,
        "run_state": run_state,
    }


@router.delete("/{job_id}/errors")
def clear_run_errors(job_id: str):
    """Archive stale failures when the user explicitly continues or retries a run."""
    run = _existing_run(job_id)
    path = run / "errors.json"
    if path.exists():
        archived = run / "errors.previous.json"
        path.replace(archived)
    return {"job_id": job_id, "status": "ok"}
