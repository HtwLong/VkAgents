import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from cvmodellearning.paths import RUNS_ROOT
from cvmodellearning.jobs.job_manager import JOB_MANAGER
from cvmodellearning.jobs.run_control import mark_stopped, read_run_state, request_cancellation
from cvmodellearning.graphrag.decision_evidence import build_dataset_selection_decision_evidence
from cvmodellearning.evaluation.result_report import DETECTION_REPORT_SCHEMA_VERSION, normalize_report
from cvmodellearning.models.registry import enabled_models
from cvmodellearning.schemas.post_training_assessment import (
    AssessmentEligibility,
    PostTrainingAssessment,
)
from cvmodellearning.schemas.revision import RevisionPlan
from cvmodellearning.llm_config import ASSESSMENT_MODEL


router = APIRouter(prefix="/runs", tags=["4 - Runs"])
ASSESSMENT_FILE = "artifacts/post_training_assessment.json"

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


def _assessment_eligibility(run: Path) -> AssessmentEligibility:
    report = _json_object(run / "artifacts" / "evaluation_report.json")
    state = _latest_state(run)
    if report is None:
        return AssessmentEligibility(
            eligible=False,
            reason="A completed evaluation report is required.",
        )
    if not isinstance(state, dict) or not str(state.get("user_query") or "").strip():
        return AssessmentEligibility(
            eligible=False,
            reason="The original user request is unavailable.",
        )
    planning = run / "artifacts" / "planning"
    has_restart_state = any(
        (planning / name).is_file()
        for name in (
            "STATE_01_INTERPRETATION.json",
            "STATE_02_DATA_CHECK.json",
            "STATE_03_MODEL_SELECTION.json",
            "STATE_04_DATASET_SELECTION.json",
        )
    )
    return AssessmentEligibility(
        eligible=True,
        can_create_revision=has_restart_state,
        revision_reason=(
            None if has_restart_state
            else "Historical planning checkpoints are unavailable; results can be assessed but not safely forked."
        ),
    )


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


@router.get("/{job_id}/assessment")
def get_assessment(job_id: str):
    run = _existing_run(job_id)
    eligibility = _assessment_eligibility(run)
    assessment = _read_json(run / ASSESSMENT_FILE)
    return {
        "assessment": assessment if isinstance(assessment, dict) else None,
        "eligibility": eligibility.model_dump(mode="json"),
    }


def _validate_assessment_recommendation(
    assessment: PostTrainingAssessment,
    state: dict,
) -> None:
    plan = assessment.recommended_plan
    if plan is None:
        return
    allowed_targets = {"model-selection", "dataset-selection", "choose-hyperparameters"}
    targets = {change.target_step for change in plan.changes}
    if not targets <= allowed_targets or len(targets) != 1:
        raise HTTPException(
            status_code=502,
            detail=(
                "The assessment recommendation must change exactly one area: model selection, "
                "dataset selection, or hyperparameter selection."
            ),
        )
    if plan.restart_from not in targets:
        raise HTTPException(status_code=502, detail="The recommendation restart step is inconsistent.")
    for change in plan.changes:
        if change.strength != "required" or change.operation not in {"set", "include", "exclude"}:
            raise HTTPException(
                status_code=502,
                detail="Approved recommendations must contain concrete, enforceable required changes.",
            )
        if change.target_step == "model-selection" and change.field != "model_name":
            raise HTTPException(status_code=502, detail="The model recommendation field is unsupported.")
        if change.target_step == "dataset-selection" and change.field not in {
            "dataset.include", "dataset.exclude"
        }:
            raise HTTPException(status_code=502, detail="The dataset recommendation field is unsupported.")
        if change.target_step == "choose-hyperparameters":
            field = change.field.removeprefix("hpo_config.")
            if not change.field.startswith("hpo_config.") or field not in (state.get("hpo_config") or {}):
                raise HTTPException(status_code=502, detail="The hyperparameter recommendation field is unsupported.")


@router.post("/{job_id}/assessment")
async def create_assessment(job_id: str):
    run = _existing_run(job_id)
    eligibility = _assessment_eligibility(run)
    if not eligibility.eligible:
        raise HTTPException(status_code=409, detail=eligibility.reason)
    existing = _read_json(run / ASSESSMENT_FILE)
    if isinstance(existing, dict):
        return {
            "assessment": existing,
            "eligibility": eligibility.model_dump(mode="json"),
        }

    state = _latest_state(run) or {}
    report = normalize_report(_json_object(run / "artifacts" / "evaluation_report.json") or {})
    review_input = {
        "original_user_request": state.get("user_query"),
        "interpreted_requirements": {
            "task": state.get("task"),
            "classes": state.get("classes"),
            "performance_requirements": state.get("performance_requirements"),
            "deployment_constraints": state.get("deployment_constraints"),
            "available_hardware": state.get("available_hardware"),
        },
        "planning": {
            "selected_model_info": state.get("selected_model_info"),
            "selected_data": state.get("selected_data"),
            "dataset_profile": state.get("dataset_profile"),
            "hpo_config": state.get("hpo_config"),
            "executable_model_options": [
                {"id": model.id, "display_name": model.display_name, "family": model.family}
                for model in enabled_models(state.get("task"))
            ] if state.get("task") in {
                "classification", "detection", "visual question answering"
            } else [],
        },
        "evaluation_report": report,
    }
    prompt = f"""
Evaluate whether this completed computer-vision run satisfied the original user's requirements.
Use only the supplied evidence. Mark a requirement unknown when it was not measured; do not
invent metrics, gains, datasets, or configuration values. Preserve the user's requested task,
classes, deployment constraints, priorities, and other requirements. Do not optimize a metric by
violating the original request. Do not blindly increase epochs, batch size, input size, dataset
size, augmentation, or model capacity. Prefer the smallest evidence-backed intervention and
explain its trade-off. A recommendation must choose exactly ONE of these planning areas:
- model-selection: model_name (required set; use an executable ID from the supplied options)
- dataset-selection: dataset.include or dataset.exclude (required include/exclude)
- choose-hyperparameters: hpo_config.<existing configuration field> (required set)
Multiple changes are allowed only when they belong to that same area and are jointly necessary.
Set required_text or preferred_text consistently with change strengths. Recommend no plan when
the evidence does not justify a concrete actionable improvement. A concrete option presented for
approval must use strength=required with an enforcing operation (set/include/exclude), not merely
prefer or avoid. The server will verify the plan.

RUN EVIDENCE:
{json.dumps(review_input, indent=2, default=str)}
"""
    try:
        response = await AsyncOpenAI().beta.chat.completions.parse(
            model=ASSESSMENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=PostTrainingAssessment,
        )
        assessment = response.choices[0].message.parsed
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The post-training assessment could not be generated: {exc}",
        ) from exc
    if assessment is None:
        raise HTTPException(status_code=502, detail="The assessment model returned no result.")
    assessment = assessment.model_copy(update={"job_id": job_id})
    _validate_assessment_recommendation(assessment, state)
    _write_json_atomic(run / ASSESSMENT_FILE, assessment.model_dump(mode="json"))
    return {
        "assessment": assessment.model_dump(mode="json"),
        "eligibility": eligibility.model_dump(mode="json"),
    }


@router.post("/{job_id}/assessment/recommendation")
async def regenerate_recommendation(job_id: str):
    """Keep the assessment verdict and generate only an alternative revision plan."""
    run = _existing_run(job_id)
    eligibility = _assessment_eligibility(run)
    if not eligibility.eligible:
        raise HTTPException(status_code=409, detail=eligibility.reason)
    saved = _read_json(run / ASSESSMENT_FILE)
    if not isinstance(saved, dict):
        raise HTTPException(status_code=409, detail="Generate the assessment first.")
    try:
        assessment = PostTrainingAssessment.model_validate(saved)
    except Exception as exc:
        raise HTTPException(status_code=409, detail="The saved assessment is invalid.") from exc
    state = _latest_state(run) or {}
    report = normalize_report(_json_object(run / "artifacts" / "evaluation_report.json") or {})
    current_plan = (
        assessment.recommended_plan.model_dump(mode="json")
        if assessment.recommended_plan else None
    )
    prompt = f"""
Generate an alternative, concrete planning recommendation for improving this completed run.
Do NOT reassess or change the existing verdict, requirement statuses, or explanations. Preserve
the original task, classes, constraints, and priorities. Use only the supplied evidence. Do not
blindly increase epochs, batch size, input size, augmentation, dataset size, or model capacity.
Prefer the smallest evidence-backed intervention with a clear causal rationale. Avoid repeating
the previous recommendation when another supported option exists.

Choose exactly ONE area:
- model-selection: required set of model_name using an executable ID from the options
- dataset-selection: required include/exclude of dataset.include or dataset.exclude
- choose-hyperparameters: required set of hpo_config.<existing field>
Multiple changes are allowed only within the same area when jointly necessary. Return null when
no different evidence-backed recommendation is available.

ORIGINAL REQUEST: {json.dumps(state.get("user_query"), default=str)}
INTERPRETED REQUIREMENTS: {json.dumps({
    "task": state.get("task"), "classes": state.get("classes"),
    "performance_requirements": state.get("performance_requirements"),
    "deployment_constraints": state.get("deployment_constraints"),
}, indent=2, default=str)}
EXISTING ASSESSMENT: {json.dumps({
    "verdict": assessment.verdict,
    "requirements": [item.model_dump(mode="json") for item in assessment.requirements],
    "limitations": assessment.limitations,
}, indent=2, default=str)}
PREVIOUS RECOMMENDATION: {json.dumps(current_plan, indent=2, default=str)}
MODEL OPTIONS: {json.dumps([
    {"id": model.id, "display_name": model.display_name, "family": model.family}
    for model in enabled_models(state.get("task"))
] if state.get("task") in {"classification", "detection", "visual question answering"} else [])}
CURRENT HYPERPARAMETERS: {json.dumps(state.get("hpo_config") or {}, indent=2, default=str)}
SELECTED DATA: {json.dumps(state.get("selected_data") or [], indent=2, default=str)}
EVALUATION: {json.dumps(report, indent=2, default=str)}
"""
    # A small wrapper is required so structured output can represent "no alternative".
    class ParsedRecommendation(BaseModel):
        recommended_plan: RevisionPlan | None = None

    try:
        response = await AsyncOpenAI().beta.chat.completions.parse(
            model=ASSESSMENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=ParsedRecommendation,
        )
        parsed = response.choices[0].message.parsed
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The improvement recommendation could not be generated: {exc}",
        ) from exc
    if parsed is None:
        raise HTTPException(status_code=502, detail="The recommendation model returned no result.")
    updated = assessment.model_copy(update={"recommended_plan": parsed.recommended_plan})
    _validate_assessment_recommendation(updated, state)
    _write_json_atomic(run / ASSESSMENT_FILE, updated.model_dump(mode="json"))
    return {
        "assessment": updated.model_dump(mode="json"),
        "eligibility": eligibility.model_dump(mode="json"),
    }


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
    report_value = _json_object(run / "artifacts" / "evaluation_report.json")
    report_ready = report_value is not None
    rich_detection_report_ready = bool(
        report_value
        and (
            report_value.get("task") != "detection"
            or int(report_value.get("schema_version", 1)) >= DETECTION_REPORT_SCHEMA_VERSION
        )
    )
    if step_id == "running-evaluation":
        return rich_detection_report_ready
    if step_id == "preparing-trained-model":
        return model_ready
    if step_id == "preparing-results":
        return rich_detection_report_ready
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
        planning / "STATE_ACTIVE_REVISION.json",
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
    if isinstance(report, dict):
        report = normalize_report(report)
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

    terminal_complete = (
        _step_complete(run, "running-evaluation")
        and _step_complete(run, "preparing-trained-model")
    )
    if terminal_complete:
        # A valid final report and model are conclusive evidence that this linear
        # pipeline finished, including for runs created before every checkpoint existed.
        for step in steps.values():
            step["status"] = "done"
    all_complete = all(step["status"] == "done" for step in steps.values())
    any_running = any(step["status"] == "running" for step in steps.values())
    if all_complete:
        status = "done"
    elif errors:
        # A persisted request failure is stronger evidence than a stale
        # synchronous-step "running" marker left behind by the failed request.
        status = "failed"
    elif run_state and run_state.get("status") in {"running", "cancelling", "stopped"}:
        status = run_state["status"]
    elif any_running:
        status = "running"
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
        "planning_llm_usage": _read_json(
            run / "artifacts" / "planning" / "planning_llm_usage.json"
        ),
        "artifacts": artifacts,
        "errors": errors,
        "run_state": run_state,
        "post_training_assessment": _read_json(run / ASSESSMENT_FILE),
        "assessment_eligibility": _assessment_eligibility(run).model_dump(mode="json"),
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
