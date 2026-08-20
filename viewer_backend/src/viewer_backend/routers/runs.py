from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..llm import structured_call
from ..schemas import AssessmentDraft, StepTimingUpdate
from ..settings import ASSESSMENT_MODEL, RUNS_ROOT
from ..store import latest_context, planning_dir, read_json, run_dir, write_json
from .artifacts import manifest_for


router = APIRouter(prefix="/runs", tags=["Runs"])

STEP_FILES = {
    "task-interpretation": ["artifacts/planning/STATE_01_INTERPRETATION.json"],
    "check-data": ["artifacts/planning/STATE_02_DATA_CHECK.json"],
    "model-selection": ["artifacts/planning/STATE_03_MODEL_SELECTION.json"],
    "dataset-selection": ["artifacts/planning/STATE_04_DATASET_SELECTION.json"],
    "choose-hyperparameters": ["artifacts/planning/STATE_05_HYPERPARAMETERS.json", "artifacts/planning/RESULT_HYPERPARAMETERS.json"],
    "ask-change-requests": ["artifacts/planning/STATE_USER_CHANGE_REQUEST.json"],
    "download-data": ["artifacts/download_report.json"],
    "prepare-data": ["data/preparation_summary.json"],
    "train-model": ["artifacts/metrics_log.csv", "artifacts/training_log.txt"],
    "running-evaluation": ["artifacts/evaluation_report.json"],
    "preparing-trained-model": ["artifacts/best_model.pt", "artifacts/best_model.pth"],
    "preparing-results": ["artifacts/evaluation_report.json"],
}
INLINE_SUFFIXES = {".json", ".csv", ".txt", ".yaml", ".yml", ".sparql"}


def _output(path: Path) -> str | None:
    if path.suffix.lower() not in INLINE_SUFFIXES or path.stat().st_size > 1_000_000:
        return None
    try:
        return f"{path.name}:\n{path.read_text(encoding='utf-8')}"
    except OSError:
        return None


def _assessment_eligibility(base: Path) -> dict:
    report = base / "artifacts" / "evaluation_report.json"
    state = latest_context(base)
    eligible = report.is_file() and isinstance(state, dict) and bool(str(state.get("user_query") or "").strip())
    can_revise = any((base / "artifacts" / "planning" / name).is_file() for name in (
        "STATE_01_INTERPRETATION.json", "STATE_03_MODEL_SELECTION.json", "STATE_04_DATASET_SELECTION.json"
    ))
    return {
        "eligible": eligible,
        "reason": None if eligible else "A completed evaluation report and original request are required.",
        "can_create_revision": eligible and can_revise,
        "revision_reason": None if can_revise else "Planning checkpoints are unavailable.",
    }


@router.get("")
def list_runs():
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return {
        "runs": [item.name for item in sorted(RUNS_ROOT.iterdir()) if item.is_dir() and not item.name.startswith(".")]
    }


@router.get("/{job_id}")
def snapshot(job_id: str):
    base = run_dir(job_id)
    steps = {}
    for step_id, relatives in STEP_FILES.items():
        files = [base / relative for relative in relatives if (base / relative).is_file()]
        timing = read_json(base / "artifacts" / "timings" / f"{step_id}.json")
        # Historical checkpoint evidence is displayed as completed. Model weights are never
        # required to mark the presentation step complete when an evaluation report exists.
        complete = bool(files)
        if step_id == "preparing-trained-model" and (base / "artifacts" / "evaluation_report.json").is_file():
            complete = True
        steps[step_id] = {
            "status": "done" if complete else "pending",
            "outputs": [value for path in files if (value := _output(path)) is not None],
            "duration_ms": timing.get("duration_ms") if isinstance(timing, dict) else None,
        }
    context = latest_context(base)
    chosen = read_json(base / "artifacts" / "planning" / "RESULT_HYPERPARAMETERS.json")
    report = read_json(base / "artifacts" / "evaluation_report.json")
    assessment = read_json(base / "artifacts" / "post_training_assessment.json")
    evidence = {}
    if isinstance(context, dict):
        for step_id, field in (
            ("model-selection", "model_selection_decision_evidence"),
            ("dataset-selection", "dataset_selection_decision_evidence"),
            ("choose-hyperparameters", "hyperparameter_decision_evidence"),
        ):
            if context.get(field):
                evidence[step_id] = context[field]
    return {
        "job_id": job_id,
        "status": "done" if report else "waiting" if chosen else "idle",
        "steps": steps,
        "context": context,
        "chosen_parameters": chosen,
        "decision_evidence": evidence,
        "evaluation_report": report,
        "planning_llm_usage": read_json(base / "artifacts" / "planning" / "planning_llm_usage.json"),
        "post_training_assessment": assessment,
        "assessment_eligibility": _assessment_eligibility(base),
        "artifacts": manifest_for(job_id),
        "errors": read_json(base / "errors.json"),
        "run_state": None,
    }


@router.put("/{job_id}/steps/{step_id}/timing")
def update_timing(job_id: str, step_id: str, update: StepTimingUpdate):
    if step_id not in STEP_FILES:
        raise HTTPException(status_code=400, detail="Unknown pipeline step.")
    path = run_dir(job_id, create=True) / "artifacts" / "timings" / f"{step_id}.json"
    write_json(path, update.model_dump(mode="json"))
    return update.model_dump(mode="json")


@router.get("/{job_id}/assessment")
def get_assessment(job_id: str):
    base = run_dir(job_id)
    return {
        "assessment": read_json(base / "artifacts" / "post_training_assessment.json"),
        "eligibility": _assessment_eligibility(base),
    }


@router.post("/{job_id}/assessment")
async def create_assessment(job_id: str):
    base = run_dir(job_id)
    eligibility = _assessment_eligibility(base)
    if not eligibility["eligible"]:
        raise HTTPException(status_code=409, detail=eligibility["reason"])
    path = base / "artifacts" / "post_training_assessment.json"
    existing = read_json(path)
    if isinstance(existing, dict):
        return {"assessment": existing, "eligibility": eligibility}
    state = latest_context(base) or {}
    report = read_json(base / "artifacts" / "evaluation_report.json") or {}
    draft = await structured_call(
        job_id=job_id,
        operation="post_training_assessment",
        model=ASSESSMENT_MODEL,
        response_model=AssessmentDraft,
        prompt=(
            "Assess whether this historical run satisfied the original request. Use only supplied "
            "evidence; mark unmeasured requirements unknown. Recommend at most one planning-area "
            "change and never suggest executing work in this service.\n\n"
            f"PLANNING STATE: {json.dumps(state, default=str)}\n"
            f"EVALUATION REPORT: {json.dumps(report, default=str)}"
        ),
    )
    persisted = draft.persisted(job_id)
    write_json(path, persisted)
    return {"assessment": persisted, "eligibility": eligibility}


@router.post("/{job_id}/assessment/recommendation")
async def regenerate_recommendation(job_id: str):
    base = run_dir(job_id)
    path = base / "artifacts" / "post_training_assessment.json"
    current = read_json(path)
    if not isinstance(current, dict):
        raise HTTPException(status_code=409, detail="Generate the assessment first.")
    state = latest_context(base) or {}
    report = read_json(base / "artifacts" / "evaluation_report.json") or {}
    draft = await structured_call(
        job_id=job_id,
        operation="assessment_recommendation",
        model=ASSESSMENT_MODEL,
        response_model=AssessmentDraft,
        prompt=(
            "Return the same assessment verdict and requirements, but provide a different supported "
            "recommendation if one exists. Do not invent evidence.\n\n"
            f"CURRENT: {json.dumps(current, default=str)}\nSTATE: {json.dumps(state, default=str)}\n"
            f"REPORT: {json.dumps(report, default=str)}"
        ),
    )
    persisted = draft.persisted(job_id)
    write_json(path, persisted)
    return {"assessment": persisted, "eligibility": _assessment_eligibility(base)}


@router.post("/{job_id}/cancel")
def cancel(job_id: str):
    run_dir(job_id)
    return {"job_id": job_id, "status": "stopped", "message": "No execution jobs run in viewer mode."}

