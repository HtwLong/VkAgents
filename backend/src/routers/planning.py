import json
import asyncio
import re
import shutil
from uuid import uuid4
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI

# Core agent/pipeline imports
from agents import Runner  # Re-exported for existing tests that patch the shared runner.
from cvmodellearning.schemas.interpretation_schema import (
    ClassDataSelection,
    DeploymentConstraints,
    HardwareSpecModel,
    PerformanceSpecModel,
    PipelineState,
)
from cvmodellearning.training.hardware_profiles import active_training_hardware_profile
from cvmodellearning.training.resource_guard import validate_training_resource_config
from cvmodellearning.schemas.classification_hpo import (
    ClassificationConfigDraft,
    active_classification_config_fields,
)
from cvmodellearning.schemas.detection_hpo import DetectionConfigDraft, active_detection_config_fields
from cvmodellearning.schemas.vqa_hpo import VQAConfigModel
from cvmodellearning.agents.interpretation_agents import interpretation_loop, task_interpretation_agent, synonym_check_agent
from cvmodellearning.agents.model_selection_agents import classification_model_selector_agent, detection_model_selector_agent, vqq_model_selector_agent
from cvmodellearning.agents.hyperparameter_agents import (
    HpoPhaseTimeout,
    PIPELINE_OWNED_HPO_CONTEXT_FIELDS,
    apply_owned_pipeline_fields,
    generate_and_evaluate_hpo,
)
from cvmodellearning.agents.data_selection_and_augmentation_agents import (
    classification_dataset_selection_agent, detection_dataset_selection_agent,
    vqa_dataset_selection_agent,
)
from cvmodellearning.agents.agents_utils import save_json, load_unified_dataset_classes
from cvmodellearning.models.registry import (
    enabled_models,
    family_for_model_reference,
    is_executable_model_reference,
    model_ids_equivalent,
    resolve_detection_model_identity,
)
from cvmodellearning.paths import RUNS_ROOT, hpo_config_path, planning_artifacts_dir
from cvmodellearning.schemas.revision import (
    RevisionPlan,
    RevisionTarget,
    changes_for,
    earliest_revision_step,
)
from cvmodellearning.download.visionkg_utils import get_multi_class_stats
from cvmodellearning.datasets.selection import (
    DEFAULT_CLASSIFICATION_POOL_PER_CLASS,
    DEFAULT_DETECTION_POOL_PER_CLASS,
    MEDIUM_HIGH_DETECTION_POOL_PER_CLASS,
    DETECTION_SHARED_BACKBONE_MIN_COUNT,
    DETECTION_SHARED_BACKBONE_MIN_SHARE,
    DETECTION_SHARED_BACKBONE_TARGET_COUNT,
    MAX_CLASSIFICATION_POOL_PER_CLASS,
    MAX_CLASSIFICATION_SELECTED_IMAGES,
    MAX_DETECTION_SELECTED_IMAGES,
    MAX_DETECTION_POOL_PER_CLASS,
    MIN_SOURCE_POOL_FOR_PRIMARY_HOLDOUT,
    MIN_SOURCE_POOL_SHARE_FOR_PRIMARY_HOLDOUT,
    DatasetSelectionValidationError,
    build_dataset_assignments,
    build_dataset_profile,
    build_split_construction_summary,
    canonicalize_selected_dataset_ids,
    prune_ineligible_optional_sources,
    determine_detection_dataset_size,
    filter_dataset_candidates,
    infer_detection_sizing_facts,
    detection_domain_mix_findings,
    detection_source_coherence_findings,
    validate_detection_source_coherence,
    validate_dataset_selection,
)
from cvmodellearning.datasets.registry import DatasetRole, dataset_family, resolve_dataset_info
from cvmodellearning.policies.data_selection_policy import (
    DETECTION_DATA_SELECTION_POLICY,
    matched_domain_tags,
)
from cvmodellearning.graphrag.model_selection_context import (
    USE_MODEL_SELECTION_GRAPHRAG,
    build_model_selection_context,
    format_model_selection_context,
    summarize_model_selection_context,
)
from cvmodellearning.graphrag.dataset_selection_context import (
    USE_DATASET_SELECTION_GRAPHRAG,
    aggregate_selected_dataset_properties,
    build_dataset_selection_context,
)
from cvmodellearning.graphrag.hyperparameter_context import (
    USE_HYPERPARAMETER_GRAPHRAG,
    build_field_provenance,
    build_hyperparameter_context,
    format_hyperparameter_context,
    llm_controlled_fields,
    summarize_hyperparameter_context,
    validate_executable_recipe_config,
)
from cvmodellearning.graphrag.decision_evidence import (
    build_dataset_selection_decision_evidence,
    build_hyperparameter_decision_evidence,
    build_model_selection_decision_evidence,
)
from cvmodellearning.jobs.error_persistence import ErrorPersistingRoute
from cvmodellearning.llm_config import PLANNING_MODEL
from cvmodellearning.observability.planning_usage import (
    run_planning_agent,
    run_planning_completion,
)

router = APIRouter(
    prefix="/planning",
    tags=["1 - Planning & Interpretation"],
    route_class=ErrorPersistingRoute,
)

FALLBACK_HARDWARE_CATEGORY = "ConsumerCPU | EdgeDevice"
MIN_RETAINED_EXPANSION_CLASSES = 3


def deployment_coverage_warnings(state: PipelineState) -> list[dict[str, Any]]:
    """Report requested evaluation slices that lack sample-level verification."""

    query = (state.user_query or "").lower()
    dimensions = []
    if "indoor" in query or "outdoor" in query:
        dimensions.append("indoor_outdoor")
    if "light" in query:
        dimensions.append("lighting")
    if "scale" in query or "size" in query:
        dimensions.append("object_scale")
    if "occlu" in query:
        dimensions.append("occlusion")
    if any(term in query for term in ("weather", "rain", "snow", "fog")):
        dimensions.append("weather")
    if "viewpoint" in query or "view point" in query:
        dimensions.append("viewpoint")
    if not dimensions:
        return []
    return [{
        "code": "DEPLOYMENT_COVERAGE_UNVERIFIED",
        "severity": "warning",
        "dimensions": dimensions,
        "reason": (
            "The split is source-stratified, but the available sample metadata does "
            "not verify coverage of these requested deployment characteristics."
        ),
    }]

# --- Schemas ---
class CompletenessCheckRequest(BaseModel):
    job_id: str
    user_prompt: str
    user_replies: Optional[List[str]] = None

class CompletenessCheckResponse(BaseModel):
    accept: bool
    reason: Optional[str]
    suggestions: Optional[List[str]]
    context: Optional[str]

class StateRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    job_id: str
    use_graphrag: bool = True

class AddUserRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    request_text: Optional[str] = None
    job_id: Optional[str] = None

class RequestAddedResponse(BaseModel):
    context: Dict[str, Any]


class PlanRevisionRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    job_id: str
    required_changes: str = ""
    preferences: str = ""
    requested_target: Union[RevisionTarget, Literal["automatic"]] = "automatic"


class ActivateRevisionRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    plan: RevisionPlan
    job_id: str


class VerifyRevisionRequest(BaseModel):
    context: Union[str, Dict[str, Any]]


class ForkRevisionRequest(BaseModel):
    parent_job_id: str
    assessment_id: str
    plan: RevisionPlan


PREDECESSOR_CHECKPOINT = {
    "task-interpretation": None,
    "model-selection": "STATE_02_DATA_CHECK.json",
    "dataset-selection": "STATE_03_MODEL_SELECTION.json",
    "choose-hyperparameters": "STATE_04_DATASET_SELECTION.json",
}
DOWNSTREAM_PLANNING_FILES = {
    "task-interpretation": [
        "STATE_01_INTERPRETATION.json", "STATE_02_DATA_CHECK.json",
        "STATE_03_MODEL_SELECTION.json", "STATE_04_DATASET_SELECTION.json",
        "STATE_04_PREPROCESSING.json", "STATE_05_HYPERPARAMETERS.json",
        "RESULT_HYPERPARAMETERS.json", "planning_rationales.txt",
    ],
    "model-selection": [
        "STATE_03_MODEL_SELECTION.json", "STATE_04_DATASET_SELECTION.json",
        "STATE_04_PREPROCESSING.json", "STATE_05_HYPERPARAMETERS.json",
        "RESULT_HYPERPARAMETERS.json", "planning_rationales.txt",
    ],
    "dataset-selection": [
        "STATE_04_DATASET_SELECTION.json", "STATE_04_PREPROCESSING.json",
        "STATE_05_HYPERPARAMETERS.json", "RESULT_HYPERPARAMETERS.json",
        "planning_rationales.txt",
    ],
    "choose-hyperparameters": [
        "STATE_05_HYPERPARAMETERS.json", "RESULT_HYPERPARAMETERS.json",
        "planning_rationales.txt",
    ],
}

PLANNING_CHECKPOINTS = [
    ("task-interpretation", "STATE_01_INTERPRETATION.json"),
    ("check-data", "STATE_02_DATA_CHECK.json"),
    ("model-selection", "STATE_03_MODEL_SELECTION.json"),
    ("dataset-selection", "STATE_04_DATASET_SELECTION.json"),
]


# --- Helper Functions ---
def get_state(request_context: Union[str, Dict[str, Any]]) -> PipelineState:
    if isinstance(request_context, str):
        try:
            state = PipelineState(**json.loads(request_context))
        except (json.JSONDecodeError, TypeError, ValidationError):
            state = PipelineState(user_query=request_context)
    else:
        state = PipelineState(**request_context)
    if state.training_hardware is None:
        state.training_hardware = active_training_hardware_profile()
    return state

def save_checkpoint(state: PipelineState, job_id: str, filename: str):
    state.last_updated = datetime.now().isoformat()
    save_json(state.model_dump(), planning_artifacts_dir(job_id), filename)
    if filename != "STATE_ACTIVE_REVISION.json":
        active_revision = planning_artifacts_dir(job_id) / "STATE_ACTIVE_REVISION.json"
        if active_revision.is_file():
            active_revision.unlink()


def _revision_value(state: PipelineState, field: str) -> Any:
    if field == "model_name":
        model = (state.selected_model_info or {}).get("model") or {}
        return model.get("model_architecture")
    if field.startswith("hpo_config."):
        return (state.hpo_config or {}).get(field.removeprefix("hpo_config."))
    if field == "dataset.include":
        return sorted({
            source.dataset_name
            for assignment in state.selected_data or []
            for source in assignment.sources
        })
    if field == "dataset.exclude":
        return sorted({
            source.dataset_name
            for assignment in state.selected_data or []
            for source in assignment.sources
        })
    current: Any = state.model_dump(mode="json")
    for part in field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _change_satisfied(change, actual: Any) -> bool:
    if change.operation == "include":
        expected = change.value if isinstance(change.value, list) else [change.value]
        return all(item in (actual or []) for item in expected)
    if change.operation == "exclude":
        expected = change.value if isinstance(change.value, list) else [change.value]
        return all(item not in (actual or []) for item in expected)
    if change.operation == "avoid":
        expected = change.value if isinstance(change.value, list) else [change.value]
        if isinstance(actual, list):
            return all(item not in actual for item in expected)
        return all(actual != item for item in expected)
    return actual == change.value


def _validate_revision_plan(plan: RevisionPlan, state: PipelineState) -> None:
    if plan.restart_from != earliest_revision_step(plan.changes):
        raise HTTPException(status_code=422, detail="Revision restart step is inconsistent with its changes.")
    task_fields = {
        "task", "classes", "application_domain", "performance_requirements",
        "deployment_constraints", "available_hardware",
    }
    hpo_schema = {
        "classification": ClassificationConfigDraft,
        "detection": DetectionConfigDraft,
        "visual question answering": VQAConfigModel,
    }.get(state.task)
    errors = []
    for change in plan.changes:
        valid = False
        if change.target_step == "task-interpretation":
            valid = change.field in task_fields and change.operation == "set"
        elif change.target_step == "model-selection":
            valid = change.field == "model_name" and change.operation in {"set", "prefer", "avoid"}
        elif change.target_step == "dataset-selection":
            valid = change.field in {"dataset.include", "dataset.exclude"}
            valid = valid and change.operation in {"include", "exclude", "prefer", "avoid"}
        elif change.target_step == "choose-hyperparameters" and hpo_schema is not None:
            field = change.field.removeprefix("hpo_config.")
            valid = (
                change.field.startswith("hpo_config.")
                and field in hpo_schema.model_fields
                and change.operation in {"set", "prefer", "avoid"}
            )
        if not valid:
            errors.append({
                "change_id": change.id,
                "target_step": change.target_step,
                "field": change.field,
                "operation": change.operation,
            })
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "The revision contains unsupported changes.", "changes": errors},
        )


def _canonicalize_revision_plan(plan: RevisionPlan, state: PipelineState) -> RevisionPlan:
    """Resolve accepted model aliases before they become durable required values."""
    changes = []
    for change in plan.changes:
        if (
            state.task == "detection"
            and change.target_step == "model-selection"
            and change.field == "model_name"
            and change.operation == "set"
            and change.value is not None
        ):
            identity = resolve_detection_model_identity(str(change.value))
            if identity is not None:
                change = change.model_copy(update={"value": identity.executable_id})
        changes.append(change)
    return plan.model_copy(update={
        "changes": changes,
        "restart_from": earliest_revision_step(changes),
    })


@router.post("/plan-revision")
async def plan_revision(request: PlanRevisionRequest):
    if not request.required_changes.strip() and not request.preferences.strip():
        raise HTTPException(status_code=422, detail="Enter a required change or preference.")
    state = get_state(request.context)
    current = {
        "task": state.task,
        "selected_model_info": state.selected_model_info,
        "selected_dataset_ids": sorted({
            source.dataset_name
            for assignment in state.selected_data or []
            for source in assignment.sources
        }),
        "hpo_config": state.hpo_config,
    }
    target_instruction = (
        "Choose the earliest affected step."
        if request.requested_target == "automatic"
        else f"Use target_step={request.requested_target!r} for changes belonging to that scope; "
             "other explicitly requested fields may use their actual owning step."
    )
    prompt = f"""
Convert the user's planning revision into atomic, typed changes.
{target_instruction}
Valid target steps are task-interpretation, model-selection, dataset-selection, and choose-hyperparameters.
Use field='model_name' for a concrete model. Use field='dataset.include' or 'dataset.exclude'
for dataset IDs. Use field='hpo_config.<schema field>' for hyperparameters. Use operation set,
include, exclude, prefer, or avoid. A required concrete value uses set; dataset.include uses include
and dataset.exclude uses exclude. Text under REQUIRED must have strength required; text under
PREFERRED must have strength preferred. Never invent a concrete value. Set restart_from to the
earliest target_step in changes. Give every change a stable short id and plain-English summary.

Current planning state:
{json.dumps(current, indent=2, default=str)}

REQUIRED:
{request.required_changes.strip() or '(none)'}

PREFERRED:
{request.preferences.strip() or '(none)'}
"""
    try:
        response = await run_planning_completion(
            job_id=request.job_id,
            operation="planning_revision",
            model=PLANNING_MODEL,
            awaitable=AsyncOpenAI().beta.chat.completions.parse(
                model=PLANNING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=RevisionPlan,
            ),
        )
        plan = response.choices[0].message.parsed
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The revision request could not be interpreted: {exc}",
        ) from exc
    if plan is None:
        raise HTTPException(status_code=422, detail="The revision request was not actionable.")
    plan = plan.model_copy(update={
        "required_text": request.required_changes.strip(),
        "preferred_text": request.preferences.strip(),
        "restart_from": earliest_revision_step(plan.changes),
    })
    _validate_revision_plan(plan, state)
    return {"plan": plan.model_dump(mode="json")}


@router.post("/activate-revision")
def activate_revision(request: ActivateRevisionRequest):
    planning = planning_artifacts_dir(request.job_id)
    if not planning.is_dir():
        raise HTTPException(status_code=404, detail="Planning run was not found.")
    run_dir = planning.parent.parent
    if any((run_dir / relative).exists() for relative in (
        "artifacts/download_report.json", "data/dataset_manifest.json",
        "progress.json", "artifacts/best_model.pt", "artifacts/best_model.pth",
    )):
        raise HTTPException(
            status_code=409,
            detail="Planning revisions are only supported before execution begins.",
        )

    current_state = get_state(request.context)
    _validate_revision_plan(request.plan, current_state)
    predecessor = PREDECESSOR_CHECKPOINT[request.plan.restart_from]
    if predecessor:
        path = planning / predecessor
        if not path.is_file():
            raise HTTPException(status_code=409, detail=f"Required checkpoint {predecessor} is missing.")
        try:
            state = PipelineState.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise HTTPException(status_code=409, detail=f"Could not restore {predecessor}: {exc}") from exc
    else:
        state = get_state(request.context)

    revision_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    archive = planning / "revisions" / revision_id
    for filename in DOWNSTREAM_PLANNING_FILES[request.plan.restart_from]:
        source = planning / filename
        if source.is_file():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(archive / filename))

    history = list(current_state.revision.history)
    history.append(request.plan)
    state.revision = state.revision.model_copy(update={
        "active": request.plan,
        "history": history,
    })
    state.step_history.append(f"Planning revision activated: {request.plan.summary}")
    save_checkpoint(state, request.job_id, "STATE_ACTIVE_REVISION.json")
    return {"context": state.model_dump(mode="json"), "revision_id": revision_id}


@router.post("/fork-revision")
def fork_revision(request: ForkRevisionRequest):
    """Create an immutable child run that resumes at a planning step."""
    if not request.parent_job_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in request.parent_job_id
    ):
        raise HTTPException(status_code=400, detail="Invalid parent job ID.")
    parent = (RUNS_ROOT / request.parent_job_id).resolve()
    if parent.parent != RUNS_ROOT or not parent.is_dir():
        raise HTTPException(status_code=404, detail="Parent run was not found.")
    assessment_path = parent / "artifacts" / "post_training_assessment.json"
    try:
        assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=409, detail="Generate an assessment before forking.") from exc
    if assessment.get("assessment_id") != request.assessment_id:
        raise HTTPException(status_code=409, detail="The assessment no longer matches this run.")

    parent_planning = parent / "artifacts" / "planning"
    state_path = next(
        (
            parent_planning / name for name in (
                "STATE_05_HYPERPARAMETERS.json", "STATE_04_DATASET_SELECTION.json",
                "STATE_03_MODEL_SELECTION.json", "STATE_02_DATA_CHECK.json",
                "STATE_01_INTERPRETATION.json",
            ) if (parent_planning / name).is_file()
        ),
        None,
    )
    if state_path is None:
        raise HTTPException(status_code=409, detail="No reusable planning state is available.")
    try:
        parent_state = PipelineState.model_validate_json(state_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise HTTPException(status_code=409, detail=f"The parent planning state is invalid: {exc}") from exc

    plan = _canonicalize_revision_plan(request.plan, parent_state)
    _validate_revision_plan(plan, parent_state)
    restart_index = [item[0] for item in PLANNING_CHECKPOINTS].index(plan.restart_from)
    required_predecessors = PLANNING_CHECKPOINTS[:restart_index]
    missing = [name for _, name in required_predecessors if not (parent_planning / name).is_file()]
    if missing:
        raise HTTPException(
            status_code=409,
            detail={"message": "Required historical planning checkpoints are missing.", "files": missing},
        )

    child_job_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:10]}"
    child_planning = planning_artifacts_dir(child_job_id)
    for _, filename in required_predecessors:
        shutil.copy2(parent_planning / filename, child_planning / filename)

    if required_predecessors:
        restored_path = child_planning / required_predecessors[-1][1]
        state = PipelineState.model_validate_json(restored_path.read_text(encoding="utf-8"))
    else:
        state = parent_state
    history = list(parent_state.revision.history)
    history.append(plan)
    state.revision = state.revision.model_copy(update={"active": plan, "history": history})
    state.hpo_config = None
    state.hpo_decision = None
    state.step_history.append(
        f"Forked from {request.parent_job_id}: {plan.summary}"
    )
    save_checkpoint(state, child_job_id, "STATE_ACTIVE_REVISION.json")
    lineage = {
        "job_id": child_job_id,
        "parent_job_id": request.parent_job_id,
        "root_job_id": request.parent_job_id,
        "assessment_id": request.assessment_id,
        "restart_from": plan.restart_from,
        "created_at": datetime.now().isoformat(),
    }
    parent_lineage_path = parent / "lineage.json"
    if parent_lineage_path.is_file():
        try:
            parent_lineage = json.loads(parent_lineage_path.read_text(encoding="utf-8"))
            lineage["root_job_id"] = parent_lineage.get("root_job_id", request.parent_job_id)
        except (OSError, json.JSONDecodeError):
            pass
    (child_planning.parent.parent / "lineage.json").write_text(
        json.dumps(lineage, indent=2), encoding="utf-8"
    )
    return {
        "job_id": child_job_id,
        "parent_job_id": request.parent_job_id,
        "context": state.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "restart_from": plan.restart_from,
        "reused_steps": [step for step, _ in required_predecessors],
    }


@router.post("/verify-revision")
def verify_revision(request: VerifyRevisionRequest):
    state = get_state(request.context)
    plan = state.revision.active
    if plan is None:
        raise HTTPException(status_code=409, detail="No active revision exists.")
    checks = []
    for change in plan.changes:
        actual = _revision_value(state, change.field)
        satisfied = _change_satisfied(change, actual)
        checks.append({
            "change_id": change.id,
            "field": change.field,
            "strength": change.strength,
            "expected": change.value,
            "actual": actual,
            "satisfied": satisfied,
            "summary": change.summary,
        })
    required_ok = all(check["satisfied"] for check in checks if check["strength"] == "required")
    return {"satisfied": required_ok, "checks": checks}


def sanitize_english_rationale(value: str) -> str:
    """Remove accidental CJK fragments from an English agent rationale."""

    value = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", " ", value)
    return re.sub(r"[ \t]+", " ", value).strip()


def save_hpo_result(state: PipelineState, job_id: str):
    save_json(state.hpo_config, hpo_config_path(job_id).parent, hpo_config_path(job_id).name)


def ensure_default_hardware_filter(state: PipelineState) -> None:
    hardware = state.available_hardware
    if hardware is None:
        state.available_hardware = HardwareSpecModel(hardware_category=FALLBACK_HARDWARE_CATEGORY)
        return

    # Apple Silicon Macs are consumer computers, not embedded edge devices.  The
    # interpretation model can otherwise over-weight words such as "local",
    # "integrated GPU", or "Metal" and incorrectly emit EdgeDevice, which makes
    # every detection model requiring ConsumerCPU fail the hard hardware filter.
    hardware_description = " ".join(
        str(value or "")
        for value in (hardware.gpu_type, hardware.details, state.user_query)
    ).lower()
    apple_mac_markers = (
        "macbook",
        "mac mini",
        "mac studio",
        "mac pro",
        "imac",
    )
    if (
        hardware.hardware_category == "EdgeDevice"
        and any(marker in hardware_description for marker in apple_mac_markers)
    ):
        hardware.hardware_category = "ConsumerCPU"

    if not hardware.hardware_category and hardware.vram_gb is None:
        hardware.hardware_category = FALLBACK_HARDWARE_CATEGORY


def apply_qualitative_constraint_fallbacks(extracted, user_query: str | None) -> None:
    """Preserve explicit qualitative intent if the interpretation agent omits it."""

    query = (user_query or "").lower()

    accuracy_target = re.search(
        r"\baccuracy\b[^0-9]{0,30}(\d+(?:\.\d+)?)\s*%",
        query,
    )
    if accuracy_target:
        matched_text = accuracy_target.group(0)
        extracted.performance_requirements = (
            extracted.performance_requirements
            or PerformanceSpecModel(primary_metric="accuracy")
        )
        extracted.performance_requirements.primary_metric = "accuracy"
        extracted.performance_requirements.target_value = (
            float(accuracy_target.group(1)) / 100.0
        )
        extracted.performance_requirements.target_is_hard = bool(
            re.search(
                r"\b(must|required|at least|minimum|cannot be below)\b",
                matched_text,
            )
        )

    latency_limit = re.search(
        r"(?:within|below|under|at most|less than|cannot exceed)\s+(?:approximately\s+|roughly\s+|about\s+)?"
        r"(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|s)\b",
        query,
    )
    if latency_limit:
        value = float(latency_limit.group(1))
        if latency_limit.group(2) in {"second", "seconds", "s"}:
            value *= 1000
        extracted.deployment_constraints = extracted.deployment_constraints or DeploymentConstraints()
        extracted.deployment_constraints.max_cpu_latency_ms = value
        if not re.search(r"\b(approximately|roughly|about|preferably|desirable)\b", latency_limit.group(0)):
            extracted.deployment_constraints.hard_limits = sorted({
                *extracted.deployment_constraints.hard_limits,
                "max_cpu_latency_ms",
            })
        if (
            extracted.performance_requirements
            and extracted.performance_requirements.primary_metric
            and extracted.performance_requirements.primary_metric.lower() in {"latency", "latency_ms"}
        ):
            extracted.performance_requirements.target_value = None
            extracted.performance_requirements.target_is_hard = False

    runtime_memory_limit = re.search(
        r"\b(?:inference|runtime)?\s*(?:memory|vram)(?:\s+(?:usage|use|footprint))?"
        r"[^.\n]{0,50}?\b(?:below|under|at most|less than|cannot exceed|maximum(?: of)?)\s+"
        r"(?:approximately\s+|roughly\s+|about\s+)?(\d+(?:\.\d+)?)\s*(gb|gib|mb|mib)\b",
        query,
    )
    if runtime_memory_limit:
        value = float(runtime_memory_limit.group(1))
        if runtime_memory_limit.group(2) in {"gb", "gib"}:
            value *= 1024
        extracted.deployment_constraints = extracted.deployment_constraints or DeploymentConstraints()
        extracted.deployment_constraints.max_runtime_memory_mb = value
        matched_text = runtime_memory_limit.group(0)
        if (
            re.search(r"\b(must|required|mandatory|cannot exceed|at most|maximum)\b", matched_text)
            and not re.search(r"\b(approximately|roughly|about|preferably|ideally|desirable)\b", matched_text)
        ):
            extracted.deployment_constraints.hard_limits = sorted({
                *extracted.deployment_constraints.hard_limits,
                "max_runtime_memory_mb",
            })
    map_target = re.search(
        r"\bmap(?:@0?\.5(?::0?\.95)?)?[^0-9]{0,40}(0?\.\d+|1(?:\.0+)?)",
        query,
    )
    if map_target:
        extracted.performance_requirements = (
            extracted.performance_requirements
            or PerformanceSpecModel(primary_metric="mAP@0.5:0.95")
        )
        extracted.performance_requirements.primary_metric = "mAP@0.5:0.95"
        extracted.performance_requirements.target_value = float(map_target.group(1))
        extracted.performance_requirements.target_is_hard = bool(
            re.search(r"\b(must|required|mandatory)\b", query)
        )

    if any(phrase in query for phrase in ("reliable accuracy", "good accuracy", "maintain accuracy")):
        extracted.performance_requirements = (
            extracted.performance_requirements
            or PerformanceSpecModel(primary_metric="accuracy")
        )
        if extracted.performance_requirements.accuracy_category is None:
            extracted.performance_requirements.accuracy_category = "MediumHigh"

    if extracted.deployment_constraints is None:
        if "very low memory" in query:
            extracted.deployment_constraints = DeploymentConstraints(memory_category="VeryLow")
        elif "low memory" in query:
            extracted.deployment_constraints = DeploymentConstraints(memory_category="Low")

    # Deterministic P5: wording strength is planning evidence, not an LLM-only guess.
    hard_words = r"\b(must|required|mandatory|at least|at most|cannot|maximum|minimum)\b"
    preference_words = r"\b(preferably|ideally|desirable|would like|aim for)\b"
    soft_words = r"\b(should|important|reliable|good|maintain)\b"

    def strength_for(pattern: str) -> str:
        matches = list(re.finditer(pattern, query))
        if not matches:
            return "unspecified"
        snippets = [
            query[max(0, match.start() - 80):min(len(query), match.end() + 80)]
            for match in matches
        ]
        local_text = " ".join(snippets)
        if re.search(hard_words, local_text):
            return "hard"
        if re.search(preference_words, local_text):
            return "preference"
        if re.search(soft_words, local_text):
            return "soft"
        return "soft"

    strengths = extracted.constraint_strengths
    if re.search(r"\b(accuracy|precis(?:ion|e)|recall|f1|map|quality)\b", query):
        strengths.accuracy = strength_for(
            r"\b(accuracy|precis(?:ion|e)|recall|f1|map|quality)\b"
        )
    if re.search(r"\b(latency|real[ -]?time|fast|fps|milliseconds?|\bms\b)\b", query):
        strengths.latency = strength_for(
            r"\b(latency|real[ -]?time|fast|fps|milliseconds?|\bms\b)\b"
        )
    if re.search(
        r"\b(?:inference time|latency|speed)\b[^.\n]{0,30}\b(?:is\s+)?not\s+important\b",
        query,
    ):
        strengths.latency = "unspecified"
        if extracted.performance_requirements:
            extracted.performance_requirements.latency_category = None
    if re.search(r"\b(memory|vram|ram)\b", query):
        strengths.runtime_memory = strength_for(r"\b(memory|vram|ram)\b")
    if re.search(r"\b(model size|small model|parameters?)\b", query):
        strengths.model_size = strength_for(r"\b(model size|small model|parameters?)\b")

    # Deterministic P6: fill a missing category from a detection mAP target.
    performance = extracted.performance_requirements
    if (
        performance
        and performance.accuracy_category is None
        and performance.target_value is not None
        and "map" in str(performance.primary_metric or "").lower()
    ):
        target = float(performance.target_value)
        performance.accuracy_category = (
            "High" if target > 0.45 else
            "MediumHigh" if target >= 0.30 else
            "Medium" if target >= 0.20 else
            "Low"
        )

    # Deterministic P7: normalize common robustness synonyms into fixed dimensions.
    robustness = extracted.robustness_requirements
    if re.search(r"\b(night|nighttime|dark)\b", query):
        robustness.lighting = sorted({*robustness.lighting, "night"})
    if re.search(r"\b(low[ -]?light|dim|poorly lit)\b", query):
        robustness.lighting = sorted({*robustness.lighting, "low_light"})
    for pattern, value in ((r"\brain(?:y|fall)?\b", "rain"), (r"\bfog(?:gy)?\b", "fog"), (r"\bsnow(?:y)?\b", "snow")):
        if re.search(pattern, query):
            robustness.weather = sorted({*robustness.weather, value})
    if re.search(r"\b(small|tiny|distant|far away)\b", query):
        robustness.object_scale = sorted({*robustness.object_scale, "small"})
    if re.search(r"\b(dense|crowded|many objects?)\b", query):
        robustness.scene_density = sorted({*robustness.scene_density, "dense"})
    robustness.motion_blur = robustness.motion_blur or bool(re.search(r"\b(motion blur|moving fast)\b", query))
    robustness.occlusion = robustness.occlusion or bool(re.search(r"\b(occlusion|occluded|partially hidden)\b", query))


def explicit_images_per_class(user_query: str | None) -> int | None:
    """Return an explicit per-class image request, if the prompt contains one."""

    match = re.search(
        r"\b([1-9][0-9,]*)\s+images?\s+per\s+class\b",
        user_query or "",
        flags=re.IGNORECASE,
    )
    return int(match.group(1).replace(",", "")) if match else None


# --- Endpoints ---
@router.post("/completenesscheck", response_model=CompletenessCheckResponse)
async def completenesscheck(request: CompletenessCheckRequest):
    context, decision = await interpretation_loop(
        request.user_prompt,
        job_id=request.job_id,
        user_replies=request.user_replies,
    )
    return CompletenessCheckResponse(
        accept=decision.accept,
        reason=getattr(decision, "reason", None),
        suggestions=getattr(decision, "suggestions", None),
        context=context if decision.accept else None,
    )

@router.post("/task-interpret")
async def task_interpret(request: StateRequest):
    state = get_state(request.context)
    state.use_graphrag = request.use_graphrag
    res = await run_planning_agent(
        job_id=request.job_id,
        operation="task_interpretation",
        agent=task_interpretation_agent,
        input=state.model_dump_json(),
    )
    extracted = res.final_output
    interpretation_attempts = [extracted.model_dump(mode="json")]
    apply_qualitative_constraint_fallbacks(extracted, state.user_query)
    if (
        not extracted.performance_requirements
        or (
            extracted.performance_requirements.latency_category is None
            and extracted.performance_requirements.accuracy_category is None
        )
    ) and extracted.deployment_constraints is None:
        extracted.performance_requirements = extracted.performance_requirements or PerformanceSpecModel(
            primary_metric="latency",
        )
        extracted.performance_requirements.latency_category = "Low"
    
    for interpretation_attempt in range(1, 3):
        valid_classes = load_unified_dataset_classes(extracted.task)
        extracted_classes = list(extracted.classes or [])
        if "traffic" in str(extracted.application_domain or "").lower():
            extracted_classes = [
                "pedestrian" if item.strip().lower() == "person" else item
                for item in extracted_classes
            ]
        final_classes = []
        class_expansions: Dict[str, List[str]] = {}
        invalid_classes: list[str] = []
        valid_classes_str = ", ".join(sorted(valid_classes))

        # Resolve every class so one repair response receives all ontology problems.
        for cls in extracted_classes:
            cls_clean = cls.strip().lower()
            if cls_clean in valid_classes:
                final_classes.append(cls_clean)
                continue

            syn_res = await run_planning_agent(
                job_id=request.job_id,
                operation="synonym_check",
                agent=synonym_check_agent,
                input=(
                    f"User Class: '{cls}'. Allowed: [{valid_classes_str}]. "
                    f"Application Domain: {extracted.application_domain}"
                ),
            )
            matched_classes = [
                dataset_class.strip().lower()
                for dataset_class in (syn_res.final_output.dataset_classes or [])
                if dataset_class and dataset_class.strip().lower() in valid_classes
            ][:20]
            if syn_res.final_output.found_match and matched_classes:
                final_classes.extend(matched_classes)
                if len(matched_classes) > 1:
                    class_expansions[cls_clean] = matched_classes
            else:
                invalid_classes.append(cls)

        if not invalid_classes:
            break
        if interpretation_attempt == 2:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Class ontology mapping failed after one repair round.",
                    "invalid_classes": invalid_classes,
                    "allowed_classes": sorted(valid_classes),
                    "interpretation_attempts": interpretation_attempts,
                },
            )
        repair_res = await run_planning_agent(
            job_id=request.job_id,
            operation="task_interpretation_repair",
            agent=task_interpretation_agent,
            input=json.dumps({
                "user_prompt": state.user_query,
                "previous_interpretation": extracted.model_dump(mode="json"),
                "invalid_classes": invalid_classes,
                "allowed_classes": sorted(valid_classes),
                "repair_instructions": (
                    "Correct all invalid classes in one response using exact semantically equivalent "
                    "labels from allowed_classes. Preserve all otherwise valid extracted fields."
                ),
            }),
        )
        extracted = repair_res.final_output
        interpretation_attempts.append(extracted.model_dump(mode="json"))
        apply_qualitative_constraint_fallbacks(extracted, state.user_query)

    extracted_patch = extracted.model_dump(exclude_unset=True)
    if extracted.performance_requirements and state.performance_requirements:
        existing_performance = state.performance_requirements.model_dump(exclude_unset=True)
        extracted_performance = extracted.performance_requirements.model_dump(
            exclude_none=True,
            exclude_unset=True,
        )
        extracted_patch["performance_requirements"] = {
            **existing_performance,
            **extracted_performance,
        }
    if extracted.available_hardware and state.available_hardware:
        existing_hardware = state.available_hardware.model_dump(exclude_unset=True)
        extracted_hardware = extracted.available_hardware.model_dump(
            exclude_none=True,
            exclude_unset=True,
        )
        extracted_patch["available_hardware"] = {
            **existing_hardware,
            **extracted_hardware,
        }
    if extracted.deployment_constraints and state.deployment_constraints:
        extracted_patch["deployment_constraints"] = {
            **state.deployment_constraints.model_dump(exclude_unset=True),
            **extracted.deployment_constraints.model_dump(exclude_none=True, exclude_unset=True),
        }

    available_hardware_patch = extracted_patch.get("available_hardware")
    if available_hardware_patch is None:
        available_hardware_patch = (
            state.available_hardware.model_dump(exclude_none=True, exclude_unset=True)
            if state.available_hardware
            else {}
        )
    if (
        not available_hardware_patch.get("hardware_category")
        and available_hardware_patch.get("vram_gb") is None
    ):
        available_hardware_patch["hardware_category"] = FALLBACK_HARDWARE_CATEGORY
    if available_hardware_patch:
        extracted_patch["available_hardware"] = available_hardware_patch

    state = PipelineState(**{**state.model_dump(), **extracted_patch})
    state.classes = list(dict.fromkeys(final_classes))
    state.class_expansions = class_expansions
    interpretation_updates = {
        change.field: change.value
        for change in changes_for(state, "task-interpretation", strength="required")
        if change.operation == "set"
        and change.field in {
            "task", "classes", "application_domain", "performance_requirements",
            "deployment_constraints", "available_hardware",
        }
    }
    if interpretation_updates:
        candidate_state = state.model_dump(mode="json")
        candidate_state.update(interpretation_updates)
        state = PipelineState.model_validate(candidate_state)
        if "classes" in interpretation_updates:
            allowed_classes = load_unified_dataset_classes(state.task)
            invalid = set(state.classes) - set(allowed_classes)
            if invalid:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "Required classes are not available for the selected task.",
                        "classes": sorted(invalid),
                    },
                )
    state.step_history.append("Task Interpretation Completed")
    
    save_checkpoint(state, request.job_id, "STATE_01_INTERPRETATION.json")
    return {
        "context": state.model_dump(),
        "llm_attempts": interpretation_attempts,
    }

@router.post("/check-data")
async def check_data(request: StateRequest):
    state = get_state(request.context)
    state.use_graphrag = request.use_graphrag

    if state.classes and not state.task:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Data availability checking requires an interpreted task.",
            },
        )
    
    if state.classes:
        query_path = planning_artifacts_dir(request.job_id) / "DATA_CHECK_QUERY.sparql"
        raw_stats = get_multi_class_stats(
            state.classes,
            query_output_path=query_path,
            task=state.task,
        )
        
        formatted_stats = []
        for cls_name, sources_dict in raw_stats.items():
            sources_list = [
                {"dataset_name": d_name, "count": count} 
                for d_name, count in sources_dict.items()
            ]
            formatted_stats.append({
                "class_name": cls_name, 
                "sources": sources_list
            })

        # Keep only official, downloadable datasets compatible with the
        # interpreted task before exposing availability to downstream steps.
        state.available_data = filter_dataset_candidates(
            [ClassDataSelection.model_validate(item) for item in formatted_stats],
            state.task,
        )
        
    state.step_history.append("Data Availability Checked")
    save_checkpoint(state, request.job_id, "STATE_02_DATA_CHECK.json")
    
    return {"context": state.model_dump()}


def _small_objects_requested(state: PipelineState) -> bool:
    robustness = state.robustness_requirements
    if not robustness:
        return False
    scales = robustness.get("object_scale", []) if isinstance(robustness, dict) else robustness.object_scale
    return "small" in {str(value).strip().lower() for value in scales}


def _required_detection_comparison_contract(
    state: PipelineState,
    candidates: list[dict[str, Any]],
) -> tuple[int, set[str], dict[str, str]]:
    """Return required count/types and candidate-to-architecture mapping."""
    architecture_by_id = {
        str((candidate.get("model") or {}).get("id")): str(
            (candidate.get("model") or {}).get("architecture_type") or "unknown"
        )
        for candidate in candidates
    }
    architecture_types = set(architecture_by_id.values())
    minimum = min(2, len(architecture_by_id))
    required_types: set[str] = set()
    if state.task == "detection" and _small_objects_requested(state):
        minimum = min(3, len(architecture_by_id))
        if "TwoStageRegionProposalDetector" in architecture_types:
            required_types.add("TwoStageRegionProposalDetector")
    return minimum, required_types, architecture_by_id


def _inference_memory_used_as_training_evidence(rationale: str) -> bool:
    """Detect the prohibited inference-memory -> training-headroom argument."""
    positive_claims = ("headroom", "ample margin", "leaves margin", "fits training", "training feasible")
    training_topics = ("train", "batch", "augment", "tiling", "multi-scale", "multiscale")
    inference_topics = ("inference memory", "inference vram", "inference footprint")
    safe_negations = ("does not", "cannot", "must not", "not evidence", "not proof", "separate from")
    for paragraph in re.split(r"\n\s*\n", rationale.lower()):
        if (
            any(term in paragraph for term in inference_topics)
            and any(term in paragraph for term in training_topics)
            and any(term in paragraph for term in positive_claims)
            and not any(term in paragraph for term in safe_negations)
        ):
            return True
    return False


def _small_object_uncertainty_recorded(uncertainties: list[str]) -> bool:
    text = " ".join(str(item).lower() for item in uncertainties)
    return "small" in text and any(term in text for term in ("unverified", "unknown", "evidence", "ap-small", "ap_small"))

@router.post("/select-model")
async def select_model(request: StateRequest):
    state = get_state(request.context)
    ensure_default_hardware_filter(state)
    from cvmodellearning.schemas.revision import (
        explicit_required_model_id,
        explicit_required_model_reference,
        initial_hpo_override_values,
    )
    explicit_model_reference = explicit_required_model_reference(state)
    explicit_model_id = explicit_required_model_id(state)
    required_model_repair_attempts: list[dict[str, Any]] = []
    selection_attempts: list[dict[str, Any]] = []
    if explicit_model_reference and explicit_model_id is None:
        unresolved_model_reference = explicit_model_reference
        repair_res = await run_planning_agent(
            job_id=request.job_id,
            operation="required_model_interpretation_repair",
            agent=task_interpretation_agent,
            input=json.dumps({
                "user_prompt": state.user_query,
                "task": state.task,
                "previous_model_requirements": [
                    item.model_dump(mode="json") for item in (state.model_requirements or [])
                ],
                "available_models": [
                    {"id": model.id, "display_name": model.display_name}
                    for model in enabled_models(state.task)
                ],
                "repair_instructions": (
                    "Repair the required model requirement once. Use the original user wording as "
                    "authoritative evidence and preserve an explicit architecture identifier exactly. "
                    "Do not guess when the wording is ambiguous; preserve all other extracted fields."
                ),
            }),
        )
        required_model_repair_attempts.append(repair_res.final_output.model_dump(mode="json"))
        if repair_res.final_output.model_requirements is not None:
            state.model_requirements = repair_res.final_output.model_requirements
        explicit_model_reference = explicit_required_model_reference(state)
        explicit_model_id = explicit_required_model_id(state)
        if explicit_model_id is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Requested model '{explicit_model_reference or unresolved_model_reference}' is not executable for {state.task}.",
                    "available_models": [model.id for model in enabled_models(state.task)],
                    "required_model_repair_attempts": required_model_repair_attempts,
                },
            )
    if (
        state.task == "classification"
        and explicit_model_id
        and initial_hpo_override_values(state).get("training_mode") == "lora"
    ):
        from cvmodellearning.models.registry import LORA_CLASSIFICATION_MODEL_IDS
        if explicit_model_id not in LORA_CLASSIFICATION_MODEL_IDS:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"LoRA is not executable for classification model '{explicit_model_id}'.",
                    "lora_supported_models": sorted(LORA_CLASSIFICATION_MODEL_IDS),
                },
            )
    use_graphrag = request.use_graphrag and USE_MODEL_SELECTION_GRAPHRAG
    state = state.model_copy(update={
        "model_selection_graph_context": None,
        "model_selection_decision_evidence": None,
        "use_graphrag": use_graphrag,
    })

    graph_context: dict[str, Any] = {}
    if use_graphrag:
        graph_context = build_model_selection_context(state)
        candidates = graph_context.get("candidate_models") or []
        if not candidates and "benchmark_target" in graph_context.get("filters", {}):
            relaxed_performance = state.performance_requirements.model_copy(
                update={"target_is_hard": False}
            )
            relaxed_state = state.model_copy(
                update={"performance_requirements": relaxed_performance}
            )
            relaxed_context = build_model_selection_context(relaxed_state)
            candidates = relaxed_context.get("candidate_models") or []
            if candidates:
                graph_context = {
                    **relaxed_context,
                    "requested_filters": graph_context.get("filters", {}),
                    "constraint_warnings": [
                        *relaxed_context.get("constraint_warnings", []),
                        "No catalog benchmark proves the requested hard performance target; "
                        "the target remains unverified and was not used to exclude executable models."
                    ],
                }
        if not candidates:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "No executable model satisfies the requested hard constraints.",
                    "filters": graph_context.get("filters", {}),
                    "rejected_counts": graph_context.get("rejected_counts", {}),
                },
            )
        state.model_selection_graph_context = {
            **graph_context,
            "llm_context": format_model_selection_context(graph_context),
        }
        state.step_history.append(summarize_model_selection_context(graph_context))
    else:
        state.step_history.append("GraphRAG Model Suggestions: disabled for this run.")
    
    agent_map = {
        "classification": classification_model_selector_agent,
        "detection": detection_model_selector_agent,
        "visual question answering": vqq_model_selector_agent
    }
    
    agent = agent_map.get(state.task)
    if agent:
        agent_input = state.model_dump_json()
        contract_error: dict[str, Any] | None = None
        selected_candidate_id: str | None = None
        comparison_warnings: list[str] = []
        for selection_attempt in range(1, 3):
            res = await run_planning_agent(
                job_id=request.job_id,
                operation="model_selection",
                agent=agent,
                input=agent_input,
            )
            selection_attempts.append(res.final_output.model_dump(mode="json"))
            model_patch_dict = res.final_output.model_dump(exclude_unset=True)
            model_rationale = sanitize_english_rationale(
                model_patch_dict.pop("rationale", "No rationale provided.")
            )
            evaluated_candidates = model_patch_dict.pop("evaluated_candidates", [])
            model_uncertainties = model_patch_dict.pop("uncertainties", [])
            selected_candidate_id = model_patch_dict.pop("selected_candidate_id", None)
            model_payload = model_patch_dict.setdefault("model", {})
            selected_id = model_payload.get("model_architecture")
            contract_error = None

            if not is_executable_model_reference(state.task, str(selected_id or "")):
                contract_error = {
                    "message": "The LLM selected a model that is not executable for the task.",
                    "selected_model": selected_id,
                    "available_models": [model.id for model in enabled_models(state.task)],
                }
            elif use_graphrag:
                candidate_ids = {
                    str((candidate.get("model") or {}).get("id"))
                    for candidate in graph_context.get("candidate_models") or []
                }
                candidate_reference = str(selected_candidate_id or selected_id or "")
                matched_candidate_id = next(
                    (
                        candidate_id for candidate_id in candidate_ids
                        if model_ids_equivalent(candidate_reference, candidate_id)
                    ),
                    None,
                )
                compared_ids = [
                    str(item.get("candidate_id", "")) for item in evaluated_candidates
                ]
                unknown_comparisons = [
                    compared_id for compared_id in compared_ids
                    if not any(
                        model_ids_equivalent(compared_id, candidate_id)
                        for candidate_id in candidate_ids
                    )
                ]
                minimum_comparisons, required_architecture_types, architecture_by_id = (
                    _required_detection_comparison_contract(
                        state,
                        graph_context.get("candidate_models") or [],
                    )
                )
                unique_compared = {
                    candidate_id
                    for compared_id in compared_ids
                    for candidate_id in candidate_ids
                    if model_ids_equivalent(compared_id, candidate_id)
                }
                selected_compared = bool(matched_candidate_id) and any(
                    model_ids_equivalent(matched_candidate_id, compared_id)
                    for compared_id in compared_ids
                )
                incomplete = [
                    item.get("candidate_id") for item in evaluated_candidates
                    if not item.get("advantages") or not item.get("risks")
                ]
                compared_architecture_types = {
                    architecture_by_id[candidate_id]
                    for candidate_id in unique_compared
                    if candidate_id in architecture_by_id
                }
                missing_architecture_types = sorted(
                    required_architecture_types - compared_architecture_types
                )
                insufficient_architecture_diversity = (
                    state.task == "detection"
                    and _small_objects_requested(state)
                    and len(compared_architecture_types) < min(2, len(set(architecture_by_id.values())))
                )
                third_architecture_preference_unmet = (
                    state.task == "detection"
                    and _small_objects_requested(state)
                    and len(set(architecture_by_id.values())) >= 3
                    and len(compared_architecture_types) < 3
                )
                inference_training_conflation = (
                    state.task == "detection"
                    and _inference_memory_used_as_training_evidence(model_rationale)
                )
                missing_small_object_uncertainty = (
                    state.task == "detection"
                    and _small_objects_requested(state)
                    and graph_context.get("filters", {}).get("small_object_benchmark_status")
                    == "unverified_without_ap_small"
                    and not _small_object_uncertainty_recorded(model_uncertainties)
                )
                if (
                    matched_candidate_id is None
                    or unknown_comparisons
                    or len(unique_compared) < minimum_comparisons
                    or not selected_compared
                    or incomplete
                    or missing_architecture_types
                    or insufficient_architecture_diversity
                    or missing_small_object_uncertainty
                ):
                    contract_error = {
                        "message": "The LLM must select and compare an exact feasible GraphRAG candidate.",
                        "selected_model": selected_id,
                        "selected_candidate_id": selected_candidate_id,
                        "candidate_models": sorted(candidate_ids),
                        "minimum_comparisons": minimum_comparisons,
                        "unknown_comparison_ids": unknown_comparisons,
                        "selected_candidate_was_compared": selected_compared,
                        "comparisons_missing_advantages_or_risks": incomplete,
                        "required_architecture_types": sorted(required_architecture_types),
                        "compared_architecture_types": sorted(compared_architecture_types),
                        "missing_architecture_types": missing_architecture_types,
                        "insufficient_architecture_diversity": insufficient_architecture_diversity,
                        "third_architecture_preference_unmet": third_architecture_preference_unmet,
                        "inference_memory_used_as_training_evidence": inference_training_conflation,
                        "missing_small_object_uncertainty": missing_small_object_uncertainty,
                    }
                else:
                    selected_candidate_id = matched_candidate_id
                    comparison_warnings = []
                    if third_architecture_preference_unmet:
                        comparison_warnings.append(
                            "A third detector architecture type was available but not compared; "
                            "the required three candidates, two architecture types, and feasible "
                            "two-stage challenger were still covered."
                        )
                    if inference_training_conflation:
                        comparison_warnings.append(
                            "Inference-memory estimates do not establish training-memory feasibility; "
                            "training memory must be verified independently at runtime."
                        )

            if contract_error is None:
                break
            if selection_attempt == 2:
                contract_error["selection_attempts"] = selection_attempts
                raise HTTPException(status_code=422, detail=contract_error)
            agent_input = json.dumps({
                "pipeline_state": state.model_dump(mode="json"),
                "previous_invalid_proposal": res.final_output.model_dump(mode="json"),
                "validation_error": contract_error,
                "repair_instructions": (
                    "Return a corrected model-selection response. When GraphRAG is enabled, "
                    "selected_candidate_id must exactly equal one candidate_models ID, and that "
                    "same ID must appear in evaluated_candidates. Preserve valid reasoning where possible."
                    " For small-object detection, compare the minimum number of candidates, cover at least two "
                    "distinct architecture types, and satisfy required_architecture_types. Never use inference "
                    "memory as evidence of training headroom. "
                    "When AP-small is unavailable, record that limitation explicitly in uncertainties."
                ),
            })

        if use_graphrag and selected_candidate_id:
            model_patch_dict["selected_candidate_id"] = selected_candidate_id
            if state.task == "detection":
                identity = resolve_detection_model_identity(selected_candidate_id)
                if identity is None:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "message": "The selected GraphRAG candidate has no executable detection identity.",
                            "selected_candidate_id": selected_candidate_id,
                        },
                    )
                model_payload["model_architecture"] = identity.executable_id
        model_changes = changes_for(state, "model-selection")
        required_model = next(
            (
                change.value for change in model_changes
                if change.strength == "required"
                and change.field == "model_name"
                and change.operation == "set"
            ),
            explicit_required_model_id(state),
        )
        if required_model is not None:
            requested_model = str(required_model)
            if not is_executable_model_reference(state.task, requested_model):
                available = [model.id for model in enabled_models(state.task)]
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": f"Requested model '{requested_model}' is not executable for {state.task}.",
                        "available_models": available,
                    },
                )
            model_payload["model_architecture"] = requested_model
            model_payload["description"] = "Model explicitly required by the user."
            if use_graphrag and any(
                model_ids_equivalent(
                    requested_model,
                    str((candidate.get("model") or {}).get("id", "")),
                )
                for candidate in graph_context.get("candidate_models") or []
            ):
                model_patch_dict["selected_candidate_id"] = requested_model
            model_rationale = (
                f"User-required model override selected '{requested_model}'. "
                f"{model_rationale}"
            )
        selected_id = model_payload.get("model_architecture")
        selected_family = family_for_model_reference(state.task, selected_id)
        if selected_family is None:
            raise HTTPException(
                status_code=422,
                detail=f"No registered architecture family for model '{selected_id}'.",
            )
        model_payload["architecture_family"] = selected_family
        state.selected_model_info = model_patch_dict
        decision_evidence = build_model_selection_decision_evidence(
            model_patch_dict,
            model_rationale,
            graph_context if use_graphrag else {},
        )
        decision_evidence["evaluated_candidates"] = evaluated_candidates
        decision_evidence["uncertainties"] = model_uncertainties
        decision_evidence["comparison_warnings"] = comparison_warnings
        decision_evidence["selection_confidence"] = (
            "conditional"
            if use_graphrag
            and state.task == "detection"
            and _small_objects_requested(state)
            and graph_context.get("filters", {}).get("small_object_benchmark_status")
            == "unverified_without_ap_small"
            else "standard"
        )
        state.model_selection_decision_evidence = decision_evidence
        
        state.step_history.append(f"Model Selection Rationale: {model_rationale}")
        state.step_history.append("Model Selection Completed")
    
    save_checkpoint(state, request.job_id, "STATE_03_MODEL_SELECTION.json")
    return {
        "context": state.model_dump(),
        "decision_evidence": state.model_selection_decision_evidence,
        "llm_attempts": [*required_model_repair_attempts, *selection_attempts],
    }

@router.post("/select-datasets")
async def select_datasets(request: StateRequest):
    state = get_state(request.context)
    state.dataset_selection_graph_context = None
    state.data_plan_constraints = None
    if not state.task or not state.classes:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Dataset selection requires an interpreted task and classes.",
            },
        )

    requested_classes = set(state.classes)
    eligible_data = [
        item
        for item in filter_dataset_candidates(state.available_data or [], state.task)
        if item.class_name in requested_classes
    ]
    dataset_changes = changes_for(state, "dataset-selection")
    required_includes = {
        str(value)
        for change in dataset_changes
        if change.strength == "required" and change.field == "dataset.include"
        for value in (change.value if isinstance(change.value, list) else [change.value])
    }
    required_excludes = {
        str(value)
        for change in dataset_changes
        if change.strength == "required" and change.field == "dataset.exclude"
        for value in (change.value if isinstance(change.value, list) else [change.value])
    }
    if required_includes & required_excludes:
        raise HTTPException(status_code=422, detail="A dataset cannot be both required and excluded.")
    known_dataset_ids = {
        source.dataset_name for item in eligible_data for source in item.sources
    }
    unknown_includes = required_includes - known_dataset_ids
    if unknown_includes:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more required datasets are not eligible for this task and class set.",
                "dataset_ids": sorted(unknown_includes),
                "eligible_dataset_ids": sorted(known_dataset_ids),
            },
        )
        decision_evidence["evaluated_candidates"] = evaluated_candidates
        decision_evidence["uncertainties"] = model_uncertainties
    if required_excludes:
        eligible_data = [
            item.model_copy(update={
                "sources": [
                    source for source in item.sources
                    if source.dataset_name not in required_excludes
                ]
            })
            for item in eligible_data
        ]

    eligible_by_class = {item.class_name: item for item in eligible_data}
    unavailable_classes = [
        class_name
        for class_name in state.classes
        if (
            class_name not in eligible_by_class
            or sum(
                source.count
                for source in eligible_by_class[class_name].sources
                if (info := resolve_dataset_info(source.dataset_name)) is not None
                and info.role.value == "train"
            ) < 3
        )
    ]
    expanded_classes = {
        class_name
        for mapped_classes in state.class_expansions.values()
        for class_name in mapped_classes
    }
    droppable_classes = [
        class_name for class_name in unavailable_classes if class_name in expanded_classes
    ]
    retained_classes = [
        class_name for class_name in state.classes if class_name not in droppable_classes
    ]
    if (
        droppable_classes
        and len(retained_classes) >= MIN_RETAINED_EXPANSION_CLASSES
        and all(class_name in expanded_classes for class_name in unavailable_classes)
    ):
        state.classes = retained_classes
        state.available_data = [
            item for item in (state.available_data or []) if item.class_name in retained_classes
        ]
        eligible_data = [
            item for item in eligible_data if item.class_name in set(retained_classes)
        ]
        unavailable_classes = []
        state.step_history.append(
            "Dataset Selection: omitted unavailable inferred expansion classes: "
            + ", ".join(droppable_classes)
        )
    if unavailable_classes:
        if state.task == "classification":
            message = (
                "No compatible image-classification datasets contain enough training "
                "images for all classes in the given user prompt."
            )
            reason = (
                "Classification requires datasets marked for image classification in "
                "the dataset registry; detection datasets cannot be used as image-level "
                "classification data."
            )
        else:
            message = "Insufficient eligible training data is available for one or more requested classes."
            reason = "Each requested class requires a compatible official training source."
        raise HTTPException(
            status_code=422,
            detail={
                "message": message,
                "classes": unavailable_classes,
                "reason": reason,
            },
        )
    
    agent_map = {
        "classification": classification_dataset_selection_agent,
        "detection": detection_dataset_selection_agent,
        "visual question answering": vqa_dataset_selection_agent,
    }
    
    selection_agent = agent_map.get(state.task)
    validation_findings = []
    fallback_validation_findings = []
    proposed_selection_snapshot = None
    fallback_selection_snapshot = None
    selection_mode = "llm_validated"
    selection_rationale = "No dataset selection rationale was returned."
    graphrag_used = False
    graph_context: Dict[str, Any] = {}
    explicit_pool_target = explicit_images_per_class(state.user_query)
    accuracy_category = (
        state.performance_requirements.accuracy_category
        if state.performance_requirements is not None
        else None
    )
    coverage_warnings = deployment_coverage_warnings(state)
    requested_robustness_dimensions = {
        dimension
        for warning in coverage_warnings
        for dimension in warning.get("dimensions", [])
    }
    robustness = state.robustness_requirements
    if robustness:
        if robustness.lighting:
            requested_robustness_dimensions.add("lighting")
        if robustness.weather:
            requested_robustness_dimensions.add("weather")
        if robustness.scene_density:
            requested_robustness_dimensions.add("scene_density")
        if robustness.motion_blur:
            requested_robustness_dimensions.add("motion_blur")
        if robustness.occlusion:
            requested_robustness_dimensions.add("occlusion")
        if robustness.viewpoint:
            requested_robustness_dimensions.add("viewpoint")
    detection_sizing_recommendation = None
    recommended_detection_target = None
    if state.task == "detection":
        selected_model = (state.selected_model_info or {}).get("model") or {}
        small_object_requested = bool(
            robustness
            and "small" in {
                str(value).strip().lower() for value in robustness.object_scale
            }
        )
        dataset_profile = state.dataset_profile
        small_fraction = (
            dataset_profile.small_object_fraction if dataset_profile else None
        )
        median_short_side = (
            dataset_profile.median_short_side_px_at_640 if dataset_profile else None
        )
        observed_small_objects = bool(
            (small_fraction is not None and small_fraction >= 0.5)
            or (median_short_side is not None and median_short_side < 32)
        )
        sizing_facts = infer_detection_sizing_facts(
            classes=state.classes,
            model_reference=selected_model.get("model_architecture"),
            application_domain=state.application_domain,
            use_case_description=state.use_case_description or state.user_query,
            accuracy_category=accuracy_category,
            other_constraints=(
                state.performance_requirements.other_constraints or ()
                if state.performance_requirements is not None
                else ()
            ),
            robustness_dimensions=requested_robustness_dimensions,
            object_size_risk=(
                "high" if observed_small_objects
                else "medium" if small_object_requested
                else "low"
            ),
            object_size_evidence=(
                "observed" if observed_small_objects
                else "explicit_requirement" if small_object_requested
                else "unknown"
            ),
            small_object_fraction=small_fraction,
            median_short_side_px_at_640=median_short_side,
        )
        detection_sizing_recommendation = determine_detection_dataset_size(sizing_facts)
        recommended_detection_target = (
            detection_sizing_recommendation.target_images_per_class
        )
    detection_policy_active = state.task == "detection"
    data_selection_policy = {
        "policy_ids": [
            "data.detection.robust_pool.v1",
            "data.detection.source_sufficiency.v1",
            "data.detection.lineage_safety.v1",
            DETECTION_DATA_SELECTION_POLICY.policy_id,
        ] if detection_policy_active else [],
        "target_images_per_class": None,
        "min_mixed_source_count": (
            MIN_SOURCE_POOL_FOR_PRIMARY_HOLDOUT if detection_policy_active else None
        ),
        "min_mixed_source_share": (
            MIN_SOURCE_POOL_SHARE_FOR_PRIMARY_HOLDOUT if detection_policy_active else None
        ),
        "reject_unpaired_original_derived_mix": detection_policy_active,
        "requested_robustness_dimensions": sorted(requested_robustness_dimensions),
    }
    if detection_policy_active:
        data_selection_policy.update(DETECTION_DATA_SELECTION_POLICY.as_context())
    if state.task == "classification":
        selection_target_images_per_class = min(
            explicit_pool_target or DEFAULT_CLASSIFICATION_POOL_PER_CLASS,
            MAX_CLASSIFICATION_POOL_PER_CLASS,
        )
    elif state.task == "detection":
        selection_target_images_per_class = min(
            explicit_pool_target or recommended_detection_target,
            MAX_DETECTION_POOL_PER_CLASS,
        )
    else:
        selection_target_images_per_class = explicit_pool_target
    data_selection_policy["target_images_per_class"] = selection_target_images_per_class
    if detection_sizing_recommendation is not None:
        data_selection_policy["dataset_sizing"] = {
            **detection_sizing_recommendation.as_context(),
            "calculated_target_images_per_class": recommended_detection_target,
            "effective_target_images_per_class": selection_target_images_per_class,
            "explicit_user_override": explicit_pool_target is not None,
            "calculation_stage": "select_datasets_prerequisite",
        }
    if selection_agent:
        state_context = state.model_dump(mode="json")
        selection_context = {
            key: state_context.get(key)
            for key in (
                "task",
                "application_domain",
                "user_query",
                "use_case_description",
                "classes",
                "performance_requirements",
                "deployment_constraints",
                "available_hardware",
                "training_hardware",
                "selected_model_info",
            )
        }
        eligible_dataset_ids = sorted({
            source.dataset_name
            for item in eligible_data
            for source in item.sources
        })
        selection_context["allowed_sources_by_class"] = {
            item.class_name: [
                {
                    "dataset_id": source.dataset_name,
                    "available_count": source.count,
                    "role": info.role.value,
                    "native_task": info.task,
                    "domains": list(info.domains),
                    "domain_alignment": {
                        "matched_tags": sorted(matched_domain_tags(
                            info.domains, state.application_domain
                        )),
                        "aligned": bool(matched_domain_tags(
                            info.domains, state.application_domain
                        )),
                    },
                    "description": info.description,
                    "family": dataset_family(source.dataset_name),
                    "lineage": {
                        "canonical_family": info.canonical_family,
                        "derived_from": info.derived_from,
                        "synthetic": info.synthetic,
                        "paired_sample_ids_available": info.paired_sample_ids_available,
                    },
                }
                for source in item.sources
                if (info := resolve_dataset_info(source.dataset_name)) is not None
            ]
            for item in eligible_data
        }
        selection_context["supported_group_isolation_keys"] = []
        mutually_exclusive_source_pairs = set()
        for dataset_id in eligible_dataset_ids:
            info = resolve_dataset_info(dataset_id)
            if (
                info is not None
                and info.derived_from
                and not info.paired_sample_ids_available
                and info.derived_from in eligible_dataset_ids
            ):
                mutually_exclusive_source_pairs.add(
                    tuple(sorted((dataset_id, info.derived_from)))
                )
        selection_context["mutually_exclusive_source_pairs"] = [
            {
                "dataset_ids": list(pair),
                "reason": (
                    "Parent and derived sources cannot both be selected because "
                    "pair/group identifiers are unavailable for leakage-safe splitting."
                ),
            }
            for pair in sorted(mutually_exclusive_source_pairs)
        ]
        selection_context["target_images_per_class"] = selection_target_images_per_class
        selection_context["data_selection_policy"] = data_selection_policy
        selection_context["user_revision"] = (
            state.revision.active.model_dump(mode="json")
            if state.revision.active else None
        )
        selection_context["required_dataset_ids"] = sorted(required_includes)
        selection_context["excluded_dataset_ids"] = sorted(required_excludes)
        training_classes_by_family: dict[str, set[str]] = defaultdict(set)
        for item in eligible_data:
            for source in item.sources:
                info = resolve_dataset_info(source.dataset_name)
                if info is not None and info.role == DatasetRole.TRAIN:
                    training_classes_by_family[dataset_family(source.dataset_name)].add(
                        item.class_name
                    )
        selection_context["training_class_coverage_by_family"] = {
            family: sorted(class_names)
            for family, class_names in sorted(training_classes_by_family.items())
        }
        if request.use_graphrag and USE_DATASET_SELECTION_GRAPHRAG:
            try:
                graph_context = build_dataset_selection_context(state, eligible_data)
                state.dataset_selection_graph_context = graph_context
                selection_context["dataset_guidance"] = {
                    item["dataset_id"]: item
                    for item in graph_context.get("candidate_guidance", [])
                    if item.get("dataset_id") in eligible_dataset_ids
                }
                graphrag_used = True
            except Exception as exc:
                graph_context = {
                    "enabled": False,
                    "warning": f"Dataset GraphRAG enrichment was unavailable: {exc}",
                }
                state.dataset_selection_graph_context = graph_context
        selection_rationale = ""
        validation_findings: list[dict[str, Any]] = []
        dataset_repair_history: list[dict[str, Any]] = []
        dataset_identifier_normalizations: list[dict[str, str]] = []
        dataset_fallback_adjustments: list[dict[str, Any]] = []
        dataset_advisory_findings: list[dict[str, Any]] = []
        proposed_selection_snapshot: list[dict[str, Any]] = []
        for selection_attempt in range(1, 3):
            attempt_context = dict(selection_context)
            if validation_findings:
                attempt_context = {
                    "task": selection_context.get("task"),
                    "classes": selection_context.get("classes"),
                    "application_domain": selection_context.get("application_domain"),
                    "target_images_per_class": selection_context.get("target_images_per_class"),
                    "data_selection_policy": selection_context.get("data_selection_policy"),
                    "allowed_sources_by_class": selection_context.get("allowed_sources_by_class"),
                    "previous_invalid_proposal": proposed_selection_snapshot,
                    "validation_errors": validation_findings,
                }
                attempt_context["repair_instructions"] = (
                    "Repair the previous dataset selection. Change only selected_data and its "
                    "rationale, use only allowed sources/counts, and address every validation error. "
                    "The output sources[].dataset_name value must be copied from the exact "
                    "allowed_sources_by_class[].dataset_id value, never from display_name. "
                    "The ideal target is not mandatory: omit an invalid secondary source when "
                    "the remaining valid sources meet dataset_sizing.minimum_images_per_class."
                )
            res_sel = await run_planning_agent(
                job_id=request.job_id,
                operation="dataset_selection",
                agent=selection_agent,
                input=json.dumps(attempt_context),
            )
            # Findings from the previous attempt were input to this repair. Start a
            # fresh collection and combine every independently detectable problem
            # in the new proposal before spending the next LLM attempt.
            validation_findings = []
            selection_rationale = res_sel.final_output.rationale
            proposed_sources = res_sel.final_output.selected_data
            proposed_sources, identifier_normalizations = canonicalize_selected_dataset_ids(
                proposed_sources,
                eligible_data,
            )
            dataset_identifier_normalizations.extend(identifier_normalizations)
            proposed_selection_snapshot = [
                item.model_dump(mode="json")
                if hasattr(item, "model_dump") else item
                for item in proposed_sources
            ]
            max_total = {
                "classification": MAX_CLASSIFICATION_SELECTED_IMAGES,
                "detection": MAX_DETECTION_SELECTED_IMAGES,
            }.get(state.task)
            max_per_class = {
                "classification": MAX_CLASSIFICATION_POOL_PER_CLASS,
                "detection": MAX_DETECTION_POOL_PER_CLASS,
            }.get(state.task)
            class_totals = {
                item.class_name: sum(source.count for source in item.sources)
                for item in proposed_sources
            }
            validation_findings.extend([
                {
                    "code": "CLASS_POOL_LIMIT_EXCEEDED",
                    "field": "selected_data",
                    "class_name": class_name,
                    "selected_count": count,
                    "maximum_count": max_per_class,
                    "reason": "Selected class pool exceeds the execution download limit.",
                }
                for class_name, count in class_totals.items()
                if max_per_class is not None and count > max_per_class
            ])
            total_selected = sum(class_totals.values())
            if max_total is not None and total_selected > max_total:
                validation_findings.append({
                    "code": "TOTAL_POOL_LIMIT_EXCEEDED",
                    "field": "selected_data",
                    "selected_count": total_selected,
                    "maximum_count": max_total,
                    "reason": "Selected dataset allocation exceeds the execution download budget.",
                })
            try:
                if validation_findings:
                    raise DatasetSelectionValidationError(validation_findings)
                selected_sources = validate_dataset_selection(proposed_sources, eligible_data)
                if state.task == "detection":
                    selected_sources = validate_detection_source_coherence(
                        selected_sources,
                        eligible_data,
                    )
                    dataset_advisory_findings.extend(
                        detection_source_coherence_findings(
                            selected_sources, eligible_data,
                        )
                    )
                    dataset_advisory_findings.extend(detection_domain_mix_findings(
                        selected_sources,
                        eligible_data,
                        state.application_domain,
                    ))
                state.selected_data = build_dataset_assignments(
                    selected_sources,
                    eligible_data,
                )
                validation_findings = []
                selection_mode = "llm_validated"
                break
            except DatasetSelectionValidationError as exc:
                validation_findings = exc.findings
                dataset_repair_history.extend(validation_findings)
        if (
            validation_findings
            and state.task == "detection"
            and detection_sizing_recommendation is not None
            and all(
                finding.get("field") == "dataset_name"
                and "not an eligible available source" in str(finding.get("reason", ""))
                for finding in validation_findings
            )
        ):
            repaired_sources, fallback_adjustments = prune_ineligible_optional_sources(
                proposed_sources,
                eligible_data,
                minimum_images_per_class=(
                    detection_sizing_recommendation.minimum_images_per_class
                ),
            )
            if repaired_sources is not None:
                try:
                    selected_sources = validate_dataset_selection(repaired_sources, eligible_data)
                    selected_sources = validate_detection_source_coherence(
                        selected_sources, eligible_data,
                    )
                    dataset_advisory_findings.extend(detection_domain_mix_findings(
                        selected_sources, eligible_data, state.application_domain,
                    ))
                    state.selected_data = build_dataset_assignments(selected_sources, eligible_data)
                    dataset_fallback_adjustments.extend(fallback_adjustments)
                    validation_findings = []
                    selection_mode = "validated_invalid_source_pruning"
                except DatasetSelectionValidationError:
                    pass
        if validation_findings:
            failure = {
                "job_id": request.job_id,
                "message": "Dataset selection remained invalid after two scoped LLM attempts.",
                "proposed_selection": proposed_selection_snapshot,
                "validation_findings": validation_findings,
                "data_selection_policy": data_selection_policy,
            }
            save_json(
                failure,
                planning_artifacts_dir(request.job_id),
                "STATE_04_DATASET_SELECTION_FAILURE.json",
            )
            raise HTTPException(status_code=422, detail=failure)
        state.dataset_profile = build_dataset_profile(state.selected_data)
        selected_ids = {
            source.dataset_name
            for assignment in state.selected_data or []
            for source in assignment.sources
        }
        missing_required = required_includes - selected_ids
        if missing_required:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Dataset selection did not satisfy the required dataset override.",
                    "missing_dataset_ids": sorted(missing_required),
                },
            )
        if graphrag_used:
            aggregated = aggregate_selected_dataset_properties(state)
            active = [item for item in aggregated if item["active"]]
            state.dataset_profile = state.dataset_profile.model_copy(update={
                "characteristics": [item["property_id"] for item in active],
                "characteristic_support": {
                    item["property_id"]: item["support_ratio"] for item in active
                },
            })
        state.step_history.append(f"Data Selection Rationale: {selection_rationale}")
        state.step_history.append("Dataset Selection Completed")

    decision_evidence = build_dataset_selection_decision_evidence(
        [item.model_dump() for item in state.selected_data or []],
        selection_rationale,
        graph_context if graphrag_used else {},
    )
    decision_evidence.update({
            "mode": selection_mode,
            "correctness_source": "local_registry",
            "graphrag_requested": request.use_graphrag,
            "graphrag_used": graphrag_used,
            "guidance_source": "dataset_graphrag" if graphrag_used else "local_registry",
            "validation_findings": dataset_repair_history,
            "advisory_findings": dataset_advisory_findings,
            "dataset_identifier_normalizations": dataset_identifier_normalizations,
            "fallback_adjustments": dataset_fallback_adjustments,
            "fallback_validation_findings": fallback_validation_findings,
            "proposed_selection": proposed_selection_snapshot,
            "fallback_selection": fallback_selection_snapshot,
            "dropped_inferred_classes": droppable_classes if not unavailable_classes else [],
            "eligible_dataset_ids": sorted({
                source.dataset_name
                for item in eligible_data
                for source in item.sources
            }),
            "selected_dataset_ids": sorted({
                source.dataset_name
                for item in state.selected_data or []
                for source in item.sources
            }),
            "dataset_profile": state.dataset_profile.model_dump() if state.dataset_profile else None,
            "split_construction_summary": build_split_construction_summary(
                state.selected_data or []
            ),
            "source_selection_rationale": selection_rationale,
            "deployment_coverage_warnings": coverage_warnings,
            "data_selection_policy": data_selection_policy,
            "assignments_authoritative": True,
            "split_policy": {
                "official_splits_preserved": True,
                "missing_holdouts": "derived_from_train",
                "derived_holdout_sizing": "adaptive_by_selected_training_pool",
                "derived_holdouts_source_stratified": True,
                "multi_family_primary_holdouts": (
                    "derived_from_sufficiently_represented_training_sources"
                ),
                "single_family_primary_holdouts": "prefer_compatible_official_splits",
                "official_holdouts_require_selected_training_family": True,
                "classification_default_pool_per_class": DEFAULT_CLASSIFICATION_POOL_PER_CLASS,
                "classification_max_pool_per_class": MAX_CLASSIFICATION_POOL_PER_CLASS,
                "classification_max_selected_images": MAX_CLASSIFICATION_SELECTED_IMAGES,
                "detection_default_pool_per_class": DEFAULT_DETECTION_POOL_PER_CLASS,
                "detection_medium_high_pool_per_class": (
                    MEDIUM_HIGH_DETECTION_POOL_PER_CLASS
                ),
                "detection_recommended_pool_per_class": recommended_detection_target,
                "detection_sizing_recommendation": (
                    data_selection_policy.get("dataset_sizing")
                ),
                "selection_target_images_per_class": selection_target_images_per_class,
                "detection_max_pool_per_class": MAX_DETECTION_POOL_PER_CLASS,
                "detection_max_selected_image_allocations": MAX_DETECTION_SELECTED_IMAGES,
                "detection_max_instances_per_class": None,
                "detection_instance_limit_enforced": False,
                "detection_shared_backbone_required_when_available": True,
                "detection_shared_backbone_min_available_images": (
                    DETECTION_SHARED_BACKBONE_MIN_COUNT
                ),
                "detection_shared_backbone_min_share": DETECTION_SHARED_BACKBONE_MIN_SHARE,
                "detection_shared_backbone_target_images": (
                    DETECTION_SHARED_BACKBONE_TARGET_COUNT
                ),
                "explicit_images_per_class": explicit_pool_target,
                "test_used_for_model_selection": False,
            },
    })
    state.dataset_selection_decision_evidence = decision_evidence
    save_checkpoint(state, request.job_id, "STATE_04_DATASET_SELECTION.json")
    return {
        "context": state.model_dump(),
        "decision_evidence": decision_evidence,
    }

@router.post("/choose-hyperparameters")
async def choose_hyperparameters(request: StateRequest):
    state = get_state(request.context)
    graph_context: Dict[str, Any] = {}
    use_graphrag = request.use_graphrag and USE_HYPERPARAMETER_GRAPHRAG
    state = state.model_copy(update={
        "hyperparameter_graph_context": None,
        "hyperparameter_decision_evidence": None,
        "use_graphrag": use_graphrag,
    })

    if use_graphrag:
        graph_context = build_hyperparameter_context(state)
        if graph_context.get("critical_materialization_errors"):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "The selected ontology recipe contains critical fields that are not executable.",
                    "model_name": graph_context.get("selected_model_id"),
                    "errors": graph_context["critical_materialization_errors"],
                },
            )
        if state.task in {"classification", "detection"} and not graph_context.get("base_recipe"):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "No executable pretrained fine-tuning recipe is available for the selected model.",
                    "model_name": graph_context.get("selected_model_id")
                    or (graph_context.get("selected_model") or {}).get("id"),
                    "reason": graph_context.get("warning")
                    or "Only incompatible or non-fine-tuning recipes were found.",
                },
            )
        state.hyperparameter_graph_context = {
            **graph_context,
            "llm_context": format_hyperparameter_context(graph_context),
        }
        state.step_history.append(summarize_hyperparameter_context(graph_context))
    else:
        state.step_history.append("Hyperparameter GraphRAG: disabled for this run.")

    field_provenance: Dict[str, Any] = {}
    try:
        candidate, decision = await asyncio.wait_for(
            generate_and_evaluate_hpo(state.model_dump_json(), job_id=request.job_id),
            timeout=480,
        )
    except HpoPhaseTimeout as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "message": "A hyperparameter model call timed out.",
                "phase": exc.phase,
                "round": exc.round_idx,
                "timeout_seconds": exc.timeout_seconds,
                "attempts": exc.attempts,
                "reason": (
                    "Both attempts for the individual optimizer/evaluator request timed out; "
                    "earlier successful phases are not evidence of a general API outage."
                ),
            },
        ) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "message": "The complete hyperparameter negotiation exceeded 480 seconds.",
                "phase": "overall_negotiation",
                "timeout_seconds": 480,
            },
        ) from exc

    if candidate is None:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Hyperparameter generation did not produce a valid candidate configuration.",
                "reason": "The generated values could not satisfy the executable configuration schema.",
                "diagnostics": getattr(decision, "_diagnostics", []) if decision else [],
                "suggestions": [
                    "Retry generation or inspect the server-side schema diagnostics."
                ],
            },
        )

    if decision is None:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "No good hyperparameters could be found.",
                "reason": "The evaluator did not return a decision for the proposed hyperparameters.",
                "suggestions": ["Retry hyperparameter generation or check evaluator/model availability."],
                "last_candidate": candidate.model_dump(),
            },
        )

    if decision is not None and not decision.accept:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No good hyperparameters could be found.",
                "reason": decision.reason,
                "suggestions": decision.suggestions or [],
                "diagnostics": getattr(decision, "_diagnostics", []),
                "evaluator_findings": [
                    finding.model_dump(mode="json") for finding in decision.findings
                ],
                "last_candidate": candidate.model_dump(),
            },
        )
    
    try:
        if state.task in {"classification", "detection"}:
            candidate = apply_owned_pipeline_fields(candidate, state.model_dump(mode="json"))
            required_hpo = {
                change.field.removeprefix("hpo_config."): change.value
                for change in changes_for(state, "choose-hyperparameters", strength="required")
                if change.operation == "set" and change.field.startswith("hpo_config.")
            }
            mismatches = {
                field: {"expected": expected, "actual": getattr(candidate, field, None)}
                for field, expected in required_hpo.items()
                if getattr(candidate, field, None) != expected
            }
            if mismatches:
                raise ValueError(
                    "Required user hyperparameters conflict with runtime-owned constraints: "
                    f"{mismatches}"
                )
            validate_executable_recipe_config(candidate.model_dump(mode="json"))
            validate_training_resource_config(candidate.model_dump(mode="json"))
        if state.task in {"classification", "detection"} and use_graphrag:
            authorized_repairs = set(
                getattr(decision, "_authorized_repair_fields", set())
            )
            authorized_repairs.update(required_hpo)
            candidate_data = candidate.model_dump(mode="json")
            reference_configuration = graph_context.get("reference_configuration") or {}
            changed_repairs = {
                field
                for field in authorized_repairs
                if candidate_data.get(field) != reference_configuration.get(field)
            }
            active_fields = (
                active_classification_config_fields(candidate_data)
                if state.task == "classification"
                else active_detection_config_fields(candidate_data)
            )
            optimizer_owned_fields = active_fields - PIPELINE_OWNED_HPO_CONTEXT_FIELDS
            llm_explanations = [
                item
                for item in getattr(candidate, "llm_field_rationales", [])
                if item.field in optimizer_owned_fields
            ]
            explained_fields = {item.field for item in llm_explanations}
            required_explanations = llm_controlled_fields(
                candidate_data,
                graph_context,
                type(candidate),
            ) | changed_repairs
            required_explanations &= optimizer_owned_fields
            missing_explanations = required_explanations - explained_fields
            if missing_explanations:
                raise ValueError(
                    "Missing LLM field rationale for completed or adjusted fields: "
                    f"{sorted(missing_explanations)}."
                )
            if llm_explanations:
                explanation_lines = "\n".join(
                    f"- {item.field}: {item.reason}"
                    for item in llm_explanations
                )
                candidate = candidate.model_copy(update={
                    "rationale": (
                        f"{candidate.rationale.rstrip()}\n\n"
                        f"LLM-completed or adjusted fields:\n{explanation_lines}"
                    )
                })
            runtime_config = candidate.runtime_config()
            llm_adapted_fields = {
                field for field in optimizer_owned_fields
                if field in reference_configuration
                and candidate_data.get(field) != reference_configuration.get(field)
            }
            all_provenance = build_field_provenance(
                candidate_data,
                graph_context,
                llm_adjusted_fields=changed_repairs | llm_adapted_fields,
            )
            field_provenance = {
                field: details
                for field, details in all_provenance.items()
                if field in active_fields
            }
            state.hpo_config = runtime_config
        else:
            state.hpo_config = candidate.runtime_config() if hasattr(candidate, "runtime_config") else candidate.model_dump()
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Hyperparameter generation returned a configuration that failed final recipe validation; "
                    "it was not saved."
                ),
                "reason": str(exc),
                "last_candidate": candidate.model_dump(mode="json"),
            },
        ) from exc

    state.hpo_decision = decision.model_dump() if decision else None
    evidence_context = dict(graph_context)

    state.hyperparameter_decision_evidence = build_hyperparameter_decision_evidence(
        state.hpo_config or {},
        str(getattr(candidate, "rationale", decision.reason if decision else "")),
        evidence_context,
        field_provenance=field_provenance,
    )
    state.step_history.append("Hyperparameter Optimization Completed")
    
    save_checkpoint(state, request.job_id, "STATE_05_HYPERPARAMETERS.json")
    save_hpo_result(state, request.job_id)
    return {
        "context": state.model_dump(),
        "decision_evidence": state.hyperparameter_decision_evidence,
    }

@router.post("/add-user-request", response_model=RequestAddedResponse)
async def add_user_request(request: AddUserRequest):
    state = get_state(request.context)
    text = (request.request_text or "").strip()

    if text:
        existing = state.model_extra.get("user_change_requests", []) if state.model_extra else []
        if not isinstance(existing, list): 
            existing = [str(existing)]
        
        updated_extras = state.model_extra or {}
        updated_extras["user_change_requests"] = existing + [text + "\n"]
        
        state_dict = state.model_dump()
        state_dict.update(updated_extras)
        state = PipelineState(**state_dict)

        state.step_history.append(f"User requested change: '{text}'")
        save_checkpoint(state, request.job_id, "STATE_USER_CHANGE_REQUEST.json")

    return {"context": state.model_dump()}
