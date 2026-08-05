import json
import asyncio
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

# Core agent/pipeline imports
from agents import Runner
from cvmodellearning.schemas.interpretation_schema import (
    ClassDataSelection,
    DeploymentConstraints,
    HardwareSpecModel,
    PerformanceSpecModel,
    PipelineState,
)
from cvmodellearning.training.hardware_profiles import active_training_hardware_profile
from cvmodellearning.schemas.classification_hpo import active_classification_config_fields
from cvmodellearning.schemas.detection_hpo import active_detection_config_fields
from cvmodellearning.agents.interpretation_agents import interpretation_loop, task_interpretation_agent, synonym_check_agent
from cvmodellearning.agents.model_selection_agents import classification_model_selector_agent, detection_model_selector_agent, vqq_model_selector_agent
from cvmodellearning.agents.hyperparameter_agents import (
    apply_owned_pipeline_fields,
    generate_and_evaluate_hpo,
)
from cvmodellearning.agents.data_selection_and_augmentation_agents import (
    classification_dataset_selection_agent, detection_dataset_selection_agent, 
    vqa_dataset_selection_agent,
)
from cvmodellearning.agents.agents_utils import save_json, load_unified_dataset_classes
from cvmodellearning.models.registry import family_for_model_reference
from cvmodellearning.paths import hpo_config_path, planning_artifacts_dir
from cvmodellearning.download.visionkg_utils import get_multi_class_stats
from cvmodellearning.datasets.selection import (
    DEFAULT_CLASSIFICATION_POOL_PER_CLASS,
    DEFAULT_DETECTION_POOL_PER_CLASS,
    DETECTION_SHARED_BACKBONE_MIN_COUNT,
    DETECTION_SHARED_BACKBONE_MIN_SHARE,
    DETECTION_SHARED_BACKBONE_TARGET_COUNT,
    MAX_CLASSIFICATION_POOL_PER_CLASS,
    MAX_CLASSIFICATION_SELECTED_IMAGES,
    MAX_DETECTION_SELECTED_IMAGES,
    MAX_DETECTION_POOL_PER_CLASS,
    DatasetSelectionValidationError,
    build_default_dataset_selection,
    build_dataset_assignments,
    build_dataset_profile,
    build_split_construction_summary,
    filter_dataset_candidates,
    limit_selected_source_pools,
    validate_detection_source_coherence,
    validate_dataset_selection,
)
from cvmodellearning.datasets.registry import DatasetRole, dataset_family, resolve_dataset_info
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
    validate_detection_graph_grounded_config,
    validate_graph_grounded_config,
)
from cvmodellearning.graphrag.decision_evidence import (
    build_dataset_selection_decision_evidence,
    build_hyperparameter_decision_evidence,
    build_model_selection_decision_evidence,
)
from cvmodellearning.policies.hyperparameter_policy_registry import (
    build_hyperparameter_policy_context,
    policy_fields,
)
from cvmodellearning.jobs.error_persistence import ErrorPersistingRoute

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
    use_policy_registry: bool = True

class AddUserRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    request_text: Optional[str] = None
    job_id: Optional[str] = None

class RequestAddedResponse(BaseModel):
    context: Dict[str, Any]


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
        request.user_prompt, user_replies=request.user_replies
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
    state.use_policy_registry = request.use_policy_registry
    res = await Runner.run(task_interpretation_agent, input=state.model_dump_json())
    extracted = res.final_output
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
    
    valid_classes = load_unified_dataset_classes(extracted.task)
    final_classes = []
    class_expansions: Dict[str, List[str]] = {}
    valid_classes_str = ", ".join(sorted(list(valid_classes)))

    # Check each extracted class directly first. If it is not directly valid,
    # ask the ontology matcher for synonym/subcategory/supercategory mappings.
    for cls in (extracted.classes or []):
        cls_clean = cls.strip().lower()
        if cls_clean in valid_classes:
            final_classes.append(cls_clean)
            continue
            
        syn_res = await Runner.run(synonym_check_agent, input=f"User Class: '{cls}'. Allowed: [{valid_classes_str}]")
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
            raise HTTPException(status_code=400, detail=f"Class '{cls}' not found.")

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
    state.step_history.append("Task Interpretation Completed")
    
    save_checkpoint(state, request.job_id, "STATE_01_INTERPRETATION.json")
    return {"context": state.model_dump()}

@router.post("/check-data")
async def check_data(request: StateRequest):
    state = get_state(request.context)
    state.use_graphrag = request.use_graphrag
    state.use_policy_registry = request.use_policy_registry

    if state.classes and not state.task:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Data availability checking requires an interpreted task.",
            },
        )
    
    if state.classes:
        query_path = planning_artifacts_dir(request.job_id) / "DATA_CHECK_QUERY.sparql"
        raw_stats = get_multi_class_stats(state.classes, query_output_path=query_path)
        
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

@router.post("/select-model")
async def select_model(request: StateRequest):
    state = get_state(request.context)
    ensure_default_hardware_filter(state)
    use_graphrag = request.use_graphrag and USE_MODEL_SELECTION_GRAPHRAG
    state = state.model_copy(update={
        "model_selection_graph_context": None,
        "model_selection_decision_evidence": None,
        "use_graphrag": use_graphrag,
        "use_policy_registry": request.use_policy_registry,
    })

    if use_graphrag:
        graph_context = build_model_selection_context(state)
        recommendation = graph_context.get("deterministic_recommendation")
        if recommendation is None and "benchmark_target" in graph_context.get("filters", {}):
            relaxed_performance = state.performance_requirements.model_copy(
                update={"target_is_hard": False}
            )
            relaxed_state = state.model_copy(
                update={"performance_requirements": relaxed_performance}
            )
            relaxed_context = build_model_selection_context(relaxed_state)
            recommendation = relaxed_context.get("deterministic_recommendation")
            if recommendation is not None:
                graph_context = {
                    **relaxed_context,
                    "requested_filters": graph_context.get("filters", {}),
                    "constraint_warnings": [
                        *relaxed_context.get("constraint_warnings", []),
                        "No catalog benchmark proves the requested hard performance target; "
                        "the target remains unverified and was not used to exclude executable models."
                    ],
                }
        if recommendation is None:
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
        res = await Runner.run(agent, input=state.model_dump_json())
        model_patch_dict = res.final_output.model_dump(exclude_unset=True)
        
        # Pop rationale to prevent floating JSON keys
        model_rationale = sanitize_english_rationale(
            model_patch_dict.pop("rationale", "No rationale provided.")
        )
        if use_graphrag:
            recommendation = graph_context["deterministic_recommendation"]
            recommended_id = recommendation["model_id"]
            model_payload = model_patch_dict.setdefault("model", {})
            agent_choice = model_payload.get("model_architecture")
            model_payload["model_architecture"] = recommended_id
            model_payload["description"] = (
                f"{recommendation['model_name']} selected by the deterministic "
                f"{recommendation['policy']} policy."
            )
            if agent_choice and agent_choice != recommended_id:
                model_rationale = (
                    f"{model_rationale} Local constraint validation replaced the agent choice "
                    f"'{agent_choice}' with '{recommended_id}'."
                )
        model_payload = model_patch_dict.setdefault("model", {})
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
        state.model_selection_decision_evidence = decision_evidence
        
        state.step_history.append(f"Model Selection Rationale: {model_rationale}")
        state.step_history.append("Model Selection Completed")
    
    save_checkpoint(state, request.job_id, "STATE_03_MODEL_SELECTION.json")
    return {
        "context": state.model_dump(),
        "decision_evidence": state.model_selection_decision_evidence,
    }

@router.post("/select-datasets")
async def select_datasets(request: StateRequest):
    state = get_state(request.context)
    state.dataset_selection_graph_context = None
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
    selection_mode = "llm_validated"
    selection_rationale = "No dataset selection rationale was returned."
    graphrag_used = False
    graph_context: Dict[str, Any] = {}
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
                    "description": info.description,
                    "family": dataset_family(source.dataset_name),
                }
                for source in item.sources
                if (info := resolve_dataset_info(source.dataset_name)) is not None
            ]
            for item in eligible_data
        }
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
        res_sel = await Runner.run(selection_agent, input=json.dumps(selection_context))
        selection_rationale = res_sel.final_output.rationale
        try:
            proposed_sources = res_sel.final_output.selected_data
            if state.task == "classification":
                proposed_sources = limit_selected_source_pools(
                    proposed_sources,
                    max_total_images=MAX_CLASSIFICATION_SELECTED_IMAGES,
                    max_images_per_class=MAX_CLASSIFICATION_POOL_PER_CLASS,
                )
            elif state.task == "detection":
                proposed_sources = limit_selected_source_pools(
                    proposed_sources,
                    max_total_images=MAX_DETECTION_SELECTED_IMAGES,
                    max_images_per_class=MAX_DETECTION_POOL_PER_CLASS,
                )
            selected_sources = validate_dataset_selection(
                proposed_sources,
                eligible_data,
            )
            if state.task == "detection":
                selected_sources = validate_detection_source_coherence(
                    selected_sources,
                    eligible_data,
                )
            state.selected_data = build_dataset_assignments(selected_sources, eligible_data)
        except DatasetSelectionValidationError as exc:
            validation_findings = exc.findings
            selection_mode = "deterministic_fallback"
            selection_rationale = (
                "The proposed dataset selection failed local eligibility, count, or "
                "source-coherence validation, so the deterministic locally registered "
                "selection was used."
            )
            fallback_sources = build_default_dataset_selection(
                eligible_data,
                target_images_per_class=(
                    DEFAULT_CLASSIFICATION_POOL_PER_CLASS
                    if state.task == "classification"
                    else DEFAULT_DETECTION_POOL_PER_CLASS
                ),
                prefer_shared_training_family=state.task == "detection",
            )
            if state.task in {"classification", "detection"}:
                fallback_sources = limit_selected_source_pools(
                    fallback_sources,
                    max_total_images=(
                        MAX_CLASSIFICATION_SELECTED_IMAGES
                        if state.task == "classification"
                        else MAX_DETECTION_SELECTED_IMAGES
                    ),
                    max_images_per_class=(
                        MAX_CLASSIFICATION_POOL_PER_CLASS
                        if state.task == "classification"
                        else MAX_DETECTION_POOL_PER_CLASS
                    ),
                )
            fallback_sources = validate_dataset_selection(fallback_sources, eligible_data)
            state.selected_data = build_dataset_assignments(fallback_sources, eligible_data)
        state.dataset_profile = build_dataset_profile(state.selected_data)
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
            "validation_findings": validation_findings,
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
            "deployment_coverage_warnings": deployment_coverage_warnings(state),
            "assignments_authoritative": True,
            "split_policy": {
                "official_splits_preserved": True,
                "missing_holdouts": "derived_from_train",
                "derived_holdout_sizing": "adaptive_by_selected_training_pool",
                "derived_holdouts_source_stratified": True,
                "multi_family_primary_holdouts": "derived_from_all_training_sources",
                "single_family_primary_holdouts": "prefer_compatible_official_splits",
                "official_holdouts_require_selected_training_family": True,
                "classification_default_pool_per_class": DEFAULT_CLASSIFICATION_POOL_PER_CLASS,
                "classification_max_pool_per_class": MAX_CLASSIFICATION_POOL_PER_CLASS,
                "classification_max_selected_images": MAX_CLASSIFICATION_SELECTED_IMAGES,
                "detection_default_pool_per_class": DEFAULT_DETECTION_POOL_PER_CLASS,
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
                "explicit_images_per_class": explicit_images_per_class(state.user_query),
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
    use_policy_registry = request.use_policy_registry
    state = state.model_copy(update={
        "hyperparameter_graph_context": None,
        "hyperparameter_policy_context": None,
        "hyperparameter_decision_evidence": None,
        "use_graphrag": use_graphrag,
        "use_policy_registry": use_policy_registry,
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

    if use_policy_registry:
        policy_context = build_hyperparameter_policy_context(state)
        state.hyperparameter_policy_context = policy_context
        if use_graphrag:
            graph_context["hyperparameter_policy_context"] = policy_context
            graph_context["fields_available_for_policy_guidance"] = sorted(
                policy_fields(policy_context) - set(graph_context.get("base_configuration") or {})
            )
            state.hyperparameter_graph_context = {
                **graph_context,
                "llm_context": format_hyperparameter_context(graph_context),
            }
        state.step_history.append("Hyperparameter Policy Registry: enabled for this run.")
    else:
        graph_context.pop("hyperparameter_policy_context", None)
        graph_context["fields_available_for_policy_guidance"] = []
        if use_graphrag:
            state.hyperparameter_graph_context = {
                **graph_context,
                "llm_context": format_hyperparameter_context(graph_context),
            }
        state.step_history.append("Hyperparameter Policy Registry: disabled for this run.")

    field_provenance: Dict[str, Any] = {}
    try:
        candidate, decision = await asyncio.wait_for(
            generate_and_evaluate_hpo(state.model_dump_json(), job_id=request.job_id),
            timeout=360,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Hyperparameter generation timed out after 360 seconds. Check the OpenAI API connection/model availability and try again.",
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
            validate_executable_recipe_config(candidate.model_dump(mode="json"))
        if state.task in {"classification", "detection"} and use_graphrag:
            authorized_repairs = set(
                getattr(decision, "_authorized_repair_fields", set())
            )
            candidate_data = candidate.model_dump(mode="json")
            base_configuration = graph_context.get("base_configuration") or {}
            changed_repairs = {
                field
                for field in authorized_repairs
                if candidate_data.get(field) != base_configuration.get(field)
            }
            active_fields = (
                active_classification_config_fields(candidate_data)
                if state.task == "classification"
                else active_detection_config_fields(candidate_data)
            )
            llm_explanations = [
                item
                for item in getattr(candidate, "llm_field_rationales", [])
                if item.field in active_fields
            ]
            explained_fields = {item.field for item in llm_explanations}
            required_explanations = llm_controlled_fields(
                candidate_data,
                graph_context,
                type(candidate),
            ) | changed_repairs
            required_explanations &= active_fields
            missing_explanations = required_explanations - explained_fields
            if missing_explanations:
                raise ValueError(
                    "Missing LLM field rationale for completed or adjusted fields: "
                    f"{sorted(missing_explanations)}."
                )
            if state.task == "classification":
                validate_graph_grounded_config(
                    candidate.model_dump(mode="json"),
                    graph_context,
                    additional_allowed_fields=changed_repairs,
                )
            else:
                validate_detection_graph_grounded_config(
                    candidate.model_dump(mode="json"),
                    graph_context,
                    additional_allowed_fields=changed_repairs,
                )
            if llm_explanations:
                explanation_lines = "\n".join(
                    f"- {item.field}: {item.reason}"
                    + (
                        f" [policies: {', '.join(item.applied_policy_ids)}]"
                        if item.applied_policy_ids else ""
                    )
                    for item in llm_explanations
                )
                candidate = candidate.model_copy(update={
                    "rationale": (
                        f"{candidate.rationale.rstrip()}\n\n"
                        f"LLM-completed or adjusted fields:\n{explanation_lines}"
                    )
                })
            runtime_config = candidate.runtime_config()
            all_provenance = build_field_provenance(
                candidate_data,
                graph_context,
                llm_adjusted_fields=changed_repairs,
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
    if use_policy_registry and state.hyperparameter_policy_context:
        evidence_context["hyperparameter_policy_context"] = (
            state.hyperparameter_policy_context
        )

    if (
        use_policy_registry
        and not use_graphrag
        and state.task in {"classification", "detection"}
    ):
        candidate_data = candidate.model_dump(mode="json")
        active_fields = (
            active_classification_config_fields(candidate_data)
            if state.task == "classification"
            else active_detection_config_fields(candidate_data)
        )
        field_provenance = {
            field: details
            for field, details in build_field_provenance(
                candidate_data,
                evidence_context,
            ).items()
            if field in active_fields
        }

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
