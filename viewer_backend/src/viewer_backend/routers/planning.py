from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from fastapi import APIRouter, HTTPException

from ..graphrag.ontology import get_ontology
from ..graphrag.evidence import decision_evidence
from ..llm import structured_call
from ..schemas import (
    CompletenessDecision,
    CompletenessRequest,
    DatasetPlan,
    HyperparameterPlan,
    ModelPlan,
    PlanRevisionRequest,
    ActivateRevisionRequest,
    VerifyRevisionRequest,
    ForkRevisionRequest,
    RevisionPlan,
    StateRequest,
    TaskInterpretation,
)
from ..store import planning_dir, read_json, run_dir, write_json
from ..registries import get_registry
from ..visionkg import query_class_availability
from ..dataset_planning import availability_candidates, build_split_assignments, preprocessing_plan
from ..hpo_planning import materialize_hpo
from ..hpo_evaluation import decision_from_findings, evaluate_hpo, repair_hpo
from ..data_strategy import build_data_plan_conflicts, build_data_strategy


router = APIRouter(prefix="/planning", tags=["Planning"])

STEP_ORDER = ("task-interpretation", "model-selection", "dataset-selection", "choose-hyperparameters")
PREDECESSOR = {
    "task-interpretation": None,
    "model-selection": "STATE_02_DATA_CHECK.json",
    "dataset-selection": "STATE_03_MODEL_SELECTION.json",
    "choose-hyperparameters": "STATE_04_DATASET_SELECTION.json",
}
DOWNSTREAM = {
    "task-interpretation": ["STATE_01_INTERPRETATION.json", "STATE_02_DATA_CHECK.json", "DATA_CHECK_QUERY.sparql", "STATE_03_MODEL_SELECTION.json", "STATE_04_DATASET_SELECTION.json", "STATE_04_PREPROCESSING.json", "STATE_05_HYPERPARAMETERS.json", "RESULT_HYPERPARAMETERS.json", "HYPERPARAMETER_PROPOSAL.json"],
    "model-selection": ["STATE_03_MODEL_SELECTION.json", "STATE_04_DATASET_SELECTION.json", "STATE_04_PREPROCESSING.json", "STATE_05_HYPERPARAMETERS.json", "RESULT_HYPERPARAMETERS.json", "HYPERPARAMETER_PROPOSAL.json"],
    "dataset-selection": ["STATE_04_DATASET_SELECTION.json", "STATE_04_PREPROCESSING.json", "STATE_05_HYPERPARAMETERS.json", "RESULT_HYPERPARAMETERS.json", "HYPERPARAMETER_PROPOSAL.json"],
    "choose-hyperparameters": ["STATE_05_HYPERPARAMETERS.json", "RESULT_HYPERPARAMETERS.json", "HYPERPARAMETER_PROPOSAL.json"],
}

MODEL_CATALOG = {
    "classification": [
        {"id": "resnet50", "display_name": "ResNet-50", "family": "ResNet"},
        {"id": "efficientnet_b0", "display_name": "EfficientNet-B0", "family": "EfficientNet"},
        {"id": "convnext_tiny", "display_name": "ConvNeXt Tiny", "family": "ConvNeXt"},
        {"id": "mobilenet_v3_large", "display_name": "MobileNet V3 Large", "family": "MobileNet"},
    ],
    "detection": [
        {"id": "yolov11", "display_name": "YOLO11", "family": "YOLO"},
        {"id": "faster-rcnn_r50_fpn_1x_coco", "display_name": "Faster R-CNN R50 FPN", "family": "Faster R-CNN"},
        {"id": "rtdetr_hgnetv2_l", "display_name": "RT-DETR-L", "family": "RT-DETR"},
        {"id": "ssd300_coco", "display_name": "SSD300 VGG16", "family": "SSD"},
    ],
    "visual question answering": [
        {"id": "Qwen3-VL-2B-Instruct", "display_name": "Qwen3-VL 2B Instruct", "family": "Qwen-VL"},
    ],
}


def _history(context: dict[str, Any], entry: str) -> None:
    history = context.setdefault("step_history", [])
    if isinstance(history, list):
        history.append(entry)


def _checkpoint(job_id: str, filename: str, context: dict[str, Any]) -> None:
    directory = planning_dir(job_id)
    write_json(directory / filename, context)
    sections = ["Viewer planning rationale log", f"Latest checkpoint: {filename}", ""]
    sections.extend(f"- {entry}" for entry in context.get("step_history", []) if isinstance(entry, str))
    for title, field in (
        ("Model decision", "model_selection_decision_evidence"),
        ("Dataset decision", "dataset_selection_decision_evidence"),
        ("Hyperparameter decision", "hyperparameter_decision_evidence"),
    ):
        evidence = context.get(field)
        if isinstance(evidence, dict):
            sections.extend(["", title, str(evidence.get("rationale") or evidence.get("summary") or "")])
            if evidence.get("uncertainties"):
                sections.append("Uncertainties: " + "; ".join(map(str, evidence["uncertainties"])))
    (directory / "planning_rationales.txt").write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def _evidence(title: str, rationale: str, candidates: list[dict] | None = None, uncertainties: list[str] | None = None) -> dict:
    return {
        "title": title,
        "summary": rationale,
        "rationale": rationale,
        "evaluated_candidates": candidates or [],
        "uncertainties": uncertainties or [],
        "evidence": [],
    }


def _canonical_reference(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _normalize_interpretation(document: dict[str, Any], query: str) -> dict[str, Any]:
    """Apply deterministic interpretation rules that should not depend on an LLM."""
    value = dict(document)
    task_aliases = {"object detection": "detection", "image classification": "classification", "vqa": "visual question answering"}
    value["task"] = task_aliases.get(str(value.get("task", "")).lower(), value.get("task"))
    value["classes"] = list(dict.fromkeys(
        str(item).strip().lower() for item in value.get("classes", []) if str(item).strip()
    ))
    robustness = dict(value.get("robustness_requirements") or {})
    lowered = query.lower()
    if re.search(r"\b(night|dark)\b", lowered):
        robustness["lighting"] = sorted({*robustness.get("lighting", []), "night"})
    if re.search(r"\b(low[ -]?light|dim|poorly lit)\b", lowered):
        robustness["lighting"] = sorted({*robustness.get("lighting", []), "low_light"})
    for pattern, label in ((r"\brain(?:y)?\b", "rain"), (r"\bfog(?:gy)?\b", "fog"), (r"\bsnow(?:y)?\b", "snow")):
        if re.search(pattern, lowered):
            robustness["weather"] = sorted({*robustness.get("weather", []), label})
    if re.search(r"\b(small|tiny|distant|far away)\b", lowered):
        robustness["object_scale"] = sorted({*robustness.get("object_scale", []), "small"})
    robustness["occlusion"] = bool(robustness.get("occlusion") or re.search(r"\b(occluded|occlusion|partially hidden)\b", lowered))
    robustness["motion_blur"] = bool(robustness.get("motion_blur") or "motion blur" in lowered)
    value["robustness_requirements"] = robustness
    performance = dict(value.get("performance_requirements") or {})
    if performance.get("target_value") is not None and "map" in str(performance.get("primary_metric", "")).lower() and not performance.get("accuracy_category"):
        target = float(performance["target_value"])
        performance["accuracy_category"] = "High" if target > .45 else "MediumHigh" if target >= .30 else "Medium" if target >= .20 else "Low"
    value["performance_requirements"] = performance
    strengths = dict(value.get("constraint_strengths") or {})
    mandatory = bool(re.search(r"\b(must|required|at least|no more than|maximum)\b", lowered))
    preferred = bool(re.search(r"\b(prefer|desirable|ideally|approximately|roughly)\b", lowered))
    if performance.get("target_value") is not None and strengths.get("accuracy", "unspecified") == "unspecified":
        strengths["accuracy"] = "hard" if mandatory else "preference" if preferred else "soft"
    if (performance.get("latency_category") or (value.get("deployment_constraints") or {}).get("max_cpu_latency_ms")) and strengths.get("latency", "unspecified") == "unspecified":
        strengths["latency"] = "hard" if mandatory else "preference" if preferred else "soft"
    value["constraint_strengths"] = strengths
    hardware = dict(value.get("available_hardware") or {})
    if not hardware.get("hardware_category") and hardware.get("vram_gb") is None:
        hardware["hardware_category"] = "ConsumerCPU | EdgeDevice"
    value["available_hardware"] = hardware
    return value


def _merge_interpretation(context: dict[str, Any], interpretation: dict[str, Any]) -> dict[str, Any]:
    merged = dict(context)
    nested = {
        "performance_requirements", "constraint_strengths", "robustness_requirements",
        "deployment_constraints", "available_hardware",
    }
    for key, value in interpretation.items():
        if key in nested and isinstance(value, dict):
            merged[key] = {**(merged.get(key) or {}), **{
                field: field_value for field, field_value in value.items() if field_value is not None
            }}
        elif value not in (None, [], ""):
            merged[key] = value
    return merged


def _revision_changes(context: dict[str, Any], step: str, strength: str | None = None) -> list[dict[str, Any]]:
    active = ((context.get("revision") or {}).get("active") or {})
    return [change for change in active.get("changes", []) if change.get("target_step") == step and (strength is None or change.get("strength") == strength)]


def _validate_revision(plan: RevisionPlan, context: dict[str, Any]) -> None:
    if not plan.changes:
        raise HTTPException(status_code=422, detail="The revision contains no actionable changes.")
    earliest = min((change.target_step for change in plan.changes), key=STEP_ORDER.index)
    if plan.restart_from != earliest:
        raise HTTPException(status_code=422, detail="Revision restart_from is not the earliest affected step.")
    hpo_fields = {
        "classification": 54, "detection": 70, "visual question answering": 28,
    }
    from ..planning_contracts import ClassificationHPOConfig, DetectionHPOConfig, VQAHPOConfig
    schemas = {"classification": ClassificationHPOConfig, "detection": DetectionHPOConfig, "visual question answering": VQAHPOConfig}
    errors = []
    for change in plan.changes:
        valid = (
            change.target_step == "task-interpretation" and change.field in {"task", "classes", "application_domain", "performance_requirements", "deployment_constraints", "available_hardware"} and change.operation == "set"
        ) or (
            change.target_step == "model-selection" and change.field == "model_name" and change.operation in {"set", "prefer", "avoid"}
        ) or (
            change.target_step == "dataset-selection" and change.field in {"dataset.include", "dataset.exclude"} and change.operation in {"include", "exclude", "prefer", "avoid"}
        ) or (
            change.target_step == "choose-hyperparameters" and change.field.startswith("hpo_config.") and change.field.removeprefix("hpo_config.") in getattr(schemas.get(context.get("task")), "model_fields", {}) and change.operation in {"set", "prefer", "avoid"}
        )
        if not valid:
            errors.append({"id": change.id, "field": change.field, "operation": change.operation})
    if errors:
        raise HTTPException(status_code=422, detail={"message": "Unsupported revision changes.", "changes": errors})


@router.post("/completenesscheck")
async def completeness_check(request: CompletenessRequest):
    decision = await structured_call(
        job_id=request.job_id,
        operation="completeness_check",
        response_model=CompletenessDecision,
        prompt=(
            "Determine whether this request contains enough information to begin a high-level "
            "computer-vision plan. At minimum, the intended visual task or outcome must be clear. "
            "Do not require training hardware because this service only plans.\n\n"
            f"REQUEST: {request.user_prompt}\nREPLIES: {json.dumps(request.user_replies)}"
        ),
    )
    context = None
    if decision.accept:
        context = {
            "user_query": request.user_prompt,
            "user_replies": request.user_replies,
            "use_graphrag": False,
            "step_history": ["Completeness check accepted"],
        }
        _checkpoint(request.job_id, "STATE_00_COMPLETENESS.json", context)
    return {**decision.model_dump(mode="json"), "context": context}


@router.post("/task-interpret")
async def task_interpret(request: StateRequest):
    context = dict(request.context)
    interpreted = await structured_call(
        job_id=request.job_id,
        operation="task_interpretation",
        response_model=TaskInterpretation,
        prompt=(
            "Interpret the complete computer-vision planning request. Use 'visual question answering' "
            "exactly for VQA. Extract only explicitly requested or directly entailed classes. Preserve "
            "performance targets, whether constraints are hard/soft/preferences, deployment limits, "
            "available hardware, robustness dimensions, model/training-mode requirements, preprocessing, "
            "and VQA question requirements. Deployment hardware is not training execution hardware.\n\n"
            f"REQUEST: {context.get('user_query')}"
        ),
    )
    normalized = _normalize_interpretation(
        interpreted.model_dump(mode="json"), str(context.get("user_query") or "")
    )
    context = _merge_interpretation(context, normalized)
    for change in _revision_changes(context, "task-interpretation", "required"):
        if change.get("operation") == "set":
            context[change["field"]] = change.get("value")
    context["use_graphrag"] = request.use_graphrag
    _history(context, "Task Interpretation Completed (lightweight planner)")
    _checkpoint(request.job_id, "STATE_01_INTERPRETATION.json", context)
    return {"context": context}


@router.post("/check-data")
async def check_data(request: StateRequest):
    context = dict(request.context)
    if not context.get("task"):
        raise HTTPException(status_code=422, detail="Task interpretation is required first.")
    query_path = planning_dir(request.job_id) / "DATA_CHECK_QUERY.sparql"
    try:
        context["available_data"] = query_class_availability(
            context.get("classes", []), query_output_path=query_path
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail={"message": "VisionKG SPARQL availability query failed.", "reason": str(exc)},
        ) from exc
    context["data_availability_note"] = (
        "Counts were queried from VisionKG metadata. No images or annotations were downloaded."
    )
    _history(context, "Data Availability Checked (VisionKG metadata only)")
    _checkpoint(request.job_id, "STATE_02_DATA_CHECK.json", context)
    return {"context": context}


@router.post("/select-model")
async def select_model(request: StateRequest):
    context = dict(request.context)
    task = str(context.get("task") or "")
    graph_context = get_ontology().model_context(context) if request.use_graphrag else None
    graph_candidates = (graph_context or {}).get("candidate_models") or []
    candidates = [
        {
            "id": item["id"],
            "display_name": item.get("model_name") or item["id"],
            "family": item.get("model_family") or "unknown",
            "ontology_evidence": item,
        }
        for item in graph_candidates
    ] if graph_candidates else MODEL_CATALOG.get(task)
    if not candidates:
        raise HTTPException(status_code=422, detail=f"No lightweight model catalog for task: {task}")
    plan = await structured_call(
        job_id=request.job_id,
        operation="model_selection",
        response_model=ModelPlan,
        prompt=(
            "Select exactly one model from CANDIDATES. The model_id and displayed metadata must "
            "match a candidate. Treat runtime and accuracy as estimates, not measurements.\n\n"
            f"CONTEXT: {json.dumps(context, default=str)}\nCANDIDATES: {json.dumps(candidates)}\n"
            f"GRAPH CONTEXT: {json.dumps(graph_context, default=str)}"
        ),
    )
    resolved = get_registry().resolve_model(plan.model_id, task)
    required_models = [change.get("value") for change in _revision_changes(context, "model-selection", "required") if change.get("operation") == "set"]
    if required_models:
        required = get_registry().resolve_model(str(required_models[-1]), task)
        if required is None:
            raise HTTPException(status_code=422, detail=f"Required model is not a supported {task} model: {required_models[-1]}")
        resolved = required
    selected = next((item for item in candidates if item["id"] == (resolved.id if resolved else plan.model_id)), None)
    if selected is None:
        raise HTTPException(status_code=502, detail="Planning model selected an unknown model ID.")
    context["selected_model_info"] = {**selected, "rationale": plan.rationale}
    if graph_context:
        context["model_selection_graph_context"] = graph_context
    evidence = decision_evidence(
        decision_type="model_selection", selected_id=selected["id"], rationale=plan.rationale,
        candidates=candidates, graph_context=graph_context, uncertainties=plan.uncertainties,
        decision=context["selected_model_info"],
    )
    context["model_selection_decision_evidence"] = evidence
    _history(context, f"Model Selection Rationale: {plan.rationale}")
    _history(context, "Model Selection Completed (lightweight planner)")
    _checkpoint(request.job_id, "STATE_03_MODEL_SELECTION.json", context)
    return {"context": context, "decision_evidence": evidence}


@router.post("/select-datasets")
async def select_datasets(request: StateRequest):
    context = dict(request.context)
    if not context.get("classes"):
        raise HTTPException(status_code=422, detail="Classes are required for dataset planning.")
    graph_context = get_ontology().dataset_context(context) if request.use_graphrag else None
    live_candidates = availability_candidates(context)
    excluded = {
        str(value) for change in _revision_changes(context, "dataset-selection", "required")
        if change.get("operation") == "exclude"
        for value in (change.get("value") if isinstance(change.get("value"), list) else [change.get("value")])
    }
    live_candidates = [item for item in live_candidates if item["dataset_id"] not in excluded]
    planning_candidates = live_candidates or (graph_context or {}).get("candidate_datasets") or []
    plan = await structured_call(
        job_id=request.job_id,
        operation="dataset_selection",
        response_model=DatasetPlan,
        prompt=(
            "Select dataset sources for this request. Use only exact dataset_id values from "
            "CANDIDATES. VisionKG counts are availability evidence, not proof of unique images.\n\n"
            f"CONTEXT: {json.dumps(context, default=str)}\n"
            f"CANDIDATES: {json.dumps(planning_candidates, default=str)}\n"
            f"ONTOLOGY CONTEXT: {json.dumps(graph_context, default=str)}"
        ),
    )
    if planning_candidates:
        candidates = planning_candidates
        references: dict[str, set[str]] = {}
        for item in candidates:
            for reference in (item["dataset_id"], item.get("display_name") or ""):
                references.setdefault(_canonical_reference(reference), set()).add(item["dataset_id"])
        live_ids = {item["dataset_id"] for item in live_candidates}
        for item in (graph_context or {}).get("candidate_datasets", []):
            if item.get("dataset_id") in live_ids:
                for reference in (item["dataset_id"], item.get("display_name") or ""):
                    references.setdefault(_canonical_reference(reference), set()).add(item["dataset_id"])
        normalized_sources = []
        unknown = []
        for source in plan.sources:
            matches = references.get(_canonical_reference(source.dataset_name), set())
            if len(matches) != 1:
                unknown.append(source.dataset_name)
                continue
            normalized_sources.append(source.model_copy(update={"dataset_name": next(iter(matches))}))
        if unknown:
            raise HTTPException(
                status_code=502,
                detail={"message": "Planning model selected datasets outside the GraphRAG candidates.", "dataset_ids": unknown},
            )
        plan = plan.model_copy(update={"sources": normalized_sources})
    selected_sources = [item.model_dump(mode="json") for item in plan.sources]
    if live_candidates:
        strategy = build_data_strategy(context)
        conflicts = build_data_plan_conflicts(context)
        context["data_strategy"] = strategy
        context["data_plan_constraints"] = {
            "minimum_unique_pool_images": strategy["minimum_unique_pool_images"],
            "preferred_unique_pool_images": strategy["preferred_unique_pool_images"],
            "conflicts": conflicts,
        }
        required_includes = {
            str(value) for change in _revision_changes(context, "dataset-selection", "required")
            if change.get("operation") == "include"
            for value in (change.get("value") if isinstance(change.get("value"), list) else [change.get("value")])
        }
        unavailable = required_includes - {item["dataset_id"] for item in live_candidates}
        if unavailable:
            raise HTTPException(status_code=422, detail={
                "message": "Required revision datasets are not available for this plan.",
                "dataset_ids": sorted(unavailable),
            })
        try:
            assignments, profile = build_split_assignments(
                context, [source.dataset_name for source in plan.sources] + sorted(required_includes)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        context["selected_data"] = assignments
        context["dataset_profile"] = profile
        preprocessing = preprocessing_plan(context)
        context["preprocessing_plan"] = preprocessing
        context["preprocessing"] = json.dumps(preprocessing, separators=(",", ":"))
    else:
        context["selected_data"] = selected_sources
        context["dataset_profile"] = {
            "status": "conceptual_no_visionkg_availability",
            "total_selected_images": 0,
            "number_of_sources": len(selected_sources),
            "limitations": ["Run check-data before dataset planning to produce split assignments."],
        }
    if graph_context:
        context["dataset_selection_graph_context"] = graph_context
    evidence = decision_evidence(
        decision_type="dataset_selection", selected_id=selected_sources[0]["dataset_name"] if selected_sources else None,
        rationale=plan.rationale, candidates=planning_candidates, graph_context=graph_context,
        uncertainties=plan.uncertainties, decision=context.get("selected_data") or selected_sources,
    )
    context["dataset_selection_decision_evidence"] = evidence
    _history(context, f"Data Selection Rationale: {plan.rationale}")
    _history(context, "Dataset Selection Completed (conceptual only)")
    _checkpoint(request.job_id, "STATE_04_DATASET_SELECTION.json", context)
    if live_candidates:
        write_json(planning_dir(request.job_id) / "STATE_04_PREPROCESSING.json", {
            "preprocessing": context["preprocessing_plan"],
            "selected_data": context["selected_data"],
            "data_strategy": context["data_strategy"],
        })
    return {"context": context, "decision_evidence": evidence}


@router.post("/choose-hyperparameters")
async def choose_hyperparameters(request: StateRequest):
    context = dict(request.context)
    selected = context.get("selected_model_info") or {}
    if not selected.get("id"):
        raise HTTPException(status_code=422, detail="Model selection is required first.")
    graph_context = get_ontology().recipe_context(context) if request.use_graphrag else None
    recipes = (graph_context or {}).get("candidate_recipes") or []
    selected_recipe = recipes[0] if recipes else None
    plan = await structured_call(
        job_id=request.job_id,
        operation="hyperparameter_planning",
        response_model=HyperparameterPlan,
        prompt=(
            "Create a conservative illustrative fine-tuning configuration. model_name must equal "
            "the selected model id. The configuration will be displayed but cannot be executed by "
            "this service.\n\n"
            f"CONTEXT: {json.dumps(context, default=str)}\n"
            f"ONTOLOGY RECIPES: {json.dumps(graph_context, default=str)}\n"
            "When an ontology recipe is supplied, keep every numeric value within its documented bounds."
        ),
    )
    if plan.model_name != selected["id"]:
        raise HTTPException(status_code=502, detail="Hyperparameter plan changed the selected model.")
    core = plan.model_dump(mode="json")
    try:
        config, field_provenance = materialize_hpo(context, core, selected_recipe)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": "A complete split-aware dataset plan is required before HPO.", "reason": str(exc)},
        ) from exc
    for change in _revision_changes(context, "choose-hyperparameters", "required"):
        if change.get("operation") == "set":
            field = str(change["field"]).removeprefix("hpo_config.")
            config[field] = change.get("value")
            field_provenance[field] = {"source": "required_revision", "change_id": change.get("id")}
    initial_findings = evaluate_hpo(config, context, selected_recipe)
    repaired_fields: list[str] = []
    if any(item["severity"] == "hard_error" for item in initial_findings):
        config, repaired_fields = repair_hpo(config, initial_findings)
        for field in repaired_fields:
            field_provenance[field] = {"source": "evaluator_repair", "recipe_id": (selected_recipe or {}).get("id")}
    findings = evaluate_hpo(config, context, selected_recipe)
    decision = decision_from_findings(findings, repaired_fields)
    if not decision["accept"]:
        raise HTTPException(status_code=422, detail={
            "message": "No acceptable hyperparameter proposal was found after one repair round.",
            "decision": decision, "last_candidate": config, "initial_findings": initial_findings,
        })
    if selected_recipe:
        context["hyperparameter_graph_context"] = graph_context
    context["hpo_config"] = config
    context["hpo_decision"] = decision
    evidence = decision_evidence(
        decision_type="hyperparameter_selection", selected_id=selected_recipe.get("id") if selected_recipe else selected["id"],
        rationale=plan.rationale, candidates=recipes or [{"id": selected["id"]}],
        graph_context=graph_context, uncertainties=plan.uncertainties,
        decision=config, field_provenance=field_provenance,
    )
    context["hyperparameter_decision_evidence"] = evidence
    _history(context, f"Hyperparameter Rationale: {plan.rationale}")
    _history(context, "Hyperparameter Planning Completed (not executed)")
    directory = planning_dir(request.job_id)
    _checkpoint(request.job_id, "STATE_05_HYPERPARAMETERS.json", context)
    write_json(directory / "RESULT_HYPERPARAMETERS.json", config)
    write_json(directory / "HYPERPARAMETER_PROPOSAL.json", {
        "candidate": config,
        "decision": decision,
        "field_provenance": field_provenance,
        "decision_evidence": evidence,
    })
    return {
        "context": context,
        "candidate": config,
        "decision": decision,
        "field_provenance": field_provenance,
        "decision_evidence": evidence,
    }


@router.post("/plan-revision")
async def plan_revision(request: PlanRevisionRequest):
    if not request.required_changes.strip() and not request.preferences.strip():
        raise HTTPException(status_code=422, detail="Enter a required change or preference.")
    plan = await structured_call(
        job_id=request.job_id,
        operation="planning_revision",
        response_model=RevisionPlan,
        prompt=(
            "Convert the requested revision into atomic changes. Valid targets are task-interpretation, "
            "model-selection, dataset-selection, and choose-hyperparameters. Use model_name, "
            "dataset.include/dataset.exclude, or hpo_config.<field>. Never invent a value. "
            "restart_from must be the earliest affected step.\n\n"
            f"CURRENT: {json.dumps(request.context, default=str)}\n"
            f"REQUIRED: {request.required_changes or '(none)'}\n"
            f"PREFERENCES: {request.preferences or '(none)'}\n"
            f"REQUESTED TARGET: {request.requested_target}"
        ),
    )
    plan = plan.model_copy(update={
        "required_text": request.required_changes.strip(),
        "preferred_text": request.preferences.strip(),
        "restart_from": min((change.target_step for change in plan.changes), key=STEP_ORDER.index),
    })
    _validate_revision(plan, request.context)
    return {"plan": plan.model_dump(mode="json")}


@router.post("/activate-revision")
def activate_revision(request: ActivateRevisionRequest):
    _validate_revision(request.plan, request.context)
    directory = planning_dir(request.job_id)
    run = directory.parent.parent
    if any((run / relative).exists() for relative in (
        "artifacts/download_report.json", "data/dataset_manifest.json", "progress.json",
        "artifacts/best_model.pt", "artifacts/best_model.pth",
    )):
        raise HTTPException(status_code=409, detail="Planning revisions are unavailable after execution artifacts exist.")
    predecessor = PREDECESSOR[request.plan.restart_from]
    if predecessor:
        restored = read_json(directory / predecessor)
        if not isinstance(restored, dict):
            raise HTTPException(status_code=409, detail=f"Required checkpoint {predecessor} is missing.")
        context = restored
    else:
        context = dict(request.context)
    revision_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    archive = directory / "revisions" / revision_id
    archived = []
    for filename in DOWNSTREAM[request.plan.restart_from]:
        source = directory / filename
        if source.is_file():
            archive.mkdir(parents=True, exist_ok=True)
            source.replace(archive / filename)
            archived.append(filename)
    prior = (request.context.get("revision") or {}).get("history") or []
    context["revision"] = {"active": request.plan.model_dump(mode="json"), "history": [*prior, request.plan.model_dump(mode="json")]}
    _history(context, f"Planning revision activated: {request.plan.summary}")
    write_json(directory / "STATE_ACTIVE_REVISION.json", context)
    return {"context": context, "revision_id": revision_id, "archived_files": archived}


def _revision_value(context: dict[str, Any], field: str):
    if field == "model_name":
        return (context.get("selected_model_info") or {}).get("id")
    if field.startswith("hpo_config."):
        return (context.get("hpo_config") or {}).get(field.removeprefix("hpo_config."))
    if field.startswith("dataset."):
        return sorted({source.get("dataset_name") for item in context.get("selected_data") or [] for source in item.get("sources", [])})
    current: Any = context
    for part in field.split("."):
        current = current.get(part) if isinstance(current, dict) else None
    return current


@router.post("/verify-revision")
def verify_revision(request: VerifyRevisionRequest):
    active = (request.context.get("revision") or {}).get("active")
    if not active:
        raise HTTPException(status_code=409, detail="No active revision exists.")
    checks = []
    for change in active.get("changes", []):
        actual = _revision_value(request.context, change["field"])
        expected = change.get("value")
        values = expected if isinstance(expected, list) else [expected]
        operation = change.get("operation", "set")
        if operation == "include":
            satisfied = all(value in (actual or []) for value in values)
        elif operation in {"exclude", "avoid"}:
            satisfied = all(value not in actual for value in values) if isinstance(actual, list) else actual not in values
        else:
            satisfied = actual == expected
        checks.append({**change, "expected": expected, "actual": actual, "satisfied": satisfied})
    required_ok = all(item["satisfied"] for item in checks if item.get("strength") == "required")
    return {"satisfied": required_ok, "checks": checks}


@router.post("/fork-revision")
def fork_revision(request: ForkRevisionRequest):
    parent = run_dir(request.parent_job_id)
    assessment = read_json(parent / "artifacts" / "post_training_assessment.json")
    if not isinstance(assessment, dict):
        raise HTTPException(status_code=409, detail="Generate a post-training assessment before forking.")
    if assessment.get("assessment_id") != request.assessment_id:
        raise HTTPException(status_code=409, detail="The assessment ID does not match the stored assessment.")
    parent_planning = parent / "artifacts" / "planning"
    state_path = next((parent_planning / name for name in (
        "STATE_05_HYPERPARAMETERS.json", "STATE_04_DATASET_SELECTION.json",
        "STATE_03_MODEL_SELECTION.json", "STATE_02_DATA_CHECK.json", "STATE_01_INTERPRETATION.json",
    ) if (parent_planning / name).is_file()), None)
    if state_path is None:
        raise HTTPException(status_code=409, detail="No reusable planning state is available.")
    parent_context = read_json(state_path)
    if not isinstance(parent_context, dict):
        raise HTTPException(status_code=409, detail="The latest planning checkpoint is invalid.")
    _validate_revision(request.plan, parent_context)
    reusable = {
        "task-interpretation": [],
        "model-selection": ["STATE_01_INTERPRETATION.json", "STATE_02_DATA_CHECK.json", "DATA_CHECK_QUERY.sparql"],
        "dataset-selection": ["STATE_01_INTERPRETATION.json", "STATE_02_DATA_CHECK.json", "DATA_CHECK_QUERY.sparql", "STATE_03_MODEL_SELECTION.json"],
        "choose-hyperparameters": ["STATE_01_INTERPRETATION.json", "STATE_02_DATA_CHECK.json", "DATA_CHECK_QUERY.sparql", "STATE_03_MODEL_SELECTION.json", "STATE_04_DATASET_SELECTION.json", "STATE_04_PREPROCESSING.json"],
    }[request.plan.restart_from]
    required_json = [name for name in reusable if name.startswith("STATE_") and name != "STATE_04_PREPROCESSING.json"]
    missing = [name for name in required_json if not (parent_planning / name).is_file()]
    if missing:
        raise HTTPException(status_code=409, detail={"message": "Required historical checkpoints are missing.", "files": missing})
    child_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:10]
    child_planning = planning_dir(child_id)
    for filename in reusable:
        source = parent_planning / filename
        if source.is_file():
            shutil.copy2(source, child_planning / filename)
    if required_json:
        context = read_json(child_planning / required_json[-1]) or {}
    else:
        context = dict(parent_context)
    history = (parent_context.get("revision") or {}).get("history") or []
    context["revision"] = {"active": request.plan.model_dump(mode="json"), "history": [*history, request.plan.model_dump(mode="json")]}
    context["hpo_config"] = None
    context["hpo_decision"] = None
    if request.plan.restart_from == "task-interpretation":
        for field in (
            "available_data", "selected_data", "selected_model_info", "dataset_profile",
            "model_selection_graph_context", "dataset_selection_graph_context",
            "hyperparameter_graph_context", "model_selection_decision_evidence",
            "dataset_selection_decision_evidence", "hyperparameter_decision_evidence",
            "data_strategy", "data_plan_constraints", "preprocessing_plan",
        ):
            context.pop(field, None)
    _history(context, f"Forked from {request.parent_job_id}: {request.plan.summary}")
    _checkpoint(child_id, "STATE_ACTIVE_REVISION.json", context)
    parent_lineage = read_json(parent / "lineage.json") or {}
    lineage = {
        "job_id": child_id,
        "parent_job_id": request.parent_job_id,
        "root_job_id": parent_lineage.get("root_job_id", request.parent_job_id),
        "assessment_id": request.assessment_id,
        "restart_from": request.plan.restart_from,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir(child_id) / "lineage.json", lineage)
    reused_steps = [step for step in ("task-interpretation", "check-data", "model-selection", "dataset-selection") if {
        "task-interpretation": "STATE_01_INTERPRETATION.json",
        "check-data": "STATE_02_DATA_CHECK.json",
        "model-selection": "STATE_03_MODEL_SELECTION.json",
        "dataset-selection": "STATE_04_DATASET_SELECTION.json",
    }[step] in reusable]
    return {
        "job_id": child_id, "parent_job_id": request.parent_job_id,
        "context": context, "plan": request.plan.model_dump(mode="json"),
        "restart_from": request.plan.restart_from, "reused_steps": reused_steps,
        "lineage": lineage,
    }
