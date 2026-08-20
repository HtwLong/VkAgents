from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from ..graphrag.ontology import get_ontology
from ..llm import structured_call
from ..schemas import (
    CompletenessDecision,
    CompletenessRequest,
    DatasetPlan,
    HyperparameterPlan,
    ModelPlan,
    StateRequest,
    TaskInterpretation,
)
from ..store import planning_dir, write_json


router = APIRouter(prefix="/planning", tags=["Planning"])

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
    write_json(planning_dir(job_id) / filename, context)


def _evidence(title: str, rationale: str, candidates: list[dict] | None = None, uncertainties: list[str] | None = None) -> dict:
    return {
        "title": title,
        "summary": rationale,
        "rationale": rationale,
        "evaluated_candidates": candidates or [],
        "uncertainties": uncertainties or [],
        "evidence": [],
    }


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
            "Interpret the computer-vision request. Use 'visual question answering' exactly for "
            "VQA. Extract only classes explicitly requested or directly entailed.\n\n"
            f"REQUEST: {context.get('user_query')}"
        ),
    )
    context.update(interpreted.model_dump(mode="json"))
    context["use_graphrag"] = request.use_graphrag
    _history(context, "Task Interpretation Completed (lightweight planner)")
    _checkpoint(request.job_id, "STATE_01_INTERPRETATION.json", context)
    return {"context": context}


@router.post("/check-data")
async def check_data(request: StateRequest):
    context = dict(request.context)
    if not context.get("task"):
        raise HTTPException(status_code=422, detail="Task interpretation is required first.")
    # The hosted planner deliberately performs no dataset queries or downloads. It records
    # candidates as unverified so later planning cannot mistake estimates for measured data.
    context["available_data"] = [
        {"class_name": name, "sources": [], "availability": "not_queried"}
        for name in context.get("classes", [])
    ]
    context["data_availability_note"] = (
        "Dataset availability was not queried in viewer mode; all source counts remain unverified."
    )
    _history(context, "Data Availability Check Skipped (viewer mode)")
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
    selected = next((item for item in candidates if item["id"] == plan.model_id), None)
    if selected is None:
        raise HTTPException(status_code=502, detail="Planning model selected an unknown model ID.")
    context["selected_model_info"] = {**selected, "rationale": plan.rationale}
    if graph_context:
        context["model_selection_graph_context"] = graph_context
    evidence = _evidence("Model selection", plan.rationale, candidates, plan.uncertainties)
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
    plan = await structured_call(
        job_id=request.job_id,
        operation="dataset_selection",
        response_model=DatasetPlan,
        prompt=(
            "Propose public dataset sources for this request. This is a conceptual plan only: "
            "do not invent image counts or claim availability/download verification.\n\n"
            f"CONTEXT: {json.dumps(context, default=str)}\n"
            f"GRAPH CANDIDATES: {json.dumps(graph_context, default=str)}\n"
            "When graph candidates are supplied, use only their dataset_id values."
        ),
    )
    if graph_context:
        allowed = {item["dataset_id"] for item in graph_context["candidate_datasets"]}
        unknown = sorted({item.dataset_name for item in plan.sources} - allowed)
        if unknown:
            raise HTTPException(
                status_code=502,
                detail={"message": "Planning model selected datasets outside the GraphRAG candidates.", "dataset_ids": unknown},
            )
    selected = [item.model_dump(mode="json") for item in plan.sources]
    context["selected_data"] = selected
    if graph_context:
        context["dataset_selection_graph_context"] = graph_context
    context["dataset_profile"] = {
        "status": "planned_not_materialized",
        "total_selected_images": 0,
        "number_of_sources": len(selected),
        "classes": context.get("classes", []),
    }
    evidence = _evidence("Dataset selection", plan.rationale, selected, plan.uncertainties)
    context["dataset_selection_decision_evidence"] = evidence
    _history(context, f"Data Selection Rationale: {plan.rationale}")
    _history(context, "Dataset Selection Completed (conceptual only)")
    _checkpoint(request.job_id, "STATE_04_DATASET_SELECTION.json", context)
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
    config = plan.model_dump(mode="json", exclude={"rationale", "uncertainties"})
    if selected_recipe:
        validation_errors = get_ontology().validate_hyperparameters(config, selected_recipe)
        if validation_errors:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "The generated hyperparameters violate ontology recipe bounds.",
                    "recipe_id": selected_recipe.get("id"),
                    "errors": validation_errors,
                },
            )
        config["ontology_recipe_id"] = selected_recipe.get("id")
        context["hyperparameter_graph_context"] = graph_context
    context["hpo_config"] = config
    evidence = _evidence("Hyperparameter planning", plan.rationale, [config], plan.uncertainties)
    context["hyperparameter_decision_evidence"] = evidence
    _history(context, f"Hyperparameter Rationale: {plan.rationale}")
    _history(context, "Hyperparameter Planning Completed (not executed)")
    directory = planning_dir(request.job_id)
    _checkpoint(request.job_id, "STATE_05_HYPERPARAMETERS.json", context)
    write_json(directory / "RESULT_HYPERPARAMETERS.json", {"hyperparameter_candidate": config})
    return {"context": context, "decision_evidence": evidence}
