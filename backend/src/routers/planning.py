import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Core agent/pipeline imports
from agents import Runner
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.agents.interpretation_agents import interpretation_loop, task_interpretation_agent, synonym_check_agent
from cvmodellearning.agents.model_selection_agents import classification_model_selector_agent, detection_model_selector_agent, vqq_model_selector_agent
from cvmodellearning.agents.hyperparameter_agents import generate_and_evaluate_hpo
from cvmodellearning.agents.data_selection_and_augmentation_agents import (
    detection_data_preprocessing_agent, classification_data_preprocessing_agent, 
    classification_dataset_selection_agent, detection_dataset_selection_agent, 
    vqa_dataset_selection_agent, vqa_data_preprocessing_agent
)
from cvmodellearning.agents.agents_utils import save_json, log_planning_step, load_unified_dataset_classes
from cvmodellearning.paths import planning_artifacts_dir
from cvmodellearning.download.visionkg_utils import get_multi_class_stats

router = APIRouter(
    prefix="/planning",
    tags=["1 - Planning & Interpretation"],
)

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
            return PipelineState(**json.loads(request_context))
        except:
            return PipelineState(user_query=request_context)
    return PipelineState(**request_context)

def save_checkpoint(state: PipelineState, job_id: str, filename: str):
    state.last_updated = datetime.now().isoformat()
    save_json(state.model_dump(), planning_artifacts_dir(job_id), filename)


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
    res = await Runner.run(task_interpretation_agent, input=state.model_dump_json())
    extracted = res.final_output 
    
    valid_classes = load_unified_dataset_classes()
    final_classes = []
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
        ][:10]
        if syn_res.final_output.found_match and matched_classes:
            final_classes.extend(matched_classes)
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

    state = PipelineState(**{**state.model_dump(), **extracted_patch})
    state.classes = final_classes
    state.step_history.append("Task Interpretation Completed")
    
    save_checkpoint(state, request.job_id, "STATE_01_INTERPRETATION.json")
    return {"context": state.model_dump()}

@router.post("/check-data")
async def check_data(request: StateRequest):
    state = get_state(request.context)
    
    if state.classes:
        raw_stats = get_multi_class_stats(state.classes)
        
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
            
        state.available_data = formatted_stats
        
    state.step_history.append("Data Availability Checked")
    save_checkpoint(state, request.job_id, "STATE_02_DATA_CHECK.json")
    
    return {"context": state.model_dump()}

@router.post("/select-model")
async def select_model(request: StateRequest):
    state = get_state(request.context)
    
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
        model_rationale = model_patch_dict.pop("rationale", "No rationale provided.")
        state.selected_model_info = model_patch_dict
        
        state.step_history.append(f"Model Selection Rationale: {model_rationale}")
        state.step_history.append("Model Selection Completed")
    
    save_checkpoint(state, request.job_id, "STATE_03_MODEL_SELECTION.json")
    return {"context": state.model_dump()}

@router.post("/preprocess")
async def preprocess_step(request: StateRequest):
    state = get_state(request.context)
    
    agent_map = {
        "classification": (classification_dataset_selection_agent, classification_data_preprocessing_agent),
        "detection": (detection_dataset_selection_agent, detection_data_preprocessing_agent),
        "visual question answering": (vqa_dataset_selection_agent, vqa_data_preprocessing_agent)
    }
    
    agents = agent_map.get(state.task)
    if agents:
        sel_agent, prep_agent = agents
        state_json = state.model_dump_json()
        
        # Run both agents CONCURRENTLY to save massive amounts of time
        res_sel_task = Runner.run(sel_agent, input=state_json)
        res_prep_task = Runner.run(prep_agent, input=state_json)
        
        res_sel, res_prep = await asyncio.gather(res_sel_task, res_prep_task)
        
        # Merge Data Selection
        state.selected_data = res_sel.final_output.selected_data
        
        # Merge Preprocessing (Pop rationale to prevent floating JSON keys)
        prep_dict = res_prep.final_output.model_dump(exclude_unset=True)
        prep_rationale = prep_dict.pop("rationale", "No rationale provided.")
        state = state.model_copy(update=prep_dict)
        
        state.step_history.append(f"Data Selection Rationale: {res_sel.final_output.rationale}")
        state.step_history.append(f"Preprocessing Rationale: {prep_rationale}")
        state.step_history.append("Data Selection & Preprocessing Finalized")

    save_checkpoint(state, request.job_id, "STATE_04_PREPROCESSING.json")
    return {"context": state.model_dump()}

@router.post("/choose-hyperparameters")
async def choose_hyperparameters(request: StateRequest):
    state = get_state(request.context)
    
    candidate, decision = await generate_and_evaluate_hpo(state.model_dump_json(), job_id=request.job_id)
    
    state.hpo_config = candidate.model_dump() if candidate else None
    state.hpo_decision = decision.model_dump() if decision else None
    state.step_history.append("Hyperparameter Optimization Completed")
    
    save_checkpoint(state, request.job_id, "STATE_05_HYPERPARAMETERS.json")
    return {"context": state.model_dump()}

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
