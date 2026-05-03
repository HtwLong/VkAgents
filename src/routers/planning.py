import json
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Core agent/pipeline imports
from agents import Runner
from cvmodellearning.agents.interpretation_agents_api import interpretation_loop, task_interpretation_agent, synonym_check_agent
from cvmodellearning.agents.model_agents import classification_model_selector_agent, detection_model_selector_agent, vqq_model_selector_agent
from cvmodellearning.agents.hyperparameter_agents_api import generate_and_evaluate_hpo
from cvmodellearning.agents.agents_utils import save_json
from cvmodellearning.paths import planning_artifacts_dir
from cvmodellearning.agents.preprocessing_agents import detection_data_preprocessing_agent, classification_data_preprocessing_agent, classification_dataset_selection_agent, detection_dataset_selection_agent, vqa_dataset_selection_agent, vqa_data_preprocessing_agent
from cvmodellearning.agents.agents_utils import log_planning_step, load_unified_dataset_classes
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

class TaskInterpretRequest(BaseModel):
    context: str
    job_id: str

class TaskInterpretResponse(BaseModel):
    context: Optional[Dict[str, Any]]
    
class CheckDataRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    job_id: str

class CheckDataResponse(BaseModel):
    context: Dict[str, Any]

class ModelSelectRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    job_id: str

class ModelSelectResponse(BaseModel):
    context: Dict[str, Any]

class PreprocessRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    job_id: str

class PreprocessResponse(BaseModel):
    context: Dict[str, Any]
    
class ChooseHPRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    job_id: str

class ChooseHPResponse(BaseModel):
    candidate: Dict[str, Any]
    decision: Dict[str, Any]

class AddUserRequest(BaseModel):
    context: Union[str, Dict[str, Any]]
    request_text: Optional[str] = None
    job_id: Optional[str] = None

class RequestAddedResponse(BaseModel):
    context: Dict[str, Any]


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

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

@router.post("/task-interpret", response_model=TaskInterpretResponse)
async def task_interpret(request: TaskInterpretRequest):
    # 1. Run the initial interpretation agent
    res = await Runner.run(task_interpretation_agent, input=request.context)
    out_obj = res.final_output # This is an instance of InterpretationRequirements
    
    # 2. Validation & Synonym Checking Logic
    valid_classes = load_unified_dataset_classes()
    
    final_class_list = []
    replacements_log = []
    
    # Ensure classes list exists
    current_classes = out_obj.classes if out_obj.classes else []

    # If valid_classes is huge, this string might need truncation or retrieval logic.
    # Assuming it fits in context for now:
    valid_classes_str = ", ".join(sorted(list(valid_classes)))

    for cls in current_classes:
        cls_clean = cls.strip().lower()
        
        # Direct check
        if cls_clean in valid_classes:
            final_class_list.append(cls) # Keep original casing (or use cls_clean)
            continue
            
        # If not found, ask the Synonym Agent
        synonym_input = f"User Class: '{cls}'.\nAllowed Dataset Classes: [{valid_classes_str}]"
        
        syn_res = await Runner.run(synonym_check_agent, input=synonym_input)
        syn_decision = syn_res.final_output
        
        if syn_decision.found_match and syn_decision.dataset_class:
            # We found a valid replacement
            final_class_list.append(syn_decision.dataset_class)
            replacements_log.append(f"Replaced '{cls}' with '{syn_decision.dataset_class}' (Reason: {syn_decision.reason})")
        else:
            # No match found - Reject
            raise HTTPException(
                status_code=400,
                detail=f"Class '{cls}' does not exist in the dataset and no valid synonym was found."
            )

    # 3. Update the output object with the validated list
    out_obj.classes = final_class_list
    out = out_obj.model_dump()

    save_json(out, planning_artifacts_dir(request.job_id), "RESULT_INTERPRETATION.json")

    # 4. Construct log message
    log_msg = "Extracted structured task information from user prompt."
    if replacements_log:
        log_msg += "\n\nClass Validations:\n" + "\n".join(replacements_log)

    log_planning_step(
        request.job_id, 
        "Task Interpretation", 
        request.context, 
        log_msg, 
        out
    )
    return TaskInterpretResponse(context=out)

@router.post("/check-data", response_model=CheckDataResponse)
async def check_data(request: CheckDataRequest):
    """
    Checks VisionKG for the availability and distribution of images for the requested classes.
    Updates the 'available_data' field in the context.
    """
    # Load context
    ctx = request.context if isinstance(request.context, dict) else json.loads(request.context)
    
    # Extract classes list
    classes = ctx.get("classes", [])
    
    if not classes:
        print("Warning: No classes found in context to check data for.")
        available_data = {}
    else:
        # Run SPARQL query via optimized utility function
        try:
            available_data = get_multi_class_stats(classes)
        except Exception as e:
            print(f"Error fetching data stats: {e}")
            # We might want to raise HTTPException or just return empty depending on strictness
            # For now, we return empty stats so the pipeline doesn't crash, but log the error
            available_data = {}

    # Update context
    ctx["available_data"] = available_data
    
    # Save Artifact
    save_json(ctx, planning_artifacts_dir(request.job_id), "RESULT_DATA_CHECK.json")

    # Log Planning Step
    log_planning_step(
        request.job_id,
        "Data Availability Check",
        f"Checking data availability for classes: {classes}",
        f"Retrieved image counts for {len(available_data)} classes from VisionKG.",
        {"available_data": available_data}
    )

    return CheckDataResponse(context=ctx)

@router.post("/select-model", response_model=ModelSelectResponse)
async def select_model(request: ModelSelectRequest):
    ctx_obj = request.context if isinstance(request.context, dict) else json.loads(request.context)
    ctx = request.context if isinstance(request.context, str) else json.dumps(request.context)

    model_selector_agent = None
    task_value = ctx_obj.get('task')
    
    if task_value == "classification":
        model_selector_agent = classification_model_selector_agent
    elif task_value == "detection":
        model_selector_agent = detection_model_selector_agent
    elif task_value == "segmentation":
        pass
    elif task_value == "visual question answering":
        model_selector_agent = vqq_model_selector_agent
    else:
        pass
    if model_selector_agent:
        res = await Runner.run(model_selector_agent, input=ctx)
        model_info_obj = res.final_output
        model_info = model_info_obj.model_dump()
        
        log_planning_step(
            request.job_id,
            "Model Selection",
            ctx_obj.get("description", "No description"),
            getattr(model_info_obj, "rationale", "No rationale."),
            model_info
        )
        
        save_json(model_info, planning_artifacts_dir(request.job_id), "RESULT_MODEL.json")
        return ModelSelectResponse(context=model_info)
    
    return ModelSelectResponse(context=ctx_obj)

@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_step(request: PreprocessRequest):
    """
    Runs a two-step data preparation pipeline:
    1. Dataset Selection: Decides which datasets and how many images to use (fills 'selected_data').
    2. Preprocessing Strategy: Decides on augmentation and resizing (fills 'augmentation' & 'preprocessing').
    """
    # Load context
    ctx_obj = request.context if isinstance(request.context, dict) else json.loads(request.context)
    ctx_str = request.context if isinstance(request.context, str) else json.dumps(request.context)
    
    task_value = ctx_obj.get('task')
    
    # Identify the correct agent pair based on task
    dataset_agent = None
    preprocessing_agent = None
    
    if task_value == "classification":
        dataset_agent = classification_dataset_selection_agent
        preprocessing_agent = classification_data_preprocessing_agent
    elif task_value == "detection":
        dataset_agent = detection_dataset_selection_agent
        preprocessing_agent = detection_data_preprocessing_agent
    elif task_value == "segmentation":
        # Add segmentation agents here when ready
        pass
    elif task_value == "visual question answering":
        dataset_agent = vqa_dataset_selection_agent
        preprocessing_agent = vqa_data_preprocessing_agent
    
    # If we have valid agents for this task, run the chain
    if dataset_agent and preprocessing_agent:
        
        # --- STEP 1: Dataset Selection ---
        print(f"Running Dataset Selection for {task_value}...")
        res_select = await Runner.run(dataset_agent, input=ctx_str)
        select_output_obj = res_select.final_output
        select_output_dict = select_output_obj.model_dump()

        # Log Rationale for Dataset Selection
        log_planning_step(
            request.job_id,
            "Dataset Selection",
            f"Selecting subset from available data for {task_value}",
            getattr(select_output_obj, "rationale", "No rationale provided."),
            {"selected_data": select_output_dict.get("selected_data")}
        )

        # --- STEP 2: Preprocessing & Augmentation ---
        # We pass the OUTPUT of Step 1 (which now has selected_data) as the INPUT to Step 2
        # This ensures the preprocessor sees which data was actually selected.
        step2_input_str = json.dumps(select_output_dict)
        
        print(f"Running Preprocessing Strategy for {task_value}...")
        res_prep = await Runner.run(preprocessing_agent, input=step2_input_str)
        final_output_obj = res_prep.final_output
        final_output_dict = final_output_obj.model_dump()
        
        # Log Rationale for Preprocessing
        log_planning_step(
            request.job_id,
            "Preprocessing Strategy",
            f"Determining augmentation for {task_value}",
            getattr(final_output_obj, "rationale", "No rationale provided."),
            {
                "augmentation": final_output_dict.get("augmentation"), 
                "preprocessing": final_output_dict.get("preprocessing")
            }
        )

        # Save final result
        save_json(final_output_dict, planning_artifacts_dir(request.job_id), "RESULT_PREPROCESSING.json")
        return PreprocessResponse(context=final_output_dict)
        
    # Fallback if task is not supported
    return PreprocessResponse(context=ctx_obj)


@router.post("/choose-hyperparameters", response_model=ChooseHPResponse)
async def choose_hyperparameters(request: ChooseHPRequest):
    ctx = request.context if isinstance(request.context, str) else json.dumps(request.context)
    
    candidate, decision = await generate_and_evaluate_hpo(ctx, job_id=request.job_id)

    combined_output = {
        "hyperparameter_candidate": candidate.model_dump(),
        "decision_summary": decision.model_dump()
    }
    save_json(combined_output, planning_artifacts_dir(request.job_id), "RESULT_HYPERPARAMETERS.json")

    return ChooseHPResponse(
        candidate=candidate.model_dump(), decision=decision.model_dump()
    )

@router.post("/add-user-request", response_model=RequestAddedResponse)
async def add_user_request(request: AddUserRequest):
    ctx = request.context if isinstance(request.context, dict) else json.loads(request.context)
    text = (request.request_text or "").strip()

    if "user_change_requests" in ctx and text:
        existing = ctx.get("user_change_requests", [])
        if not isinstance(existing, list): existing = [str(existing)]
        ctx["user_change_requests"] = existing + [text + "\n"]

    if request.job_id and text:
         log_planning_step(
            request.job_id,
            "User Change Request",
            "User Input",
            f"User explicitly requested: '{text}'",
            "Context updated."
        )

    return RequestAddedResponse(context=ctx)