import json
from agents import Agent, Runner
from cvmodellearning.schemas.decision_schema import Decision
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel  
from cvmodellearning.agents.agents_utils import log_planning_step


hyperparameter_revision_agent = Agent(
    name="Hyperparameter Revision Elicitor",
    instructions=(
        "Given the CURRENT training config JSON and the user's free-form change request, "
        "produce a concise list of targeted 'suggestions' that can be appended to the constraints "
        "for hyperparameter search. Summarize the intent in 'reason'."
        "Return only a pydantic Decision."
    ),
    output_type=Decision,
    model="gpt-4o-mini",
)


classification_hyperparameter_suggest_agent = Agent(
    name="Hyperparameters Suggester (Classification)",
    instructions=(
        "Suggest a combination of hyperparameters for the given model, task and data information "
        "and suggestions if available. "
        "Fill the 'rationale' field explaining concisely your suggestion choices."
        "CRITICAL OPTIMIZER GUIDELINES: "
        "1. For 'adamw' or 'rmsprop', set 'eps' (epsilon) to a very small value (e.g., 1e-8). "
        "2. Never set 'eps' > 1e-3 unless explicitly requested. A value of 1.0 is invalid for standard training. "
        "3. Ensure learning_rate is appropriate for the chosen optimizer (e.g., 1e-3 to 1e-4 for AdamW)."
    ),
    model="gpt-4o-mini",
    output_type=ClassificationConfigModel
)

detection_hyperparameter_suggest_agent = Agent(
    name="Hyperparameters Suggester (Detection)",
    instructions=(
        "Suggest a combination of hyperparameters for the given model, task and data information. "
        "Fill in the selected_data from the input json."
        "Make sure the number of images is enough to train a good enough model with the suggested hyperparameters. "
        "Fill the 'rationale' field explaining concisely your suggestion choices."
        "CRITICAL: If 'task_type' is 'detection', you MUST leave 'loss_mask' as null and 'lambda_mask' as 0.0. "
        "Only set mask parameters if 'task_type' is 'segmentation'. "
        "OPTIMIZER RULES: "
        "1. If using 'adamw', keep 'eps' extremely small (default 1e-8). Do not use large values like 0.1 or 1.0. "
        "2. Adjust 'weight_decay' appropriately (e.g., 0.0005 for detection tasks)."
    ),
    model="gpt-4o-mini",
    output_type=DetectionConfigModel
)


hyperparameter_opt_agent = Agent(
    name="Hyperparameter Optimizer",
    handoff_description="Specialist agent for optimizing hyperparameters",
    instructions=(
        "Predict whether the hyperparameters are likely to return good results. "
        "Sanity check the optimizer settings: "
        "1. Reject the configuration if 'eps' is > 1e-3 for AdamW/RMSprop. Suggest changing it to 1e-8. "
        "2. Check if learning rate is too high (> 0.1) or too low (< 1e-6) without reason. "
        "3. Ensure batch size matches the model size (smaller batch for larger models)."
    ),
    model="gpt-4o-mini",
    output_type=Decision
)


async def choose_hyperparameters_loop(json_data: str, job_id: str, max_rounds: int = 5):
    try:
        data = json.loads(json_data)
        task = data.get("task")
    except (json.JSONDecodeError, AttributeError):
        return None, None

    TASK_AGENT_MAP = {
        "classification": classification_hyperparameter_suggest_agent,
        "detection": detection_hyperparameter_suggest_agent,
    }

    hyperparameter_suggest_agent = TASK_AGENT_MAP.get(task)
    
    constraints = json_data
    last_reason = None
    candidate = None
    
    for r in range(max_rounds):
        round_idx = r + 1
        
        # 1. GENERATE SUGGESTION
        prompt = (
            f"Task/context:\n{constraints}\n"
            f"{'Prior feedback: ' + last_reason if last_reason else ''}\n"
            "Propose one valid hyperparameter configuration and make sure user change requests are considered."
        )
        suggest_res = await Runner.run(hyperparameter_suggest_agent, prompt)
        candidate = suggest_res.final_output
        
        # Extract rationale from the candidate model
        candidate_rationale = getattr(candidate, "rationale", "No rationale provided.")

        # 2. EVALUATE SUGGESTION
        decision_res = await Runner.run(
            hyperparameter_opt_agent,
            f"Task/context: \n{constraints}\n Evaluate the candidate hyperparameter combination (JSON below) for the Json task description above:\n{candidate.model_dump_json()} and suggest improvements to get better results."
        )
        decision = decision_res.final_output

        # 3. LOG THE CONVERSATION ROUND
        # We combine the Proposer's rationale and the Optimizer's Decision/Reasoning
        round_log_rationale = (
            f"--- Suggester Rationale ---\n{candidate_rationale}\n\n"
            f"--- Optimizer Decision ---\nAccepted: {decision.accept}\n"
            f"Reasoning: {decision.reason}\n"
            f"Suggestions: {decision.suggestions}"
        )
        
        log_planning_step(
            job_id=job_id,
            step_name="Hyperparameter Negotiation",
            input_context=f"Constraints:\n{constraints}\n\nPrior Feedback:\n{last_reason}",
            rationale=round_log_rationale,
            output_summary=candidate.model_dump(),
            round_num=round_idx
        )

        if decision.accept:
            return candidate, decision

        if decision.suggestions:
            constraints += f"\nPrefer these changes: {decision.suggestions}"
        last_reason = decision.reason

    return candidate, decision