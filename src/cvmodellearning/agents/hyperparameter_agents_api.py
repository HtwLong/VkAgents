import json
from agents import Agent, Runner
from cvmodellearning.schemas.decision_schema import Decision
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel  
from cvmodellearning.schemas.vqa_hpo import VQAConfigModel
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
    model="gpt-5-nano",
)

# ==========================================
# 1. SUGGESTER AGENTS
# ==========================================

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
    model="gpt-5-nano",
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
    model="gpt-5-nano",
    output_type=DetectionConfigModel
)

vqa_hyperparameter_suggest_agent = Agent(
    name="Hyperparameters Suggester (VQA)",
    instructions=(
        "Suggest a combination of hyperparameters for the given Vision-Language Model (e.g., Qwen-VL), task, and data information. "
        "Fill the 'rationale' field explaining concisely your suggestion choices. "
        "CRITICAL VLM GUIDELINES: "
        "1. Recommend LoRA parameters for efficient fine-tuning (e.g., lora_r=16, lora_alpha=32) unless full fine-tuning is explicitly requested. "
        "2. Keep learning rates very small (e.g., 1e-5 to 5e-5) to avoid catastrophic forgetting in pre-trained VLMs. "
        "3. Strongly recommend 'bf16' precision to save memory while maintaining stability. "
        "4. Adjust 'max_seq_length' based on expected image + question + answer tokens (e.g., 2048 or 4096)."
    ),
    model="gpt-5-nano",
    output_type=VQAConfigModel
)

# ==========================================
# 2. OPTIMIZER / CRITIC AGENTS
# ==========================================

classification_hyperparameter_opt_agent = Agent(
    name="Hyperparameter Optimizer (Classification)",
    handoff_description="Specialist agent for optimizing classification hyperparameters",
    instructions=(
        "Predict whether the classification hyperparameters are likely to return good results. "
        "Sanity check the settings: "
        "1. Reject if 'eps' is > 1e-3 for AdamW/RMSprop. Suggest 1e-8. "
        "2. Check if the learning rate is too high (> 0.1) or too low (< 1e-6) without reason. "
        "3. Ensure batch size matches the model size (smaller batch for larger models like vit_b_16). "
        "4. Ensure standard classification losses are used (e.g., Cross Entropy)."
    ),
    model="gpt-5-nano",
    output_type=Decision
)

detection_hyperparameter_opt_agent = Agent(
    name="Hyperparameter Optimizer (Detection)",
    handoff_description="Specialist agent for optimizing detection hyperparameters",
    instructions=(
        "Predict whether the detection hyperparameters are likely to return good results. "
        "Sanity check the settings: "
        "1. Reject if 'eps' is > 1e-3 for AdamW/RMSprop. Suggest 1e-8. "
        "2. Ensure image_size is appropriate (often 640 or higher for YOLO models). "
        "3. Check that 'loss_mask' and 'lambda_mask' are properly null/0.0 unless the task specifically requires segmentation. "
        "4. Verify that weight decay is standard for object detection (e.g., ~0.0005)."
    ),
    model="gpt-5-nano",
    output_type=Decision
)

vqa_hyperparameter_opt_agent = Agent(
    name="Hyperparameter Optimizer (VQA)",
    handoff_description="Specialist agent for optimizing VQA/VLM hyperparameters",
    instructions=(
        "Predict whether the VQA hyperparameters are likely to return good results for a Vision-Language Model. "
        "Sanity check the settings: "
        "1. MEMORY: VLMs are huge. If 'use_lora' is False, immediately warn about extreme memory usage unless batch_size is extremely small (e.g., 1 or 2). "
        "2. PRECISION: Strongly encourage 'bf16' over 'fp32' or 'fp16' to prevent numerical instability in LLMs. "
        "3. LEARNING RATE: Reject if the learning rate is > 1e-3. Pre-trained VLMs need small LRs (like 1e-5 to 5e-5) to preserve pre-training knowledge. "
        "4. CONTEXT: Ensure 'max_seq_length' is sufficiently large to hold image tokens and text (e.g., >= 1024)."
    ),
    model="gpt-5-nano",
    output_type=Decision
)

# ==========================================
# 3. LOOP EXECUTION
# ==========================================

async def choose_hyperparameters_loop(json_data: str, job_id: str, max_rounds: int = 5):
    try:
        data = json.loads(json_data)
        task = data.get("task")
    except (json.JSONDecodeError, AttributeError):
        return None, None

    # Route to the correct Suggestion Agent
    TASK_SUGGEST_AGENT_MAP = {
        "classification": classification_hyperparameter_suggest_agent,
        "detection": detection_hyperparameter_suggest_agent,
        "visual question answering": vqa_hyperparameter_suggest_agent,
    }

    # Route to the correct Optimizer (Critic) Agent
    TASK_OPT_AGENT_MAP = {
        "classification": classification_hyperparameter_opt_agent,
        "detection": detection_hyperparameter_opt_agent,
        "visual question answering": vqa_hyperparameter_opt_agent,
    }

    hyperparameter_suggest_agent = TASK_SUGGEST_AGENT_MAP.get(task)
    hyperparameter_opt_agent = TASK_OPT_AGENT_MAP.get(task)
    
    # Fallback/Safety Check
    if not hyperparameter_suggest_agent or not hyperparameter_opt_agent:
        return None, None
    
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