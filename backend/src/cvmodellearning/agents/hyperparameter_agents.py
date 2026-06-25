import json
from typing import Union
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

# Import your schemas and logging utility
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel
from cvmodellearning.schemas.vqa_hpo import VQAConfigModel
from cvmodellearning.schemas.decision_schema import Decision
from cvmodellearning.agents.agents_utils import log_planning_step

# --- Knowledge Base Constant ---
PIPELINE_STATE_BLUEPRINT = """
### PIPELINE STATE STRUCTURE (Input Context):
You are receiving a full PipelineState JSON containing:
- `task`, `application_domain`, `classes`
- `performance_requirements`: Metrics, targets, and normalized optimization priority such as LatencyFirst, AccuracyFirst, ThroughputFirst, or Balanced.
- `available_hardware`: Compute constraints, including `hardware_category` (ConsumerCPU, ConsumerGPU, EdgeDevice, DataCenterGPU, or fallback ConsumerCPU | EdgeDevice) and `vram_gb`.
- `selected_data`: The image counts chosen for training.
- `selected_model_info`: The architecture chosen in previous steps.
- `augmentation`, `preprocessing`: The data transformation strategies.
"""

async def generate_and_evaluate_hpo(json_data: str, job_id: str, max_rounds: int = 5) -> tuple[Union[BaseModel, None], Union[Decision, None]]:
    """
    Uses the Evaluator-Optimizer pattern to guess and validate ML hyperparameters,
    logging the negotiation process for every round.
    """
    client = AsyncOpenAI() 
    
    # 1. Parse the input to determine the task
    try:
        context_data = json.loads(json_data)
        task = context_data.get("task", "").lower()
    except json.JSONDecodeError:
        print("Error: Invalid JSON data provided.")
        return None, None

    # 2. Map the task to the correct Pydantic schema
    schema_mapping = {
        "classification": ClassificationConfigModel,
        "detection": DetectionConfigModel,
        "visual question answering": VQAConfigModel
    }
    
    if task not in schema_mapping:
        print(f"Error: Unsupported task '{task}'.")
        return None, None
    
    TargetSchema = schema_mapping[task]

    # 3. Setup the initial Optimizer conversation history
    optimizer_messages = [
        {
            "role": "system",
            "content": (
                f"{PIPELINE_STATE_BLUEPRINT}\n\n"
                "You are a strict, deterministic Machine Learning configuration engine. "
                "Review the `selected_model_info`, `selected_data`, and `task` from the state. "
                "Based on this context, generate a safe hyperparameter configuration. "
                "You must rely ONLY on standard, universally accepted heuristics. Do not attempt creative, novel, or experimental configurations. "
                "If a parameter is standard, use the standard value. Hallucination or guessing outside of the provided context is strictly prohibited. "
                "Pay strict attention to memory constraints and learning rates for the selected architecture. "
                "Respect `performance_requirements.priority` when present: favor stronger training choices for AccuracyFirst, "
                "efficient settings for LatencyFirst or ThroughputFirst, and moderate defaults for Balanced. "
                "Always fill the 'rationale' field explaining concisely your standard choices."
            )
        },
        {
            "role": "user",
            "content": f"Task Context:\n{json_data}"
        }
    ]

    evaluator_system_prompt = (
        f"{PIPELINE_STATE_BLUEPRINT}\n\n"
        "You are a strict Senior Machine Learning Reviewer. Your job is to review proposed hyperparameters against the provided PipelineState. "
        "Look for catastrophic errors: Out of Memory risks based on the chosen model, exploding gradients, or logical mismatches. "
        "Check that the proposal is consistent with `performance_requirements.priority` when it is present. "
        "Be ruthless but constructive. If it is safe and adheres to standard practices, accept it."
    )

    print(f"Starting Multi-Agent Optimization for task: {task.upper()} (Job ID: {job_id})")

    last_reason = None
    last_suggestions = None
    rejection_count = 0

    # 4. The Evaluator-Optimizer Loop
    for r in range(max_rounds):
        round_idx = r + 1
        print(f"\n--- Round {round_idx}/{max_rounds} ---")
        
        # Phase A: The Optimizer proposes a configuration (wrapped in try/except for self-healing)
        try:
            optimizer_response = await client.beta.chat.completions.parse(
                model="gpt-5-nano",
                messages=optimizer_messages,
                response_format=TargetSchema
            )
            
            opt_message = optimizer_response.choices[0].message
            
            if opt_message.refusal:
                print(f"❌ Optimizer refused to generate a configuration: {opt_message.refusal}")
                return None, None
                
            proposal = opt_message.parsed
            
        except ValidationError as e:
            # The LLM violated our Pydantic rules (e.g., set the learning rate too high)
            error_msg = e.errors()[0]['msg']
            print(f"⚠️ Pydantic Validation Error in Round {round_idx}: {error_msg}")
            
            # Feed the exact validation error back to the LLM so it can fix it
            error_feedback = (
                f"Your previous JSON output failed strict schema validation.\n"
                f"Error: {error_msg}\n"
                f"Please adjust your parameters to strictly obey the validation rules."
            )
            optimizer_messages.append({"role": "user", "content": error_feedback})
            continue 

        # Record the successful proposal in the optimizer's history
        optimizer_messages.append({
            "role": "assistant",
            "content": opt_message.content
        })

        # Phase B: The Evaluator reviews the proposal
        evaluator_messages = [
            {"role": "system", "content": evaluator_system_prompt},
            {"role": "user", "content": f"Task Context: {json_data}\n\nProposed Configuration:\n{proposal.model_dump_json(indent=2)}"}
        ]
        
        evaluator_response = await client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=evaluator_messages,
            response_format=Decision
        )
        
        eval_message = evaluator_response.choices[0].message
        
        if eval_message.refusal:
             print(f"❌ Evaluator refused to generate a decision: {eval_message.refusal}")
             return proposal, None
             
        decision = eval_message.parsed
        
        # Phase C: Format and execute the planning log step
        candidate_rationale = getattr(proposal, "rationale", "No rationale provided.")
        
        input_context_str = f"Constraints:\n{json_data}\n"
        if last_reason:
            input_context_str += f"\nPrior Feedback applied this round:\nReason: {last_reason}\nSuggestions: {last_suggestions}"
            
        round_log_rationale = (
            f"--- Suggester Rationale ---\n{candidate_rationale}\n\n"
            f"--- Optimizer Decision ---\nAccepted: {decision.accept}\n"
            f"Reasoning: {decision.reason}\n"
            f"Suggestions: {decision.suggestions}"
        )
        
        output_summary_dict = {
            "proposal": proposal.model_dump(),
            "decision": decision.model_dump()
        }

        log_planning_step(
            job_id=job_id,
            step_name="Hyperparameter Negotiation",
            input_context=input_context_str,
            rationale=round_log_rationale,
            output_summary=output_summary_dict,
            round_num=round_idx
        )

        # Phase D: Evaluate the decision to continue or break
        if decision.accept:
            print("✅ Evaluator accepted the configuration!")
            return proposal, decision
        else:
            print(f"❌ Evaluator rejected the configuration. Reason: {decision.reason}")
            
            if decision.reason == last_reason:
                rejection_count += 1
            else:
                rejection_count = 1
                
            feedback = (
                f"Your proposal was rejected by the reviewer.\n"
                f"Reason: {decision.reason}\n"
                f"Suggestions to fix: {decision.suggestions}\n"
                "Please generate a new configuration incorporating these fixes."
            )
            
            if rejection_count >= 2:
                feedback = f"CRITICAL WARNING: You have been rejected multiple times for the exact same reason:\n'{decision.reason}'.\nYou MUST fundamentally change your approach to address this constraint." + feedback
                
            optimizer_messages.append({
                "role": "user",
                "content": feedback
            })
            
            last_reason = decision.reason
            last_suggestions = decision.suggestions

    print("\n⚠️ Max rounds reached. The agents could not agree on a safe configuration.")
    return proposal, decision
