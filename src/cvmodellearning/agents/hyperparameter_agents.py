import json
from typing import Union
from openai import OpenAI
from pydantic import BaseModel

# Import your schemas and logging utility
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel
from cvmodellearning.schemas.vqa_hpo import VQAConfigModel
from cvmodellearning.schemas.decision_schema import Decision
from cvmodellearning.agents.agents_utils import log_planning_step

def generate_and_evaluate_hpo(json_data: str, job_id: str, max_rounds: int = 5) -> tuple[Union[BaseModel, None], Union[Decision, None]]:
    """
    Uses the Evaluator-Optimizer pattern to guess and validate ML hyperparameters,
    logging the negotiation process for every round.
    """
    client = OpenAI()
    
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
                "You are an expert Machine Learning architect. Based on the provided dataset "
                "and hardware context, generate a highly optimal, safe hyperparameter configuration. "
                "Pay strict attention to memory constraints, learning rates, and standard best practices. "
                "Always fill the 'rationale' field explaining concisely your suggestion choices."
            )
        },
        {
            "role": "user",
            "content": f"Task Context:\n{json_data}"
        }
    ]

    evaluator_system_prompt = (
        "You are a Senior Machine Learning Reviewer. Your job is to review proposed hyperparameters. "
        "Look for catastrophic errors: Out of Memory risks, exploding gradients, or logical mismatches. "
        "Be ruthless but constructive. If it is safe and reasonably optimal, accept it."
    )

    print(f"Starting Multi-Agent Optimization for task: {task.upper()} (Job ID: {job_id})")

    # Tracking variables for the log
    last_reason = None
    last_suggestions = None

    # 4. The Evaluator-Optimizer Loop
    for r in range(max_rounds):
        round_idx = r + 1
        print(f"\n--- Round {round_idx}/{max_rounds} ---")
        
        # Phase A: The Optimizer proposes a configuration
        optimizer_response = client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=optimizer_messages,
            response_format=TargetSchema,
            temperature=0.2 
        )
        proposal = optimizer_response.choices[0].message.parsed
        
        # Record the proposal in the optimizer's history
        optimizer_messages.append({
            "role": "assistant",
            "content": optimizer_response.choices[0].message.content
        })

        # Phase B: The Evaluator reviews the proposal
        evaluator_messages = [
            {"role": "system", "content": evaluator_system_prompt},
            {"role": "user", "content": f"Task Context: {json_data}\n\nProposed Configuration:\n{proposal.model_dump_json(indent=2)}"}
        ]
        
        evaluator_response = client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=evaluator_messages,
            response_format=Decision,
            temperature=0.1
        )
        decision = evaluator_response.choices[0].message.parsed
        
        # Phase C: Format and execute the planning log step
        candidate_rationale = getattr(proposal, "rationale", "No rationale provided.")
        
        # Build the strings for the log based on what happened this round
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

        # Call your custom logging function
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
            
            # Format the feedback and append it to the Optimizer's history for the next round
            feedback = (
                f"Your proposal was rejected by the reviewer.\n"
                f"Reason: {decision.reason}\n"
                f"Suggestions to fix: {decision.suggestions}\n"
                "Please generate a new configuration incorporating these fixes."
            )
            optimizer_messages.append({
                "role": "user",
                "content": feedback
            })
            
            # Update tracking variables for the next round's log
            last_reason = decision.reason
            last_suggestions = decision.suggestions

    # If the loop exhausts without an accept
    print("\n⚠️ Max rounds reached. The agents could not agree on a safe configuration.")
    return proposal, decision # Returning the last proposal and decision even if failed, for downstream inspection