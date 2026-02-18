import asyncio
import sys
from typing import List, Tuple
from agents import Agent, Runner
from cvmodellearning.schemas.decision_schema import Decision
from cvmodellearning.schemas.classification_model_requirements import StructuredOutputModel

readiness_check_agent = Agent(
    name="Task Readiness Checker",
    instructions=(
        "You decide if the provided description is sufficient to BEGIN structured extraction for a CV training workflow.\n"
        "ACCEPT only if BOTH are present:\n"
        "1) Explicit CV task type (e.g., classification, detection, segmentation, pose, OCR, tracking, keypoints).\n"
        "2) What classes they want to classify/detect/segment/track etc.\n"
        "If rejecting, return a clear 'reason' and 'suggestions' as a list of short questions to elicit exactly the missing pieces.\n"
        "Return only a pydantic Decision."
    ),
    output_type=Decision,
    model="gpt-4o-mini"
)

task_interpretation_agent = Agent(
    name="Task Interpretor",
    instructions=(
        "Extract ONLY given information from the user prompt about a computer vision task, "
        "the model and the data the model should be trained on. Leave fields empty, if the information "
        "does not exist in the user request. For the classes mentioned, turn them into singular form if they are in plural form."
    ),
    output_type=StructuredOutputModel,
    model="gpt-4o-mini"

)

async def ask_user_for_details(suggestions: List[str]) -> str:
    print("\nMore details are required. Please answer the following:\n")
    for i, q in enumerate(suggestions or [], 1):
        print(f"{i}. {q}")
    print("\nProvide answers (free-form). Press Enter to submit:")
    # Run blocking input in thread so we can stay in async
    return await asyncio.to_thread(sys.stdin.readline)

async def interpretation_loop(initial_context: str, max_rounds: int = 3) -> Tuple[str, Decision]:
    context = initial_context
    last_decision = None
    for r in range(1, max_rounds + 1):
        res = await Runner.run(readiness_check_agent, context)
        decision: Decision = res.final_output
        last_decision = decision

        if decision.accept:
            print(f"Readiness check passed (round {r}).")
            return context, decision

        print(f"Readiness check failed (round {r}): {decision.reason}")
        user_addon = await ask_user_for_details(decision.suggestions or [])
        # Append new info to prior description so the Task Interpreter sees the full context
        context += f"\n\nAdditional details (round {r}):\n{user_addon.strip()}"

    # If still not acceptable, return the latest context and decision (caller can decide how to proceed)
    return context, last_decision