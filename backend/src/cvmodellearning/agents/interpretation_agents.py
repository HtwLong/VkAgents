from typing import List, Tuple, Optional, Literal
from pydantic import BaseModel, Field
from agents import Agent, Runner
from cvmodellearning.schemas.decision_schema import Decision
from cvmodellearning.schemas.interpretation_schema import (
    HardwareSpecModel,
    ModelSpecModel,
    PerformanceSpecModel,
    SynonymMatch,
)

# --- 1. Define Targeted 'Patch' Schema ---
class TaskExtractionPatch(BaseModel):
    task: Literal["classification", "detection", "segmentation", "visual question answering"] = Field(...)
    application_domain: Optional[str] = Field(None, description="The real-world domain (e.g., medical, traffic).")
    use_case_description: Optional[str] = Field(None, description="Overall goal.")
    questions_list: Optional[List[str]] = Field(None, description="Specific VQA questions if provided.")
    classes: List[str] = Field(default_factory=list, description="Target objects to detect/classify.")
    performance_requirements: Optional[PerformanceSpecModel] = Field(
        None,
        description="Performance targets or constraints explicitly mentioned by the user.",
    )
    available_hardware: Optional[HardwareSpecModel] = Field(
        None,
        description="Available compute hardware explicitly mentioned by the user.",
    )
    model_requirements: Optional[List[ModelSpecModel]] = Field(
        None,
        description="User-stated model architecture, framework, backbone, or model preference requirements.",
    )

# --- 2. Agents ---
readiness_check_agent = Agent(
    name="Task Readiness Checker",
    instructions=(
        "You decide if the provided description is sufficient to BEGIN structured extraction for a CV training workflow.\n"
        "ACCEPT only if following question is answered:\n"
        "Does the user want to classify, detect, segment images/videos or create a visual question answering model? It **HAS** to be one of them.\n"
        "If rejecting, return a clear 'reason'. As 'suggestions' ask the user for the missing information and nothing else. \n"
        "Return only a pydantic Decision."
    ),
    output_type=Decision,
    model="gpt-5-nano"
)

task_interpretation_agent = Agent(
    name="Task Interpreter",
    instructions=(
        "Extract ONLY the requested information from the user prompt. "
        "Leave fields empty if the information does not exist. "
        "For the classes mentioned, turn them into singular form. "
        "If classes are not explicitly mentioned, try to infer them. "
        "Also extract explicit performance requirements, available hardware, and model requirements if provided."
    ),
    output_type=TaskExtractionPatch, 
    model="gpt-5-nano"
)

synonym_check_agent = Agent(
    name="Synonym Checker",
    instructions=(
        "You are an expert ontology matcher for computer vision datasets. "
        "You will be given a User Class and a list of Allowed Dataset Classes. "
        "Task: Determine if the User Class is a synonym, alternative name, or sub-type.\n"
        "1. If exact match exists, select it.\n"
        "2. If a clear semantic match exists, select it.\n"
        "3. If NO match exists, set found_match=False."
    ),
    output_type=SynonymMatch,
    model="gpt-5-nano"
)

async def interpretation_loop(initial_context: str, user_replies: Optional[List[str]] = None, max_rounds: int = 3) -> Tuple[str, Decision]:
    context = initial_context
    last_decision = None
    user_replies = user_replies or []
    for r in range(1, max_rounds + 1):
        res = await Runner.run(readiness_check_agent, context)
        decision: Decision = res.final_output
        last_decision = decision
        if decision.accept:
            return context, decision
        if len(user_replies) < r:
            return context, decision
        reply = user_replies[r - 1]
        if reply:
            context += f"\n\nAdditional details (round {r}):\n{reply.strip()}"
    return context, last_decision
