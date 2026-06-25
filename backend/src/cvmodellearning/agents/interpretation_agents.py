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
        description=(
            "Performance targets, constraints, or inferred optimization priority. "
            "Infer priority from user intent when possible, even if no numeric target is provided."
        ),
    )
    available_hardware: Optional[HardwareSpecModel] = Field(
        None,
        description=(
            "Available compute hardware explicitly mentioned by the user, including "
            "hardware_category and vram_gb when stated or inferable."
        ),
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
        "Also extract explicit performance requirements, available hardware, and model requirements if provided. "
        "For available_hardware.hardware_category, use exactly one of ConsumerCPU, ConsumerGPU, EdgeDevice, "
        "or DataCenterGPU when the prompt states or implies it. Extract GPU memory as vram_gb. "
        "If there is no data regarding hardware_category or VRAM, set available_hardware.hardware_category "
        "to ConsumerCPU | EdgeDevice. "
        "Infer performance_requirements.priority when the prompt signals an optimization preference: "
        "use LatencyFirst for real-time, low-latency, edge, mobile, interactive, or fast-response requirements; "
        "use ThroughputFirst for high-FPS, high-volume, batch, or many-stream processing; "
        "use AccuracyFirst for best quality, highest accuracy, precision/recall/mAP, or safety-critical correctness; "
        "use Balanced when the user asks for a good trade-off between speed and quality. "
        "If you infer a priority without a stated metric, set primary_metric to the most relevant metric "
        "such as latency, throughput, accuracy, F1, mAP, or balanced_performance."
    ),
    output_type=TaskExtractionPatch, 
    model="gpt-5-nano"
)

synonym_check_agent = Agent(
    name="Class Ontology Matcher",
    instructions=(
        "You are an expert ontology matcher for computer vision datasets. "
        "You will be given a User Class and a list of Allowed Dataset Classes. "
        "Task: Determine if the User Class maps to one or more allowed dataset classes.\n"
        "Return only exact strings from the Allowed Dataset Classes in dataset_classes.\n"
        "1. If the User Class is a synonym or alternative name of one allowed class, "
        "set found_match=True and return that one valid class.\n"
        "2. If the User Class is a subcategory of one allowed class, set found_match=True "
        "and return that broader valid class. Example: 'sports car' may map to 'car' if 'car' is allowed.\n"
        "3. If the User Class is a supercategory of multiple allowed classes, set found_match=True "
        "and return the most appropriate non-overlapping allowed classes, with a maximum of ten classes. "
        "Do not return both a parent and child class if both are allowed; choose the non-overlapping set.\n"
        "4. If no synonym, subcategory, or supercategory relationship exists, set found_match=False "
        "and return an empty dataset_classes list."
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
