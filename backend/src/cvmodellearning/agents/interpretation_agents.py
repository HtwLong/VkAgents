from typing import List, Tuple, Optional, Literal
from pydantic import BaseModel, Field
from agents import Agent
from cvmodellearning.llm_config import PLANNING_MODEL
from cvmodellearning.observability.planning_usage import run_planning_agent
from cvmodellearning.schemas.decision_schema import Decision
from cvmodellearning.schemas.interpretation_schema import (
    DeploymentConstraints,
    HardwareSpecModel,
    ModelSpecModel,
    PerformanceSpecModel,
    ConstraintStrengths,
    RobustnessRequirements,
    SynonymMatch,
)

# --- 1. Define Targeted 'Patch' Schema ---
class TaskExtractionPatch(BaseModel):
    task: Literal["classification", "detection", "visual question answering"] = Field(...)
    application_domain: Optional[str] = Field(None, description="The real-world domain (e.g., medical, traffic).")
    use_case_description: Optional[str] = Field(None, description="Overall goal.")
    questions_list: Optional[List[str]] = Field(None, description="Specific VQA questions if provided.")
    classes: List[str] = Field(default_factory=list, description="Target objects to detect/classify.")
    performance_requirements: Optional[PerformanceSpecModel] = Field(
        None,
        description=(
            "Performance targets, constraints, or inferred latency/accuracy categories. "
            "Infer categories from user intent when possible, even if no numeric target is provided."
        ),
    )
    constraint_strengths: ConstraintStrengths = Field(default_factory=ConstraintStrengths)
    robustness_requirements: RobustnessRequirements = Field(default_factory=RobustnessRequirements)
    deployment_constraints: Optional[DeploymentConstraints] = Field(
        None,
        description="Explicit or qualitative inference memory, model-size, parameter, or CPU-latency limits.",
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
    model=PLANNING_MODEL
)

task_interpretation_agent = Agent(
    name="Task Interpreter",
    instructions=(
        "Extract ONLY the requested information from the user prompt. "
        "Leave fields empty if the information does not exist. "
        "For the classes mentioned, turn them into singular form. "
        "If classes are not explicitly mentioned, try to infer them. "
        "Also extract explicit performance requirements, available hardware, and model requirements if provided. "
        "For each model requirement, set requirement_strength=required for directives such as 'must use', "
        "'use exactly', or 'please use', and preferred for 'prefer', 'ideally', or 'if possible'. "
        "Normalize an explicit LoRA request as training_mode='lora' and extract rank, alpha, or dropout when stated. "
        "For performance_requirements.latency_category and performance_requirements.accuracy_category, "
        "use only VeryLow, Low, Medium, MediumHigh, or High. "
        "Use latency_category for speed, real-time, low-latency, edge, mobile, interactive, fast-response, "
        "high-FPS, high-volume, batch, or many-stream requirements. "
        "Use accuracy_category for best quality, highest accuracy, precision/recall/mAP, or safety-critical correctness. "
        "Treat wording such as 'reliable accuracy', 'good accuracy', or 'maintain accuracy' as a soft "
        "MediumHigh accuracy_category. Reserve High for explicit high, critical, maximum, or best-possible accuracy. "
        "Set performance_requirements.target_is_hard=true only for an explicit mandatory numeric target such as "
        "'must achieve at least 90%'; qualitative statements such as 'accuracy is important' are soft preferences. "
        "Represent how strongly objectives are stated in constraint_strengths using hard, soft, preference, or "
        "unspecified. Extract robustness into robustness_requirements using normalized dimensions for lighting, "
        "weather, object_scale, scene_density, motion_blur, occlusion, and viewpoint. "
        "Also extract augmentation semantics only when explicitly supported: set color_semantics=true when "
        "changing colors could change labels or meaning; set horizontal_flip_safe=false when mirroring would "
        "change meaning; and set text_or_symbols_present=true when readable text or directional symbols must "
        "be preserved. Do not infer these properties merely from a class name. "
        "Extract inference footprint separately into deployment_constraints: 'very low memory footprint' means "
        "memory_category=VeryLow, 'low memory' or 'small model' means memory_category=Low, and explicit limits map "
        "to max_runtime_memory_mb, max_model_size_mb, max_parameters_m, or max_cpu_latency_ms. "
        "Add a numeric field name to deployment_constraints.hard_limits only when mandatory wording "
        "such as 'must', 'at most', or 'cannot exceed' makes that specific limit hard. Wording such "
        "as 'aim for', 'around', 'approximately', 'desirable', or 'preferably' is a soft preference: "
        "extract its value but do not add it to hard_limits. "
        "Do not treat relaxed latency as relaxed memory. "
        "Leave category fields empty when the prompt does not state or clearly imply them. "
        "For available_hardware.hardware_category, use exactly one of ConsumerCPU, ConsumerGPU, EdgeDevice, "
        "or DataCenterGPU when the prompt states or implies it. Extract GPU memory as vram_gb. "
        "Classify laptop and desktop computers, including Apple Silicon Macs (MacBook, iMac, Mac mini, "
        "Mac Studio, or Mac Pro), as ConsumerCPU even when they have an integrated GPU or Metal support. "
        "Reserve EdgeDevice for mobile, embedded, and dedicated edge hardware such as phones, tablets, "
        "Raspberry Pi, and Jetson devices. "
        "If there is no data regarding hardware_category or VRAM, set available_hardware.hardware_category "
        "to ConsumerCPU | EdgeDevice. "
        "If you infer a performance intent without a stated metric, set primary_metric to the most relevant metric "
        "such as latency, throughput, accuracy, F1, mAP, or balanced_performance."
    ),
    output_type=TaskExtractionPatch, 
    model=PLANNING_MODEL
)

synonym_check_agent = Agent(
    name="Class Ontology Matcher",
    instructions=(
        "You are an expert ontology matcher for computer vision datasets. "
        "You will be given a User Class and a list of Allowed Dataset Classes. "
        "Task: Determine if the User Class maps to one or more allowed dataset classes in the given application domain.\n"
        "Return only exact strings from the Allowed Dataset Classes in dataset_classes.\n"
        "1. If the User Class is a synonym or alternative name of one allowed class, "
        "set found_match=True and return that one valid class.\n"
        "2. If the User Class is a subcategory of one allowed class, set found_match=True "
        "and return that broader valid class. \n"
        "3. If the User Class is a supercategory of multiple allowed classes, set found_match=True "
        "and return the most appropriate non-overlapping allowed classes but only if they also fall "
        "into the given application domain, with a maximum of ten classes. "
        "Do not return both a parent and child class if both are allowed; choose the non-overlapping set.\n"
        "Treat equivalent textual and symbolic number labels as aliases when the symbolic label is allowed; "
        "for example, 'zero' maps to '0', 'one' to '1', through 'nine' to '9'.\n"
        "4. If no synonym, subcategory, supercategory, or representation-equivalence relationship exists, "
        "set found_match=False "
        "and return an empty dataset_classes list."
    ),
    output_type=SynonymMatch,
    model=PLANNING_MODEL
)

async def interpretation_loop(
    initial_context: str,
    *,
    job_id: str,
    user_replies: Optional[List[str]] = None,
    max_rounds: int = 3,
) -> Tuple[str, Decision]:
    context = initial_context
    last_decision = None
    user_replies = user_replies or []
    for r in range(1, max_rounds + 1):
        res = await run_planning_agent(
            job_id=job_id,
            operation="completeness_check",
            agent=readiness_check_agent,
            input=context,
        )
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
