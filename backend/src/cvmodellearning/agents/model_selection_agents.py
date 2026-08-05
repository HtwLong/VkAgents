from pydantic import BaseModel, ConfigDict, Field, create_model
from agents import Agent

# Import ONLY the specific sub-schemas
from cvmodellearning.models.registry import format_available_models
from cvmodellearning.schemas.classification_model_requirements import ModelSpecModel
from cvmodellearning.schemas.detection_model_requirements import ObjectDetectionModelSpecModel
from cvmodellearning.schemas.vqa_model_requirements import VQAModelSpecModel


def _agent_model_spec(name: str, source: type[BaseModel]) -> type[BaseModel]:
    """Build an agent-facing model spec without registry-derived metadata."""
    fields = {
        field_name: (field.annotation, field)
        for field_name, field in source.model_fields.items()
        if field_name != "architecture_family"
    }
    return create_model(
        name,
        __config__=ConfigDict(extra="forbid", use_enum_values=True),
        **fields,
    )


ClassificationAgentModelSpec = _agent_model_spec(
    "ClassificationAgentModelSpec", ModelSpecModel
)
DetectionAgentModelSpec = _agent_model_spec(
    "DetectionAgentModelSpec", ObjectDetectionModelSpecModel
)
VQAAgentModelSpec = _agent_model_spec("VQAAgentModelSpec", VQAModelSpecModel)


# --- 1. Define Targeted 'Patch' Schemas ---
class ClassificationModelPatch(BaseModel):
    model: ClassificationAgentModelSpec = Field(..., description="The single chosen model architecture.")
    rationale: str = Field(..., description="Explanation of why this model was chosen.")

class DetectionModelPatch(BaseModel):
    model: DetectionAgentModelSpec = Field(..., description="The single chosen model architecture.")
    rationale: str = Field(..., description="Explanation of why this model was chosen.")

class VQAModelPatch(BaseModel):
    model: VQAAgentModelSpec = Field(..., description="The single chosen VQA model architecture.")
    rationale: str = Field(..., description="Explanation for the VQA configuration and LoRA settings.")

# --- 2. Blueprint Constant ---
PIPELINE_STATE_BLUEPRINT = """
### PIPELINE STATE STRUCTURE (Input Context):
You will receive a JSON object with the following fields. Some may be null:
- `task`: The core CV task.
- `application_domain`: The real-world use case.
- `user_query`: The original prompt.
- `classes`: Target objects/classes.
- `performance_requirements`: Metrics, targets, whether a numeric target is hard, and normalized latency/accuracy preferences.
- `deployment_constraints`: Hard inference footprint limits, including qualitative memory category and explicit memory/model-size/parameter/CPU-latency limits.
- `available_hardware`: User-provided inference and deployment constraints.
- `training_hardware`: Server-selected training hardware; use it as a soft feasibility preference, not as a deployment constraint.
- `available_data`: Datasets found containing these classes.
- `model_selection_graph_context`: GraphRAG shortlist from the NetworkX knowledge graph. It contains up to 7 ranked candidate models with model metadata, inference-memory estimates, benchmark results, task evaluation metrics, hardware profiles, and evidence sources. Use this as grounded evidence for model choice and rationale.
"""

RATIONALE_LANGUAGE_INSTRUCTION = (
    "Write the rationale in clear English using ASCII characters only. "
    "Do not insert words or characters from another language. "
)

CLASSIFICATION_MODEL_INSTRUCTIONS = format_available_models("classification")
DETECTION_MODEL_INSTRUCTIONS = format_available_models("detection")
VQA_MODEL_INSTRUCTIONS = format_available_models("visual question answering")

# --- 3. Agents ---
classification_model_selector_agent = Agent(
    name="Model Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{RATIONALE_LANGUAGE_INSTRUCTION}"
        "Review the 'task', 'application_domain', and 'use_case_description'. "
        "When present, use `model_selection_graph_context.deterministic_recommendation` as the architecture; local code has already applied hard constraints and a balanced capacity policy. Clearly distinguish measured latency from unverified CPU feasibility and mention any fallback model. "
        "Choose the best architecture. Respect `performance_requirements.latency_category` and "
        "`performance_requirements.accuracy_category` when present: prefer efficient models for "
        "VeryLow or Low latency requirements, stronger models for MediumHigh or High accuracy "
        "requirements, and practical trade-offs when both matter. "
        f"Available: {CLASSIFICATION_MODEL_INSTRUCTIONS}."
    ),
    output_type=ClassificationModelPatch,
    model="gpt-5-nano"
)

detection_model_selector_agent = Agent(
    name="Detection Model Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{RATIONALE_LANGUAGE_INSTRUCTION}"
        "Observe the extracted 'classes' and 'application_domain'. "
        "When present, use `model_selection_graph_context.deterministic_recommendation` as the architecture; local code has already applied hard constraints and a balanced capacity policy. Clearly distinguish measured latency from unverified CPU feasibility and mention any fallback model. "
        "Select a model architecture. Respect `performance_requirements.latency_category` and "
        "`performance_requirements.accuracy_category` when present: prefer efficient detectors for "
        "VeryLow or Low latency requirements, stronger detectors for MediumHigh or High accuracy "
        "requirements, and practical trade-offs when both matter. "
        f"Available: {DETECTION_MODEL_INSTRUCTIONS}."
    ),
    output_type=DetectionModelPatch,
    model="gpt-5-nano"
)

vqq_model_selector_agent = Agent(
    name="VQA Model Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{RATIONALE_LANGUAGE_INSTRUCTION}"
        f"Review the state. For Visual Question Answering, select from: {VQA_MODEL_INSTRUCTIONS}. "
        "When present, use `model_selection_graph_context.deterministic_recommendation` as the architecture; local code has already applied hard constraints and a balanced capacity policy. Clearly distinguish measured latency from unverified CPU feasibility and mention any fallback model. "
        "Respect `performance_requirements.latency_category` and `performance_requirements.accuracy_category` "
        "when present when choosing model size and LoRA settings. "
        "Fill the rationale based on the user's specific 'questions_list' if present in the state."
    ),
    output_type=VQAModelPatch,
    model="gpt-5-nano"
)
