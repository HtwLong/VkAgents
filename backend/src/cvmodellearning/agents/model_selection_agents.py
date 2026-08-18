from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, create_model
from agents import Agent
from cvmodellearning.llm_config import PLANNING_MODEL

# Import ONLY the specific sub-schemas
from cvmodellearning.models.registry import format_available_models
from cvmodellearning.schemas.classification_model_requirements import ModelSpecModel
from cvmodellearning.schemas.detection_model_requirements import ObjectDetectionModelSpecModel
from cvmodellearning.schemas.vqa_model_requirements import VQAModelSpecModel
from cvmodellearning.skills import load_cv_skill


def _agent_model_spec(name: str, source: type[BaseModel]) -> type[BaseModel]:
    """Build an architecture-only selector spec; HPO owns every training field."""
    selector_fields = {"model_architecture", "description"}
    fields = {
        field_name: (field.annotation, field)
        for field_name, field in source.model_fields.items()
        if field_name in selector_fields
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
class CandidateComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(..., min_length=1)
    advantages: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    constraint_status: Literal["feasible", "uncertain"]


class ClassificationModelPatch(BaseModel):
    selected_candidate_id: Optional[str] = Field(None, min_length=1)
    model: ClassificationAgentModelSpec = Field(..., description="The single chosen model architecture.")
    rationale: str = Field(..., description="Explanation of why this model was chosen.")
    evaluated_candidates: list[CandidateComparison] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

class DetectionModelPatch(BaseModel):
    selected_candidate_id: Optional[str] = Field(None, min_length=1)
    model: DetectionAgentModelSpec = Field(..., description="The single chosen model architecture.")
    rationale: str = Field(..., description="Explanation of why this model was chosen.")
    evaluated_candidates: list[CandidateComparison] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

class VQAModelPatch(BaseModel):
    selected_candidate_id: Optional[str] = Field(None, min_length=1)
    model: VQAAgentModelSpec = Field(..., description="The single chosen VQA model architecture.")
    rationale: str = Field(..., description="Explanation for the VQA configuration and LoRA settings.")
    evaluated_candidates: list[CandidateComparison] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

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
- `training_hardware`: Server-selected training hardware. GraphRAG may use its evidence-backed training VRAM recommendation as a feasibility constraint; it is separate from deployment hardware.
- `available_data`: Datasets found containing these classes.
- `model_selection_graph_context`: GraphRAG shortlist from the NetworkX knowledge graph. It contains up to 7 complementary candidate models in neutral alphabetical order, with shortlist roles, criterion assessments, model metadata, inference-memory estimates, benchmarks, metrics, and evidence. Candidate position is not a preference.

Model selection owns architecture choice only. Never emit epochs, patience, optimizer, learning rate, precision, losses, LoRA settings, or augmentation fields; the HPO stage owns all training configuration.
"""

RATIONALE_LANGUAGE_INSTRUCTION = (
    "Write the rationale in clear English using ASCII characters only. "
    "Do not insert words or characters from another language. "
)

MODEL_SELECTION_SKILLS = (
    f"\n\n{load_cv_skill('diagnose')}\n\n"
    f"{load_cv_skill('model-selection')}\n\n"
    f"{load_cv_skill('data-problems')}\n"
)

COMPARISON_INSTRUCTION = (
    "When GraphRAG candidates are present, compare at least two candidates (or every candidate if fewer than two) "
    "in evaluated_candidates before selecting. Use exact candidate IDs, give concrete advantages and risks, mark "
    "uncertain claims as uncertain, and explain why the chosen trade-off best fits the use case. Do not infer preference "
    "from candidate order or shortlist roles. Set selected_candidate_id to the exact ID of the chosen GraphRAG "
    "candidate. For detection this concrete candidate may be more specific than model.model_architecture; for "
    "example, selected_candidate_id='yolo12s' and model.model_architecture='yolov12'. "
    "A soft numeric target is not a ceiling: continue to prefer stronger comparable benchmark values when "
    "accuracy matters. Treat models that are within a soft memory limit as equally satisfying that limit; do "
    "not prefer the smallest footprint unless model size, compute, or latency was requested. If the context "
    "marks small-object evidence unverified, explicitly state that AP-small is unavailable and never claim one "
    "candidate has superior small-object performance from overall mAP or architecture alone. "
    "Inference-memory estimates are deployment facts only. Never use inference VRAM to justify training "
    "feasibility, training headroom, batch size, augmentation, tiling, or multi-scale training; use only "
    "model_training_hardware_requirement for those claims. For detection with requested small objects, compare "
    "at least three candidates when available, cover at least two distinct architecture types, and include a "
    "feasible TwoStageRegionProposalDetector candidate when one is retrieved. A third architecture type is "
    "preferred but is not required."
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
        f"{MODEL_SELECTION_SKILLS}"
        f"{COMPARISON_INSTRUCTION}"
        "Choose the best architecture. Respect `performance_requirements.latency_category` and "
        "`performance_requirements.accuracy_category` when present: prefer efficient models for "
        "VeryLow or Low latency requirements, stronger models for MediumHigh or High accuracy "
        "requirements, and practical trade-offs when both matter. "
        f"Available: {CLASSIFICATION_MODEL_INSTRUCTIONS}."
    ),
    output_type=ClassificationModelPatch,
    model=PLANNING_MODEL
)

detection_model_selector_agent = Agent(
    name="Detection Model Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{RATIONALE_LANGUAGE_INSTRUCTION}"
        "Observe the extracted 'classes' and 'application_domain'. "
        f"{MODEL_SELECTION_SKILLS}"
        f"{COMPARISON_INSTRUCTION}"
        "Select a model architecture. Respect `performance_requirements.latency_category` and "
        "`performance_requirements.accuracy_category` when present: prefer efficient detectors for "
        "VeryLow or Low latency requirements, stronger detectors for MediumHigh or High accuracy "
        "requirements, and practical trade-offs when both matter. "
        f"Available: {DETECTION_MODEL_INSTRUCTIONS}."
    ),
    output_type=DetectionModelPatch,
    model=PLANNING_MODEL
)

vqq_model_selector_agent = Agent(
    name="VQA Model Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{RATIONALE_LANGUAGE_INSTRUCTION}"
        f"Review the state. For Visual Question Answering, select from: {VQA_MODEL_INSTRUCTIONS}. "
        f"{MODEL_SELECTION_SKILLS}"
        f"{COMPARISON_INSTRUCTION}"
        "Respect `performance_requirements.latency_category` and `performance_requirements.accuracy_category` "
        "when present when choosing model size and LoRA settings. "
        "Fill the rationale based on the user's specific 'questions_list' if present in the state."
    ),
    output_type=VQAModelPatch,
    model=PLANNING_MODEL
)
