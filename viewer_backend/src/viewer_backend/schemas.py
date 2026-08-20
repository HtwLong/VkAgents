from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from .planning_contracts import (
    ClassDataAssignment,
    ClassDataSelection,
    ConstraintStrengths as ContractConstraintStrengths,
    DeploymentConstraints as ContractDeploymentConstraints,
    HardwareSpec as ContractHardwareSpec,
    PerformanceSpec as ContractPerformanceSpec,
    RevisionPlan as ContractRevisionPlan,
    RobustnessSpec as ContractRobustnessSpec,
)


class CompletenessRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=160)
    user_prompt: str = Field(min_length=1, max_length=12_000)
    user_replies: list[str] = Field(default_factory=list)


class StateRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=160)
    context: str | dict[str, Any]
    use_graphrag: bool = True

    @field_validator("context", mode="after")
    @classmethod
    def deserialize_context(cls, value: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, str):
            import json
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("context JSON must contain an object")
            return parsed
        return value


class PlanRevisionRequest(BaseModel):
    context: str | dict[str, Any]
    job_id: str = Field(min_length=1, max_length=160)
    required_changes: str = ""
    preferences: str = ""
    requested_target: RevisionTarget | Literal["automatic"] = "automatic"

    _deserialize_context = field_validator("context", mode="after")(StateRequest.deserialize_context.__func__)


class ActivateRevisionRequest(BaseModel):
    context: str | dict[str, Any]
    plan: ContractRevisionPlan
    job_id: str = Field(min_length=1, max_length=160)

    _deserialize_context = field_validator("context", mode="after")(StateRequest.deserialize_context.__func__)


class VerifyRevisionRequest(BaseModel):
    context: str | dict[str, Any]

    _deserialize_context = field_validator("context", mode="after")(StateRequest.deserialize_context.__func__)


class CompletenessDecision(BaseModel):
    accept: bool
    reason: str | None = None
    suggestions: list[str] = Field(default_factory=list)


class PerformanceRequirements(BaseModel):
    primary_metric: str | None = None
    target_value: float | None = None
    target_is_hard: bool = False
    latency_category: str | None = None
    accuracy_category: str | None = None
    other_constraints: list[str] = Field(default_factory=list)


class DeploymentConstraints(BaseModel):
    memory_category: str | None = None
    max_runtime_memory_mb: float | None = None
    max_model_size_mb: float | None = None
    max_parameters_m: float | None = None
    max_cpu_latency_ms: float | None = None
    hard_limits: list[str] = Field(default_factory=list)


class AvailableHardware(BaseModel):
    hardware_category: str | None = None
    cpu_cores: int | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None
    vram_gb: float | None = None
    ram_gb: float | None = None
    storage_gb: float | None = None
    details: str | None = None


class RobustnessRequirements(BaseModel):
    object_scale: list[str] = Field(default_factory=list)
    lighting: list[str] = Field(default_factory=list)
    weather: list[str] = Field(default_factory=list)
    scene_density: list[str] = Field(default_factory=list)
    viewpoint: list[str] = Field(default_factory=list)
    motion_blur: bool = False
    occlusion: bool = False
    color_semantics: bool = False
    horizontal_flip_safe: bool | None = None
    text_or_symbols_present: bool = False


class ConstraintStrengths(BaseModel):
    accuracy: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"
    latency: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"
    runtime_memory: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"
    model_size: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"
    training_time: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"


class ModelRequirement(BaseModel):
    name: str | None = None
    framework: str | None = None
    backbone: str | None = None
    description: str | None = None
    requirement_strength: Literal["required", "preferred"] = "required"
    training_mode: Literal["fine_tune_pretrained", "staged_fine_tune", "head_only", "lora", "train_from_scratch"] | None = None


class TaskInterpretation(BaseModel):
    model_config = {"extra": "forbid"}
    task: Literal["classification", "detection", "visual question answering"] | None = None
    classes: list[str] = Field(default_factory=list)
    application_domain: str | None = None
    use_case_description: str | None = None
    questions_list: list[str] | None = None
    available_data: list[ClassDataSelection] | None = None
    selected_data: list[ClassDataAssignment] | None = None
    performance_requirements: ContractPerformanceSpec | None = None
    deployment_constraints: ContractDeploymentConstraints | None = None
    available_hardware: ContractHardwareSpec | None = None
    robustness_requirements: ContractRobustnessSpec = Field(default_factory=ContractRobustnessSpec)
    constraint_strengths: ContractConstraintStrengths = Field(default_factory=ContractConstraintStrengths)
    # OpenAI strict structured outputs cannot represent the original contract's
    # arbitrary hyperparameters dictionary. The persisted PipelineState validates
    # against ContractModelRequirement after extraction.
    model_requirements: list[ModelRequirement] | None = None
    augmentation: str | None = None
    preprocessing: str | None = None
    num_qa_pairs: int | None = None


class ModelPlan(BaseModel):
    model_id: str
    display_name: str
    family: str
    rationale: str
    uncertainties: list[str] = Field(default_factory=list)


class DatasetSource(BaseModel):
    dataset_name: str
    classes: list[str] = Field(default_factory=list)
    rationale: str


class DatasetPlan(BaseModel):
    sources: list[DatasetSource] = Field(default_factory=list)
    rationale: str
    uncertainties: list[str] = Field(default_factory=list)


class HyperparameterPlan(BaseModel):
    model_name: str
    epochs: int = Field(ge=1, le=500)
    batch_size: int = Field(ge=1, le=1024)
    learning_rate: float = Field(gt=0, le=1)
    optimizer: str
    image_size: int | None = Field(default=None, ge=32, le=4096)
    training_mode: Literal["full", "head", "lora"] = "full"
    rationale: str
    uncertainties: list[str] = Field(default_factory=list)


RevisionTarget = Literal[
    "task-interpretation", "model-selection", "dataset-selection", "choose-hyperparameters"
]


class RevisionChange(BaseModel):
    id: str = Field(min_length=1)
    target_step: RevisionTarget
    field: str = Field(min_length=1)
    operation: Literal["set", "include", "exclude", "prefer", "avoid"] = "set"
    value: str | int | float | bool | list[str] | None = None
    strength: Literal["required", "preferred"]
    summary: str = Field(min_length=1)


class RevisionPlan(BaseModel):
    required_text: str = ""
    preferred_text: str = ""
    summary: str = Field(min_length=1)
    restart_from: RevisionTarget
    changes: list[RevisionChange] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_content(self):
        if not self.required_text.strip() and not self.preferred_text.strip():
            raise ValueError("At least one required change or preference is required.")
        if not self.changes:
            raise ValueError("The request did not produce any actionable changes.")
        return self


class ForkRevisionRequest(BaseModel):
    parent_job_id: str = Field(min_length=1, max_length=160)
    assessment_id: str = Field(min_length=1)
    plan: ContractRevisionPlan


class RequirementAssessment(BaseModel):
    requirement: str
    status: Literal["satisfied", "not_satisfied", "unknown"]
    evidence: list[str] = Field(default_factory=list)
    explanation: str


class AssessmentDraft(BaseModel):
    verdict: Literal["satisfied", "partially_satisfied", "not_satisfied", "unknown"]
    summary: str
    requirements: list[RequirementAssessment] = Field(default_factory=list)
    recommended_plan: RevisionPlan | None = None
    limitations: list[str] = Field(default_factory=list)

    def persisted(self, job_id: str) -> dict[str, Any]:
        return {
            "assessment_id": uuid4().hex,
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **self.model_dump(mode="json"),
        }


class StepTimingUpdate(BaseModel):
    duration_ms: int = Field(ge=0)
    status: str
