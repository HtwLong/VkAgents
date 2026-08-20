from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class CompletenessRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=160)
    user_prompt: str = Field(min_length=1, max_length=12_000)
    user_replies: list[str] = Field(default_factory=list)


class StateRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=160)
    context: dict[str, Any]
    use_graphrag: bool = True


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
    deployment_target: str | None = None
    max_runtime_memory_mb: float | None = None
    max_model_size_mb: float | None = None
    max_parameters_m: float | None = None
    max_cpu_latency_ms: float | None = None
    details: str | None = None


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
    lighting_conditions: list[str] = Field(default_factory=list)
    weather_conditions: list[str] = Field(default_factory=list)
    viewpoints: list[str] = Field(default_factory=list)
    occlusion: bool | None = None
    other_requirements: list[str] = Field(default_factory=list)


class TaskInterpretation(BaseModel):
    task: Literal["classification", "detection", "visual question answering"]
    classes: list[str] = Field(default_factory=list)
    application_domain: str | None = None
    performance_requirements: PerformanceRequirements = Field(default_factory=PerformanceRequirements)
    deployment_constraints: DeploymentConstraints = Field(default_factory=DeploymentConstraints)
    available_hardware: AvailableHardware = Field(default_factory=AvailableHardware)
    robustness_requirements: RobustnessRequirements = Field(default_factory=RobustnessRequirements)


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
    id: str
    target_step: RevisionTarget
    field: str
    operation: Literal["set", "include", "exclude", "prefer", "avoid"] = "set"
    value: str | int | float | bool | list[str] | None = None
    strength: Literal["required", "preferred"]
    summary: str


class RevisionPlan(BaseModel):
    required_text: str = ""
    preferred_text: str = ""
    summary: str
    restart_from: RevisionTarget
    changes: list[RevisionChange] = Field(default_factory=list)


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
