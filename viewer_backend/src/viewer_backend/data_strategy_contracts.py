"""PyTorch-free copies of the original backend's data-strategy contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .planning_contracts import ClassDataSelection


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassCoverageObjective(StrictModel):
    class_name: str
    minimum_positive_images: int = Field(ge=1)
    preferred_positive_images: int = Field(ge=1)
    maximum_positive_images: int = Field(ge=1)
    priority: Literal["standard", "high", "critical"] = "standard"
    rationale: str

    @model_validator(mode="after")
    def ordered_bounds(self):
        if not self.minimum_positive_images <= self.preferred_positive_images <= self.maximum_positive_images:
            raise ValueError("Coverage must satisfy minimum <= preferred <= maximum.")
        return self


class DatasetClassUseOverride(StrictModel):
    class_name: str
    role: Literal["primary", "training_supplement", "external_evaluation"]
    activation: Literal[
        "required", "if_needed_for_minimum", "if_needed_for_preferred", "optional"
    ] | None = None
    minimum_images_when_used: int | None = Field(None, ge=0)
    allow_derived_validation: bool = False
    allow_derived_test: bool = False
    rationale: str

    @model_validator(mode="after")
    def coherent_class_use(self):
        if self.role == "external_evaluation" and (self.allow_derived_validation or self.allow_derived_test):
            raise ValueError("External evaluation uses an official split.")
        return self


class DatasetSourceDecision(StrictModel):
    dataset_id: str
    role: Literal["primary", "training_supplement", "external_evaluation"]
    priority: int = Field(ge=1)
    focus_classes: list[str] = Field(default_factory=list)
    activation: Literal[
        "required", "if_needed_for_minimum", "if_needed_for_preferred", "optional"
    ]
    minimum_images_when_used: int = Field(0, ge=0)
    allow_derived_validation: bool = False
    allow_derived_test: bool = False
    class_use_overrides: list[DatasetClassUseOverride] = Field(default_factory=list)
    rationale: str

    @model_validator(mode="after")
    def coherent_source_decision(self):
        if self.role == "external_evaluation" and (self.allow_derived_validation or self.allow_derived_test):
            raise ValueError(
                "External evaluation sources use their official split and cannot provide derived holdouts."
            )
        names = [item.class_name for item in self.class_use_overrides]
        if len(names) != len(set(names)):
            raise ValueError("A dataset may contain only one override per class.")
        return self

    def use_for_class(self, class_name: str) -> dict[str, object]:
        override = next((item for item in self.class_use_overrides if item.class_name == class_name), None)
        return {
            "role": override.role if override else self.role,
            "activation": override.activation if override and override.activation is not None else self.activation,
            "minimum_images_when_used": (
                override.minimum_images_when_used
                if override and override.minimum_images_when_used is not None
                else self.minimum_images_when_used
            ),
            "allow_derived_validation": override.allow_derived_validation if override else self.allow_derived_validation,
            "allow_derived_test": override.allow_derived_test if override else self.allow_derived_test,
        }


class DataSplitStrategy(StrictModel):
    primary_evaluation_domain: str
    use_official_validation: bool
    use_official_test: bool
    derive_missing_holdouts: bool
    group_isolation_keys: list[str] = Field(default_factory=list)
    preserve_natural_evaluation_frequency: bool = True
    training_balance_policy: Literal["natural", "bounded", "class_aware_sampling"] = "class_aware_sampling"


class DataStrategy(StrictModel):
    coverage_objectives: list[ClassCoverageObjective]
    source_decisions: list[DatasetSourceDecision]
    split_strategy: DataSplitStrategy
    minimum_unique_pool_images: int = Field(ge=1)
    preferred_unique_pool_images: int = Field(ge=1)
    preferred_target_is_strict: bool = False
    acceptable_compromises: list[str] = Field(default_factory=list)
    rationale: str

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "minimum_unique_pool_images" not in normalized and "minimum_unique_images" in normalized:
            normalized["minimum_unique_pool_images"] = normalized.pop("minimum_unique_images")
        if "preferred_unique_pool_images" not in normalized and "preferred_unique_images" in normalized:
            normalized["preferred_unique_pool_images"] = normalized.pop("preferred_unique_images")
        return normalized

    @model_validator(mode="after")
    def ordered_unique_image_objectives(self):
        if self.preferred_unique_pool_images < self.minimum_unique_pool_images:
            raise ValueError("preferred_unique_pool_images must be >= minimum_unique_pool_images.")
        return self


class DataPlanConflict(StrictModel):
    code: str
    class_name: str | None = None
    dataset_id: str | None = None
    reason: str
    facts: dict = Field(default_factory=dict)
    available_options: list[dict] = Field(default_factory=list)


class CompiledDataPlan(StrictModel):
    selected_data: list[ClassDataSelection]
    strategy: DataStrategy
    source_roles: dict[str, dict[str, str]]
    class_coverage: dict[str, dict[str, int]]
    warnings: list[dict] = Field(default_factory=list)
    conflicts: list[DataPlanConflict] = Field(default_factory=list)


class DataPlanReviewFinding(StrictModel):
    code: str
    severity: Literal["blocking", "warning"]
    field: str
    reason: str


class DataPlanReview(StrictModel):
    accept: bool
    reason: str
    findings: list[DataPlanReviewFinding] = Field(default_factory=list)
