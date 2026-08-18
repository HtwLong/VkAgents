"""Shared contracts for planning and materializing dataset split assignments.

``normalize_dataset_assignments`` remains the single compatibility boundary
for older training-only selections.
"""

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class AssignmentType(str, Enum):
    OFFICIAL_SPLIT = "official_split"
    DERIVED_FROM_TRAIN = "derived_from_train"


class DatasetSourceCount(BaseModel):
    """Legacy availability/selection entry retained during the migration."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(
        ...,
        min_length=1,
        description=(
            "The exact canonical dataset_id. Human-readable display names are not identifiers."
        ),
    )
    count: int = Field(..., ge=0, description="Number of available or selected images.")


class ClassDataSelection(BaseModel):
    """Legacy class-to-source selection used by the current planning flow."""

    model_config = ConfigDict(extra="forbid")

    class_name: str = Field(..., min_length=1, description="The class or subset name.")
    sources: list[DatasetSourceCount] = Field(
        ..., description="Dataset sources and their available or selected counts."
    )


class SplitAllocation(BaseModel):
    """A planned number of samples assigned to one runtime split."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    split: DatasetSplit
    count: int = Field(..., gt=0)
    assignment_type: AssignmentType


class DatasetSourceAssignment(BaseModel):
    """All runtime allocations made from one official dataset source."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(..., min_length=1)
    allocations: list[SplitAllocation] = Field(..., min_length=1)


class ClassDataAssignment(BaseModel):
    """Canonical split-aware assignment for one requested class."""

    model_config = ConfigDict(extra="forbid")

    class_name: str = Field(..., min_length=1)
    sources: list[DatasetSourceAssignment] = Field(..., min_length=1)


class DatasetSplitCounts(BaseModel):
    """Stable summary shape used by planning, reporting, and the frontend."""

    model_config = ConfigDict(extra="forbid")

    train: int = Field(0, ge=0)
    validation: int = Field(0, ge=0)
    test: int = Field(0, ge=0)


class DatasetAssignmentValidationError(ValueError):
    def __init__(self, findings: list[dict[str, object]]):
        super().__init__("The dataset split assignment is invalid.")
        self.findings = findings


def normalize_dataset_assignments(
    selections: Iterable[ClassDataAssignment | ClassDataSelection | Mapping[str, Any]],
) -> list[ClassDataAssignment]:
    """Return canonical assignments, treating legacy selections as official train.

    Legacy normalization is intentionally conservative: the old contract could
    select only registered training sources, so its count maps to one train
    allocation.  No validation or test samples are invented here.
    """

    normalized: list[ClassDataAssignment] = []
    for value in selections:
        raw = value.model_dump() if isinstance(value, BaseModel) else dict(value)
        if all("allocations" in source for source in raw.get("sources", [])):
            normalized.append(ClassDataAssignment.model_validate(raw))
            continue

        legacy = ClassDataSelection.model_validate(raw)
        normalized.append(ClassDataAssignment(
            class_name=legacy.class_name,
            sources=[
                DatasetSourceAssignment(
                    dataset_name=source.dataset_name,
                    allocations=[SplitAllocation(
                        split=DatasetSplit.TRAIN,
                        count=source.count,
                        assignment_type=AssignmentType.OFFICIAL_SPLIT,
                    )],
                )
                for source in legacy.sources
                if source.count > 0
            ],
        ))
    return normalized


def validate_dataset_assignments(
    assignments: Iterable[ClassDataAssignment],
    source_roles: Mapping[str, str],
    *,
    available_counts: Mapping[tuple[str, str], int] | None = None,
    require_all_splits: bool = True,
) -> list[ClassDataAssignment]:
    """Validate leakage, uniqueness, coverage, and optional availability rules.

    ``source_roles`` deliberately uses plain strings so this schema module does
    not depend on the dataset registry.  The planning layer will provide the
    registry-backed mapping when it adopts this contract in Phase 2.
    """

    values = list(assignments)
    findings: list[dict[str, object]] = []
    seen_classes: set[str] = set()
    for item in values:
        if item.class_name in seen_classes:
            findings.append({"class_name": item.class_name, "reason": "Class appears more than once."})
        seen_classes.add(item.class_name)
        seen_sources: set[str] = set()
        class_splits: set[str] = set()

        for source in item.sources:
            if source.dataset_name in seen_sources:
                findings.append({
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "reason": "Dataset source appears more than once for the class.",
                })
            seen_sources.add(source.dataset_name)
            raw_role = source_roles.get(source.dataset_name)
            role = getattr(raw_role, "value", raw_role)
            if role not in {split.value for split in DatasetSplit}:
                findings.append({
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "reason": "Dataset has no supported official train/validation/test role.",
                })

            seen_allocations: set[str] = set()
            allocated_count = 0
            for allocation in source.allocations:
                split = str(allocation.split)
                assignment_type = str(allocation.assignment_type)
                class_splits.add(split)
                allocated_count += allocation.count
                if split in seen_allocations:
                    findings.append({
                        "class_name": item.class_name,
                        "dataset_name": source.dataset_name,
                        "split": split,
                        "reason": "Source has more than one allocation for the split.",
                    })
                seen_allocations.add(split)

                if assignment_type == AssignmentType.OFFICIAL_SPLIT.value and role != split:
                    findings.append({
                        "class_name": item.class_name,
                        "dataset_name": source.dataset_name,
                        "split": split,
                        "reason": f"Official source role '{role}' does not match assigned split '{split}'.",
                    })
                if assignment_type == AssignmentType.DERIVED_FROM_TRAIN.value and (
                    role != DatasetSplit.TRAIN.value or split == DatasetSplit.TRAIN.value
                ):
                    findings.append({
                        "class_name": item.class_name,
                        "dataset_name": source.dataset_name,
                        "split": split,
                        "reason": "Derived holdouts must assign an official training source to validation or test.",
                    })

            available = (available_counts or {}).get((item.class_name, source.dataset_name))
            if available is not None and allocated_count > available:
                findings.append({
                    "class_name": item.class_name,
                    "dataset_name": source.dataset_name,
                    "allocated_count": allocated_count,
                    "available_count": available,
                    "reason": "Allocated count exceeds availability.",
                })

        if require_all_splits:
            for split in DatasetSplit:
                if split.value not in class_splits:
                    findings.append({
                        "class_name": item.class_name,
                        "split": split.value,
                        "reason": "Required class has no allocation in the split.",
                    })

    if require_all_splits and not values:
        for split in DatasetSplit:
            findings.append({"split": split.value, "reason": "Required split has no allocation."})

    if findings:
        raise DatasetAssignmentValidationError(findings)
    return values


def summarize_dataset_assignments(
    assignments: Iterable[ClassDataAssignment],
) -> tuple[DatasetSplitCounts, DatasetSplitCounts, DatasetSplitCounts]:
    """Return planned, official, and derived counts for an assignment plan."""

    planned = {split.value: 0 for split in DatasetSplit}
    official = {split.value: 0 for split in DatasetSplit}
    derived = {split.value: 0 for split in DatasetSplit}
    for item in assignments:
        for source in item.sources:
            for allocation in source.allocations:
                split = str(allocation.split)
                planned[split] += allocation.count
                target = (
                    official
                    if str(allocation.assignment_type) == AssignmentType.OFFICIAL_SPLIT.value
                    else derived
                )
                target[split] += allocation.count
    return (
        DatasetSplitCounts.model_validate(planned),
        DatasetSplitCounts.model_validate(official),
        DatasetSplitCounts.model_validate(derived),
    )


def planned_split_ratios(state: Mapping[str, Any]) -> dict[str, float] | None:
    """Return compatibility ratios derived from the authoritative split plan."""

    profile = state.get("dataset_profile") or {}
    counts = profile.get("planned_counts") if isinstance(profile, Mapping) else None
    if not isinstance(counts, Mapping):
        return None
    values = {
        "train_data_ratio": int(counts.get("train", 0)),
        "val_data_ratio": int(counts.get("validation", 0)),
        "test_data_ratio": int(counts.get("test", 0)),
    }
    total = sum(values.values())
    if total <= 0 or any(value <= 0 for value in values.values()):
        return None
    return {field: value / total for field, value in values.items()}
