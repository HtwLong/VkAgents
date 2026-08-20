"""Explainable dataset strategy derived from VisionKG availability metadata."""

from __future__ import annotations

from typing import Any

from .dataset_planning import availability_candidates
from .data_strategy_contracts import DataPlanConflict, DataStrategy


def build_data_strategy(context: dict[str, Any]) -> dict[str, Any]:
    candidates = availability_candidates(context)
    classes = context.get("classes") or []
    minimum = 500 if context.get("task") == "detection" else 200
    preferred = 2000 if context.get("task") == "detection" else 1000
    objectives = []
    for class_name in classes:
        sources = [source for item in context.get("available_data") or [] if item.get("class_name") == class_name for source in item.get("sources") or []]
        training_count = sum(int(source.get("count") or 0) for source in sources if "train" in str(source.get("dataset_name", "")).casefold())
        objectives.append({
            "class_name": class_name,
            "minimum_positive_images": minimum,
            "preferred_positive_images": preferred,
            "maximum_positive_images": min(5000, max(preferred, training_count)),
            "priority": "high" if training_count < preferred else "standard",
            "rationale": "Coverage target is based on task type and reported VisionKG label counts.",
        })
    decisions = []
    for priority, candidate in enumerate(candidates, 1):
        dataset_id = candidate["dataset_id"]
        role = "external_evaluation" if candidate["dataset_role"] in {"validation", "test"} else "primary" if priority == 1 else "training_supplement"
        decisions.append({
            "dataset_id": dataset_id,
            "role": role,
            "priority": priority,
            "focus_classes": candidate["covered_classes"],
            "activation": "required" if role == "primary" else "if_needed_for_preferred",
            "minimum_images_when_used": 1,
            "allow_derived_validation": candidate["dataset_role"] == "train",
            "allow_derived_test": candidate["dataset_role"] == "train",
            "class_use_overrides": [],
            "rationale": "Ranked from live class coverage, split role, count, and available domain hints.",
        })
    strategy = DataStrategy.model_validate({
        "coverage_objectives": objectives,
        "source_decisions": decisions,
        "split_strategy": {
            "primary_evaluation_domain": str(
                context.get("application_domain") or context.get("use_case_description")
                or "requested deployment domain"
            ),
            "use_official_validation": any(item["dataset_role"] == "validation" for item in candidates),
            "use_official_test": any(item["dataset_role"] == "test" for item in candidates),
            "derive_missing_holdouts": True,
            "group_isolation_keys": ["source_dataset", "original_image_id"],
            "preserve_natural_evaluation_frequency": True,
            "training_balance_policy": "class_aware_sampling",
        },
        "minimum_unique_pool_images": minimum * max(1, len(classes)),
        "preferred_unique_pool_images": preferred * max(1, len(classes)),
        "preferred_target_is_strict": False,
        "acceptable_compromises": ["Use derived holdouts when an official compatible split is unavailable."],
        "rationale": "Metadata-only strategy; no unique-image, overlap, or sample-quality claims are made.",
    })
    return strategy.model_dump(mode="json")


def build_data_plan_conflicts(context: dict[str, Any]) -> list[dict[str, Any]]:
    classes = context.get("classes") or []
    minimum = 500 if context.get("task") == "detection" else 200
    conflicts: list[dict[str, Any]] = []
    for class_name in classes:
        sources = [
            source
            for item in context.get("available_data") or []
            if item.get("class_name") == class_name
            for source in item.get("sources") or []
        ]
        training_count = sum(
            int(source.get("count") or 0)
            for source in sources
            if "train" in str(source.get("dataset_name", "")).casefold()
        )
        if training_count < minimum:
            conflicts.append({
                "code": "INSUFFICIENT_TRAINING_AVAILABILITY",
                "class_name": class_name,
                "reason": "Reported training availability is below the conservative planning minimum.",
                "facts": {"available_count": training_count, "required_count": minimum},
                "available_options": [],
            })
    if context.get("task") == "detection" and len(classes) > 1:
        conflicts.append({
            "code": "MULTILABEL_UNIQUE_COUNT_UNVERIFIED",
            "reason": "Per-class label counts can overlap on the same image; unique pool size requires materialization.",
            "facts": {},
            "available_options": [],
        })
    return [DataPlanConflict.model_validate(item).model_dump(mode="json") for item in conflicts]
