"""Pure, metadata-only dataset split planning."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def dataset_role(dataset_id: str) -> str:
    value = dataset_id.casefold()
    if re.search(r"(?:^|_)(?:val|validation)(?:_|$)", value):
        return "validation"
    if re.search(r"(?:^|_)test(?:_|$)", value):
        return "test"
    if re.search(r"(?:^|_)train(?:_|$)", value):
        return "train"
    return "benchmark"


def dataset_family(dataset_id: str) -> str:
    value = dataset_id.casefold()
    value = re.sub(r"_(?:det|cls|vqa)?_?(?:train|val|validation|test)$", "", value)
    value = re.sub(r"_(?:train|val|validation|test)$", "", value)
    return value


def availability_candidates(context: dict[str, Any]) -> list[dict[str, Any]]:
    totals: dict[str, int] = defaultdict(int)
    classes: dict[str, set[str]] = defaultdict(set)
    for item in context.get("available_data") or []:
        class_name = str(item.get("class_name") or "")
        for source in item.get("sources") or []:
            dataset_id = str(source.get("dataset_name") or "")
            count = int(source.get("count") or 0)
            if dataset_id and count > 0:
                totals[dataset_id] += count
                classes[dataset_id].add(class_name)
    return sorted(({
        "dataset_id": dataset_id,
        "display_name": dataset_id,
        "dataset_role": dataset_role(dataset_id),
        "available_label_count": total,
        "covered_classes": sorted(classes[dataset_id]),
        "family": dataset_family(dataset_id),
        "source": "VisionKG SPARQL",
    } for dataset_id, total in totals.items()), key=lambda item: (
        item["dataset_role"] != "train", -len(item["covered_classes"]),
        -item["available_label_count"], item["dataset_id"],
    ))


def build_split_assignments(
    context: dict[str, Any], selected_dataset_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build conservative assignments from reported availability only.

    Counts are label-level availability. No claim of distinct-image counts is
    made until a materialized manifest exists.
    """
    allowed = set(selected_dataset_ids or [])
    assignments = []
    selected_counts = []
    for item in context.get("available_data") or []:
        class_name = item.get("class_name")
        sources = [source for source in item.get("sources") or [] if int(source.get("count") or 0) > 0]
        if allowed:
            preferred = [source for source in sources if source.get("dataset_name") in allowed]
            sources = preferred or sources
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source in sources:
            by_role[dataset_role(str(source["dataset_name"]))].append(source)
        for values in by_role.values():
            values.sort(key=lambda source: (-int(source["count"]), str(source["dataset_name"])))
        training = (by_role["train"] or by_role["benchmark"])
        if not training:
            continue
        train_source = training[0]
        available_train = int(train_source["count"])
        target_total = min(2000, available_train)
        validation_source = next((source for source in by_role["validation"] if dataset_family(source["dataset_name"]) == dataset_family(train_source["dataset_name"])), None)
        test_source = next((source for source in by_role["test"] if dataset_family(source["dataset_name"]) == dataset_family(train_source["dataset_name"])), None)
        validation_count = min(200, int(validation_source["count"])) if validation_source else max(1, round(target_total * .1))
        test_count = min(200, int(test_source["count"])) if test_source else max(1, round(target_total * .1))
        train_count = max(1, target_total - (0 if validation_source else validation_count) - (0 if test_source else test_count))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        grouped[train_source["dataset_name"]].append({"split": "train", "count": train_count, "assignment_type": "official_split"})
        if validation_source:
            grouped[validation_source["dataset_name"]].append({"split": "validation", "count": validation_count, "assignment_type": "official_split"})
        else:
            grouped[train_source["dataset_name"]].append({"split": "validation", "count": validation_count, "assignment_type": "derived_from_train"})
        if test_source:
            grouped[test_source["dataset_name"]].append({"split": "test", "count": test_count, "assignment_type": "official_split"})
        else:
            grouped[train_source["dataset_name"]].append({"split": "test", "count": test_count, "assignment_type": "derived_from_train"})
        assignments.append({"class_name": class_name, "sources": [
            {"dataset_name": dataset_id, "allocations": allocations}
            for dataset_id, allocations in grouped.items()
        ]})
        selected_counts.append(train_count + validation_count + test_count)
    if not assignments:
        raise ValueError("No class has a positive VisionKG source suitable for split planning.")
    planned = {
        split: sum(allocation["count"] for item in assignments for source in item["sources"] for allocation in source["allocations"] if allocation["split"] == split)
        for split in ("train", "validation", "test")
    }
    per_class = [sum(allocation["count"] for source in item["sources"] for allocation in source["allocations"]) for item in assignments]
    source_ids = {source["dataset_name"] for item in assignments for source in item["sources"]}
    profile = {
        "status": "planned_from_visionkg_counts_not_materialized",
        "total_selected_images": sum(per_class),
        "minimum_images_per_class": min(per_class),
        "maximum_images_per_class": max(per_class),
        "class_balance_ratio": min(per_class) / max(per_class),
        "number_of_sources": len(source_ids),
        "target_unique_images": max(per_class),
        "verified_unique_images": None,
        "minimum_images_by_class": dict(zip((item["class_name"] for item in assignments), per_class)),
        "verified_images_by_class": {},
        "planned_counts": planned,
        "official_counts": {split: sum(allocation["count"] for item in assignments for source in item["sources"] for allocation in source["allocations"] if allocation["split"] == split and allocation["assignment_type"] == "official_split") for split in planned},
        "derived_counts": {split: sum(allocation["count"] for item in assignments for source in item["sources"] for allocation in source["allocations"] if allocation["split"] == split and allocation["assignment_type"] == "derived_from_train") for split in planned},
        "limitations": ["VisionKG counts may be label counts rather than deduplicated images.", "Unique-image and class-overlap checks require later materialization and are not claimed by viewer mode."],
    }
    return assignments, profile


def preprocessing_plan(context: dict[str, Any]) -> dict[str, Any]:
    robustness = context.get("robustness_requirements") or {}
    task = context.get("task")
    plan = {
        "task": task,
        "resize": "letterbox_preserve_aspect_ratio" if task == "detection" else "resize_and_crop",
        "normalization": "use selected pretrained model defaults",
        "horizontal_flip": False if robustness.get("text_or_symbols_present") or robustness.get("horizontal_flip_safe") is False else True,
        "augmentations": [],
        "materialization_status": "planned_not_executed",
    }
    if robustness.get("lighting"):
        plan["augmentations"].append("conservative_brightness_contrast")
    if robustness.get("motion_blur"):
        plan["augmentations"].append("light_motion_blur")
    if "small" in robustness.get("object_scale", []):
        plan["augmentations"].append("scale_aware_crop_with_box_retention")
    return plan
