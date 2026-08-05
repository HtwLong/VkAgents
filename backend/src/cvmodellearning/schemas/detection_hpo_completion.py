from __future__ import annotations

from typing import Any, Mapping

from cvmodellearning.schemas.dataset_assignment import planned_split_ratios
from cvmodellearning.schemas.detection_hpo import (
    DetectionConfigDraft,
    DetectionConfigModel,
    detection_runtime_family,
)


def complete_detection_config(
    draft: DetectionConfigDraft,
    state: Mapping[str, Any],
    model_name: str,
) -> tuple[DetectionConfigModel, list[dict[str, Any]]]:
    """Apply pipeline- and runtime-owned values before strict validation."""
    config = draft.model_dump(mode="json")
    adjustments: list[dict[str, Any]] = []

    def apply(field: str, value: Any, reason: str) -> None:
        previous = config.get(field)
        if previous == value:
            return
        config[field] = value
        adjustments.append({
            "field": field,
            "previous": previous,
            "applied": value,
            "reason": reason,
        })

    apply("model_name", model_name, "Model selection owns the executable model identifier.")
    apply("classes", list(state.get("classes") or []), "Task interpretation owns class order.")
    apply(
        "selected_data",
        list(state.get("selected_data") or []),
        "Dataset selection owns source and split assignments.",
    )
    for field, ratio in (planned_split_ratios(state) or {}).items():
        apply(field, ratio, "Derived from the authoritative dataset assignment plan.")

    if not bool(state.get("use_graphrag", True)):
        apply("training_recipe_id", "", "Recipe provenance is empty when GraphRAG is disabled.")

    runtime_family = detection_runtime_family(model_name)
    if runtime_family in {"yolo", "rtdetr"}:
        apply(
            "single_cls",
            len(state.get("classes") or []) == 1,
            "Derived deterministically from the authoritative task class count.",
        )

    if runtime_family == "yolo":
        fixed_values = {
            "loss_box": "ciou",
            "loss_cls": "bce",
            "copy_paste": 0.0,
            "track_metric": "val_mAP",
            "scheduler_name": "linear",
        }
        if config.get("optimizer_name") == "auto":
            fixed_values.update({"learning_rate": 0.01, "momentum": 0.9})
        for field, value in fixed_values.items():
            apply(field, value, "Fixed by the executable Ultralytics YOLO contract.")

    return DetectionConfigModel.model_validate(config), adjustments
