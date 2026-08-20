"""Materialize rich, non-executable HPO proposals from planning metadata."""

from __future__ import annotations

from typing import Any

from .planning_contracts import ClassificationHPOConfig, DetectionHPOConfig, VQAHPOConfig


def _recipe_number(recipe: dict[str, Any] | None, field: str, fallback: float | int):
    raw = (recipe or {}).get(field)
    if raw in (None, ""):
        return fallback
    number = float(raw)
    return int(number) if isinstance(fallback, int) else number


def materialize_hpo(
    context: dict[str, Any], core: dict[str, Any], recipe: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = context["task"]
    common = {
        "classes": context.get("classes") or [],
        "selected_data": context.get("selected_data") or [],
        "num_epochs": int(core.get("epochs") or _recipe_number(recipe, "epochs_default", 100)),
        "patience": int(_recipe_number(recipe, "patience_default", 20)),
        "batch_size": int(core.get("batch_size") or _recipe_number(recipe, "batch_size_default", 16)),
        "model_name": core["model_name"],
        "optimizer_name": str(core.get("optimizer") or (recipe or {}).get("optimizer") or "adamw").lower(),
        "learning_rate": float(core.get("learning_rate") or _recipe_number(recipe, "learning_rate_default", 0.001)),
        "weight_decay": float(_recipe_number(recipe, "weight_decay_default", 0.0005)),
        "rationale": core.get("rationale") or "Metadata-grounded fine-tuning proposal.",
    }
    if task == "classification":
        candidate = ClassificationHPOConfig.model_validate({
            **common,
            "image_size": int(core.get("image_size") or _recipe_number(recipe, "image_size_default", 224)),
            "track_metric": "val_acc",
            "model_weights": "default",
            "training_mode": "fine_tune_pretrained",
            "training_recipe_id": (recipe or {}).get("id", ""),
            "scheduler_name": "cosine" if "cos" in str((recipe or {}).get("scheduler", "")).lower() else "none",
            "criterion_name": "cross_entropy",
            "llm_field_rationales": [],
        })
    elif task == "detection":
        family = str((context.get("selected_model_info") or {}).get("family") or "").lower()
        is_rtdetr = "rtdetr" in family.replace("-", "")
        candidate = DetectionHPOConfig.model_validate({
            **common,
            "input_size": int(core.get("image_size") or _recipe_number(recipe, "image_size_default", 640)),
            "model_weights": "coco",
            "training_recipe_id": (recipe or {}).get("id", ""),
            "scheduler_name": "linear" if is_rtdetr else "cosine",
            "loss_box": "l1_giou" if is_rtdetr else "ciou_dfl",
            "loss_cls": "varifocal" if is_rtdetr else "bce",
            "lambda_box": 5.0 if is_rtdetr else 7.5,
            "lambda_giou": 2.0 if is_rtdetr else 0.0,
            "lambda_cls": 1.0 if is_rtdetr else 0.5,
            "lambda_dfl": 0.0 if is_rtdetr else 1.5,
            "max_size": int(core.get("image_size") or _recipe_number(recipe, "image_size_default", 640)),
            "llm_field_rationales": [],
        })
    else:
        candidate = VQAHPOConfig.model_validate({
            **common,
            "batch_size": int(core.get("batch_size") or _recipe_number(recipe, "batch_size_default", 2)),
            "optimizer_name": "adamw",
        })
    config = candidate.model_dump(mode="json")
    provenance = {}
    for field in config:
        provenance[field] = {
            "source": "pipeline_state" if field in {"classes", "selected_data", "model_name"}
            else "planner" if field in {"num_epochs", "batch_size", "learning_rate", "optimizer_name", "rationale"}
            else "ontology_recipe" if recipe and field in {"weight_decay", "image_size", "input_size", "training_recipe_id"}
            else "schema_default",
            "recipe_id": (recipe or {}).get("id"),
        }
    return config, provenance
