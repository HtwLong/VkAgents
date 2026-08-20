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


def _detection_hpo_model_id(value: str) -> str:
    """Translate selection-layer IDs/aliases to the original HPO registry ID."""
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "yolov8": "yolov8_n", "yolov8n": "yolov8_n",
        "yolov10": "yolov10_n", "yolov10n": "yolov10_n",
        "yolov11": "yolov11_n", "yolo11": "yolov11_n", "yolo11n": "yolov11_n",
        "yolov12": "yolov12_n", "yolov12n": "yolov12_n",
        "retinanet_r50_fpn_1x_coco": "retinanet_r50",
        "faster_rcnn_r50_fpn_1x_coco": "faster_rcnn_r50",
        "ssd300_coco": "ssd300",
    }
    return aliases.get(normalized, normalized)


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
        model_id = _detection_hpo_model_id(str(common["model_name"]))
        is_rtdetr = model_id == "rtdetr_hgnetv2_l" or "rtdetr" in family.replace("-", "")
        is_retinanet = model_id == "retinanet_r50"
        is_faster_rcnn = model_id == "faster_rcnn_r50"
        is_ssd = model_id == "ssd300"
        input_size = 300 if is_ssd else int(
            core.get("image_size") or _recipe_number(recipe, "image_size_default", 640)
        )
        candidate = DetectionHPOConfig.model_validate({
            **common,
            "model_name": model_id,
            "task_type": "detection",
            "input_size": input_size,
            "model_weights": "imagenet_backbone" if is_ssd else "coco",
            "training_recipe_id": (recipe or {}).get("id", ""),
            "scheduler_name": "multistep" if (is_retinanet or is_faster_rcnn or is_ssd) else "linear",
            "loss_box": "l1_giou" if is_rtdetr else "l1" if is_retinanet else "smooth_l1" if (is_faster_rcnn or is_ssd) else "ciou",
            "loss_cls": "varifocal" if is_rtdetr else "focal" if is_retinanet else "cross_entropy" if (is_faster_rcnn or is_ssd) else "bce",
            "lambda_box": 5.0 if is_rtdetr else 1.0 if (is_retinanet or is_faster_rcnn or is_ssd) else 7.5,
            "lambda_giou": 2.0 if is_rtdetr else 0.0,
            "lambda_cls": 1.0 if (is_rtdetr or is_retinanet or is_faster_rcnn or is_ssd) else 0.5,
            "lambda_dfl": 0.0 if (is_rtdetr or is_retinanet or is_faster_rcnn or is_ssd) else 1.5,
            "max_size": 300 if is_ssd else 640 if is_rtdetr else 1333,
            "aspect_ratio_range": None if (is_rtdetr or is_retinanet or is_faster_rcnn or is_ssd) else [0.5, 2.0],
            "augmentation_policy": "ssd" if is_ssd else "basic",
            "amp": not is_rtdetr,
            "nms_iou_threshold": 0.0 if is_rtdetr else 0.7,
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
