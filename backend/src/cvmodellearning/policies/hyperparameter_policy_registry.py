"""Small, executable policy registry for contextual HPO completion.

Policies are deliberately advisory: recipes and matched ontology rules remain
authoritative.  The registry only guides active fields that a recipe leaves
unspecified, and deterministic schema/runtime validation remains the final
authority.
"""

from __future__ import annotations

from typing import Any

from cvmodellearning.schemas.dataset_assignment import normalize_dataset_assignments
from cvmodellearning.schemas.interpretation_schema import PipelineState


def _data_profile(state: PipelineState) -> dict[str, Any]:
    per_class: dict[str, int] = {}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for assignment in normalize_dataset_assignments(state.selected_data or []):
        class_train = 0
        for source in assignment.sources:
            for allocation in source.allocations:
                split = str(allocation.split)
                split_counts[split] += allocation.count
                if split == "train":
                    class_train += allocation.count
        per_class[assignment.class_name] = class_train

    positive = [count for count in per_class.values() if count > 0]
    minimum = min(positive, default=0)
    maximum = max(positive, default=0)
    return {
        "split_counts": split_counts,
        "num_classes": len(per_class) or len(state.classes or []),
        "train_images_per_class": per_class,
        "min_train_images_per_class": minimum,
        "max_train_images_per_class": maximum,
        "class_imbalance_ratio": round(maximum / minimum, 2) if minimum else None,
    }


def _policy(
    policy_id: str,
    fields: list[str],
    guidance: str,
    evidence_inputs: list[str],
) -> dict[str, Any]:
    return {
        "id": policy_id,
        "fields": fields,
        "guidance": guidance,
        "evidence_inputs": evidence_inputs,
        "authority": "advisory",
    }


def build_hyperparameter_policy_context(state: PipelineState) -> dict[str, Any]:
    """Return a compact problem profile and only the policies applicable to it."""
    data = _data_profile(state)
    profile = {
        "data": data,
        "training_hardware": (
            state.training_hardware.model_dump(mode="json")
            if state.training_hardware
            else None
        ),
        "deployment_hardware": (
            state.available_hardware.model_dump(mode="json")
            if state.available_hardware
            else None
        ),
        "deployment_constraints": (
            state.deployment_constraints.model_dump(mode="json")
            if state.deployment_constraints
            else None
        ),
        "objectives": {
            "primary_metric": getattr(state.performance_requirements, "primary_metric", None),
            "target_value": getattr(state.performance_requirements, "target_value", None),
            "accuracy_category": getattr(state.performance_requirements, "accuracy_category", None),
            "latency_category": getattr(state.performance_requirements, "latency_category", None),
        },
    }
    train_images = data["split_counts"]["train"]
    min_per_class = data["min_train_images_per_class"]
    imbalance = data["class_imbalance_ratio"] or 1.0
    policies: list[dict[str, Any]] = []

    policies.append(_policy(
        "hpo.common.effective_batch.v1",
        [
            "batch_size",
            "gradient_accumulation_steps",
            "optimizer_name",
            "learning_rate",
        ],
        (
            "Choose the largest hardware-safe micro-batch. For classification, use gradient "
            "accumulation only when needed to reach a moderate effective batch; "
            "effective_batch=batch_size*gradient_accumulation_steps*gpu_count. Never exceed "
            "training_hardware.max_batch_size. Preserve a grounded recipe optimizer by default. "
            "For Ultralytics optimizer='auto', do not claim control of learning_rate because "
            "Ultralytics derives and overrides it. A grounded recipe or a specifically justified "
            "LLM decision may instead select a supported explicit optimizer; in that case choose "
            "learning_rate jointly with effective batch size and treat linear batch scaling as a "
            "starting heuristic, not a mandatory formula."
        ),
        [
            "training_hardware.max_batch_size",
            "training_hardware.gpu_count",
            "data.split_counts.train",
            "hyperparameter_graph_context.base_recipe",
        ],
    ))
    policies.append(_policy(
        "hpo.common.schedule_data_size.v1",
        ["num_epochs", "patience", "warmup_epochs"],
        "Choose epochs from dataset size and transfer-learning mode. Smaller datasets may need more epochs but earlier stopping. Keep patience below num_epochs, warmup a small fraction of training, and leave enough post-warmup epochs for learning.",
        ["data.split_counts.train", "data.min_train_images_per_class", "objectives.accuracy_category"],
    ))

    if state.task == "classification":
        policies.append(_policy(
            "hpo.classification.freeze_by_data.v1",
            ["training_mode", "freeze_backbone_epochs", "head_learning_rate_multiplier"],
            (
                "For pretrained models, prefer a short staged head warm-up only for small data "
                f"(current train_images={train_images}, min_per_class={min_per_class}); otherwise fine-tune all layers. "
                "High domain shift, when explicitly known, favors immediate unfreezing. A positive freeze duration requires training_mode='staged_fine_tune' and must be shorter than num_epochs."
            ),
            ["data.split_counts.train", "data.min_train_images_per_class", "model_weights"],
        ))
        policies.append(_policy(
            "hpo.classification.metric_imbalance.v1",
            ["track_metric"],
            f"Use macro_f1 for materially imbalanced multiclass data (current imbalance_ratio={imbalance}); otherwise prefer val_acc unless the user explicitly prioritizes loss calibration.",
            ["data.class_imbalance_ratio", "data.num_classes", "objectives.primary_metric"],
        ))
        policies.append(_policy(
            "hpo.classification.regularization.v1",
            ["label_smoothing", "mixup_alpha", "cutmix_alpha", "random_erasing", "auto_augment_policy", "use_model_ema"],
            "Use conservative regularization for small datasets and stronger augmentation only with enough data. Avoid stacking strong label smoothing, MixUp, and CutMix. Preserve label semantics; when orientation/color/text safety is unknown, keep destructive transforms conservative.",
            ["data.split_counts.train", "data.min_train_images_per_class", "data.num_classes"],
        ))
        policies.append(_policy(
            "hpo.classification.optimization_stability.v1",
            ["gradient_clip_norm", "warmup_start_factor", "scheduler_name", "min_learning_rate"],
            "Use warmup and optional gradient clipping for transformer-style or otherwise unstable full-model fine-tuning. Scheduler parameters must be consistent with num_epochs and learning_rate; cosine minimum LR must remain below the base LR.",
            ["selected_model_info", "num_epochs", "learning_rate"],
        ))
    elif state.task == "detection":
        policies.append(_policy(
            "hpo.detection.resolution_tradeoff.v1",
            ["input_size", "max_size"],
            "Keep architecture-required resolution fixed. Otherwise favor larger inputs for small-object/accuracy priorities and smaller inputs for memory or latency pressure, within recipe and runtime bounds.",
            ["training_hardware.training_memory_budget_gb", "objectives.accuracy_category", "objectives.latency_category"],
        ))
        policies.append(_policy(
            "hpo.detection.transfer_depth.v1",
            ["freeze", "trainable_backbone_layers"],
            (
                "Freezing is valid only with pretrained weights. For YOLO, freeze is the number "
                "of initial layers kept frozen for the whole run, not a number of epochs. For a "
                "small dataset with explicitly supported similarity to the pretraining domain, "
                f"start with freeze=10 (current train_images={train_images}); with uncertain "
                "similarity use 0-5, and with strong domain shift, ample data, or training from "
                "scratch use 0. Treat these as validation experiments, not correctness rules. "
                "TorchVision trainable_backbone_layers has different semantics and must not be "
                "translated mechanically."
            ),
            ["data.split_counts.train", "selected_model_info", "model_weights"],
        ))
        policies.append(_policy(
            "hpo.detection.augmentation_by_data.v1",
            ["mosaic", "mixup", "cutmix", "degrees", "translate", "scale", "fliplr", "hsv_h", "hsv_s", "hsv_v", "close_mosaic"],
            "Use moderate geometry and color augmentation for box detection, scaled to dataset size. Keep semantically unsafe flips/color shifts conservative. Do not enable mask-only copy-paste for box-only data, and close mosaic before the final epoch.",
            ["data.split_counts.train", "data.min_train_images_per_class", "num_epochs"],
        ))
        policies.append(_policy(
            "hpo.detection.postprocess_density.v1",
            ["confidence_threshold", "nms_iou_threshold", "max_detections"],
            "Treat these as safe starting values, not calibrated optima. Favor higher max_detections and less aggressive suppression only when crowded-scene evidence exists; respect NMS-free architectures. Final thresholds should be calibrated on validation data.",
            ["selected_dataset_characteristics", "selected_model_info"],
        ))

    return {
        "profile": profile,
        "applicable_policies": policies,
        "policy_ids": [item["id"] for item in policies],
    }


def policy_fields(policy_context: dict[str, Any]) -> set[str]:
    return {
        field
        for policy in policy_context.get("applicable_policies", [])
        for field in policy.get("fields", [])
    }


def policy_ids_by_field(policy_context: dict[str, Any]) -> dict[str, list[str]]:
    """Return the applicable policy IDs registered for each guided field."""
    result: dict[str, list[str]] = {}
    for policy in policy_context.get("applicable_policies", []):
        policy_id = str(policy.get("id", "")).strip()
        if not policy_id:
            continue
        for field in policy.get("fields", []):
            result.setdefault(str(field), []).append(policy_id)
    return result


def normalize_policy_rationales(
    rationales: list[Any],
    policy_context: dict[str, Any],
) -> list[Any]:
    """Make policy citations match the registry's exact field mapping."""
    allowed_by_field = policy_ids_by_field(policy_context)
    return [
        item.model_copy(
            update={"applied_policy_ids": allowed_by_field.get(str(item.field), [])}
        )
        for item in rationales
    ]


def validate_policy_rationales(
    rationales: list[Any],
    required_fields: set[str],
    policy_context: dict[str, Any],
) -> list[str]:
    """Require applicable policy IDs for policy-guided LLM decisions."""
    allowed_by_field = {
        field: set(policy_ids)
        for field, policy_ids in policy_ids_by_field(policy_context).items()
    }
    by_field = {str(item.field): set(item.applied_policy_ids) for item in rationales}
    errors = []
    for field in sorted(required_fields):
        cited = by_field.get(field, set())
        if not cited:
            errors.append(f"{field} must cite at least one applicable policy ID")
        elif not cited <= allowed_by_field.get(field, set()):
            errors.append(
                f"{field} cites policies not registered for that field: "
                f"{sorted(cited - allowed_by_field.get(field, set()))}"
            )
    return errors
