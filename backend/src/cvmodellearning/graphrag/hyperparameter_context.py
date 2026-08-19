from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import networkx as nx
from pydantic import TypeAdapter

from cvmodellearning.graphrag.build_graph import build_graph
from cvmodellearning.models.registry import (
    LORA_CLASSIFICATION_MODEL_IDS,
    resolve_detection_model_identity,
)
from cvmodellearning.paths import PROJECT_ROOT
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.graphrag.dataset_selection_context import (
    aggregate_selected_dataset_properties,
)
from cvmodellearning.training.resource_guard import rank_training_shape_candidates


USE_HYPERPARAMETER_GRAPHRAG = os.getenv("USE_HYPERPARAMETER_GRAPHRAG", "true").lower() not in {
    "0", "false", "no", "off",
}

TASK_IDS = {
    "classification": "image_classification",
    "detection": "object_detection",
    "visual question answering": "visual_question_answering",
}

# Family edges intentionally expose shared recipes, but these recipes carry
# checkpoints/configurations for one concrete execution model only. Keep the
# constraint close to retrieval until the ontology CSV schema gains an
# applicable_model_ids column.
RECIPE_MODEL_ID_CONSTRAINTS = {
    "rtdetr_objects365_to_coco_finetune": {
        "rtdetr_r18",
        "rtdetr_r34",
        "rtdetr_r50_m",
        "rtdetr_r50",
        "rtdetr_r101",
    },
    "timm_dinov2_s_b14_adapted_custom_finetune": {
        "dinov2_vits14",
        "dinov2_vitb14",
    },
    "torchvision_swin_v2_t_imagenet_v1_adapted_custom_finetune": {"swin_v2_t"},
    "torchvision_swin_v2_s_imagenet_v1_adapted_custom_finetune": {"swin_v2_s"},
    "torchvision_retinanet_resnet50_fpn_coco_pretrained_custom_finetune": {
        "retinanet_resnet50_fpn"
    },
    "torchvision_fasterrcnn_resnet50_fpn_coco_pretrained_custom_finetune": {
        "fasterrcnn_resnet50_fpn"
    },
    "torchvision_ssd300_vgg16_imagenet_backbone_custom_training": {
        "ssd300_vgg16"
    },
    "ultralytics_rtdetr_l_coco_pretrained_custom_finetune": {
        "rtdetr_hgnetv2_l"
    },
    "ultralytics_rtdetr_l_coco_pretrained_custom_finetune_high_throughput": {
        "rtdetr_hgnetv2_l"
    },
}

# Hardware recommendations are deliberately scoped to the exact graph model
# whose executable recipe and memory metadata justify the suggested values.
# Family edges keep the recommendations discoverable, while this constraint
# prevents a future larger backbone from inheriting an unsafe batch size.
RULE_MODEL_ID_CONSTRAINTS = {
    "rule_fasterrcnn_low_memory_batch_lr": {"fasterrcnn_resnet50_fpn"},
    "rule_fasterrcnn_high_memory_batch_lr": {"fasterrcnn_resnet50_fpn"},
    "rule_retinanet_low_memory_batch_lr": {"retinanet_resnet50_fpn"},
    "rule_retinanet_high_memory_batch_lr": {"retinanet_resnet50_fpn"},
    "rule_rtdetr_l_low_memory_batch": {"rtdetr_hgnetv2_l"},
    "rule_rtdetr_l_high_memory_batch": {"rtdetr_hgnetv2_l"},
    "rule_ssd300_low_memory_batch": {"ssd300_vgg16"},
    "rule_ssd300_high_memory_batch": {"ssd300_vgg16"},
}

RECIPE_ALLOWED_TRAINING_MODES = {
    "timm_dinov2_s_b14_adapted_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "timm_clip_vit_b16_openai_adapted_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "torchvision_mobilenetv2_imagenet_v2_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "torchvision_mobilenetv3_imagenet_pretrained_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "torchvision_efficientnet_imagenet_pretrained_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "densenet121_imagenet_v1_adapted_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "torchvision_resnet50_imagenet_pretrained_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "torchvision_convnext_tiny_imagenet_v1_adapted_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "torchvision_vit_b16_imagenet_pretrained_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "torchvision_swin_v2_t_imagenet_v1_adapted_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
    "torchvision_swin_v2_s_imagenet_v1_adapted_custom_finetune": {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
    },
}

DETECTION_EXECUTION_TO_GRAPH_MODEL_IDS = {
    "retinanet_r50": "retinanet_resnet50_fpn",
    "faster_rcnn_r50": "fasterrcnn_resnet50_fpn",
    "ssd300": "ssd300_vgg16",
    "rtdetr_hgnetv2_l": "rtdetr_hgnetv2_l",
}


def _graph_model_id_for_recipe_validation(model_id: str) -> str:
    """Map execution-only detector IDs to the corresponding ontology node."""
    mapped = DETECTION_EXECUTION_TO_GRAPH_MODEL_IDS.get(model_id)
    if mapped:
        return mapped
    if re.fullmatch(r"yolov(?:8|10|11|12)_[nslmx]", model_id):
        mapped = model_id.replace("_", "")
        return mapped.replace("yolov11", "yolo11").replace("yolov12", "yolo12")
    return model_id

# The shared EfficientNet recipe covers B0-B7, whose pretrained weights use
# different native resolutions. Resolve that one executable field from the
# selected weight metadata instead of duplicating the otherwise identical
# fine-tuning recipe eight times.
NATIVE_WEIGHT_IMAGE_SIZE_RECIPE_IDS = {
    "torchvision_efficientnet_imagenet_pretrained_custom_finetune",
}

NON_EXECUTABLE_RECIPE_IDS: set[str] = {
    # The application executes TorchVision, not Detectron2. The original
    # TorchVision row describes COCO training with a pretrained backbone rather
    # than custom-data fine-tuning from detector weights.
    "detectron2_retinanet_r50_fpn_coco_pretrained_finetune",
    "torchvision_retinanet_resnet50_fpn_coco_training",
    # These rows remain useful reference/Detectron2 provenance. The local
    # executor uses the explicitly adapted TorchVision custom-data recipe.
    "torchvision_fasterrcnn_resnet50_fpn_coco_pretrained_finetune",
    "detectron2_fasterrcnn_r50_fpn_coco_pretrained_finetune",
    # Meta's 518px linear evaluation uses its own multi-layer classifier
    # protocol; it is benchmark provenance, not this pipeline's simple head.
    "meta_dinov2_imagenet1k_linear_eval",
    # This recipe is executable through Hugging Face Trainer, not through the
    # timm/raw-PyTorch classification pipeline registered here.
    "hf_dinov2_image_classification_finetune",
    # Retain the published 85.7% recipe as benchmark evidence. It requires exact
    # RandAugment, layer-wise LR decay and distributed effective batch 2048,
    # which the minimal custom-data executor does not faithfully reproduce.
    "ftclip_clip_vit_b16_imagenet1k_finetune",
    # This row is valuable checkpoint/benchmark provenance, but it does not
    # publish the optimizer or schedule and requires a SWAG-specific 384px
    # weight path that the current executor does not expose.
    "swag_vit_b16_imagenet1k_e2e_finetune",
    # The executor's MobileNet V2 `default` weight alias is ImageNet V2. Keep
    # the V1 adaptation as ontology evidence, but do not allow V1 provenance
    # to be attached to a configuration that actually loads V2 weights.
    "torchvision_mobilenetv2_imagenet_v1_custom_finetune",
    # Preserve the sourced QAT recipe and benchmark in the graph, but the
    # classification factory currently exposes only floating-point default or
    # uninitialized weights and has no quantized/QNNPACK execution path.
    "torchvision_mobilenetv2_imagenet_qnnpack_qat",
}

NON_EXECUTABLE_RULE_IDS = {
    # RetinaNet currently executes the documented TorchVision defaults. These
    # rules either target Detectron2 fields or require an explicit scalar policy
    # that this single-shot planner cannot infer safely from PipelineState.
    "rule_retinanet_focal_loss_for_class_imbalance",
    "rule_retinanet_low_vram_batch_lr_scaling",
    "rule_retinanet_accuracy_first_input_scale",
    "rule_retinanet_latency_first_input_scale",
    "rule_retinanet_freeze_backbone_on_small_dataset",
    "rule_detectron2_retinanet_set_num_classes",
    "rule_detectron2_retinanet_scale_lr_batch",
    "rule_detectron2_retinanet_longer_schedule_for_accuracy",
    "rule_detectron2_retinanet_increase_detections_crowded",
    "rule_detectron2_one_gpu_batch_lr_exact",
    "rule_detectron2_crowded_set_detections_per_image_to_expected_max",
    # Faster R-CNN class replacement is deterministic in the model factory;
    # the remaining rules target Detectron2 or lack an exact scalar policy.
    "rule_fasterrcnn_set_num_classes_torchvision",
    "rule_detectron2_fasterrcnn_set_num_classes",
    "rule_fasterrcnn_low_vram_batch_lr_scaling",
    "rule_fasterrcnn_longer_schedule_for_accuracy",
    "rule_fasterrcnn_increase_detections_crowded",
    "rule_fasterrcnn_freeze_backbone_on_small_dataset",
    "rule_fasterrcnn_accuracy_first_rpn_proposals",
    "rule_domain_traffic_surveillance_offline_accuracy_detector",
    "rule_domain_ecommerce_shelf_detection_dense_objects",
    # The application intentionally produces a single configuration rather than
    # launching HPO, and epoch-bounded runs do not expose Ultralytics `time`.
    "rule_yolo_hyperparameter_tuning",
    "rule_yolo_time_constrained_training",
    # Not accepted by the installed Ultralytics 8.4 configuration schema.
    "rule_class_imbalance_cls_pw",
    # Formula/post-training deployment policies are not scalar planning fields.
    "rule_linear_scaling_lr0",
    "rule_yolo_realtime_stream_buffer_false",
    "rule_yolo_tta_augment_true",
    # This mixed classification/detection example includes auto_augment and
    # erasing, which are not part of the detector transform contract here.
    "rule_yolo_disable_all_augmentations_label_sensitive",
    # The downloaded detection data contains boxes, not segmentation masks.
    "rule_small_dataset_augment",
    # These describe Hugging Face Trainer fields, automatic factory behavior,
    # or model-selection choices rather than scalar executor adjustments.
    "rule_dinov2_custom_num_labels_classifier",
    "rule_dinov2_trainer_keep_image_columns",
    "rule_dinov2_accuracy_first_consider_vitg14_or_registers",
    # Published FT-CLIP reproduction guidance retained as evidence. These rules
    # require layer decay, exact RandAugment, drop path, distributed effective
    # batch semantics, or fields the minimal classifier schema does not expose.
    "rule_clip_finetune_with_layer_decay",
    "rule_clip_finetune_effective_batch_2048",
    "rule_clip_disable_mixup_cutmix_drop_path",
    "rule_clip_use_randaug_label_smoothing_ema",
    # These remain useful ontology facts, but the current PipelineState cannot
    # prove their conditions and the ImageNet pretraining LR rule must not be
    # applied to the custom-data fine-tuning baseline.
    "rule_resnet50_finetune_all_layers_domain_shift",
    "rule_resnet50_low_vram_batch_lr_scaling",
    "rule_resnet50_accuracy_first_use_v2_recipe",
    # These remain useful model-selection or official ResNet RT-DETR facts,
    # but they are not executable adjustments for the registered Ultralytics
    # HGNetV2-L custom-data recipe.
    "rule_rtdetr_accuracy_first_variant",
    "rule_rtdetr_latency_first_decoder_or_variant",
    "rule_rtdetr_low_vram_reduce_batch",
    "rule_rtdetr_use_objects365_pretraining_for_accuracy",
    "rule_rtdetr_use_official_multiscale_list",
    "rule_rtdetr_use_10x_lower_backbone_lr",
    "rule_rtdetr_set_official_batch_sizes",
    "rule_domain_autonomous_driving_realtime_detector",
}

RECIPE_FIELD_TO_EXECUTABLE_FIELD = {
    "training_mode": "training_mode",
    "pretrained": "model_weights",
    "optimizer": "optimizer_name",
    "scheduler": "scheduler_name",
    "precision": "precision",
    "learning_rate_default": "learning_rate",
    "batch_size_default": "batch_size",
    "epochs_default": "num_epochs",
    "weight_decay_default": "weight_decay",
    "image_size_default": "image_size",
    "patience_default": "patience",
    "warmup_epochs_default": "warmup_epochs",
    "momentum_default": "momentum",
    "gradient_accumulation_default": "gradient_accumulation_steps",
    "freeze_default": "freeze_backbone_epochs",
}

RECIPE_DETAIL_TO_EXECUTABLE_FIELD = {
    "label_smoothing_default": "label_smoothing",
    "mixup_alpha_default": "mixup_alpha",
    "cutmix_alpha_default": "cutmix_alpha",
    "auto_augment_policy": "auto_augment_policy",
    "random_erasing_default": "random_erasing",
    "train_crop_size": "image_size",
}

DETECTION_RECIPE_FIELD_TO_EXECUTABLE_FIELD = {
    "optimizer": "optimizer_name",
    "scheduler": "scheduler_name",
    "pretrained": "model_weights",
    "learning_rate_default": "learning_rate",
    "batch_size_default": "batch_size",
    "epochs_default": "num_epochs",
    "weight_decay_default": "weight_decay",
    "image_size_default": "input_size",
    "patience_default": "patience",
    "warmup_epochs_default": "warmup_epochs",
    "momentum_default": "momentum",
}

DETECTION_RECIPE_DETAIL_TO_EXECUTABLE_FIELD = {
    "confidence_threshold_default": "confidence_threshold",
    "nms_iou_threshold_default": "nms_iou_threshold",
    "mosaic_default": "mosaic",
    "mixup_default": "mixup",
    "copy_paste_default": "copy_paste",
    "degrees_default": "degrees",
    "hsv_h_default": "hsv_h",
    "hsv_s_default": "hsv_s",
    "hsv_v_default": "hsv_v",
    "close_mosaic_default": "close_mosaic",
}


def _executable_recipe_value(field: str, value: Any) -> Any:
    if field == "training_mode":
        return {
            "finetunepretrained": "fine_tune_pretrained",
            "trainfromscratch": "train_from_scratch",
            "headonly": "head_only",
            "stagedfinetune": "staged_fine_tune",
        }.get(str(value).lower(), value)
    if field == "pretrained":
        normalized = str(value).lower()
        if normalized == "true":
            return "default"
        if normalized == "false":
            return "none"
        return value
    if field == "optimizer":
        return str(value).lower()
    if field == "scheduler":
        return {
            "cosineannealinglr": "cosine",
            "steplr": "step",
            "none": "none",
        }.get(str(value).lower(), value)
    if field == "precision" and str(value).lower() == "mixed_precision":
        return "mixed"
    if field == "freeze_default" and str(value).lower() == "false":
        return 0
    if field == "gradient_accumulation_default" and str(value) == "0":
        # Ontology rows use 0 to mean disabled; the executable schema expresses
        # the same behavior as one optimizer step per micro-batch.
        return 1
    return value


def _coerce_classification_config_value(field_name: str, value: Any) -> Any:
    """Coerce an ontology scalar with the same type rules as the HPO schema."""
    from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel

    field = ClassificationConfigModel.model_fields.get(field_name)
    if field is None:
        raise ValueError(f"Ontology field does not map to an executable classification field: {field_name}")
    adapter = TypeAdapter(field.rebuild_annotation())
    parsed = adapter.validate_python(value)
    return adapter.dump_python(parsed, mode="json")


def _coerce_detection_config_value(field_name: str, value: Any) -> Any:
    from cvmodellearning.schemas.detection_hpo import DetectionConfigModel

    field = DetectionConfigModel.model_fields.get(field_name)
    if field is None:
        raise ValueError(f"Ontology field does not map to an executable detection field: {field_name}")
    adapter = TypeAdapter(field.rebuild_annotation())
    parsed = adapter.validate_python(value)
    return adapter.dump_python(parsed, mode="json")


def _materialize_detection_recipe_config(context: dict[str, Any]) -> dict[str, Any]:
    from cvmodellearning.schemas.detection_hpo import DetectionConfigModel

    recipe = context.get("base_recipe") or {}
    if recipe.get("task_id") != "object_detection":
        return {}

    recipe_id = str(recipe.get("id", ""))
    materialized: dict[str, Any] = {"training_recipe_id": recipe_id}
    sources: dict[str, dict[str, str]] = {
        "training_recipe_id": {"source": "recipe", "source_id": recipe_id}
    }
    warnings: list[dict[str, Any]] = []
    for ontology_field, executable_field in DETECTION_RECIPE_FIELD_TO_EXECUTABLE_FIELD.items():
        value = recipe.get(ontology_field)
        if value in {"", None, "unknown"}:
            continue
        if ontology_field == "pretrained":
            value = "coco" if str(value).lower() == "true" else "none"
        elif ontology_field == "optimizer":
            value = str(value).lower()
        elif ontology_field == "scheduler":
            value = {"linearlr": "linear", "multisteplr": "multistep", "none": "none"}.get(
                str(value).lower(), str(value).lower()
            )
        try:
            materialized[executable_field] = _coerce_detection_config_value(executable_field, value)
            sources[executable_field] = {"source": "recipe", "source_id": recipe_id}
        except ValueError as exc:
            warnings.append({
                "source": "base_recipe",
                "ontology_field": ontology_field,
                "value": value,
                "critical": ontology_field in {
                    "pretrained", "optimizer", "learning_rate_default",
                    "batch_size_default", "epochs_default", "image_size_default",
                },
                "reason": str(exc),
            })

    if materialized.get("optimizer_name") == "auto":
        # Ultralytics AutoOptimizer derives both values from the run and ignores
        # lr0/momentum. Do not present them as graph-controlled execution fields.
        for inactive_field in ("learning_rate", "momentum"):
            materialized.pop(inactive_field, None)
            sources.pop(inactive_field, None)

    details = (context.get("recipe_details") or [{}])[0]
    detail_id = str(details.get("id", ""))
    for ontology_field, executable_field in DETECTION_RECIPE_DETAIL_TO_EXECUTABLE_FIELD.items():
        value = details.get(ontology_field)
        if value in {"", None}:
            continue
        try:
            materialized[executable_field] = _coerce_detection_config_value(executable_field, value)
            sources[executable_field] = {"source": "recipe_details", "source_id": detail_id}
        except ValueError as exc:
            warnings.append({
                "source": "recipe_details",
                "ontology_field": ontology_field,
                "value": value,
                "critical": False,
                "reason": str(exc),
            })

    precision = str(recipe.get("precision", "")).lower()
    if precision in {"mixed_precision", "fp32"}:
        materialized["amp"] = precision == "mixed_precision"
        sources["amp"] = {"source": "recipe", "source_id": recipe_id}

    for parameter in context.get("recipe_parameters") or []:
        field_name = str(parameter.get("param_name", "")).strip()
        value = parameter.get("param_value")
        if field_name not in DetectionConfigModel.model_fields or value in {"", None}:
            continue
        if field_name == "lr_milestones" and isinstance(value, str):
            value = [int(item) for item in value.split("|") if item]
        try:
            materialized[field_name] = _coerce_detection_config_value(field_name, value)
            sources[field_name] = {
                "source": "recipe_parameter",
                "source_id": str(parameter.get("id", "")),
            }
        except ValueError as exc:
            warnings.append({
                "source": "recipe_parameters",
                "ontology_field": field_name,
                "value": value,
                "critical": False,
                "reason": str(exc),
            })

    selected_family = str((context.get("selected_model") or {}).get("model_family", ""))
    if selected_family.startswith("YOLO"):
        fixed_values = {
            "loss_box": "ciou",
            "loss_cls": "bce",
            "lambda_box": 7.5,
            "lambda_cls": 0.5,
            "lambda_dfl": 1.5,
            "copy_paste": 0.0,
            "track_metric": "val_mAP",
            "scheduler_name": "linear",
        }
        for field_name, value in fixed_values.items():
            materialized[field_name] = value
            sources[field_name] = {
                "source": "system_policy",
                "source_id": "ultralytics_yolo_execution_contract",
            }
    elif selected_family == "RetinaNet":
        fixed_values = {
            "loss_box": "l1",
            "loss_cls": "focal",
            "lambda_box": 1.0,
            "lambda_cls": 1.0,
            "lambda_dfl": 0.0,
            "aspect_ratio_range": None,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "fliplr": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "close_mosaic": 0,
            "multi_scale": 0.0,
        }
        for field_name, value in fixed_values.items():
            materialized[field_name] = value
            sources[field_name] = {
                "source": "system_policy",
                "source_id": "torchvision_retinanet_loss_contract",
            }
    elif selected_family == "Faster R-CNN":
        fixed_values = {
            "loss_box": "smooth_l1",
            "loss_cls": "cross_entropy",
            "lambda_box": 1.0,
            "lambda_cls": 1.0,
            "lambda_dfl": 0.0,
            "aspect_ratio_range": None,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "fliplr": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "close_mosaic": 0,
            "multi_scale": 0.0,
        }
        for field_name, value in fixed_values.items():
            materialized[field_name] = value
            sources[field_name] = {
                "source": "system_policy",
                "source_id": "torchvision_fasterrcnn_loss_and_transform_contract",
            }
    elif selected_family == "SSD":
        fixed_values = {
            "model_weights": "imagenet_backbone",
            "loss_box": "smooth_l1",
            "loss_cls": "cross_entropy",
            "lambda_box": 1.0,
            "lambda_cls": 1.0,
            "lambda_dfl": 0.0,
            "input_size": 300,
            "max_size": 300,
            "aspect_ratio_range": None,
            "augmentation_policy": "ssd",
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "fliplr": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "close_mosaic": 0,
            "multi_scale": 0.0,
        }
        for field_name, value in fixed_values.items():
            materialized[field_name] = value
            sources[field_name] = {
                "source": "system_policy",
                "source_id": "torchvision_ssd300_custom_class_contract",
            }
    elif selected_family == "RT-DETR":
        fixed_values = {
            "loss_box": "l1_giou",
            "loss_cls": "varifocal",
            "lambda_box": 5.0,
            "lambda_giou": 2.0,
            "lambda_cls": 1.0,
            "lambda_dfl": 0.0,
            "input_size": 640,
            "max_size": 640,
            "aspect_ratio_range": None,
            "rect": False,
            "copy_paste": 0.0,
            "multi_scale": 0.0,
            "nms_iou_threshold": 0.0,
            "max_detections": 300,
            "amp": False,
            "freeze": 0,
            "lr_milestones": [],
            "scheduler_gamma": 1.0,
            "augmentation_policy": "basic",
            "trainable_backbone_layers": 0,
            "horizontal_flip_probability": 0.0,
            "topk_candidates": 400,
            "positive_fraction": 0.25,
            "matching_iou_threshold": 0.5,
        }
        for field_name, value in fixed_values.items():
            materialized[field_name] = value
            sources[field_name] = {
                "source": "system_policy",
                "source_id": "ultralytics_rtdetr_l_execution_contract",
            }

    context["materialization_warnings"] = warnings
    context["critical_materialization_errors"] = [item for item in warnings if item["critical"]]
    context["base_field_provenance"] = sources
    context["fields_requiring_llm_completion"] = ["model_name"]
    return materialized


def _native_weight_image_size(model_id: str) -> int:
    """Read the authoritative crop size without constructing the model."""
    from cvmodellearning.models.classification_model_utils import get_model_weights

    weights = get_model_weights(model_id, "default")
    return int(weights.transforms().crop_size[0])


def _normalize_inactive_classification_fields(
    config: dict[str, Any],
    field_sources: dict[str, dict[str, str]],
) -> None:
    """Backward-compatible wrapper around the shared completion policy."""
    from cvmodellearning.schemas.classification_hpo_completion import (
        normalize_inactive_classification_fields,
    )

    normalize_inactive_classification_fields(config, field_sources)


def materialize_base_recipe_config(context: dict[str, Any]) -> dict[str, Any]:
    """Convert the selected graph recipe into a typed, executable partial config."""
    recipe = context.get("base_recipe")
    selected_model = context.get("selected_model") or {}
    if recipe and recipe.get("task_id") == "object_detection":
        return _materialize_detection_recipe_config(context)
    if not recipe or recipe.get("task_id") != "image_classification":
        return {}

    materialized: dict[str, Any] = {
        "model_name": selected_model.get("id"),
        "training_recipe_id": recipe.get("id"),
        "criterion_name": "cross_entropy",
    }
    field_sources: dict[str, dict[str, str]] = {
        "model_name": {"source": "selected_model", "source_id": selected_model.get("id", "")},
        "training_recipe_id": {"source": "recipe", "source_id": recipe.get("id", "")},
        "criterion_name": {"source": "system_policy", "source_id": "single_label_classification"},
    }
    completion_fields: set[str] = set()
    warnings: list[dict[str, Any]] = []
    critical_fields = {
        "training_mode",
        "pretrained",
        "optimizer",
        "scheduler",
        "learning_rate_default",
        "batch_size_default",
        "epochs_default",
        "image_size_default",
    }
    for ontology_field, executable_field in RECIPE_FIELD_TO_EXECUTABLE_FIELD.items():
        value = recipe.get(ontology_field)
        if value in {"", None}:
            continue
        if str(value).strip().lower() == "unknown":
            completion_fields.add(executable_field)
            continue
        try:
            materialized[executable_field] = _coerce_classification_config_value(
                executable_field,
                _executable_recipe_value(ontology_field, value),
            )
            field_sources[executable_field] = {
                "source": "recipe",
                "source_id": recipe.get("id", ""),
            }
        except ValueError as exc:
            warnings.append({
                "source": "base_recipe",
                "ontology_field": ontology_field,
                "value": value,
                "critical": ontology_field in critical_fields,
                "reason": str(exc),
            })

    details = context.get("recipe_details") or []
    if details:
        for ontology_field, executable_field in RECIPE_DETAIL_TO_EXECUTABLE_FIELD.items():
            # Scalar top-level recipe defaults are authoritative. Details may
            # retain family-wide values such as pipe-delimited variant sizes
            # for retrieval without overriding an executable scalar.
            if executable_field in materialized:
                continue
            value = details[0].get(ontology_field)
            if value in {"", None}:
                continue
            try:
                materialized[executable_field] = _coerce_classification_config_value(
                    executable_field,
                    value,
                )
                field_sources[executable_field] = {
                    "source": "recipe_details",
                    "source_id": details[0].get("id", ""),
                }
            except ValueError as exc:
                warnings.append({
                    "source": "recipe_details",
                    "ontology_field": ontology_field,
                    "value": value,
                    "critical": False,
                    "reason": str(exc),
                })

    if recipe.get("id") in NATIVE_WEIGHT_IMAGE_SIZE_RECIPE_IDS:
        model_id = str(selected_model.get("id", ""))
        try:
            materialized["image_size"] = _native_weight_image_size(model_id)
            field_sources["image_size"] = {
                "source": "pretrained_weight_metadata",
                "source_id": model_id,
            }
        except ValueError as exc:
            warnings.append({
                "source": "pretrained_weight_metadata",
                "ontology_field": "image_size",
                "value": model_id,
                "critical": True,
                "reason": str(exc),
            })

    from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
    executable_fields = ClassificationConfigModel.model_fields
    for parameter in context.get("recipe_parameters") or []:
        field_name = str(parameter.get("param_name", "")).strip()
        value = parameter.get("param_value")
        if not field_name or value in {"", None}:
            continue
        if field_name not in executable_fields:
            # Structural recipe metadata such as model.fc is consumed by the
            # model factory, not copied into the scalar HPO configuration.
            continue
        try:
            materialized[field_name] = _coerce_classification_config_value(field_name, value)
            field_sources[field_name] = {
                "source": "recipe_parameter",
                "source_id": parameter.get("id", ""),
            }
        except ValueError as exc:
            warnings.append({
                "source": "recipe_parameters",
                "ontology_field": field_name,
                "value": value,
                "critical": False,
                "reason": str(exc),
            })

    _normalize_inactive_classification_fields(materialized, field_sources)

    context["materialization_warnings"] = warnings
    context["critical_materialization_errors"] = [
        warning for warning in warnings if warning["critical"]
    ]
    context["base_field_provenance"] = field_sources
    context["fields_requiring_llm_completion"] = sorted(completion_fields)

    return materialized


_PIPELINE_CONTEXT_FIELDS = {
    "classes",
    "selected_data",
    "train_data_ratio",
    "val_data_ratio",
    "test_data_ratio",
}
_SPLIT_COMPATIBILITY_FIELDS = {
    "train_data_ratio",
    "val_data_ratio",
    "test_data_ratio",
}
_EXPLANATION_FIELDS = {"rationale", "llm_field_rationales", "field_provenance"}


def llm_controlled_fields(
    config: dict[str, Any],
    context: dict[str, Any],
    schema_model: type,
) -> set[str]:
    """Return non-grounded fields that need a field-specific LLM rationale."""
    controlled = set(context.get("fields_requiring_llm_completion") or [])
    reference_configuration = context.get("reference_configuration") or {}
    grounded_fields = set(reference_configuration)
    # An allowed departure from a grounded baseline is still an LLM decision
    # and must carry field-specific rationale/provenance.
    for field_name in set(context.get("allowed_adjustment_fields") or []):
        expected = reference_configuration.get(field_name)
        if field_name in config and config.get(field_name) != expected:
            controlled.add(field_name)
    for field_name, value in config.items():
        if (
            field_name in grounded_fields
            or field_name in _PIPELINE_CONTEXT_FIELDS
            or field_name in _EXPLANATION_FIELDS
            or field_name == "task_type"
        ):
            continue
        schema_field = schema_model.model_fields.get(field_name)
        if schema_field is None:
            continue
        if schema_field.is_required():
            controlled.add(field_name)
            continue
        default = schema_field.get_default(call_default_factory=True)
        if value != default:
            controlled.add(field_name)
    return controlled


def build_field_provenance(
    config: dict[str, Any],
    context: dict[str, Any],
    *,
    llm_adjusted_fields: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Describe the deterministic origin of every saved HPO field."""
    from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
    from cvmodellearning.schemas.detection_hpo import DetectionConfigModel

    recipe_task = str((context.get("base_recipe") or {}).get("task_id", ""))
    schema_model = DetectionConfigModel if recipe_task == "object_detection" else ClassificationConfigModel

    adjusted = llm_adjusted_fields or set()
    base_sources = context.get("base_field_provenance") or {}
    rule_sources = context.get("adjustment_rule_provenance") or {}
    hardware_sources = context.get("training_hardware_adjustment_provenance") or {}
    completion_fields = set(context.get("fields_requiring_llm_completion") or [])
    llm_reasons = {
        str(item.get("field")): str(item.get("reason"))
        for item in config.get("llm_field_rationales", [])
        if isinstance(item, dict) and item.get("field")
    }
    general_rationale = str(config.get("rationale", "LLM-selected value."))
    provenance: dict[str, dict[str, Any]] = {}
    records = [
        context.get("selected_model") or {},
        context.get("base_recipe") or {},
        *(context.get("recipe_parameters") or []),
        *(context.get("recipe_details") or []),
        *(context.get("matched_adjustment_rules") or []),
        *(context.get("model_variants") or []),
    ]
    records_by_id = {
        str(record.get("id")): record for record in records if record.get("id")
    }

    def enrich(entry: dict[str, Any]) -> dict[str, Any]:
        source = entry.get("source")
        source_id = str(entry.get("source_id", ""))
        record = records_by_id.get(source_id, {})
        evidence_ids = [
            item.strip()
            for item in str(record.get("evidence_ids", "")).split("|")
            if item.strip()
        ]
        support_type = {
            "llm_adjustment": "llm_judgment",
            "llm_completion": "llm_judgment",
            "adjustment_rule": "inferred",
            "pipeline_state": "user_constraint",
            "schema_default": "schema_default",
            "system_policy": "system_policy",
            "pretrained_weight_metadata": "derived",
        }.get(str(source), "direct_evidence" if evidence_ids else "internal_assertion")
        return {**entry, "support_type": support_type, "evidence_ids": evidence_ids}

    for field_name, value in config.items():
        if field_name in _EXPLANATION_FIELDS:
            continue
        if field_name in adjusted:
            provenance[field_name] = enrich({
                "source": "llm_adjustment",
                "source_id": "evaluator_authorized_repair",
                "reason": llm_reasons.get(field_name, general_rationale),
            })
        elif field_name in hardware_sources:
            provenance[field_name] = enrich({
                "source": "system_policy",
                "source_id": str(hardware_sources[field_name]),
                "reason": "Adjusted for the server-selected training hardware profile.",
            })
        elif field_name in rule_sources:
            provenance[field_name] = enrich({
                "source": "adjustment_rule",
                "source_id": str(rule_sources[field_name]),
                "reason": f"Set by matched adjustment rule {rule_sources[field_name]}.",
            })
        elif field_name in base_sources:
            source = base_sources[field_name]
            provenance[field_name] = enrich({
                **source,
                "reason": f"Materialized from {source['source']} {source['source_id']}.",
            })
        elif field_name in _SPLIT_COMPATIBILITY_FIELDS:
            provenance[field_name] = enrich({
                "source": "system_policy",
                "source_id": "dataset_assignment_plan",
                "reason": "Derived from the authoritative planned split counts for execution compatibility.",
            })
        elif field_name in _PIPELINE_CONTEXT_FIELDS:
            provenance[field_name] = enrich({
                "source": "pipeline_state",
                "source_id": "user_and_data_selection_context",
                "reason": "Copied from the interpreted task and selected data.",
            })
        elif field_name == "model_name" and context.get("model_variants"):
            normalized_value = re.sub(r"[^a-z0-9]", "", str(value).lower())
            variant = next(
                (
                    item
                    for item in context["model_variants"]
                    if re.sub(r"[^a-z0-9]", "", str(item.get("id", "")).lower())
                    .replace("yolo11", "yolov11")
                    .replace("yolo12", "yolov12")
                    == normalized_value
                ),
                {},
            )
            provenance[field_name] = enrich({
                "source": "llm_completion",
                "source_id": str(variant.get("id", "missing_recipe_field")),
                "reason": llm_reasons.get(field_name, general_rationale),
            })
        elif field_name in completion_fields:
            provenance[field_name] = enrich({
                "source": "llm_completion",
                "source_id": "missing_recipe_field",
                "reason": llm_reasons.get(field_name, general_rationale),
            })
        else:
            schema_field = schema_model.model_fields.get(field_name)
            default = schema_field.get_default(call_default_factory=True) if schema_field else None
            if schema_field and not schema_field.is_required() and value == default:
                provenance[field_name] = enrich({
                    "source": "schema_default",
                    "source_id": schema_model.__name__,
                    "reason": "Used the deterministic executable schema default.",
                })
            else:
                provenance[field_name] = enrich({
                    "source": "llm_completion",
                    "source_id": "unmaterialized_configuration_field",
                    "reason": llm_reasons.get(field_name, general_rationale),
                })
    return provenance


_MEMORY_BUDGET_THRESHOLD = re.compile(
    r"(?:vram_gb|training_memory_budget_gb)\s*(<=|>=)\s*(\d+(?:\.\d+)?)"
    r"(?:\s*;\s*training_mode\s*!=\s*head_only)?",
    re.IGNORECASE,
)
_MAX_MIN_IMAGES_PER_CLASS = re.compile(
    r"min_images_per_class\s*<=\s*(\d+)", re.IGNORECASE
)
_RANGE_MIN_IMAGES_PER_CLASS = re.compile(
    r"(\d+)\s*<\s*min_images_per_class\s*<=\s*(\d+)", re.IGNORECASE
)
_MAX_TOTAL_SELECTED_IMAGES = re.compile(
    r"total_selected_images\s*<=\s*(\d+)", re.IGNORECASE
)
_RANGE_TOTAL_SELECTED_IMAGES = re.compile(
    r"(\d+)\s*<\s*total_selected_images\s*<=\s*(\d+)", re.IGNORECASE
)
_ASSIGNMENT = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;,]+)")


def _minimum_selected_images_per_class(state: PipelineState) -> int | None:
    totals = []
    for selection in state.selected_data or []:
        sources = selection.sources if hasattr(selection, "sources") else selection.get("sources", [])
        count = 0
        for source in sources:
            if hasattr(source, "allocations"):
                value = sum(allocation.count for allocation in source.allocations)
            elif hasattr(source, "count"):
                value = source.count
            else:
                value = source.get("count", 0)
            count += max(0, int(value))
        totals.append(count)
    return min(totals) if totals else None


def _total_selected_images(state: PipelineState) -> int | None:
    counts = []
    for selection in state.selected_data or []:
        sources = selection.sources if hasattr(selection, "sources") else selection.get("sources", [])
        for source in sources:
            if hasattr(source, "allocations"):
                value = sum(allocation.count for allocation in source.allocations)
            elif hasattr(source, "count"):
                value = source.count
            else:
                value = source.get("count", 0)
            counts.append(max(0, int(value)))
    return sum(counts) if counts else None


def _materialize_matched_rule(
    rule: dict[str, Any],
    state: PipelineState,
    reference_configuration: dict[str, Any],
    active_dataset_properties: set[str] | None = None,
) -> dict[str, Any] | None:
    """Materialize explicit scalar rules whose conditions are provable from state."""
    condition = str(rule.get("condition_value", "")).strip()
    condition_type = rule.get("condition_type")

    if condition_type == "HardwareConstraint":
        threshold_match = _MEMORY_BUDGET_THRESHOLD.fullmatch(condition)
        hardware = state.training_hardware
        memory_budget_gb = hardware.training_memory_budget_gb if hardware else None
        if (
            not threshold_match
            or memory_budget_gb is None
            or hardware.hardware_category not in {"ConsumerGPU", "DataCenterGPU"}
        ):
            return None
        operator, threshold = threshold_match.groups()
        if operator == "<=" and float(memory_budget_gb) > float(threshold):
            return None
        if operator == ">=" and float(memory_budget_gb) < float(threshold):
            return None
    elif condition_type == "DatasetProperty":
        images_per_class = _minimum_selected_images_per_class(state)
        total_selected_images = _total_selected_images(state)
        maximum_match = _MAX_MIN_IMAGES_PER_CLASS.fullmatch(condition)
        range_match = _RANGE_MIN_IMAGES_PER_CLASS.fullmatch(condition)
        total_match = _MAX_TOTAL_SELECTED_IMAGES.fullmatch(condition)
        total_range_match = _RANGE_TOTAL_SELECTED_IMAGES.fullmatch(condition)
        if total_match:
            if total_selected_images is None or total_selected_images > int(total_match.group(1)):
                return None
        elif total_range_match:
            if total_selected_images is None:
                return None
            lower, upper = map(int, total_range_match.groups())
            if not lower < total_selected_images <= upper:
                return None
        elif images_per_class is None:
            return None
        elif maximum_match:
            if images_per_class > int(maximum_match.group(1)):
                return None
        elif range_match:
            lower, upper = map(int, range_match.groups())
            if not lower < images_per_class <= upper:
                return None
        elif condition not in (active_dataset_properties or set()):
            return None
        else:
            pass
    else:
        return None

    coerce = (
        _coerce_detection_config_value
        if state.task == "detection"
        else _coerce_classification_config_value
    )
    adjustments: dict[str, Any] = {}
    for field_name, raw_value in _ASSIGNMENT.findall(str(rule.get("adjustment_value", ""))):
        value = raw_value.strip()
        if value == "$num_epochs":
            value = reference_configuration.get("num_epochs")
        if value is None:
            return None
        try:
            adjustments[field_name] = coerce(field_name, value)
        except ValueError:
            return None
    if not adjustments:
        return None
    return {**rule, "executable_adjustments": adjustments}


def _resolve_matched_rule_conflicts(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove hardware tactics that are incompatible with a selected training mode."""
    head_only_selected = any(
        rule["executable_adjustments"].get("training_mode") == "head_only"
        for rule in rules
    )
    if not head_only_selected:
        lora_selected = any(
            rule["executable_adjustments"].get("training_mode") == "lora"
            for rule in rules
        )
        if not lora_selected:
            return rules
        return [
            rule
            for rule in rules
            if rule["executable_adjustments"].get("training_mode") in {None, "lora"}
        ]
    return [
        rule
        for rule in rules
        if not (
            rule.get("condition_type") == "HardwareConstraint" and (
                rule["executable_adjustments"].get("use_activation_checkpointing") is True
                or rule["executable_adjustments"].get("training_mode") == "lora"
            )
        )
    ]


def _training_hardware_adjustments(
    state: PipelineState,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Apply deterministic execution limits from the snapshotted profile."""
    hardware = state.training_hardware
    if hardware is None:
        return {}

    adjustments: dict[str, Any] = {}
    batch_size = configuration.get("batch_size")
    if isinstance(batch_size, (int, float)) and batch_size > hardware.max_batch_size:
        adjustments["batch_size"] = hardware.max_batch_size
    if state.task == "detection":
        adjustments["workers"] = hardware.workers
        identity = resolve_detection_model_identity(
            str(configuration.get("model_name") or _selected_model_id(state) or "")
        )
        if (
            identity is not None
            and identity.runtime_family == "yolo"
            and 0 < hardware.training_memory_budget_gb <= 6.0
        ):
            robustness = state.robustness_requirements
            requested_scales = {
                str(value).strip().lower()
                for value in (
                    robustness.get("object_scale", [])
                    if isinstance(robustness, dict)
                    else robustness.object_scale
                )
            }
            small_objects_requested = "small" in requested_scales or any(
                phrase in str(state.user_query or "").lower()
                for phrase in ("small object", "small and far", "far away", "distant object")
            )
            input_size = configuration.get("input_size")
            if isinstance(input_size, (int, float)) and input_size > 768:
                adjustments["input_size"] = 768
            elif small_objects_requested and isinstance(input_size, (int, float)) and input_size < 768:
                # A bounded high-resolution profile for 6 GiB cards. The paired
                # batch reduction is validated again by the resource guard.
                adjustments["input_size"] = 768
            if small_objects_requested:
                adjustments["batch_size"] = min(2, hardware.max_batch_size)
                adjustments["translate"] = 0.05
                adjustments["scale"] = 0.25
                adjustments["fliplr"] = 0.5
            adjustments["multi_scale"] = 0.0
    if configuration.get("amp") is True and not hardware.supports_amp:
        adjustments["amp"] = False
    if configuration.get("precision") == "mixed" and not hardware.supports_amp:
        adjustments["precision"] = "fp32"
    return adjustments


def validate_executable_recipe_config(config: dict[str, Any]) -> None:
    """Validate optional recipe provenance against executable graph semantics."""
    recipe_id = str(config.get("training_recipe_id", "")).strip()
    if not recipe_id:
        return

    graph = get_hyperparameter_graph()
    if recipe_id not in graph or graph.nodes[recipe_id].get("source_csv") != "training_recipes.csv":
        raise ValueError(f"Unknown training_recipe_id: {recipe_id}")
    if recipe_id in NON_EXECUTABLE_RECIPE_IDS:
        raise ValueError(f"Training recipe is not executable by this registry: {recipe_id}")

    model_id = str(config.get("model_name", ""))
    graph_model_id = _graph_model_id_for_recipe_validation(model_id)
    related_recipe_ids = {
        recipe["id"] for recipe in _related_nodes(graph, graph_model_id, "has_training_recipe")
    }
    permitted_models = RECIPE_MODEL_ID_CONSTRAINTS.get(recipe_id)
    if recipe_id not in related_recipe_ids or (
        permitted_models is not None and graph_model_id not in permitted_models
    ):
        raise ValueError(
            f"training_recipe_id='{recipe_id}' is not compatible with model_name='{model_id}'."
        )

    recipe = graph.nodes[recipe_id]
    if recipe.get("task_id") == "object_detection":
        pretrained = str(recipe.get("pretrained", "")).lower()
        if pretrained == "true" and config.get("model_weights") not in {
            "coco", "default", "imagenet_backbone"
        }:
            raise ValueError(
                f"training_recipe_id='{recipe_id}' requires pretrained model weights."
            )
        if pretrained == "false" and config.get("model_weights") != "none":
            raise ValueError(
                f"training_recipe_id='{recipe_id}' requires model_weights='none'."
            )
        for field, minimum_field, maximum_field in (
            ("learning_rate", "learning_rate_min", "learning_rate_max"),
            ("batch_size", "batch_size_min", "batch_size_max"),
            ("num_epochs", "epochs_min", "epochs_max"),
            ("weight_decay", "weight_decay_min", "weight_decay_max"),
            ("input_size", "image_size_min", "image_size_max"),
        ):
            value = config.get(field)
            if value is None or (field == "batch_size" and value == -1):
                continue
            minimum = recipe.get(minimum_field)
            maximum = recipe.get(maximum_field)
            if minimum not in {"", None} and float(value) < float(minimum):
                raise ValueError(f"training_recipe_id='{recipe_id}' requires {field} >= {minimum}.")
            if maximum not in {"", None} and float(value) > float(maximum):
                raise ValueError(f"training_recipe_id='{recipe_id}' requires {field} <= {maximum}.")
        return

    expected_mode = _executable_recipe_value("training_mode", recipe.get("training_mode", ""))
    allowed_modes = set(RECIPE_ALLOWED_TRAINING_MODES.get(recipe_id, {expected_mode}))
    if model_id in LORA_CLASSIFICATION_MODEL_IDS:
        allowed_modes.add("lora")
    if expected_mode in {
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
        "train_from_scratch",
    } and config.get("training_mode") not in allowed_modes:
        raise ValueError(
            f"training_recipe_id='{recipe_id}' requires training_mode in {sorted(allowed_modes)}."
        )

    pretrained = str(recipe.get("pretrained", "")).lower()
    if pretrained in {"true", "false"}:
        expected_weights = _executable_recipe_value("pretrained", pretrained)
        if config.get("model_weights") != expected_weights:
            raise ValueError(
                f"training_recipe_id='{recipe_id}' requires model_weights='{expected_weights}'."
            )

    bounded_fields = (
        ("learning_rate", "learning_rate_min", "learning_rate_max"),
        ("batch_size", "batch_size_min", "batch_size_max"),
        ("num_epochs", "epochs_min", "epochs_max"),
        ("weight_decay", "weight_decay_min", "weight_decay_max"),
        ("image_size", "image_size_min", "image_size_max"),
    )
    for executable_field, minimum_field, maximum_field in bounded_fields:
        value = config.get(executable_field)
        if value is None:
            continue
        minimum = recipe.get(minimum_field, "")
        maximum = recipe.get(maximum_field, "")
        if minimum not in {"", None} and float(value) < float(minimum):
            raise ValueError(
                f"training_recipe_id='{recipe_id}' requires {executable_field} >= {minimum}."
            )
        if maximum not in {"", None} and float(value) > float(maximum):
            raise ValueError(
                f"training_recipe_id='{recipe_id}' requires {executable_field} <= {maximum}."
            )


def validate_graph_grounded_config(
    config: dict[str, Any],
    context: dict[str, Any],
    **_ignored: Any,
) -> None:
    """Validate executable recipe provenance without making its reference values immutable."""
    validate_executable_recipe_config(config)


def validate_detection_graph_grounded_config(
    config: dict[str, Any],
    context: dict[str, Any],
    **_ignored: Any,
) -> None:
    """Validate detector recipe provenance and selected-family compatibility."""
    validate_executable_recipe_config(config)
    recipe = context.get("base_recipe") or {}
    recipe_id = str(recipe.get("id", ""))
    if not recipe_id or recipe_id in NON_EXECUTABLE_RECIPE_IDS:
        raise ValueError("No executable fine-tuning recipe was retrieved for the selected detector.")
    if config.get("training_recipe_id") != recipe_id:
        raise ValueError(f"Detection configuration must use training_recipe_id='{recipe_id}'.")
    selected = str(context.get("selected_registry_id") or "")
    candidate = str(config.get("model_name") or "")
    selected_identity = resolve_detection_model_identity(selected)
    candidate_identity = resolve_detection_model_identity(candidate)
    if (
        selected_identity is not None
        and candidate_identity is not None
        and selected_identity.family != candidate_identity.family
    ):
        raise ValueError(
            f"Selected model family '{selected}' is incompatible with HPO model_name='{candidate}'."
        )


@lru_cache(maxsize=1)
def get_hyperparameter_graph() -> nx.MultiDiGraph:
    return build_graph(PROJECT_ROOT / "ontology_data")


def _selected_model_id(state: PipelineState) -> str | None:
    selected = state.selected_model_info or {}
    models = selected.get("model") or []
    if isinstance(models, dict):
        models = [models]
    if models:
        model = models[0]
        for key in ("model_architecture", "model_name", "name", "id"):
            if model.get(key):
                return str(model[key])
    for key in ("model_id", "model_name"):
        if selected.get(key):
            return str(selected[key])
    return None


def _find_model_node(graph: nx.MultiDiGraph, state: PipelineState) -> tuple[str | None, dict[str, Any]]:
    def canonical(value: str) -> str:
        normalized = "".join(character for character in value.lower() if character.isalnum())
        aliases = {
            "retinanetr50fpn1xcoco": "retinanetresnet50fpn",
            "retinanetr50": "retinanetresnet50fpn",
            "fasterrcnnr50fpn1xcoco": "fasterrcnnresnet50fpn",
            "fasterrcnnr50": "fasterrcnnresnet50fpn",
            "ssd300coco": "ssd300vgg16",
            "ssd300": "ssd300vgg16",
            "rtdetrl": "rtdetrhgnetv2l",
            "rtdetr": "rtdetrhgnetv2l",
        }
        normalized = normalized.replace("yolo11", "yolov11").replace("yolo12", "yolov12")
        return aliases.get(normalized, normalized)

    selected_id = canonical(_selected_model_id(state) or "")
    task_id = TASK_IDS.get(state.task or "")
    if not selected_id:
        return None, {}
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("source_csv") != "models.csv" or attrs.get("task") != task_id:
            continue
        values = {str(node_id), str(attrs.get("model_name", ""))}
        normalized = {canonical(value) for value in values}
        if selected_id in normalized or any(
            selected_id.startswith(value) or value.startswith(selected_id)
            for value in normalized if value
        ):
            return str(node_id), {"id": str(node_id), **attrs}
    return None, {}


def _related_nodes(graph: nx.MultiDiGraph, source: str, relation: str) -> list[dict[str, Any]]:
    result = []
    for _, target, edge in graph.out_edges(source, data=True):
        if edge.get("relation") == relation:
            result.append({"id": target, **graph.nodes[target]})
    return result


def _recipe_score(recipe: dict[str, Any], state: PipelineState) -> tuple[int, int, str]:
    score = 0
    if str(recipe.get("training_mode", "")).lower() == "finetunepretrained":
        score += 5
    hardware = state.training_hardware.hardware_category if state.training_hardware else None
    recipe_hardware = str(recipe.get("hardware_category", ""))
    if not recipe_hardware or recipe_hardware == "Unspecified":
        score += 2
    elif hardware and recipe_hardware == hardware:
        score += 3
    priority = str(recipe.get("performance_priority", ""))
    accuracy = state.performance_requirements.accuracy_category if state.performance_requirements else None
    if accuracy in {"MediumHigh", "High"} and priority == "AccuracyFirst":
        score += 2
    if priority == "Balanced":
        score += 1
    if state.training_hardware:
        try:
            default_batch = int(recipe.get("batch_size_default") or 0)
        except (TypeError, ValueError):
            default_batch = 0
        if (
            default_batch > 0
            and default_batch <= state.training_hardware.max_batch_size
            and default_batch >= max(8, state.training_hardware.max_batch_size // 2)
        ):
            # Prefer an explicitly executable high-throughput recipe when the
            # selected server profile can support its micro-batch. Memory
            # estimation and the runtime resource guard remain authoritative.
            score += 2
    return score, len(str(recipe.get("evidence_ids", ""))), str(recipe.get("id", ""))


def build_hyperparameter_context(state: PipelineState) -> dict[str, Any]:
    """Retrieve the selected model's best recipe, parameters, details, rules, and evidence."""
    graph = get_hyperparameter_graph()
    model_id, model = _find_model_node(graph, state)
    if not model_id:
        return {
            "enabled": True,
            "source": "NetworkX knowledge graph from backend/ontology_data",
            "selected_model_id": _selected_model_id(state),
            "base_recipe": None,
            "applicable_rules": [],
            "warning": "The selected model could not be matched to a models.csv node.",
        }

    recipes = [
        recipe
        for recipe in _related_nodes(graph, model_id, "has_training_recipe")
        if recipe["id"] not in NON_EXECUTABLE_RECIPE_IDS
        and model_id in RECIPE_MODEL_ID_CONSTRAINTS.get(recipe["id"], {model_id})
    ]
    # This application generates transfer-learning configurations. Official
    # pretraining recipes remain in the graph for provenance and benchmarks,
    # but must never become the base of a custom-data fine-tuning config.
    if state.task == "classification":
        recipes = [
            candidate
            for candidate in recipes
            if str(candidate.get("training_mode", "")).lower() == "finetunepretrained"
        ]
    recipes.sort(key=lambda recipe: _recipe_score(recipe, state), reverse=True)
    recipe = recipes[0] if recipes else None
    parameters = _related_nodes(graph, recipe["id"], "has_parameter") if recipe else []
    details = _related_nodes(graph, recipe["id"], "has_recipe_details") if recipe else []

    model_variants: list[dict[str, Any]] = []
    variant_benchmarks: list[dict[str, Any]] = []
    if state.task == "detection" and str(model.get("model_family", "")).startswith("YOLO"):
        family = model.get("model_family")
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("source_csv") != "models.csv" or attrs.get("model_family") != family:
                continue
            model_variants.append({"id": str(node_id), **attrs})
            variant_benchmarks.extend(
                benchmark
                for benchmark in _related_nodes(graph, str(node_id), "has_benchmark_result")
                if str(benchmark.get("dataset", "")).lower() == "coco"
                and benchmark.get("metric_id") in {"map50_95", "latency_ms", "params_m", "flops_b"}
            )

    rules = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("source_csv") != "adjustment_rules.csv":
            continue
        if node_id in NON_EXECUTABLE_RULE_IDS:
            continue
        applies = any(
            target == model_id and edge.get("relation") == "applies_to_model"
            for _, target, edge in graph.out_edges(node_id, data=True)
        )
        if applies:
            permitted_models = RULE_MODEL_ID_CONSTRAINTS.get(node_id)
            if permitted_models is not None and model_id not in permitted_models:
                continue
            rules.append({"id": node_id, **attrs})

    evidence_ids: set[str] = set()
    for item in [
        model,
        *(recipes[:1]),
        *parameters,
        *details,
        *rules,
        *model_variants,
        *variant_benchmarks,
    ]:
        evidence_ids.update(filter(None, str(item.get("evidence_ids", "")).split("|")))
    evidence = [{"id": eid, **graph.nodes[eid]} for eid in sorted(evidence_ids) if eid in graph]

    context = {
        "enabled": True,
        "source": "NetworkX knowledge graph from backend/ontology_data",
        "retrieval_strategy": "selected model -> compatible recipes -> recipe details/parameters and applicable adjustment rules",
        "selected_model_id": model_id,
        "selected_registry_id": _selected_model_id(state),
        "selected_model": model,
        "base_recipe": recipe,
        "recipe_parameters": parameters,
        "recipe_details": details,
        "model_variants": model_variants,
        "variant_benchmarks": variant_benchmarks,
        "applicable_rules": rules,
        "evidence_sources": evidence,
        "excluded_non_executable_recipe_ids": sorted(NON_EXECUTABLE_RECIPE_IDS),
        "excluded_non_executable_rule_ids": sorted(NON_EXECUTABLE_RULE_IDS),
        "warning": (
            None
            if recipe
            else "No executable pretrained custom-data fine-tuning recipe is available for this model."
        ),
        "instructions_for_generator": (
            "Treat reference_configuration as an evidence-backed starting point. Preserve or adapt its "
            "schema-configurable values for the selected model, data, task, and hardware, explaining each "
            "choice. Treat matched_adjustment_rules as evidence-backed recommendations, not mandatory "
            "overrides: follow or adapt them using the retrieved recipe, hardware-safe candidates, and "
            "task requirements, and explain the decision. Rules absent from matched_adjustment_rules are "
            "informational only. When a matched rule pairs batch_size and learning_rate, select them "
            "jointly. For Ultralytics optimizer=auto, leave learning-rate control to Ultralytics; apply "
            "the paired LR guidance only if an explicit optimizer is selected. Runtime constraints and "
            "schema validation remain authoritative."
        ),
    }
    if state.training_hardware is not None:
        context["hardware_role_context"] = {
            "training_hardware_authority": state.training_hardware.model_dump(mode="json"),
            "deployment_hardware_not_for_training_memory": (
                state.available_hardware.model_dump(mode="json")
                if state.available_hardware is not None
                else None
            ),
            "instruction": (
                "Use training_hardware exclusively for batch size, workers, AMP, precision, "
                "and training-memory feasibility. available_hardware and deployment_constraints "
                "describe inference/deployment and must not reduce training hyperparameters."
            ),
        }
    if state.task == "detection":
        identity = resolve_detection_model_identity(_selected_model_id(state) or model_id)
        if identity is not None:
            context["model_execution_constraints"] = {
                "runtime_family": identity.runtime_family,
                "input_stride": identity.input_stride,
                "supported_input_size": {"minimum": 32, "maximum": 1280},
                "baseline_input_size": 640 if identity.runtime_family == "yolo" else None,
            }
    context["reference_configuration"] = materialize_base_recipe_config(context)
    # Ultralytics AutoOptimizer is a grounded default, not an immutable runtime
    # requirement. Permit the optimizer agent to choose an explicit supported
    # optimizer when it can justify the departure; learning_rate then becomes
    # an active, jointly selected field.
    optimizer_adjustments: set[str] = set()
    if (
        state.task == "detection"
        and context["reference_configuration"].get("optimizer_name") == "auto"
    ):
        optimizer_adjustments = {"optimizer_name", "learning_rate"}
    selected_dataset_characteristics = aggregate_selected_dataset_properties(
        state,
        graph,
    )
    context["selected_dataset_characteristics"] = selected_dataset_characteristics
    active_dataset_properties = {
        item["property_id"]
        for item in selected_dataset_characteristics
        if item["active"]
    }
    if state.task == "classification" and recipe is not None:
        from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel

        completion_fields = set(context.get("fields_requiring_llm_completion") or [])
        completion_fields.update(
            field_name
            for field_name, schema_field in ClassificationConfigModel.model_fields.items()
            if schema_field.is_required()
            and field_name not in context["reference_configuration"]
            and field_name not in _PIPELINE_CONTEXT_FIELDS
            and field_name not in _EXPLANATION_FIELDS
        )
        context["fields_requiring_llm_completion"] = sorted(completion_fields)
    matched_rules = []
    if state.task in {"classification", "detection"} and recipe is not None:
        matched_rules = _resolve_matched_rule_conflicts([
            matched
            for rule in rules
            if (
                matched := _materialize_matched_rule(
                    rule,
                    state,
                    context["reference_configuration"],
                    active_dataset_properties,
                )
            ) is not None
        ])
    suppressed_rules = []
    if state.task == "detection":
        robustness = state.robustness_requirements
        requested_scales = {
            str(value).strip().lower()
            for value in (
                robustness.get("object_scale", [])
                if isinstance(robustness, dict)
                else robustness.object_scale
            )
        }
        if "small" in requested_scales:
            retained_rules = []
            for rule in matched_rules:
                if (
                    rule.get("condition_type") == "DatasetProperty"
                    and rule.get("condition_value") == "ObjectScaleVariation"
                    and set((rule.get("executable_adjustments") or {}))
                    & {"scale", "multi_scale"}
                ):
                    suppressed_rules.append({
                        "id": rule.get("id"),
                        "reason": (
                            "Generic object-scale variation downscaling is superseded by "
                            "the explicit small-object requirement."
                        ),
                    })
                    continue
                retained_rules.append(rule)
            matched_rules = retained_rules
            context["small_object_training_policy"] = {
                "preferred_minimum_input_size": 768,
                "selection_policy": (
                    "Prefer more pixels for small objects, then choose batch_size from "
                    "hardware_safe_resolution_candidates using training_hardware only."
                ),
                "reason": (
                    "Preserve more pixels for small objects without assuming that deployment "
                    "VRAM limits the separate training GPU."
                ),
                "requires_runtime_memory_validation": True,
            }
    context["matched_adjustment_rules"] = matched_rules
    context["suppressed_adjustment_rules"] = suppressed_rules
    context["allowed_adjustment_fields"] = sorted({
        field
        for rule in matched_rules
        for field in rule["executable_adjustments"]
    } | optimizer_adjustments)
    rule_adjustments = {
        field: value for rule in matched_rules
        for field, value in rule["executable_adjustments"].items()
    }
    hardware_adjustments = _training_hardware_adjustments(
        state,
        {**context["reference_configuration"], **rule_adjustments},
    )
    context["allowed_adjustment_fields"] = sorted(
        set(context["allowed_adjustment_fields"]) | set(hardware_adjustments)
    )
    context["adjustment_rule_provenance"] = {
        field: rule["id"]
        for rule in matched_rules
        for field in rule["executable_adjustments"]
    }
    context["training_hardware_adjustment_provenance"] = (
        {
            field: state.training_hardware.profile_id
            for field in hardware_adjustments
        }
        if state.training_hardware
        else {}
    )
    context["training_hardware_adjustments"] = hardware_adjustments
    if state.task == "detection":
        selected_identity = resolve_detection_model_identity(
            _selected_model_id(state) or model_id
        )
        memory_model_id = (
            selected_identity.executable_id if selected_identity is not None else model_id
        )
        reference_input_size = context["reference_configuration"].get("input_size")
        recipe_min_size = recipe.get("image_size_min") if recipe else None
        recipe_max_size = recipe.get("image_size_max") if recipe else None
        minimum_size = int(recipe_min_size) if str(recipe_min_size).isdigit() else None
        maximum_size = int(recipe_max_size) if str(recipe_max_size).isdigit() else None
        proposed_sizes: tuple[int | float | None, ...] = (reference_input_size, 640, 768)
        if (
            selected_identity is not None
            and selected_identity.runtime_family == "yolo"
            and state.training_hardware is not None
            and state.training_hardware.training_memory_budget_gb >= 32
        ):
            proposed_sizes = (
                reference_input_size, 640, 768, 896, 960, 1024, 1280,
            )
        candidate_image_sizes = tuple(dict.fromkeys(
            int(value)
            for value in proposed_sizes
            if isinstance(value, (int, float)) and value > 0
            and (minimum_size is None or int(value) >= minimum_size)
            and (maximum_size is None or int(value) <= maximum_size)
        ))
        context["hardware_safe_resolution_candidates"] = rank_training_shape_candidates(
            {
                **context["reference_configuration"],
                **rule_adjustments,
                **hardware_adjustments,
                "model_name": memory_model_id,
            },
            image_sizes=candidate_image_sizes,
        )
    return context


def format_hyperparameter_context(context: dict[str, Any]) -> str:
    recipe = context.get("base_recipe")
    if not recipe:
        return f"Hyperparameter GraphRAG found no base recipe. {context.get('warning', '')}".strip()
    lines = [
        "Hyperparameter GraphRAG context",
        f"Selected model: {context['selected_model'].get('model_name')} ({context['selected_model'].get('id')})",
        f"Base recipe: {recipe.get('recipe_name')} ({recipe.get('id')})",
        f"- training_recipe_id: {recipe.get('id')}",
        "Base defaults/ranges:",
    ]
    lines.append(f"Reference configuration: {context.get('reference_configuration', {})}")
    if context.get("hardware_role_context"):
        lines.append(f"Hardware role authority: {context['hardware_role_context']}")
    lines.append(
        f"Fields requiring bounded LLM completion: {context.get('fields_requiring_llm_completion', [])}"
    )
    field_map = (
        DETECTION_RECIPE_FIELD_TO_EXECUTABLE_FIELD
        if recipe.get("task_id") == "object_detection"
        else RECIPE_FIELD_TO_EXECUTABLE_FIELD
    )
    for field in (
        "training_mode", "pretrained", "optimizer", "scheduler", "precision", "learning_rate_min",
        "learning_rate_default", "learning_rate_max", "batch_size_min", "batch_size_default",
        "batch_size_max", "epochs_min", "epochs_default", "epochs_max", "weight_decay_default",
        "image_size_default", "patience_default", "warmup_epochs_default", "momentum_default",
        "lora_profile", "lora_rank_default", "lora_alpha_default", "gradient_accumulation_default",
        "freeze_default", "task_specific_params",
    ):
        value = recipe.get(field)
        if value != "" and value is not None:
            executable_field = field_map.get(field)
            if executable_field:
                lines.append(
                    f"- {executable_field}: {_executable_recipe_value(field, value)} "
                    f"(ontology field: {field})"
                )
            else:
                lines.append(f"- {field}: {value}")
    if context.get("recipe_details"):
        lines.append(f"Recipe details: {context['recipe_details']}")
        lines.append("Executable recipe-detail defaults:")
        detail_map = (
            DETECTION_RECIPE_DETAIL_TO_EXECUTABLE_FIELD
            if recipe.get("task_id") == "object_detection"
            else RECIPE_DETAIL_TO_EXECUTABLE_FIELD
        )
        for detail_field, executable_field in detail_map.items():
            value = context["recipe_details"][0].get(detail_field)
            if value != "" and value is not None:
                lines.append(f"- {executable_field}: {value} (ontology field: {detail_field})")
    if context.get("recipe_parameters"):
        lines.append(f"Structured parameters: {context['recipe_parameters']}")
    if context.get("model_variants"):
        lines.append(f"Executable variants in the selected family: {context['model_variants']}")
        lines.append(f"Variant benchmark evidence: {context.get('variant_benchmarks', [])}")
    if recipe.get("task_id") == "image_classification":
        lines.append(
            "Value normalization: CosineAnnealingLR -> scheduler_name=cosine; "
            "StepLR -> scheduler_name=step; "
            "mixed_precision -> precision=mixed; freeze_default=false -> freeze_backbone_epochs=0. "
            "FineTunePretrained -> training_mode=fine_tune_pretrained with model_weights=default; "
            "TrainFromScratch -> training_mode=train_from_scratch with model_weights=none. "
            "Use random_resized_crop_scale_min=0.6 and horizontal_flip_probability=0.5 when the "
            "grounded context has no more specific value; set horizontal_flip_probability=0 for "
            "orientation-sensitive labels or domains. "
            "Use exact param_name values from Structured parameters as executable configuration fields."
        )
    if context.get("selected_dataset_characteristics"):
        lines.append(
            "Aggregated selected-dataset characteristics "
            f"(support is weighted by selected class-image allocations): "
            f"{context['selected_dataset_characteristics']}"
        )
    lines.append("Matched evidence-backed recommendations (the condition is proven by PipelineState):")
    for rule in context.get("matched_adjustment_rules", []):
        lines.append(
            f"- {rule.get('id')}: IF {rule.get('condition_type')}={rule.get('condition_value')} "
            f"CONSIDER {rule.get('executable_adjustments')} "
            f"(confidence={rule.get('confidence')})"
        )
    lines.append(f"Fields the generator may adjust from the base: {context.get('allowed_adjustment_fields', [])}")
    return "\n".join(lines)


def summarize_hyperparameter_context(context: dict[str, Any]) -> str:
    recipe = context.get("base_recipe")
    if not recipe:
        return "Hyperparameter GraphRAG: no compatible base recipe found."
    return (
        f"Hyperparameter GraphRAG: selected recipe {recipe.get('id')} with "
        f"{len(context.get('applicable_rules', []))} model-scoped rules."
    )
