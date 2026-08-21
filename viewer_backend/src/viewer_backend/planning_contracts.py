"""PyTorch-free copies of the full backend's planning data contracts.

These models deliberately live in the viewer package and import only Pydantic.
They describe planning documents; they do not instantiate models, optimizers,
datasets, or any other execution object.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


HardwareCategory = Literal[
    "ConsumerCPU", "ConsumerGPU", "EdgeDevice", "DataCenterGPU",
    "ConsumerCPU | EdgeDevice",
]
PerformanceCategory = Literal["VeryLow", "Low", "Medium", "MediumHigh", "High"]
DeploymentLimit = Literal[
    "max_runtime_memory_mb", "max_model_size_mb", "max_parameters_m", "max_cpu_latency_ms",
]
ClassificationModelId = Literal[
    "resnet50", "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "densenet121", "convnext_tiny", "clip_vit_b16", "dinov2_vits14",
    "dinov2_vitb14", "vit_b_16", "swin_v2_t", "swin_v2_s",
]
VQAModelId = Literal["Qwen3-VL-2B-Instruct"]
DetectionHPOModelId = Literal[
    "yolov8_n", "yolov8_s", "yolov8_m", "yolov8_l", "yolov8_x",
    "yolov10_n", "yolov10_s", "yolov10_m", "yolov10_l", "yolov10_x",
    "yolov11_n", "yolov11_s", "yolov11_m", "yolov11_l", "yolov11_x",
    "yolov12_n", "yolov12_s", "yolov12_m", "yolov12_l", "yolov12_x",
    "retinanet_r50", "faster_rcnn_r50", "ssd300", "rtdetr_hgnetv2_l",
]
MAX_IMAGE_SIDE = 4096


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrderedSchemaModel(StrictModel):
    schema_field_order: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler(core_schema)
        properties = schema.get("properties")
        if properties and cls.schema_field_order:
            schema["properties"] = {
                name: properties[name] for name in cls.schema_field_order if name in properties
            }
        return schema


class DatasetSourceCount(StrictModel):
    dataset_name: str = Field(min_length=1)
    count: int = Field(ge=0)


class ClassDataSelection(StrictModel):
    class_name: str = Field(min_length=1)
    sources: list[DatasetSourceCount]


class SplitAllocation(StrictModel):
    split: Literal["train", "validation", "test"]
    count: int = Field(gt=0)
    assignment_type: Literal["official_split", "derived_from_train"]


class DatasetSourceAssignment(StrictModel):
    dataset_name: str = Field(min_length=1)
    allocations: list[SplitAllocation] = Field(min_length=1)


class ClassDataAssignment(StrictModel):
    class_name: str = Field(min_length=1)
    sources: list[DatasetSourceAssignment] = Field(min_length=1)


class DatasetSplitCounts(StrictModel):
    train: int = Field(0, ge=0)
    validation: int = Field(0, ge=0)
    test: int = Field(0, ge=0)


class HardwareSpec(StrictModel):
    hardware_category: HardwareCategory | None = None
    cpu_cores: int | None = Field(None, ge=1)
    gpu_type: str | None = None
    gpu_count: int | None = Field(None, ge=0)
    vram_gb: float | None = Field(None, ge=0)
    ram_gb: float | None = Field(None, ge=1)
    storage_gb: float | None = Field(None, ge=1)
    details: str | None = None

    @field_validator("hardware_category", mode="before")
    @classmethod
    def normalize_hardware_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().replace("-", "_").replace(" ", "_").lower()
        return {
            "consumer_cpu": "ConsumerCPU", "consumercpu": "ConsumerCPU", "cpu": "ConsumerCPU",
            "cpu_only": "ConsumerCPU", "consumer_gpu": "ConsumerGPU",
            "consumergpu": "ConsumerGPU", "gpu": "ConsumerGPU", "edge_device": "EdgeDevice",
            "edgedevice": "EdgeDevice", "edge": "EdgeDevice", "mobile": "EdgeDevice",
            "embedded": "EdgeDevice", "data_center_gpu": "DataCenterGPU",
            "datacenter_gpu": "DataCenterGPU", "datacentergpu": "DataCenterGPU",
            "data_center": "DataCenterGPU", "datacenter": "DataCenterGPU",
            "server_gpu": "DataCenterGPU",
            "consumer_cpu_|_edge_device": "ConsumerCPU | EdgeDevice",
            "consumercpu|edgedevice": "ConsumerCPU | EdgeDevice",
            "consumer_cpu_edge_device": "ConsumerCPU | EdgeDevice",
        }.get(normalized, value)


class TrainingHardwareSpec(StrictModel):
    profile_id: str
    accelerator: Literal["cpu", "mps", "cuda"]
    hardware_category: HardwareCategory
    gpu_type: str | None = None
    gpu_count: int = Field(0, ge=0)
    vram_gb: float | None = Field(None, ge=0)
    ram_gb: float | None = Field(None, ge=1)
    unified_memory: bool = False
    training_memory_budget_gb: float = Field(gt=0)
    max_batch_size: int = Field(ge=1)
    workers: int = Field(ge=0)
    supports_amp: bool = False


class PerformanceSpec(StrictModel):
    primary_metric: str | None
    target_value: float | None = Field(None, ge=0, le=1)
    target_is_hard: bool = False
    latency_category: PerformanceCategory | None = None
    accuracy_category: PerformanceCategory | None = None
    other_constraints: list[str] | None = None

    @field_validator("latency_category", "accuracy_category", mode="before")
    @classmethod
    def normalize_performance_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().replace("-", "").replace("_", "").replace(" ", "").lower()
        return {"verylow": "VeryLow", "low": "Low", "medium": "Medium",
                "mediumhigh": "MediumHigh", "high": "High"}.get(normalized, value)


class ConstraintStrengths(StrictModel):
    accuracy: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"
    latency: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"
    runtime_memory: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"
    model_size: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"
    training_time: Literal["hard", "soft", "preference", "unspecified"] = "unspecified"


class RobustnessSpec(StrictModel):
    lighting: list[str] = Field(default_factory=list)
    weather: list[str] = Field(default_factory=list)
    object_scale: list[str] = Field(default_factory=list)
    scene_density: list[str] = Field(default_factory=list)
    motion_blur: bool = False
    occlusion: bool = False
    viewpoint: list[str] = Field(default_factory=list)
    color_semantics: bool = False
    horizontal_flip_safe: bool | None = None
    text_or_symbols_present: bool = False


class DeploymentConstraints(StrictModel):
    memory_category: PerformanceCategory | None = None
    max_runtime_memory_mb: float | None = Field(None, gt=0)
    max_model_size_mb: float | None = Field(None, gt=0)
    max_parameters_m: float | None = Field(None, gt=0)
    max_cpu_latency_ms: float | None = Field(None, gt=0)
    hard_limits: list[DeploymentLimit] = Field(default_factory=list)

    @field_validator("memory_category", mode="before")
    @classmethod
    def normalize_memory_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().replace("-", "").replace("_", "").replace(" ", "").lower()
        return {"verylow": "VeryLow", "low": "Low", "medium": "Medium",
                "mediumhigh": "MediumHigh", "high": "High"}.get(normalized, value)


class ModelRequirement(StrictModel):
    name: str | None
    framework: str | None = None
    backbone: str | None = None
    hyperparameters: dict[str, str | float | int | bool] | None = None
    description: str | None = None
    requirement_strength: Literal["required", "preferred"] = "required"
    training_mode: Literal[
        "fine_tune_pretrained", "staged_fine_tune", "head_only", "lora", "train_from_scratch"
    ] | None = None
    lora_rank: int | None = Field(None, ge=1, le=256)
    lora_alpha: int | None = Field(None, ge=1, le=1024)
    lora_dropout: float | None = Field(None, ge=0, lt=1)


RevisionValue = (
    str | int | float | bool | list[str]
    | dict[str, str | int | float | bool | list[str]] | None
)


class RevisionChange(BaseModel):
    id: str = Field(min_length=1)
    target_step: Literal[
        "task-interpretation", "model-selection", "dataset-selection", "choose-hyperparameters"
    ]
    field: str = Field(min_length=1)
    operation: Literal["set", "include", "exclude", "prefer", "avoid"] = "set"
    value: RevisionValue = None
    strength: Literal["required", "preferred"]
    summary: str = Field(min_length=1)


class RevisionPlan(BaseModel):
    required_text: str = ""
    preferred_text: str = ""
    summary: str = Field(min_length=1)
    restart_from: Literal[
        "task-interpretation", "model-selection", "dataset-selection", "choose-hyperparameters"
    ]
    changes: list[RevisionChange] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_content(self):
        if not self.required_text.strip() and not self.preferred_text.strip():
            raise ValueError("At least one required change or preference is required.")
        if not self.changes:
            raise ValueError("The request did not produce any actionable changes.")
        return self


class RevisionState(BaseModel):
    active: RevisionPlan | None = None
    history: list[RevisionPlan] = Field(default_factory=list)


class DatasetProfile(StrictModel):
    total_selected_images: int = Field(ge=0)
    minimum_images_per_class: int = Field(ge=0)
    maximum_images_per_class: int = Field(ge=0)
    class_balance_ratio: float = Field(ge=0, le=1)
    number_of_sources: int = Field(ge=0)
    domains: list[str] = Field(default_factory=list)
    multi_domain: bool = False
    characteristics: list[str] = Field(default_factory=list)
    characteristic_support: dict[str, float] = Field(default_factory=dict)
    target_unique_images: int = Field(0, ge=0)
    verified_unique_images: int | None = Field(None, ge=0)
    minimum_images_by_class: dict[str, int] = Field(default_factory=dict)
    verified_images_by_class: dict[str, int] = Field(default_factory=dict)
    small_object_fraction: float | None = Field(None, ge=0, le=1)
    median_short_side_px_at_640: float | None = Field(None, ge=0)
    fraction_below_8px_at_640: float | None = Field(None, ge=0, le=1)
    planned_counts: DatasetSplitCounts = Field(default_factory=DatasetSplitCounts)
    official_counts: DatasetSplitCounts = Field(default_factory=DatasetSplitCounts)
    derived_counts: DatasetSplitCounts = Field(default_factory=DatasetSplitCounts)


class InterpretationRequirements(StrictModel):
    task: Literal["classification", "detection", "visual question answering"] | None = None
    application_domain: str | None = None
    user_query: str | None = None
    use_case_description: str | None = None
    questions_list: list[str] | None = None
    classes: list[str] = Field(default_factory=list)
    available_data: list[ClassDataSelection] | None = None
    selected_data: list[ClassDataAssignment] | None = None
    performance_requirements: PerformanceSpec | None = None
    constraint_strengths: ConstraintStrengths = Field(default_factory=ConstraintStrengths)
    robustness_requirements: RobustnessSpec = Field(default_factory=RobustnessSpec)
    deployment_constraints: DeploymentConstraints | None = None
    available_hardware: HardwareSpec | None = None
    model_requirements: list[ModelRequirement] | None = None
    augmentation: str | None = None
    preprocessing: str | None = None
    num_qa_pairs: int | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_selected_data(cls, value: Any) -> Any:
        return _normalize_legacy_selected_data(value)


class PipelineState(InterpretationRequirements):
    """Full, extensible planning state used by the original API contract."""

    model_config = ConfigDict(extra="allow")
    training_hardware: TrainingHardwareSpec | None = None
    class_expansions: dict[str, list[str]] = Field(default_factory=dict)
    model_selection_graph_context: dict[str, Any] | None = None
    dataset_selection_graph_context: dict[str, Any] | None = None
    hyperparameter_graph_context: dict[str, Any] | None = None
    use_graphrag: bool = True
    selected_model_info: dict[str, Any] | None = None
    hpo_config: dict[str, Any] | None = None
    hpo_decision: dict[str, Any] | None = None
    model_selection_decision_evidence: dict[str, Any] | None = None
    dataset_selection_decision_evidence: dict[str, Any] | None = None
    hyperparameter_decision_evidence: dict[str, Any] | None = None
    dataset_profile: DatasetProfile | None = None
    data_plan_constraints: dict[str, Any] | None = None
    step_history: list[str] = Field(default_factory=list)
    last_updated: str | None = None
    revision: RevisionState = Field(default_factory=RevisionState)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_selected_data(cls, value: Any) -> Any:
        """Accept historical count-only selections without discarding them.

        Full split assignment is performed by dataset planning later. At the
        contract boundary, legacy selections remain valid planning evidence.
        """
        return _normalize_legacy_selected_data(value)


def _normalize_legacy_selected_data(value: Any) -> Any:
    if not isinstance(value, dict) or value.get("selected_data") is None:
        return value
    document = dict(value)
    converted = []
    for item in document["selected_data"]:
        if all("allocations" in source for source in item.get("sources", [])):
            converted.append(item)
            continue
        converted.append({
            "class_name": item["class_name"],
            "sources": [{
                "dataset_name": source["dataset_name"],
                "allocations": [{"split": "train", "count": source["count"],
                                 "assignment_type": "official_split"}],
            } for source in item.get("sources", []) if source.get("count", 0) > 0],
        })
    document["selected_data"] = converted
    return document


class LLMFieldRationale(StrictModel):
    field: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class CommonHPOConfig(OrderedSchemaModel):
    classes: list[str] = Field(min_length=1)
    selected_data: list[ClassDataAssignment] = Field(min_length=1)
    train_data_ratio: float = Field(0.8, ge=0, lt=1)
    val_data_ratio: float = Field(0.1, ge=0, lt=1)
    test_data_ratio: float = Field(0.1, ge=0, lt=1)
    num_epochs: int = Field(ge=1)
    patience: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    model_name: str
    optimizer_name: str
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(0, ge=0)
    rationale: str

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_selected_data(cls, value: Any) -> Any:
        return _normalize_legacy_selected_data(value)


class ClassificationHPOConfig(CommonHPOConfig):
    schema_field_order = (
        "classes", "selected_data", "train_data_ratio", "val_data_ratio", "test_data_ratio",
        "num_epochs", "patience", "batch_size", "image_size", "precision", "scheduler_name",
        "min_learning_rate", "scheduler_step_size", "scheduler_gamma", "warmup_epochs",
        "warmup_start_factor", "gradient_accumulation_steps", "gradient_clip_norm",
        "freeze_backbone_epochs", "head_learning_rate_multiplier", "mixup_alpha", "cutmix_alpha",
        "random_erasing", "auto_augment_policy", "random_resized_crop_scale_min",
        "horizontal_flip_probability", "use_model_ema", "model_ema_decay", "model_ema_steps",
        "repeated_augmentation_repetitions", "use_activation_checkpointing", "track_metric",
        "model_name", "model_weights", "training_mode", "training_recipe_id", "lora_rank",
        "lora_alpha", "lora_dropout", "optimizer_name", "learning_rate", "weight_decay", "eps",
        "beta1", "beta2", "nesterov", "momentum", "alpha", "centered", "criterion_name",
        "label_smoothing", "pos_weight", "rationale", "llm_field_rationales",
    )
    train_data_ratio: float = Field(0.8, gt=0, lt=1)
    val_data_ratio: float = Field(0.1, gt=0, lt=1)
    test_data_ratio: float = Field(0.1, gt=0, lt=1)
    batch_size: int = Field(32, ge=1)
    model_name: ClassificationModelId
    optimizer_name: Literal["adamw", "sgd", "rmsprop"]
    image_size: int = Field(224, ge=32, le=MAX_IMAGE_SIDE)
    precision: Literal["fp32", "mixed"] = "fp32"
    scheduler_name: Literal["none", "cosine", "step"] = "none"
    min_learning_rate: float = Field(0, ge=0)
    scheduler_step_size: int = Field(7, ge=1)
    scheduler_gamma: float = Field(0.1, gt=0, le=1)
    warmup_epochs: int = Field(0, ge=0)
    warmup_start_factor: float = Field(0.01, gt=0, le=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    gradient_clip_norm: float = Field(0, ge=0)
    freeze_backbone_epochs: int = Field(0, ge=0)
    head_learning_rate_multiplier: float = Field(1, gt=0)
    mixup_alpha: float = Field(0, ge=0)
    cutmix_alpha: float = Field(0, ge=0)
    random_erasing: float = Field(0, ge=0, le=1)
    auto_augment_policy: Literal["none", "ta_wide"] = "none"
    random_resized_crop_scale_min: float = Field(0.6, gt=0, le=1)
    horizontal_flip_probability: float = Field(0.5, ge=0, le=1)
    use_model_ema: bool = False
    model_ema_decay: float = Field(0.99998, gt=0, lt=1)
    model_ema_steps: int = Field(32, ge=1)
    repeated_augmentation_repetitions: int = Field(1, ge=1)
    use_activation_checkpointing: bool = False
    track_metric: Literal["val_acc", "val_loss", "macro_f1", "micro_f1"]
    model_weights: Literal["default", "none"] = "default"
    training_mode: Literal[
        "fine_tune_pretrained", "staged_fine_tune", "head_only", "lora", "train_from_scratch"
    ] = "fine_tune_pretrained"
    training_recipe_id: str = ""
    lora_rank: int = Field(8, ge=1, le=256)
    lora_alpha: int = Field(16, ge=1, le=1024)
    lora_dropout: float = Field(0.05, ge=0, lt=1)
    eps: float = Field(1e-8, gt=0, le=1e-2)
    beta1: float = Field(0.9, gt=0, lt=1)
    beta2: float = Field(0.999, gt=0, lt=1)
    nesterov: bool = False
    momentum: float = Field(0, ge=0)
    alpha: float = Field(0.99, gt=0, lt=1)
    centered: bool = False
    criterion_name: Literal["cross_entropy", "bce_with_logit"]
    label_smoothing: float = Field(0, ge=0, le=1)
    pos_weight: float = Field(1, gt=0)
    llm_field_rationales: list[LLMFieldRationale] = Field(default_factory=list)


class DetectionDataPlanConstraints(StrictModel):
    minimum_unique_pool_images: int = Field(0, ge=0)
    preferred_unique_pool_images: int = Field(0, ge=0)
    preferred_target_is_strict: bool = False
    group_isolation_keys: list[str] = Field(default_factory=list)

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


def _normalize_detection_inactive_fields(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    model_name = str(normalized.get("model_name", ""))
    if model_name.startswith(("yolov8_", "yolov10_", "yolov11_", "yolov12_")):
        normalized.update({
            "aspect_ratio_range": None, "lr_milestones": [], "lambda_giou": 0.0,
            "max_size": 1333, "trainable_backbone_layers": 0,
            "horizontal_flip_probability": 0.0, "augmentation_policy": "basic",
            "topk_candidates": 400, "positive_fraction": 0.25,
            "matching_iou_threshold": 0.5,
        })
        if normalized.get("scheduler_name", "linear") != "multistep":
            normalized["scheduler_gamma"] = 0.1
    elif model_name == "rtdetr_hgnetv2_l":
        normalized.update({
            "lr_milestones": [], "scheduler_gamma": 1.0,
            "trainable_backbone_layers": 0, "horizontal_flip_probability": 0.0,
            "augmentation_policy": "basic", "topk_candidates": 400,
            "positive_fraction": 0.25, "matching_iou_threshold": 0.5,
        })
    elif model_name in {"retinanet_r50", "faster_rcnn_r50", "ssd300"}:
        normalized.update({
            "final_learning_rate_factor": 0.01, "warmup_momentum": 0.8,
            "lambda_giou": 0.0, "single_cls": False, "rect": False,
            "multi_scale": 0.0, "freeze": None, "mosaic": 0.0, "mixup": 0.0,
            "cutmix": 0.0, "copy_paste": 0.0, "degrees": 0.0, "translate": 0.0,
            "scale": 0.0, "fliplr": 0.0, "hsv_h": 0.0, "hsv_s": 0.0,
            "hsv_v": 0.0, "close_mosaic": 0,
        })
    if str(normalized.get("optimizer_name", "adamw")) != "adamw":
        normalized["beta1"] = 0.9
    if normalized.get("scheduler_name", "linear") != "multistep":
        gamma = normalized.get("scheduler_gamma", 0.1)
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or gamma <= 0:
            normalized["scheduler_gamma"] = 0.1
    return normalized


class DetectionHPOConfig(CommonHPOConfig):
    schema_field_order = (
        "task_type", "classes", "selected_data", "train_data_ratio", "val_data_ratio",
        "test_data_ratio", "num_epochs", "patience", "batch_size", "input_size",
        "aspect_ratio_range", "track_metric", "model_name", "model_weights", "training_recipe_id",
        "optimizer_name", "learning_rate", "weight_decay", "beta1", "momentum", "scheduler_name",
        "lr_milestones", "scheduler_gamma", "final_learning_rate_factor", "data_plan_constraints",
        "training_mode", "lora_rank", "lora_alpha", "lora_dropout", "lora_target_profile",
        "train_detection_head", "warmup_epochs", "warmup_momentum", "amp", "loss_box", "loss_cls",
        "lambda_box", "lambda_cls", "lambda_giou", "lambda_dfl", "mosaic", "mixup", "cutmix",
        "copy_paste", "degrees", "translate", "scale", "fliplr", "hsv_h", "hsv_s", "hsv_v",
        "close_mosaic", "single_cls", "rect", "multi_scale", "confidence_threshold",
        "nms_iou_threshold", "max_detections", "workers", "seed", "max_size",
        "trainable_backbone_layers", "horizontal_flip_probability", "augmentation_policy",
        "topk_candidates", "positive_fraction", "matching_iou_threshold", "freeze", "rationale",
        "llm_field_rationales",
    )
    task_type: Literal["detection"]
    model_name: DetectionHPOModelId
    batch_size: int = Field(16, ge=-1)
    input_size: int = Field(640, ge=32, le=MAX_IMAGE_SIDE)
    aspect_ratio_range: list[float] | None = Field(default_factory=lambda: [0.5, 2.0])
    track_metric: Literal["val_mAP", "val_mAP_50", "val_mAP_75", "val_loss"] = "val_mAP"
    model_weights: Literal["default", "none", "coco", "imagenet_backbone"] = "coco"
    training_recipe_id: str = ""
    optimizer_name: Literal["auto", "adamw", "sgd", "rmsprop"] = "adamw"
    learning_rate: float = Field(0.01, gt=0)
    weight_decay: float = Field(0.0005, ge=0)
    beta1: float = Field(0.9, gt=0, lt=1)
    momentum: float = Field(0.9, ge=0)
    scheduler_name: Literal["none", "linear", "multistep"] = "linear"
    lr_milestones: list[int] = Field(default_factory=lambda: [16, 22])
    scheduler_gamma: float = Field(0.1, gt=0, le=1)
    final_learning_rate_factor: float = Field(0.01, gt=0, le=1)
    data_plan_constraints: DetectionDataPlanConstraints = Field(default_factory=DetectionDataPlanConstraints)
    training_mode: Literal["full_finetune", "lora"] = "full_finetune"
    lora_rank: int = Field(8, ge=1, le=64)
    lora_alpha: int = Field(16, ge=1, le=256)
    lora_dropout: float = Field(0.05, ge=0, lt=1)
    lora_target_profile: Literal["decoder_attention", "decoder_attention_and_ffn"] = "decoder_attention"
    train_detection_head: bool = True
    warmup_epochs: float = Field(3.0, ge=0)
    warmup_momentum: float = Field(0.8, ge=0, lt=1)
    amp: bool = True
    loss_box: Literal["l1", "l1_giou", "smooth_l1", "giou", "diou", "ciou"] = "ciou"
    loss_cls: Literal["cross_entropy", "bce", "focal", "varifocal"] = "bce"
    lambda_box: float = Field(7.5, gt=0)
    lambda_cls: float = Field(0.5, gt=0)
    lambda_giou: float = Field(0, ge=0)
    lambda_dfl: float = Field(1.5, ge=0)
    mosaic: float = Field(1, ge=0, le=1)
    mixup: float = Field(0, ge=0, le=1)
    cutmix: float = Field(0, ge=0, le=1)
    copy_paste: float = Field(0, ge=0, le=1)
    degrees: float = Field(0, ge=0, le=180)
    translate: float = Field(0.1, ge=0, le=1)
    scale: float = Field(0.5, ge=0)
    fliplr: float = Field(0.5, ge=0, le=1)
    hsv_h: float = Field(0.015, ge=0, le=1)
    hsv_s: float = Field(0.7, ge=0, le=1)
    hsv_v: float = Field(0.4, ge=0, le=1)
    close_mosaic: int = Field(10, ge=0)
    single_cls: bool = False
    rect: bool = False
    multi_scale: float = Field(0, ge=0, le=1)
    confidence_threshold: float = Field(0.25, ge=0, le=1)
    nms_iou_threshold: float = Field(0.7, ge=0, le=1)
    max_detections: int = Field(300, ge=1)
    workers: int = Field(8, ge=0)
    seed: int = Field(0, ge=0)
    max_size: int = Field(1333, ge=32, le=MAX_IMAGE_SIDE)
    trainable_backbone_layers: int = Field(3, ge=0, le=5)
    horizontal_flip_probability: float = Field(0.5, ge=0, le=1)
    augmentation_policy: Literal["basic", "ssd"] = "basic"
    topk_candidates: int = Field(400, ge=1)
    positive_fraction: float = Field(0.25, gt=0, lt=1)
    matching_iou_threshold: float = Field(0.5, gt=0, lt=1)
    freeze: int | None = Field(None, ge=0)
    llm_field_rationales: list[LLMFieldRationale] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_detection_draft(cls, value: Any) -> Any:
        return _normalize_detection_inactive_fields(_normalize_legacy_selected_data(value))


class VQAHPOConfig(CommonHPOConfig):
    schema_field_order = (
        "task_type", "classes", "selected_data", "train_data_ratio", "val_data_ratio",
        "test_data_ratio", "num_epochs", "patience", "batch_size", "max_seq_length", "track_metric",
        "model_name", "precision", "use_lora", "lora_r", "lora_alpha", "lora_dropout",
        "optimizer_name", "learning_rate", "weight_decay", "eps", "beta1", "beta2", "nesterov",
        "momentum", "alpha", "centered", "rationale",
    )
    task_type: Literal["visual question answering"] = "visual question answering"
    classes: list[str]
    batch_size: int = Field(2, ge=1)
    model_name: VQAModelId
    optimizer_name: Literal["adamw", "paged_adamw_8bit", "rmsprop", "sgd"] = "adamw"
    learning_rate: float = Field(2e-5, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    max_seq_length: int = Field(2048, ge=128)
    track_metric: Literal["val_loss", "exact_match", "f1", "meteor", "rouge", "cider"] = "val_loss"
    precision: Literal["bf16", "fp16", "fp32", "fp8"] = "bf16"
    use_lora: bool = True
    lora_r: int = Field(16, ge=1)
    lora_alpha: int = Field(32, ge=1)
    lora_dropout: float = Field(0.05, ge=0, lt=1)
    eps: float = Field(1e-8, gt=0, lt=0.01)
    beta1: float = Field(0.9, gt=0, lt=1)
    beta2: float = Field(0.999, gt=0, lt=1)
    nesterov: bool = False
    momentum: float = Field(0, ge=0)
    alpha: float = Field(0.99, gt=0, lt=1)
    centered: bool = False

    @model_validator(mode="after")
    def validate_combinations(self):
        total_ratio = self.train_data_ratio + self.val_data_ratio + self.test_data_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
            raise ValueError(
                "train_data_ratio, val_data_ratio, and test_data_ratio must sum to 1.0. "
                f"Current sum: {total_ratio}"
            )
        if self.learning_rate > 1e-3:
            raise ValueError(
                "Learning rate is suspiciously high for a pre-trained VLM. It should "
                "typically be <= 1e-3 to avoid catastrophic forgetting."
            )
        return self


class HPOFinding(StrictModel):
    field: str
    severity: Literal["hard_error", "safety_warning", "preference"]
    reason: str
    recommended_value: str | None = None
    rule_id: str | None = None


class HPODecision(StrictModel):
    accept: bool
    reason: str
    findings: list[HPOFinding] = Field(default_factory=list)
    suggestions: list[str] | None = None


class HyperparameterProposal(StrictModel):
    """Rich confirmation payload retained by the viewer planning API."""

    task: Literal["classification", "detection", "visual question answering"]
    hpo_config: ClassificationHPOConfig | DetectionHPOConfig | VQAHPOConfig
    decision: HPODecision
    graph_context: dict[str, Any] | None = None
    decision_evidence: dict[str, Any] | None = None
    field_provenance: dict[str, Any] = Field(default_factory=dict)
