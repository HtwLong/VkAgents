"""PyTorch-free copies of the full backend's planning data contracts.

These models deliberately live in the viewer package and import only Pydantic.
They describe planning documents; they do not instantiate models, optimizers,
datasets, or any other execution object.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    hardware_category: str | None = None
    cpu_cores: int | None = Field(None, ge=1)
    gpu_type: str | None = None
    gpu_count: int | None = Field(None, ge=0)
    vram_gb: float | None = Field(None, ge=0)
    ram_gb: float | None = Field(None, ge=1)
    storage_gb: float | None = Field(None, ge=1)
    details: str | None = None


class TrainingHardwareSpec(StrictModel):
    profile_id: str
    accelerator: Literal["cpu", "mps", "cuda"]
    hardware_category: str
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
    latency_category: str | None = None
    accuracy_category: str | None = None
    other_constraints: list[str] | None = None


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
    memory_category: str | None = None
    max_runtime_memory_mb: float | None = Field(None, gt=0)
    max_model_size_mb: float | None = Field(None, gt=0)
    max_parameters_m: float | None = Field(None, gt=0)
    max_cpu_latency_ms: float | None = Field(None, gt=0)
    hard_limits: list[str] = Field(default_factory=list)


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


class RevisionChange(StrictModel):
    id: str
    target_step: Literal[
        "task-interpretation", "model-selection", "dataset-selection", "choose-hyperparameters"
    ]
    field: str
    operation: Literal["set", "include", "exclude", "prefer", "avoid"] = "set"
    value: Any = None
    strength: Literal["required", "preferred"]
    summary: str


class RevisionPlan(StrictModel):
    required_text: str = ""
    preferred_text: str = ""
    summary: str
    restart_from: Literal[
        "task-interpretation", "model-selection", "dataset-selection", "choose-hyperparameters"
    ]
    changes: list[RevisionChange] = Field(default_factory=list)


class RevisionState(StrictModel):
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


class PipelineState(BaseModel):
    """Full, extensible planning state used by the original API contract."""

    model_config = ConfigDict(extra="allow")
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


class LLMFieldRationale(StrictModel):
    field: str
    reason: str


class CommonHPOConfig(StrictModel):
    classes: list[str]
    selected_data: list[ClassDataAssignment]
    train_data_ratio: float = 0.8
    val_data_ratio: float = 0.1
    test_data_ratio: float = 0.1
    num_epochs: int = Field(ge=1)
    patience: int = Field(ge=0)
    batch_size: int
    model_name: str
    optimizer_name: str
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    rationale: str


class ClassificationHPOConfig(CommonHPOConfig):
    image_size: int = Field(224, ge=32)
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
    training_mode: str = "fine_tune_pretrained"
    training_recipe_id: str = ""
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    nesterov: bool = False
    momentum: float = 0
    alpha: float = 0.99
    centered: bool = False
    criterion_name: Literal["cross_entropy", "bce_with_logit"]
    label_smoothing: float = 0
    pos_weight: float = 1
    llm_field_rationales: list[LLMFieldRationale] = Field(default_factory=list)


class DetectionHPOConfig(CommonHPOConfig):
    task_type: Literal["detection"] = "detection"
    input_size: int = Field(640, ge=32)
    aspect_ratio_range: list[float] | None = Field(default_factory=lambda: [0.5, 2.0])
    track_metric: str = "val_mAP"
    model_weights: str = "coco"
    training_recipe_id: str = ""
    beta1: float = 0.9
    momentum: float = 0.937
    scheduler_name: str = "cosine"
    lr_milestones: list[int] = Field(default_factory=list)
    scheduler_gamma: float = 0.1
    final_learning_rate_factor: float = 0.01
    data_plan_constraints: dict[str, Any] = Field(default_factory=dict)
    training_mode: str = "fine_tune_pretrained"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_profile: str = "decoder_attention"
    train_detection_head: bool = True
    warmup_epochs: int = 0
    warmup_momentum: float = 0.8
    amp: bool = False
    loss_box: str
    loss_cls: str
    lambda_box: float = 1
    lambda_cls: float = 1
    lambda_giou: float = 0
    lambda_dfl: float = 0
    mosaic: float = 0
    mixup: float = 0
    cutmix: float = 0
    copy_paste: float = 0
    degrees: float = 0
    translate: float = 0
    scale: float = 0
    fliplr: float = 0
    hsv_h: float = 0
    hsv_s: float = 0
    hsv_v: float = 0
    close_mosaic: int = 0
    single_cls: bool = False
    rect: bool = False
    multi_scale: float = 0
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.7
    max_detections: int = 300
    workers: int = 0
    seed: int = 0
    max_size: int = 640
    trainable_backbone_layers: int = 0
    horizontal_flip_probability: float = 0
    augmentation_policy: str = "basic"
    topk_candidates: int = 400
    positive_fraction: float = 0.25
    matching_iou_threshold: float = 0.5
    freeze: int | None = None
    llm_field_rationales: list[LLMFieldRationale] = Field(default_factory=list)


class VQAHPOConfig(CommonHPOConfig):
    task_type: Literal["visual question answering"] = "visual question answering"
    max_seq_length: int = Field(2048, ge=128)
    track_metric: str = "val_loss"
    precision: Literal["bf16", "fp16", "fp32", "fp8"] = "bf16"
    use_lora: bool = True
    lora_r: int = Field(16, ge=1)
    lora_alpha: int = Field(32, ge=1)
    lora_dropout: float = Field(0.05, ge=0, lt=1)
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    nesterov: bool = False
    momentum: float = 0
    alpha: float = 0.99
    centered: bool = False


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

