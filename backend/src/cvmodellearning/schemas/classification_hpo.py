import math
from typing import List, Literal, Self
from pydantic import BaseModel, Field, ConfigDict, model_validator
from cvmodellearning.models.registry import (
    ClassificationModelId,
    FREEZABLE_CLASSIFICATION_MODEL_IDS,
    HEAD_LR_MULTIPLIER_MODEL_IDS,
    LORA_CLASSIFICATION_MODEL_IDS,
)
from cvmodellearning.models.classification_capabilities import classification_capabilities
from cvmodellearning.schemas.hpo_runtime import build_runtime_hpo_config
from cvmodellearning.schemas.dataset_assignment import (
    ClassDataAssignment,
    normalize_dataset_assignments,
)
from cvmodellearning.training.resource_guard import MAX_IMAGE_SIDE

CLASSIFICATION_OPTIMIZER_PARAM_FIELDS = {
    "adamw": ("learning_rate", "weight_decay", "eps", "beta1", "beta2"),
    "sgd": ("learning_rate", "weight_decay", "momentum", "nesterov"),
    "rmsprop": ("learning_rate", "weight_decay", "eps", "momentum", "alpha", "centered"),
}

CLASSIFICATION_CRITERION_PARAM_FIELDS = {
    "cross_entropy": ("label_smoothing",),
    "bce_with_logit": ("pos_weight",),
}

class LLMFieldRationale(BaseModel):
    """The LLM's explanation for a field it completed or repaired."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., min_length=1, description="Top-level configuration field being explained.")
    reason: str = Field(..., min_length=1, description="Why this value fits the model, data, and hardware.")


class ClassificationConfigFields(BaseModel):
    """
    Union-free schema designed for structured outputs:
    - Uses registry-backed enums for selector fields (no unions/oneOf).
    - Keeps parameters as simple primitives.
    - Provides the shared fields for LLM drafts and executable configs.
    """
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    # Data/task
    classes: List[str] = Field(
        ..., min_length=1,
        description="Provide class names in training label order; list must be non-empty."
    )
    
    # CHANGED: Replaced Dict with strictly typed List
    selected_data: List[ClassDataAssignment] = Field(
        ..., min_length=1,
        description="Authoritative class/source train, validation, and test assignments from dataset planning."
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_selected_data(cls, value):
        if isinstance(value, dict) and "selected_data" in value:
            value = dict(value)
            value["selected_data"] = [
                item.model_dump(mode="json")
                for item in normalize_dataset_assignments(value["selected_data"] or [])
            ]
        return value
    
    train_data_ratio: float = Field(
        0.8, gt=0.0, lt=1.0,
        description="Deprecated execution compatibility ratio derived from the planned assignments."
    )
    val_data_ratio: float = Field(
        0.1, gt=0.0, lt=1.0,
        description="Deprecated execution compatibility ratio derived from the planned assignments."
    )
    test_data_ratio: float = Field(
        0.1, gt=0.0, lt=1.0,
        description="Deprecated execution compatibility ratio derived from the planned assignments."
    )
    
    # Training loop
    num_epochs: int = Field(
        ..., ge=1,
        description="Set the maximum number of training epochs; must be ≥ 1."
    )
    patience: int = Field(
        ..., ge=0,
        description=(
            "Set epochs without improvement to wait before early stopping; "
            "use 0 to disable early stopping."
        )
    )
    batch_size: int = Field(
        32, ge=1,
        description="Set mini-batch size in samples; must be ≥ 1."
    )
    image_size: int = Field(
        224, ge=32, le=MAX_IMAGE_SIDE,
        description=(
            f"Set square input image size in pixels after resize/crop; must be between 32 "
            f"and {MAX_IMAGE_SIDE}."
        )
    )
    precision: Literal["fp32", "mixed"] = Field(
        "fp32",
        description="Use full precision or automatic mixed precision (AMP; CUDA only).",
    )
    scheduler_name: Literal["none", "cosine", "step"] = Field(
        "none",
        description="Select no scheduler, cosine decay, or fixed-interval step decay.",
    )
    min_learning_rate: float = Field(
        0.0, ge=0.0,
        description="Set the final cosine learning rate; used when scheduler_name='cosine'.",
    )
    scheduler_step_size: int = Field(
        7, ge=1,
        description="Decay the learning rate after this many epochs when scheduler_name='step'.",
    )
    scheduler_gamma: float = Field(
        0.1, gt=0.0, le=1.0,
        description="Multiply the learning rate by this factor at each StepLR boundary.",
    )
    warmup_epochs: int = Field(
        0, ge=0,
        description="Linearly warm up the learning rate for this many initial epochs.",
    )
    warmup_start_factor: float = Field(
        0.01, gt=0.0, le=1.0,
        description="Initial learning-rate fraction during linear warmup.",
    )
    gradient_accumulation_steps: int = Field(
        1, ge=1,
        description="Accumulate this many micro-batches before each optimizer update.",
    )
    gradient_clip_norm: float = Field(
        0.0, ge=0.0,
        description="Clip the global gradient norm; 0 disables clipping.",
    )
    freeze_backbone_epochs: int = Field(
        0, ge=0,
        description="Train only the classification head for these initial epochs, then unfreeze the backbone.",
    )
    head_learning_rate_multiplier: float = Field(
        1.0, gt=0.0,
        description="Multiply the base learning rate for the newly initialized classification head.",
    )
    mixup_alpha: float = Field(
        0.0, ge=0.0,
        description="MixUp alpha for batch augmentation; 0 disables MixUp.",
    )
    cutmix_alpha: float = Field(
        0.0, ge=0.0,
        description="CutMix alpha for batch augmentation; 0 disables CutMix.",
    )
    random_erasing: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Probability of random erasing in the training transform.",
    )
    auto_augment_policy: Literal["none", "ta_wide"] = Field(
        "none",
        description="Select no automatic augmentation or TorchVision TrivialAugmentWide.",
    )
    random_resized_crop_scale_min: float = Field(
        0.6, gt=0.0, le=1.0,
        description=(
            "Set the minimum retained image-area fraction for RandomResizedCrop. "
            "A conservative 0.6 default reduces label-destroying crops during fine-tuning."
        ),
    )
    horizontal_flip_probability: float = Field(
        0.5, ge=0.0, le=1.0,
        description=(
            "Set horizontal-flip probability; use 0 for text, laterality, direction, or other "
            "orientation-sensitive labels."
        ),
    )
    use_model_ema: bool = Field(
        False,
        description="Maintain an exponential moving average of model parameters for validation and checkpoints.",
    )
    model_ema_decay: float = Field(
        0.99998, gt=0.0, lt=1.0,
        description="Decay used by model exponential moving average.",
    )
    model_ema_steps: int = Field(
        32, ge=1,
        description="Update model EMA after this many optimizer steps.",
    )
    repeated_augmentation_repetitions: int = Field(
        1, ge=1,
        description=(
            "Build this many independently augmented candidates per source image while keeping each epoch "
            "approximately one dataset traversal; 1 disables repeated augmentation."
        ),
    )
    use_activation_checkpointing: bool = Field(
        False,
        description="Checkpoint Swin Transformer blocks to reduce activation memory at the cost of extra compute.",
    )
    track_metric: Literal["val_acc", "val_loss", "macro_f1", "micro_f1"] = Field(
        ...,
        description="Choose the validation metric to monitor for early stopping/checkpointing."
    )

    # Model selection (no unions)
    model_name: ClassificationModelId = Field(
        ...,
        description="Select the backbone architecture identifier."
    )
    model_weights: Literal["default", "none"] = Field(
        "default",
        description="Choose pretrained weights policy: 'default' for ImageNet-style pretrain or 'none' for random init."
    )
    training_mode: Literal[
        "fine_tune_pretrained",
        "staged_fine_tune",
        "head_only",
        "lora",
        "train_from_scratch",
    ] = Field(
        "fine_tune_pretrained",
        description=(
            "Declare how the model is initialized and trained. This must agree with model_weights and "
            "freeze_backbone_epochs."
        ),
    )
    training_recipe_id: str = Field(
        "",
        description="Record the ontology training-recipe ID used to ground this configuration, when available.",
    )
    lora_rank: int = Field(
        8, ge=1, le=256,
        description="Set the LoRA low-rank dimension; used only when training_mode='lora'.",
    )
    lora_alpha: int = Field(
        16, ge=1, le=1024,
        description="Set the LoRA scaling numerator; used only when training_mode='lora'.",
    )
    lora_dropout: float = Field(
        0.05, ge=0.0, lt=1.0,
        description="Set dropout on LoRA branches; used only when training_mode='lora'.",
    )

    # Optimizer selection (no unions)
    optimizer_name: Literal["adamw", "sgd", "rmsprop"] = Field(
        ...,
        description="Select the optimizer algorithm."
    )

    # Core optimizer params (all present; irrelevant ones are ignored by code)
    learning_rate: float = Field(
        ..., gt=0,
        description="Set the base learning rate; must be > 0."
    )
    weight_decay: float = Field(
        0.0, ge=0,
        description="Set L2 regularization coefficient (weight decay); must be ≥ 0."
    )

    # AdamW
    eps: float = Field(
        1e-8, gt=0, le=1e-2,
        description="Set AdamW epsilon for numerical stability; must be in (0, 1e-2]. Only use when optimizer_name='adamw'."
    )
    beta1: float = Field(
        0.9, gt=0, lt=1,
        description="Set AdamW β1 (first-moment momentum); must be in (0, 1). Only use when optimizer_name='adamw'."
    )
    beta2: float = Field(
        0.999, gt=0, lt=1,
        description="Set AdamW β2 (second-moment momentum); must be in (0, 1). Only use when optimizer_name='adamw'."
    )

    # SGD
    nesterov: bool = Field(
        False,
        description="Enable Nesterov momentum for SGD when True. Only use when optimizer_name='sgd'."
    )
    momentum: float = Field(
        0.0, ge=0,
        description="Set momentum factor; must be ≥ 0. Only use when optimizer_name='sgd' or optimizer_name='rmsprop'."
    )

    # RMSprop
    alpha: float = Field(
        0.99, gt=0, lt=1,
        description="Set RMSprop smoothing constant α; must be in (0, 1). Only use when optimizer_name='rmsprop'."
    )
    centered: bool = Field(
        False,
        description="Use centered RMSprop variant when True. Only use when optimizer_name='rmsprop'."
    )

    # Criterion (no unions)
    criterion_name: Literal["cross_entropy", "bce_with_logit"] = Field(
        ...,
        description="Select the loss function."
    )

    # Criterion-specific (kept simple; always present)
    label_smoothing: float = Field(
        0.0, ge=0, le=1,
        description="Set label smoothing for cross-entropy; 0.0 disables smoothing; must be in [0, 1]. Used when criterion_name='cross_entropy'"
    )  # cross_entropy
    pos_weight: float = Field(
        1.0, gt=0,
        description="Set positive class weight for BCEWithLogits; use 1.0 for balanced classes; must be > 0. Used when criterion_name='bce_with_logit'"
    )  # bce_with_logit

    rationale: str = Field(
        ..., 
        description="Explain the choice of hyperparameters. Mention specific heuristics."
    )
    llm_field_rationales: List[LLMFieldRationale] = Field(
        default_factory=list,
        description=(
            "Explain every field listed in GraphRAG fields_requiring_llm_completion and every "
            "field changed during an evaluator-authorized repair."
        ),
    )

class ClassificationConfigDraft(ClassificationConfigFields):
    """LLM structured output with field-level validation only."""


class ClassificationConfigModel(ClassificationConfigFields):
    """Executable configuration with all cross-field constraints enforced."""

    @model_validator(mode="after")
    def _validate_combinations(self) -> Self:
        # Validate data split ratios
        total_ratio = self.train_data_ratio + self.val_data_ratio + self.test_data_ratio
        # Allow small floating-point noise; accept sums within an absolute tolerance.
        if not math.isclose(total_ratio, 1.0, abs_tol=1e-4):
            raise ValueError(f"train_data_ratio, val_data_ratio, and test_data_ratio must sum to 1.0. Current sum: {total_ratio}")

        if self.optimizer_name == "adamw":
            pass
        elif self.optimizer_name == "sgd":
            pass
        elif self.optimizer_name == "rmsprop":
            pass

        # Criterion compatibility
        if self.criterion_name == "cross_entropy":
            if self.pos_weight != 1.0:
                raise ValueError("For cross_entropy, pos_weight must be 1.0 (unused).")
        elif self.criterion_name == "bce_with_logit":
            if getattr(self, "label_smoothing", 0.0) not in (0.0,):
                raise ValueError("label_smoothing must be 0.0 for bce_with_logit.")

        if self.criterion_name != "cross_entropy":
            raise ValueError(
                "The current classification pipeline supports only cross_entropy because datasets emit "
                "one integer class target per image; BCEWithLogitsLoss requires multi-hot targets."
            )

        if self.warmup_epochs > self.num_epochs:
            raise ValueError("warmup_epochs cannot exceed num_epochs.")
        if self.patience and self.patience >= self.num_epochs:
            raise ValueError("patience must be lower than num_epochs when early stopping is enabled.")
        if self.scheduler_name == "cosine" and self.min_learning_rate >= self.learning_rate:
            raise ValueError("min_learning_rate must be lower than learning_rate for cosine scheduling.")
        if self.scheduler_name == "cosine" and self.warmup_epochs >= self.num_epochs:
            raise ValueError("Cosine scheduling requires warmup_epochs to be lower than num_epochs.")
        if self.scheduler_name == "step" and self.warmup_epochs >= self.num_epochs:
            raise ValueError("Step scheduling requires warmup_epochs to be lower than num_epochs.")
        if (
            self.scheduler_name == "step"
            and self.scheduler_step_size > self.num_epochs - self.warmup_epochs
        ):
            raise ValueError(
                "scheduler_step_size cannot exceed the post-warmup training epochs."
            )
        if self.scheduler_name == "none" and self.min_learning_rate != 0.0:
            raise ValueError("min_learning_rate must be 0 when scheduler_name='none'.")
        if self.scheduler_name == "step" and self.min_learning_rate != 0.0:
            raise ValueError("min_learning_rate must be 0 when scheduler_name='step'.")
        if self.criterion_name != "cross_entropy" and (self.mixup_alpha > 0 or self.cutmix_alpha > 0):
            raise ValueError("MixUp and CutMix currently require criterion_name='cross_entropy'.")
        if (self.mixup_alpha > 0 or self.cutmix_alpha > 0) and self.batch_size < 2:
            raise ValueError("MixUp and CutMix require batch_size >= 2.")
        if self.use_activation_checkpointing and not str(self.model_name).startswith("swin_v2_"):
            raise ValueError("use_activation_checkpointing is currently supported only for Swin V2 models.")
        if self.training_mode == "head_only" and self.use_activation_checkpointing:
            raise ValueError("Activation checkpointing has no benefit during head-only training.")

        model_name = str(self.model_name)
        is_swin_v2 = model_name.startswith("swin_v2_")
        capabilities = classification_capabilities(model_name)
        if (
            not capabilities.configurable_image_size
            and self.image_size != capabilities.native_image_size
        ):
            raise ValueError(
                f"{model_name} supports only image_size={capabilities.native_image_size} "
                "with the registered constructor and weights."
            )
        if self.freeze_backbone_epochs > self.num_epochs:
            raise ValueError("freeze_backbone_epochs cannot exceed num_epochs.")
        requires_pretrained = self.training_mode in {
            "fine_tune_pretrained",
            "staged_fine_tune",
            "head_only",
            "lora",
        }
        if requires_pretrained and self.model_weights != "default":
            raise ValueError(
                f"training_mode='{self.training_mode}' requires model_weights='default' pretrained weights."
            )
        if self.training_mode == "train_from_scratch":
            if self.model_weights != "none":
                raise ValueError("training_mode='train_from_scratch' requires model_weights='none'.")
            if self.freeze_backbone_epochs != 0:
                raise ValueError("train_from_scratch cannot freeze a randomly initialized backbone.")
        elif self.training_mode == "fine_tune_pretrained" and self.freeze_backbone_epochs != 0:
            raise ValueError(
                "fine_tune_pretrained trains the full pretrained model immediately; use staged_fine_tune "
                "or head_only when freeze_backbone_epochs is non-zero."
            )
        elif self.training_mode == "staged_fine_tune":
            if not 0 < self.freeze_backbone_epochs < self.num_epochs:
                raise ValueError(
                    "staged_fine_tune requires freeze_backbone_epochs greater than 0 and lower than num_epochs."
                )
        elif self.training_mode == "head_only" and self.freeze_backbone_epochs != self.num_epochs:
            raise ValueError("head_only requires freeze_backbone_epochs to equal num_epochs.")
        elif self.training_mode == "lora":
            if model_name not in LORA_CLASSIFICATION_MODEL_IDS:
                raise ValueError(f"LoRA is not executable for classification model '{model_name}'.")
            if self.freeze_backbone_epochs != 0:
                raise ValueError("LoRA freezes the base model itself; freeze_backbone_epochs must be 0.")
            if self.head_learning_rate_multiplier != 1.0:
                raise ValueError("LoRA uses one optimizer rate for adapters and the classifier head.")
            if self.use_model_ema:
                raise ValueError("Model EMA is not supported for adapter-only LoRA checkpoints.")

        if (
            self.model_weights == "default"
            and self.training_mode == "fine_tune_pretrained"
            and self.image_size <= 32
            and self.batch_size <= 4
            and self.learning_rate >= 1e-3
        ):
            raise ValueError(
                "Pretrained full-model fine-tuning at image_size <= 32 with "
                "batch_size <= 4 requires learning_rate < 1e-3, staged "
                "fine-tuning, or a larger image size."
            )

        if self.training_mode != "lora" and (
            self.lora_rank != 8 or self.lora_alpha != 16 or self.lora_dropout != 0.05
        ):
            raise ValueError(
                "LoRA fields must retain their inactive defaults unless training_mode='lora'."
            )

        if (
            self.training_mode in {"staged_fine_tune", "head_only"}
            and model_name not in FREEZABLE_CLASSIFICATION_MODEL_IDS
        ):
            raise ValueError(
                "staged_fine_tune and head_only require a registered classifier-head/backbone mapping."
            )
        if (
            self.head_learning_rate_multiplier != 1.0
            and model_name not in HEAD_LR_MULTIPLIER_MODEL_IDS
        ):
            raise ValueError(
                "head_learning_rate_multiplier is not executable for the selected model."
            )

        if is_swin_v2:
            if self.image_size % 32 != 0:
                raise ValueError("Swin V2 image_size must be divisible by 32.")
            if self.model_weights == "default" and self.image_size < 256:
                raise ValueError("Pretrained Swin V2 fine-tuning requires image_size >= 256.")
            if self.freeze_backbone_epochs > 0 and self.model_weights == "none":
                raise ValueError("A randomly initialized Swin V2 backbone cannot be frozen; use model_weights='default'.")

        return self

    def runtime_config(self) -> dict:
        config = build_runtime_hpo_config(
            self.model_dump(
                exclude_none=True,
                exclude={"rationale", "llm_field_rationales"},
            ),
            CLASSIFICATION_OPTIMIZER_PARAM_FIELDS,
            CLASSIFICATION_CRITERION_PARAM_FIELDS,
        )
        for field_name in set(config) - active_classification_config_fields(config):
            config.pop(field_name, None)
        return config


def active_classification_config_fields(config: dict) -> set[str]:
    """Return fields that can affect the selected classification runtime path."""
    active = set(config) - {"rationale", "llm_field_rationales", "field_provenance"}

    if config.get("training_mode") != "lora":
        active -= {"lora_rank", "lora_alpha", "lora_dropout"}
    if not config.get("use_model_ema", False):
        active -= {"model_ema_decay", "model_ema_steps"}

    scheduler_name = config.get("scheduler_name")
    if scheduler_name != "cosine":
        active.discard("min_learning_rate")
    if scheduler_name != "step":
        active -= {"scheduler_step_size", "scheduler_gamma"}
    if int(config.get("warmup_epochs", 0)) == 0:
        active.discard("warmup_start_factor")

    optimizer_name = config.get("optimizer_name")
    optimizer_fields = set().union(*map(set, CLASSIFICATION_OPTIMIZER_PARAM_FIELDS.values()))
    active -= optimizer_fields - set(CLASSIFICATION_OPTIMIZER_PARAM_FIELDS.get(optimizer_name, ()))

    criterion_name = config.get("criterion_name")
    criterion_fields = set().union(*map(set, CLASSIFICATION_CRITERION_PARAM_FIELDS.values()))
    active -= criterion_fields - set(CLASSIFICATION_CRITERION_PARAM_FIELDS.get(criterion_name, ()))

    return active
