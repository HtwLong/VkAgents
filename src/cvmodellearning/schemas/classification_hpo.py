import math
from typing import List, Literal, Optional, Self
from pydantic import BaseModel, Field, ConfigDict, model_validator

# --- Helper Models for Strict JSON Schema ---
class DatasetSourceCount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str = Field(..., description="The name of the dataset")
    count: int = Field(..., description="Number of images from this dataset")

class ClassDataSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_name: str = Field(..., description="The name of the class or subset")
    sources: List[DatasetSourceCount] = Field(..., description="List of datasets and their counts")


class ClassificationConfigModel(BaseModel):
    """
    Union-free schema designed for structured outputs:
    - Uses Literal enums for selector fields (no unions/oneOf).
    - Keeps parameters as simple primitives.
    - Enforces cross-field constraints in a post-parse validator.
    """
    model_config = ConfigDict(extra="forbid")

    # Data/task
    classes: List[str] = Field(
        ..., min_length=1,
        description="Provide class names in training label order; list must be non-empty."
    )
    img_per_class: int = Field(
        ..., ge=1,
        description="Set target images per class used for sampling/balancing; must be ≥ 1."
    )
    
    # CHANGED: Replaced Dict with strictly typed List
    selected_data: List[ClassDataSelection] = Field(
        ..., 
        description="List of selected classes, their dataset sources, and respective image counts to download."
    )
    
    train_data_ratio: float = Field(
        0.8, ge=0.0, lt=1.0,
        description="Proportion of the dataset allocated for training; must be in [0, 1)."
    )
    val_data_ratio: float = Field(
        0.1, ge=0.0, lt=1.0,
        description="Proportion of the dataset allocated for validation; must be in [0, 1)."
    )
    test_data_ratio: float = Field(
        0.1, ge=0.0, lt=1.0,
        description="Proportion of the dataset allocated for testing; must be in [0, 1)."
    )
    
    # Training loop
    num_epochs: int = Field(
        ..., ge=1,
        description="Set the maximum number of training epochs; must be ≥ 1."
    )
    patience: int = Field(
        ..., ge=0,
        description="Set epochs without improvement to wait before early stopping; must be ≥ 0."
    )
    batch_size: int = Field(
        32, ge=1,
        description="Set mini-batch size in samples; must be ≥ 1."
    )
    image_size: int = Field(
        224, ge=16,
        description="Set square input image size in pixels after resize/crop; must be ≥ 16."
    )
    track_metric: Literal["val_acc", "val_loss", "macro_f1", "micro_f1"] = Field(
        ...,
        description="Choose the validation metric to monitor for early stopping/checkpointing."
    )

    # Model selection (no unions)
    model_name: Literal[
        "resnet50", "vgg16", "mobilenet_v2", "mobilenet_v3_large",
        "efficientnet_b0", "densenet121", "convnext_tiny", "vit_b_16",
        "swin_v2_t", "swin_v2_s", "swin_v2_b"
    ] = Field(
        ...,
        description="Select the backbone architecture identifier."
    )
    model_weights: Literal["default", "none"] = Field(
        "default",
        description="Choose pretrained weights policy: 'default' for ImageNet-style pretrain or 'none' for random init."
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
        1e-8, gt=0,
        description="Set AdamW epsilon for numerical stability; must be > 0. Only use when optimizer_name='adamw'."
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

    @model_validator(mode="after")
    def _validate_combinations(self) -> Self:
        # Validate data split ratios
        total_ratio = self.train_data_ratio + self.val_data_ratio + self.test_data_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
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

        return self