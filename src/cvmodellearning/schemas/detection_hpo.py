import math
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing_extensions import Self

# --- Unified Helper Models for Strict JSON Schema ---
class DatasetSourceCount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str = Field(..., description="The name of the dataset")
    count: int = Field(..., description="Number of images from this dataset")

class ClassDataSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_name: str = Field(..., description="The name of the class or subset")
    sources: List[DatasetSourceCount] = Field(..., description="List of datasets and their counts")


class DetectionConfigModel(BaseModel):
    """
    Structured schema for Object Detection training configuration.
    Uses Literal enums for key selector fields.
    """
    model_config = ConfigDict(extra="forbid")

    # --- Data/Task ---
    task_type: Literal["detection", "segmentation"] = Field(
        ...,
        description="The task type: 'detection' for bounding boxes, 'segmentation' for masks (e.g., Mask R-CNN)."
    )
    classes: List[str] = Field(
        ..., min_length=1,
        description="Provide class names in training label order; list must be non-empty."
    )
    
    # CHANGED: Unified schema
    selected_data: List[ClassDataSelection] = Field(
        ..., 
        description="List of selected classes, their data sources and the respective image counts."
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
    
    # --- Training Loop ---
    num_epochs: int = Field(
        ..., ge=1,
        description="Set the maximum number of training epochs; must be ≥ 1."
    )
    patience: int = Field(
        ..., ge=0,
        description="Set epochs without improvement to wait before early stopping; must be ≥ 0."
    )
    batch_size: int = Field(
        16, ge=1,
        description="Set mini-batch size in samples (often smaller than classification); must be ≥ 1."
    )
    input_size: int = Field(
        640, ge=32,
        description="Set square input image size in pixels after resize/padding (e.g., 416, 640); must be ≥ 32."
    )
    aspect_ratio_range: Optional[List[float]] = Field(
        [0.5, 2.0],
        description="Range [min, max] of aspect ratios to sample images for."
    )
    track_metric: Literal["val_mAP", "val_mAP_50", "val_mAP_75", "val_loss"] = Field(
        ...,
        description="Choose the validation metric to monitor for early stopping/checkpointing (mAP is standard)."
    )

    # --- Model Selection (Detection Models) ---
    model_name: Literal[
        "yolov8_n", "yolov8_s", "yolov8_m","yolov8_l", "yolov8_x",
        "yolov10_n", "yolov10_s", "yolov10_m","yolov10_l", "yolov10_x", 
        "yolov11_n", "yolov11_s", "yolov11_m","yolov11_l", "yolov11_x",
        "yolov12_n", "yolov12_s", "yolov12_m","yolov12_l", "yolov12_x", 
        "retinanet_r50", "faster_rcnn_r50", "mask_rcnn_r50", "ssd300", "rt_detr_r50"
    ] = Field(
        ...,
        description="Select the object detection architecture identifier."
    )
    model_weights: Literal["default", "none", "coco"] = Field(
        "coco",
        description="Choose pretrained weights policy: 'coco' for COCO pretrain, 'default' (usually ImageNet) or 'none'."
    )
    
    # --- Optimizer Selection (Adapted from Classification) ---
    optimizer_name: Literal["adamw", "sgd", "rmsprop"] = Field(
        ...,
        description="Select the optimizer algorithm."
    )
    learning_rate: float = Field(..., gt=0, description="Set the base learning rate; must be > 0.")
    weight_decay: float = Field(0.0005, ge=0, description="Set L2 regularization coefficient; often higher for detection.")
    eps: float = Field(
        1e-8, 
        gt=0, 
        lt=0.01,
        description="Set epsilon for AdamW/RMSprop. Must be small (typically 1e-8)."
    )
    beta1: float = Field(0.9, gt=0, lt=1, description="Set AdamW β1.")
    beta2: float = Field(0.999, gt=0, lt=1, description="Set AdamW β2.")
    nesterov: bool = Field(False, description="Enable Nesterov momentum for SGD.")
    momentum: float = Field(0.9, ge=0, description="Set momentum factor for SGD/RMSprop.")
    alpha: float = Field(0.99, gt=0, lt=1, description="Set RMSprop smoothing constant α.")
    centered: bool = Field(False, description="Use centered RMSprop variant.")
    
    # --- Criterion (Loss Functions for Detection) ---
    loss_box: Literal["l1", "smooth_l1", "giou", "diou", "ciou"] = Field(
        ...,
        description="Select the bounding box regression loss function (e.g., CIoU is common for modern detectors)."
    )
    loss_cls: Literal["cross_entropy", "bce", "focal"] = Field(
        ...,
        description="Select the classification loss function (Focal Loss is common for one-stage detectors)."
    )
    loss_mask: Optional[Literal["bce", "cross_entropy"]] = Field(
        None,
        description="Select the mask prediction loss function; required if task_type='segmentation'."
    )
    
    # Loss Weights
    lambda_box: float = Field(1.0, gt=0, description="Weighting factor for the box loss component.")
    lambda_cls: float = Field(1.0, gt=0, description="Weighting factor for the classification loss component.")
    lambda_mask: float = Field(1.0, ge=0, description="Weighting factor for the mask loss component (0.0 if not used).")

    # --- Transfer Learning / Freezing ---
    freeze: Optional[int] = Field(
        None, 
        ge=0,
        description="Number of initial layers to freeze for transfer learning. Set to None or 0 to train all layers."
    )

    rationale: str = Field(
        ..., 
        description="Explain the choice of hyperparameters. Mention specific heuristics."
    )

    @model_validator(mode="after")
    def _validate_combinations(self) -> Self:
        total_ratio = self.train_data_ratio + self.val_data_ratio + self.test_data_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
            raise ValueError(f"train_data_ratio, val_data_ratio, and test_data_ratio must sum to 1.0. Current sum: {total_ratio}")

        if self.task_type == "segmentation":
            segmentation_models = ["mask_rcnn_r50", "yolov8_l", "yolov10_s"]
            if self.model_name not in segmentation_models:
                raise ValueError(
                    f"Task is 'segmentation', but selected model '{self.model_name}' does not support segmentation."
                )
            if self.loss_mask is None:
                raise ValueError("Task is 'segmentation', but 'loss_mask' is not defined.")
            if self.lambda_mask <= 0.0:
                raise ValueError("Task is 'segmentation', but 'lambda_mask' is not > 0.0.")
        else: # task_type == "detection"
            if self.loss_mask is not None:
                self.loss_mask = None 
            if self.lambda_mask > 0.0:
                self.lambda_mask = 0.0

        if self.aspect_ratio_range:
            if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] <= 0 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
                 raise ValueError("aspect_ratio_range must be a list [min, max] where 0 < min <= max.")

        return self