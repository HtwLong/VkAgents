from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing_extensions import TypedDict

from cvmodellearning.training.resource_guard import MAX_IMAGE_SIDE

# ----- Optimizer specs (unchanged) -----
class AdamWSpec(TypedDict, total=False):
    name: Literal["adamw"]
    lr: float
    weight_decay: float
    betas: List[float]  # [beta1, beta2]
    eps: float

class SGDSpec(TypedDict, total=False):
    name: Literal["sgd"]
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

class RMSpropSpec(TypedDict, total=False):
    name: Literal["rmsprop"]
    lr: float
    alpha: float
    eps: float
    momentum: float
    weight_decay: float
    centered: bool

OptimizerSpec = Union[AdamWSpec, SGDSpec, RMSpropSpec]

# ----- Criterion specs (unchanged) -----
class CrossEntropySpec(TypedDict, total=False):
    name: Literal["cross_entropy"]
    label_smoothing: float
    weight: Optional[List[float]]  # 1 per class

class BCEWithLogitsSpec(TypedDict, total=False):
    name: Literal["bce_with_logit"]
    pos_weight: Optional[float]

CriterionSpec = Union[CrossEntropySpec, BCEWithLogitsSpec]

# ----- Model spec (from your model factory) -----
class ResNet50Spec(TypedDict, total=False):
    name: Literal["resnet50"]
    weights: Literal["default", "none"]
class MobileNetV2Spec(TypedDict, total=False):
    name: Literal["mobilenet_v2"]
    weights: Literal["default", "none"]
class MobileNetV3LargeSpec(TypedDict, total=False):
    name: Literal["mobilenet_v3_large"]
    weights: Literal["default", "none"]
class MobileNetV3SmallSpec(TypedDict, total=False):
    name: Literal["mobilenet_v3_small"]
    weights: Literal["default", "none"]
class EfficientNetB0Spec(TypedDict, total=False):
    name: Literal[
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
        "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    ]
    weights: Literal["default", "none"]
class DenseNet121Spec(TypedDict, total=False):
    name: Literal["densenet121"]
    weights: Literal["default", "none"]
class ConvNeXtTinySpec(TypedDict, total=False):
    name: Literal["convnext_tiny"]
    weights: Literal["default", "none"]
class ViTB16Spec(TypedDict, total=False):
    name: Literal["vit_b_16"]
    weights: Literal["default", "none"]

ModelSpec = Union[
    ResNet50Spec,
    MobileNetV2Spec, MobileNetV3LargeSpec, MobileNetV3SmallSpec,
    EfficientNetB0Spec, DenseNet121Spec,
    ConvNeXtTinySpec, ViTB16Spec,
]

class TrainingConfig(TypedDict):
    """
    Training loop configuration for supervised image classification.

    Parameters
    ----------
    classes : List[str]
        Ordered class names; defines label mapping and report headers.
    img_per_class : int
        Number of images to fetch per class from the data source; controls dataset size.
    num_epochs : int
        Maximum number of training epochs to run (upper bound before early stopping).
    patience : int
        Early-stopping patience (stop if the tracked metric shows no improvement for this many epochs).
    batch_size : Optional[int]
        Mini-batch size for DataLoaders; if None, the pipeline defaults to 32.
    optimizer : OptimizerSpec
        Optimizer selection and hyperparameters (e.g., AdamW/SGD/RMSprop and their params).
    criterion : CriterionSpec
        Loss function selection and parameters (e.g., CrossEntropy with optional label smoothing).
    model : ModelSpec
        Backbone architecture and weights flag (e.g., {"name": "resnet50", "weights": "default"|"none"}).
    track_metric : Literal["val_acc", "val_loss", "macro_f1", "micro_f1"]
        Validation metric to monitor each epoch; Used for early stopping and best-checkpoint selection.
    """

    classes: List[str]
    img_per_class: int

    num_epochs: int
    patience: int
    batch_size: Optional[int]

    model: ModelSpec
    optimizer: OptimizerSpec
    criterion: CriterionSpec
    track_metric: Literal["val_acc", "val_loss", "macro_f1", "micro_f1"]
    


class TrainingConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Data/task
    classes: List[str] = Field(..., min_length=1, description="Ordered class names")
    img_per_class: int = Field(..., ge=1, description="Images per class target")
    track_metric: Literal["val_acc", "val_loss", "macro_f1", "micro_f1"]

    # Training
    num_epochs: int = Field(..., ge=1)
    patience: int = Field(..., ge=0)
    batch_size: Optional[int] = Field(None, ge=1)

    # Model selection (no unions)
    model_name: Literal[
        "resnet50", "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
        "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
        "densenet121", "convnext_tiny", "vit_b_16"
    ]
    model_weights: Literal["default", "none"] = "default"

    # Optimizer selection (no unions)
    optimizer_name: Literal["adamw", "sgd", "rmsprop"]

    # Shared/common optimizer params
    learning_rate: float = Field(..., gt=0)
    weight_decay: Optional[float] = Field(default=0.0, ge=0)
    eps: Optional[float] = Field(default=None, gt=0)          # adamw/rmsprop
    momentum: Optional[float] = Field(default=None, ge=0)     # sgd/rmsprop
    nesterov: Optional[bool] = None                           # sgd
    beta1: Optional[float] = Field(default=None, gt=0, lt=1)  # adamw
    beta2: Optional[float] = Field(default=None, gt=0, lt=1)  # adamw
    alpha: Optional[float] = Field(default=None, gt=0, lt=1)  # rmsprop
    centered: Optional[bool] = None                           # rmsprop

    # Criterion (no unions)
    criterion_name: Literal["cross_entropy", "bce_with_logit"]
    # Criterion-specific
    label_smoothing: Optional[float] = Field(default=None, ge=0, le=1)  # cross_entropy
    pos_weight: Optional[float] = Field(default=None, gt=0)             # bce_with_logit

    # Optional: image size or other global params, kept simple
    image_size: Optional[int] = Field(default=None, ge=16, le=MAX_IMAGE_SIDE)

    @model_validator(mode="after")
    def _validate_combinations(self):
        # Validate optimizer-specific params without unions
        if self.optimizer_name == "adamw":
            if self.beta1 is None or self.beta2 is None:
                raise ValueError("adamw requires beta1 and beta2")
            if self.eps is None:
                raise ValueError("adamw requires eps")
        if self.optimizer_name == "sgd":
            if self.momentum is None:
                # allow 0.0 by default if not provided
                object.__setattr__(self, "momentum", 0.0)
            # nesterov may be None; default to False
            if self.nesterov is None:
                object.__setattr__(self, "nesterov", False)
        if self.optimizer_name == "rmsprop":
            if self.alpha is None:
                raise ValueError("rmsprop requires alpha")
            if self.eps is None:
                raise ValueError("rmsprop requires eps")
            if self.momentum is None:
                object.__setattr__(self, "momentum", 0.0)
            if self.centered is None:
                object.__setattr__(self, "centered", False)

        # Criterion-specific checks
        if self.criterion_name == "cross_entropy":
            # default label_smoothing if omitted
            if self.label_smoothing is None:
                object.__setattr__(self, "label_smoothing", 0.0)
            if self.pos_weight is not None:
                raise ValueError("pos_weight not applicable to cross_entropy")
        if self.criterion_name == "bce_with_logit":
            if self.pos_weight is None:
                # Optional: allow missing pos_weight
                pass
            if self.label_smoothing not in (None, 0.0):
                raise ValueError("label_smoothing not applicable to bce_with_logit")

        return self
