import math
from typing import Any, ClassVar, List, Literal, Mapping, Optional, Self
from pydantic import BaseModel, Field, ConfigDict, model_validator
from cvmodellearning.models.registry import DetectionHpoModelId
from cvmodellearning.schemas.hpo_runtime import build_runtime_hpo_config
from cvmodellearning.schemas.dataset_assignment import (
    ClassDataAssignment,
    normalize_dataset_assignments,
)
from cvmodellearning.training.resource_guard import MAX_IMAGE_SIDE

DETECTION_OPTIMIZER_PARAM_FIELDS = {
    "auto": (),
    "adamw": ("learning_rate", "weight_decay", "beta1"),
    "sgd": ("learning_rate", "weight_decay", "momentum"),
    "rmsprop": ("learning_rate", "weight_decay", "momentum"),
}

COMMON_DETECTION_RUNTIME_FIELDS = {
    "task_type", "classes", "selected_data", "train_data_ratio", "val_data_ratio",
    "test_data_ratio", "num_epochs", "patience", "batch_size", "input_size",
    "track_metric", "model_name", "model_weights", "training_recipe_id", "optimizer",
    "workers", "seed", "amp", "confidence_threshold", "max_detections",
}

YOLO_RUNTIME_FIELDS = {
    "scheduler_name", "final_learning_rate_factor", "warmup_epochs", "warmup_momentum",
    "lambda_box", "lambda_cls", "lambda_dfl", "mosaic", "mixup", "cutmix",
    "copy_paste", "degrees", "translate", "scale", "fliplr", "hsv_h", "hsv_s",
    "hsv_v", "close_mosaic", "single_cls", "rect", "multi_scale", "freeze",
    "nms_iou_threshold",
}

RTDETR_RUNTIME_FIELDS = {
    "scheduler_name", "final_learning_rate_factor", "warmup_epochs", "warmup_momentum",
    "mosaic", "mixup", "cutmix", "degrees", "translate", "scale", "fliplr",
    "hsv_h", "hsv_s", "hsv_v", "close_mosaic", "single_cls",
}

TORCHVISION_RUNTIME_FIELDS = {
    "scheduler_name", "lr_milestones", "scheduler_gamma", "max_size",
    "trainable_backbone_layers", "horizontal_flip_probability", "augmentation_policy",
    "topk_candidates", "positive_fraction", "matching_iou_threshold", "nms_iou_threshold",
}


def detection_runtime_family(model_name: str) -> Literal["yolo", "rtdetr", "torchvision"]:
    if model_name.startswith(("yolov8_", "yolov10_", "yolov11_", "yolov12_")):
        return "yolo"
    if model_name == "rtdetr_hgnetv2_l":
        return "rtdetr"
    if model_name in {"retinanet_r50", "faster_rcnn_r50", "ssd300"}:
        return "torchvision"
    raise ValueError(f"Unsupported detection model: {model_name}")


def active_detection_config_fields(config: Mapping[str, Any]) -> set[str]:
    """Return only fields consumed by the selected detector backend."""
    model_name = str(config.get("model_name", ""))
    family_fields = {
        "yolo": YOLO_RUNTIME_FIELDS,
        "rtdetr": RTDETR_RUNTIME_FIELDS,
        "torchvision": TORCHVISION_RUNTIME_FIELDS,
    }[detection_runtime_family(model_name)]
    active = (COMMON_DETECTION_RUNTIME_FIELDS | family_fields) & set(config)
    if "optimizer_name" in config:
        active.add("optimizer_name")
        optimizer_fields = set().union(*map(set, DETECTION_OPTIMIZER_PARAM_FIELDS.values()))
        active |= set(DETECTION_OPTIMIZER_PARAM_FIELDS.get(config.get("optimizer_name"), ()))
        active -= optimizer_fields - set(
            DETECTION_OPTIMIZER_PARAM_FIELDS.get(config.get("optimizer_name"), ())
        )
    return active


def expand_detection_config_for_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Restore deterministic schema sentinels omitted from the clean runtime config."""
    expanded = dict(config)
    model_name = str(expanded.get("model_name", ""))
    family = detection_runtime_family(model_name)

    def set_missing(values: Mapping[str, Any]) -> None:
        for field_name, value in values.items():
            expanded.setdefault(field_name, value)

    if family == "torchvision":
        set_missing({
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
            "warmup_epochs": 0.0,
            "lambda_dfl": 0.0,
            "freeze": None,
            "single_cls": False,
            "rect": False,
        })
        if model_name == "retinanet_r50":
            set_missing({"loss_box": "l1", "loss_cls": "focal"})
        else:
            set_missing({"loss_box": "smooth_l1", "loss_cls": "cross_entropy"})
    elif family == "rtdetr":
        set_missing({
            "aspect_ratio_range": None,
            "loss_box": "l1_giou",
            "loss_cls": "varifocal",
            "lambda_box": 5.0,
            "lambda_giou": 2.0,
            "lambda_cls": 1.0,
            "lambda_dfl": 0.0,
            "max_size": 640,
            "nms_iou_threshold": 0.0,
            "copy_paste": 0.0,
            "multi_scale": 0.0,
            "freeze": None,
            "rect": False,
        })

    return expanded

class LLMFieldRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    applied_policy_ids: List[str] = Field(default_factory=list)


class DetectionConfigDraft(BaseModel):
    """
    Structured schema for Object Detection training configuration.
    Uses registry-backed enums for key selector fields.
    """
    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    enforce_executable_contract: ClassVar[bool] = False

    # --- Data/Task ---
    task_type: Literal["detection"] = Field(
        ...,
        description="The supported task type is bounding-box object detection."
    )
    classes: List[str] = Field(
        ..., min_length=1,
        description="Provide class names in training label order; list must be non-empty."
    )
    
    # CHANGED: Unified schema
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
        0.8, ge=0.0, lt=1.0,
        description="Deprecated execution compatibility ratio derived from the planned assignments."
    )
    val_data_ratio: float = Field(
        0.1, ge=0.0, lt=1.0,
        description="Deprecated execution compatibility ratio derived from the planned assignments."
    )
    test_data_ratio: float = Field(
        0.1, ge=0.0, lt=1.0,
        description="Deprecated execution compatibility ratio derived from the planned assignments."
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
        16, ge=-1,
        description="Set a positive mini-batch size or -1 for Ultralytics automatic batch sizing."
    )
    input_size: int = Field(
        640, ge=32, le=MAX_IMAGE_SIDE,
        description=(
            f"Set square input image size in pixels after resize/padding (e.g., 416, 640); "
            f"must be between 32 and {MAX_IMAGE_SIDE}."
        )
    )
    aspect_ratio_range: Optional[List[float]] = Field(
        [0.5, 2.0],
        description="Range [min, max] of aspect ratios to sample images for."
    )
    track_metric: Literal["val_mAP", "val_mAP_50", "val_mAP_75", "val_loss"] = Field(
        "val_mAP",
        description="Ultralytics checkpoint fitness is dominated by validation mAP@0.5:0.95."
    )

    # --- Model Selection (Detection Models) ---
    model_name: DetectionHpoModelId = Field(
        ...,
        description="Select the object detection architecture identifier."
    )
    model_weights: Literal["default", "none", "coco", "imagenet_backbone"] = Field(
        "coco",
        description="Choose detector weights or an ImageNet-pretrained backbone for SSD custom classes."
    )
    training_recipe_id: str = Field(
        "",
        description="Ontology recipe used to produce this fine-tuning configuration.",
    )
    
    # --- Optimizer Selection (Adapted from Classification) ---
    optimizer_name: Literal["auto", "adamw", "sgd", "rmsprop"] = Field(
        "adamw",
        description="Select the optimizer algorithm."
    )
    learning_rate: float = Field(0.01, gt=0, description="Set the base learning rate; must be > 0.")
    weight_decay: float = Field(0.0005, ge=0, description="Set L2 regularization coefficient; often higher for detection.")
    beta1: float = Field(0.9, gt=0, lt=1, description="Set AdamW β1.")
    momentum: float = Field(0.9, ge=0, description="Set momentum factor for SGD/RMSprop.")
    scheduler_name: Literal["none", "linear", "multistep"] = "linear"
    lr_milestones: List[int] = Field(default_factory=lambda: [16, 22])
    scheduler_gamma: float = Field(0.1, gt=0, le=1)
    final_learning_rate_factor: float = Field(0.01, gt=0, le=1)
    warmup_epochs: float = Field(3.0, ge=0)
    warmup_momentum: float = Field(0.8, ge=0, lt=1)
    amp: bool = Field(True, description="Use Ultralytics automatic mixed precision when supported.")
    
    # --- Criterion (Loss Functions for Detection) ---
    loss_box: Literal["l1", "l1_giou", "smooth_l1", "giou", "diou", "ciou"] = Field(
        "ciou",
        description="YOLO detection uses its architecture-defined CIoU-based box objective."
    )
    loss_cls: Literal["cross_entropy", "bce", "focal", "varifocal"] = Field(
        "bce",
        description="YOLO detection uses its architecture-defined BCE classification objective."
    )
    # Loss Weights
    lambda_box: float = Field(7.5, gt=0, description="Ultralytics box-loss gain.")
    lambda_cls: float = Field(0.5, gt=0, description="Ultralytics classification-loss gain.")
    lambda_giou: float = Field(0.0, ge=0, description="Fixed GIoU loss gain for DETR-style models.")
    lambda_dfl: float = Field(1.5, ge=0, description="Ultralytics distribution-focal-loss gain; unused for RetinaNet.")

    # Ultralytics detection augmentations and final prediction settings.
    mosaic: float = Field(1.0, ge=0, le=1)
    mixup: float = Field(0.0, ge=0, le=1)
    cutmix: float = Field(0.0, ge=0, le=1)
    copy_paste: float = Field(0.0, ge=0, le=1)
    degrees: float = Field(0.0, ge=0, le=180)
    translate: float = Field(0.1, ge=0, le=1)
    scale: float = Field(0.5, ge=0)
    fliplr: float = Field(0.5, ge=0, le=1)
    hsv_h: float = Field(0.015, ge=0, le=1)
    hsv_s: float = Field(0.7, ge=0, le=1)
    hsv_v: float = Field(0.4, ge=0, le=1)
    close_mosaic: int = Field(10, ge=0)
    single_cls: bool = False
    rect: bool = False
    multi_scale: float = Field(0.0, ge=0, le=1)
    confidence_threshold: float = Field(0.25, ge=0, le=1)
    nms_iou_threshold: float = Field(0.7, ge=0, le=1)
    max_detections: int = Field(300, ge=1)
    workers: int = Field(8, ge=0)
    seed: int = Field(0, ge=0)
    max_size: int = Field(
        1333, ge=32, le=MAX_IMAGE_SIDE,
        description=f"Maximum image side used by TorchVision detectors; must be <= {MAX_IMAGE_SIDE}.",
    )
    trainable_backbone_layers: int = Field(3, ge=0, le=5)
    horizontal_flip_probability: float = Field(0.5, ge=0, le=1)
    augmentation_policy: Literal["basic", "ssd"] = "basic"
    topk_candidates: int = Field(400, ge=1)
    positive_fraction: float = Field(0.25, gt=0, lt=1)
    matching_iou_threshold: float = Field(0.5, gt=0, lt=1)

    # --- Transfer Learning / Freezing ---
    freeze: Optional[int] = Field(
        None, 
        ge=0,
        description=(
            "Number of initial layers kept frozen throughout transfer-learning training; this is "
            "a layer count, not an epoch count. Set to None or 0 to train all layers."
        )
    )

    rationale: str = Field(
        ..., 
        description="Explain the choice of hyperparameters. Mention specific heuristics."
    )
    llm_field_rationales: List[LLMFieldRationale] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_combinations(self) -> Self:
        if not self.enforce_executable_contract:
            return self
        model_name = getattr(self.model_name, "value", self.model_name)
        is_yolo = str(model_name).startswith(("yolov8_", "yolov10_", "yolov11_", "yolov12_"))
        is_retinanet = model_name == "retinanet_r50"
        is_faster_rcnn = model_name == "faster_rcnn_r50"
        is_ssd = model_name == "ssd300"
        is_rtdetr = model_name == "rtdetr_hgnetv2_l"
        is_torchvision = is_retinanet or is_faster_rcnn or is_ssd
        total_ratio = self.train_data_ratio + self.val_data_ratio + self.test_data_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
            raise ValueError(f"train_data_ratio, val_data_ratio, and test_data_ratio must sum to 1.0. Current sum: {total_ratio}")

        if is_yolo:
            if self.task_type != "detection":
                raise ValueError("YOLO detection checkpoints do not support segmentation masks.")
            if self.model_weights not in {"coco", "default"}:
                raise ValueError("YOLO fine-tuning requires pretrained COCO/default weights.")
            if self.training_recipe_id not in {"", "ultralytics_yolo_detection_finetune_balanced"}:
                raise ValueError("YOLO models require the executable Ultralytics fine-tuning recipe.")
            if self.track_metric != "val_mAP":
                raise ValueError("YOLO checkpoint selection uses validation mAP fitness.")
            if self.scheduler_name != "linear":
                raise ValueError("The YOLO executor uses Ultralytics' linear learning-rate schedule.")
            if self.optimizer_name == "auto" and (
                self.learning_rate != 0.01 or self.momentum != 0.9
            ):
                raise ValueError(
                    "Ultralytics optimizer='auto' derives learning rate and momentum; "
                    "their schema sentinel values must remain 0.01 and 0.9."
                )
            if self.loss_box != "ciou" or self.loss_cls != "bce":
                raise ValueError("YOLO loss implementations are fixed to CIoU-based box loss and BCE classification.")
            if self.copy_paste != 0.0:
                raise ValueError("copy_paste requires segmentation masks and must be 0 for box-only YOLO detection.")
            if self.close_mosaic >= self.num_epochs:
                self.close_mosaic = 0

        if is_retinanet:
            if self.task_type != "detection":
                raise ValueError("The registered RetinaNet checkpoint supports bounding-box detection only.")
            if self.model_weights not in {"coco", "default"}:
                raise ValueError("RetinaNet fine-tuning requires pretrained COCO/default detector weights.")
            if self.training_recipe_id not in {
                "",
                "torchvision_retinanet_resnet50_fpn_coco_pretrained_custom_finetune",
            }:
                raise ValueError("RetinaNet requires the executable TorchVision custom fine-tuning recipe.")
            if self.loss_box != "l1" or self.loss_cls != "focal":
                raise ValueError("TorchVision RetinaNet uses L1 box regression and sigmoid focal classification loss.")
            if self.lambda_dfl != 0:
                raise ValueError("RetinaNet does not use distribution focal loss; lambda_dfl must be 0.")
            if self.track_metric != "val_mAP":
                raise ValueError("RetinaNet checkpoint selection uses validation COCO mAP.")
            if self.warmup_epochs != 0:
                raise ValueError("The minimal RetinaNet executor does not apply LR warm-up.")
            if self.aspect_ratio_range is not None:
                raise ValueError("RetinaNet preserves aspect ratio internally; aspect_ratio_range must be null.")
            inactive_augmentations = (
                self.mosaic, self.mixup, self.cutmix, self.copy_paste, self.degrees,
                self.translate, self.scale, self.fliplr, self.hsv_h, self.hsv_s,
                self.hsv_v, float(self.close_mosaic), self.multi_scale,
            )
            if any(value != 0 for value in inactive_augmentations):
                raise ValueError(
                    "YOLO-specific augmentation fields must be 0 for the TorchVision RetinaNet executor."
                )
            if self.max_size < self.input_size:
                raise ValueError("max_size must be greater than or equal to input_size.")

        if is_faster_rcnn:
            if self.task_type != "detection":
                raise ValueError("The registered Faster R-CNN checkpoint supports bounding-box detection only.")
            if self.model_weights not in {"coco", "default"}:
                raise ValueError("Faster R-CNN fine-tuning requires pretrained COCO/default detector weights.")
            if self.training_recipe_id not in {
                "",
                "torchvision_fasterrcnn_resnet50_fpn_coco_pretrained_custom_finetune",
            }:
                raise ValueError("Faster R-CNN requires the executable TorchVision custom fine-tuning recipe.")
            if self.loss_box != "smooth_l1" or self.loss_cls != "cross_entropy":
                raise ValueError(
                    "TorchVision Faster R-CNN uses Smooth L1 box regression and cross-entropy classification."
                )
            if self.lambda_dfl != 0:
                raise ValueError("Faster R-CNN does not use distribution focal loss; lambda_dfl must be 0.")
            if self.track_metric != "val_mAP":
                raise ValueError("Faster R-CNN checkpoint selection uses validation COCO mAP.")
            if self.warmup_epochs != 0:
                raise ValueError("The minimal Faster R-CNN executor does not apply LR warm-up.")
            if self.aspect_ratio_range is not None:
                raise ValueError("Faster R-CNN preserves aspect ratio internally; aspect_ratio_range must be null.")
            inactive_augmentations = (
                self.mosaic, self.mixup, self.cutmix, self.copy_paste, self.degrees,
                self.translate, self.scale, self.fliplr, self.hsv_h, self.hsv_s,
                self.hsv_v, float(self.close_mosaic), self.multi_scale,
            )
            if any(value != 0 for value in inactive_augmentations):
                raise ValueError(
                    "YOLO-specific augmentation fields must be 0 for the TorchVision Faster R-CNN executor."
                )
            if self.max_size < self.input_size:
                raise ValueError("max_size must be greater than or equal to input_size.")

        if is_ssd:
            if self.task_type != "detection":
                raise ValueError("SSD300 VGG16 supports bounding-box detection only.")
            if self.model_weights != "imagenet_backbone":
                raise ValueError(
                    "SSD300 custom-class training requires the pretrained ImageNet VGG16 backbone."
                )
            if self.training_recipe_id not in {
                "",
                "torchvision_ssd300_vgg16_imagenet_backbone_custom_training",
            }:
                raise ValueError("SSD300 requires the executable TorchVision custom-data recipe.")
            if self.loss_box != "smooth_l1" or self.loss_cls != "cross_entropy":
                raise ValueError(
                    "TorchVision SSD uses Smooth L1 box regression and cross-entropy classification."
                )
            if self.lambda_dfl != 0:
                raise ValueError("SSD does not use distribution focal loss; lambda_dfl must be 0.")
            if self.track_metric != "val_mAP":
                raise ValueError("SSD checkpoint selection uses validation COCO mAP.")
            if self.warmup_epochs != 0:
                raise ValueError("The minimal SSD executor does not apply LR warm-up.")
            if self.input_size != 300 or self.max_size != 300:
                raise ValueError("SSD300 VGG16 requires input_size=max_size=300.")
            if self.aspect_ratio_range is not None:
                raise ValueError("SSD300 uses its fixed default-box geometry; aspect_ratio_range must be null.")
            if self.augmentation_policy != "ssd":
                raise ValueError("SSD300 training requires augmentation_policy='ssd'.")
            inactive_augmentations = (
                self.mosaic, self.mixup, self.cutmix, self.copy_paste, self.degrees,
                self.translate, self.scale, self.fliplr, self.hsv_h, self.hsv_s,
                self.hsv_v, float(self.close_mosaic), self.multi_scale,
            )
            if any(value != 0 for value in inactive_augmentations):
                raise ValueError(
                    "YOLO-specific augmentation fields must be 0 for the TorchVision SSD executor."
                )
            if self.batch_size == -1:
                raise ValueError("SSD requires an explicit positive batch_size.")

        if is_torchvision:
            inactive_fields = {
                "freeze": None,
                "single_cls": False,
                "rect": False,
                "final_learning_rate_factor": 0.01,
                "warmup_momentum": 0.8,
                # TorchVision returns already-composed detector losses. The
                # trainer sums them with an implicit fixed weight of one.
                "lambda_box": 1.0,
                "lambda_cls": 1.0,
                "lambda_giou": 0.0,
            }
            for field_name, value in inactive_fields.items():
                setattr(self, field_name, value)

        if is_rtdetr:
            if self.task_type != "detection":
                raise ValueError("RT-DETR-L supports bounding-box detection only.")
            if self.model_weights not in {"coco", "default"}:
                raise ValueError("RT-DETR-L fine-tuning requires COCO-pretrained weights.")
            if self.training_recipe_id not in {
                "",
                "ultralytics_rtdetr_l_coco_pretrained_custom_finetune",
            }:
                raise ValueError("RT-DETR-L requires its executable Ultralytics fine-tuning recipe.")
            if self.optimizer_name != "adamw" or self.scheduler_name != "linear":
                raise ValueError("The executable RT-DETR-L recipe uses AdamW with a linear LR schedule.")
            if self.loss_box != "l1_giou" or self.loss_cls != "varifocal":
                raise ValueError("Ultralytics RT-DETR uses L1 plus GIoU box loss and Varifocal classification loss.")
            if (self.lambda_box, self.lambda_giou, self.lambda_cls) != (5.0, 2.0, 1.0):
                raise ValueError("RT-DETR fixes its L1, GIoU, and classification loss gains to 5, 2, and 1.")
            if self.lambda_dfl != 0:
                raise ValueError("RT-DETR does not use distribution focal loss; lambda_dfl must be 0.")
            if self.track_metric != "val_mAP":
                raise ValueError("RT-DETR checkpoint selection uses validation COCO mAP.")
            if self.input_size != 640 or self.max_size != 640:
                raise ValueError("The registered pretrained RT-DETR-L recipe requires 640px square inputs.")
            if self.aspect_ratio_range is not None or self.rect:
                raise ValueError("Ultralytics RT-DETR uses square scale-fill preprocessing, not rectangular batching.")
            if self.nms_iou_threshold != 0:
                raise ValueError("RT-DETR is NMS-free; nms_iou_threshold must be 0.")
            if self.max_detections > 300:
                raise ValueError("The pretrained RT-DETR-L decoder has 300 object queries.")
            if self.batch_size == -1:
                raise ValueError(
                    "RT-DETR-L requires an explicit batch_size because Ultralytics AutoBatch "
                    "falls back to batch 16 on CPU and MPS."
                )
            if self.copy_paste != 0:
                raise ValueError("copy_paste requires segmentation masks and must be 0 for RT-DETR detection.")
            if self.multi_scale != 0:
                raise ValueError("The minimal RT-DETR executor uses the recipe's fixed 640px input size.")
            if self.amp:
                raise ValueError("AMP is disabled because Ultralytics documents possible RT-DETR matching failures.")
            if self.freeze not in {None, 0}:
                raise ValueError("The registered RT-DETR-L recipe fine-tunes the full pretrained model.")
            # Normalize fields belonging to other detector backends. Keeping
            # them inert makes the shared output schema honest without adding
            # a separate RT-DETR schema.
            self.lr_milestones = []
            self.scheduler_gamma = 1.0
            self.augmentation_policy = "basic"
            self.trainable_backbone_layers = 0
            self.horizontal_flip_probability = 0.0
            self.topk_candidates = 400
            self.positive_fraction = 0.25
            self.matching_iou_threshold = 0.5
            if self.close_mosaic >= self.num_epochs:
                self.close_mosaic = 0

        if self.scheduler_name == "multistep":
            if not self.lr_milestones or any(step < 1 for step in self.lr_milestones):
                raise ValueError("lr_milestones must contain positive epoch numbers.")
            if self.lr_milestones != sorted(set(self.lr_milestones)):
                raise ValueError("lr_milestones must be sorted and contain no duplicates.")

        if self.batch_size == 0 or self.batch_size < -1:
            raise ValueError("batch_size must be -1 for auto batch sizing or a positive integer.")

        if self.aspect_ratio_range:
            if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] <= 0 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
                 raise ValueError("aspect_ratio_range must be a list [min, max] where 0 < min <= max.")

        return self

    def runtime_config(self) -> dict:
        config = build_runtime_hpo_config(
            self.model_dump(exclude={"rationale", "llm_field_rationales"}),
            DETECTION_OPTIMIZER_PARAM_FIELDS,
        )
        return {
            field_name: value
            for field_name, value in config.items()
            if field_name in active_detection_config_fields(config)
        }


class DetectionConfigModel(DetectionConfigDraft):
    """Strict, executable detection configuration produced after draft completion."""

    enforce_executable_contract: ClassVar[bool] = True
