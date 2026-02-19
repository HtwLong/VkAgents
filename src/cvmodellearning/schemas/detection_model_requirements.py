from typing import List, Optional, Literal
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

# --- Model Specification ---
class ObjectDetectionModelSpecModel(BaseModel):
    """
    Candidate model specification for object detection.
    """
    model_config = ConfigDict(extra="forbid")

    model_architecture: Optional[
        Literal[
            "yolov8", "yolov10", "yolov11", "yolov12",
            "retinanet_r50_fpn_1x_coco", "faster-rcnn_r50_fpn_1x_coco",
            "mask-rcnn_r50_fpn_1x_coco", "ssd300_coco",
            "rt-detr-r50_8xb2-100e_coco"
        ]
    ] = Field(
        None,
        description="Backbone architecture identifier; optional to allow other values downstream."
    )
    architecture_family: Optional[
        Literal["yolo", "retinanet", "faster-rcnn", "mask-rcnn", "ssd", "detr"]
    ] = Field(
        None,
        description="High-level architecture family; if provided with model_architecture, must be consistent."
    )
    
    loss_function_box: Optional[
        Literal["smooth_l1", "l1", "mse", "iou", "giou", "diou", "ciou"]
    ] = Field(
        None,
        description="Loss function for bounding box regression (e.g., Smooth L1, CIoU Loss)."
    )
    loss_function_cls: Optional[
        Literal["cross_entropy", "bce", "focal"]
    ] = Field(
        None,
        description="Loss function for class prediction (e.g., Cross-Entropy, Focal Loss)."
    )
    optimizer: Optional[
        Literal["sgd", "adam", "adamw", "rmsprop"]
    ] = Field(
        None,
        description="Optimization algorithm (e.g., SGD, AdamW)."
    )

    description: str = Field(..., min_length=1, description="Short model rationale/notes.")
    num_epochs: Optional[int] = Field(None, ge=1, description="Max training epochs; must be ≥ 1.")
    patience: Optional[int] = Field(None, ge=0, description="Early stopping patience (epochs without improvement); must be ≥ 0.")

    @model_validator(mode="after")
    def _check_family_consistency(self) -> Self:
        if self.model_architecture and self.architecture_family:
            family_map = {
                "yolov8": "yolo", "yolov10": "yolo", "yolov11": "yolo", "yolov12": "yolo",
                "retinanet_r50_fpn_1x_coco": "retinanet",
                "faster-rcnn_r50_fpn_1x_coco": "faster-rcnn",
                "mask-rcnn_r50_fpn_1x_coco": "mask-rcnn",
                "ssd300_coco": "ssd",
                "rt-detr-r50_8xb2-100e_coco": "detr",
            }
            inferred = family_map.get(self.model_architecture)
            if inferred and inferred != self.architecture_family:
                raise ValueError(
                    f"architecture_family='{self.architecture_family}' "
                    f"does not match model_architecture='{self.model_architecture}'"
                )
             
        return self


# --- Structured Output for Object Detection ---
class DetectionOutputModel(BaseModel):
    """
    Structured schema for LLM output, designed for Object Detection tasks.
    """
    model_config = ConfigDict(extra="forbid")

    # Problem
    task: Literal["classification", "detection", "segmentation", "visual question answering"] = Field(
        ..., description="Primary Computer Vision task; set to 'classificaiton', 'detection', 'segmentation', or 'visual question answering'."
    )
    application_domain: str = Field(..., min_length=1, description="Application domain (e.g., aerial imagery, retail shelf scanning).")
    description: str = Field(..., min_length=1, description="Problem description and objectives, including detection scope.")
    user_query: str = Field(..., min_length=1, description="Original user prompt or query for context.")

    # Dataset
    dataset_name: str = Field(..., min_length=1, description="Dataset name or identifier.")
    classes: List[str] = Field(..., min_length=1, description="Class names in label order; list must be non-empty.")
    annotations_format: Literal["coco", "yolo", "pascal_voc"] = Field(
        ..., description="The format of the ground truth bounding box/segmentation annotations."
    )
    source: Optional[str] = Field(None, description="Dataset source (URL, paper, registry, or internal ID).")
    path_images: Optional[str] = Field(None, description="Local/remote path to images if applicable.")
    path_labels: Optional[str] = Field(None, description="Local/remote path to labels (annotations) if applicable.")
    preprocessing: Optional[str] = Field(None, description="Text description of preprocessing steps (e.g., resizing, normalization).")
    augmentation: Optional[str] = Field(None, description="Text description of augmentation strategy (e.g., Mosaic, CutMix, H-Flip).")
    
    # CHANGED: Replaced Dicts with strictly typed Lists
    available_data: Optional[List[ClassDataSelection]] = Field(None, description="List mapping class names to dataset sources and their respective image counts.")
    selected_data: Optional[List[ClassDataSelection]] = Field(None, description="The subset of available_data selected for training.")

    # Model candidates
    model: List[ObjectDetectionModelSpecModel] = Field(
        ..., min_length=1, description="One or more candidate object detection model specs."
    )

    # Optional user change requests
    user_change_requests: Optional[List[str]] = Field(
        default=None,
        description="Optional list of user-provided change requests; use [] or null when not provided."
    )

    rationale: str = Field(
        ..., 
        description="A clear explanation of why you've chosen specific field values. Cite sources or dataset characteristics."
    )

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:                
        if self.task == "instance_segmentation":
            valid_is_models = ["mask-rcnn", "yolo"]
            
            if not any(
                m.architecture_family in valid_is_models 
                or m.model_architecture in ["yolov8", "yolov10", "yolov11", "yolov12", "mask-rcnn_r50_fpn_1x_coco"]
                for m in self.model
            ):
                 raise ValueError("Task is 'instance_segmentation', but no 'mask-rcnn' or YOLO-family model is listed in 'model'.")
                 
        return self