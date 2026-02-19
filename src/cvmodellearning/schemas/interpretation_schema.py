from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing_extensions import Self


class SynonymMatch(BaseModel):
    original_class: str
    found_match: bool
    dataset_class: Optional[str] = Field(None, description="The exact string from the allowed list that matches the meaning.")
    reason: str

class HardwareSpecModel(BaseModel):
    """
    Describes the hardware resources available for training and inference.
    Uses simple primitives and dictionaries for flexibility.
    """
    model_config = ConfigDict(extra="forbid")

    cpu_cores: Optional[int] = Field(None, ge=1, description="Number of available CPU cores.")
    gpu_type: Optional[str] = Field(None, description="Specific GPU model (e.g., 'NVIDIA A100', 'RTX 3090').")
    gpu_count: Optional[int] = Field(None, ge=0, description="Number of GPUs available; 0 if only CPU is used.")
    ram_gb: Optional[float] = Field(None, ge=1.0, description="Available system RAM in GB.")
    storage_gb: Optional[float] = Field(None, ge=1.0, description="Available storage space in GB.")
    details: Optional[str] = Field(None, description="Any additional hardware details or cluster information.")


class ModelSpecModel(BaseModel):
    """
    Detailed model requirements (loosely defined without Literals).
    """
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(..., min_length=1, description="Preferred model architecture name (e.g., 'ResNet-50', 'YOLOv12', 'ViT-B/16').")
    framework: Optional[str] = Field(None, description="Preferred ML framework (e.g., 'PyTorch', 'TensorFlow', 'JAX').")
    backbone: Optional[str] = Field(None, description="Specific backbone or feature extractor (e.g., 'darknet', 'swin_t').")
    hyperparameters: Optional[Dict[str, Union[str, float, int, bool]]] = Field(
        default=None,
        description="Dictionary of required/suggested hyperparameters (e.g., {'learning_rate': 0.001, 'batch_size': 32})."
    )
    description: Optional[str] = Field(None, description="Rationale for choosing this model or constraints.")


class PerformanceSpecModel(BaseModel):
    """
    Defines the target performance metrics.
    """
    model_config = ConfigDict(extra="forbid")

    primary_metric: Optional[str] = Field(..., min_length=1, description="The main metric to optimize (e.g., 'accuracy', 'mAP@0.5', 'F1-score').")
    target_value: Optional[float] = Field(None, ge=0.0, le=1.0, description="The required minimum target value for the primary metric (0.0 to 1.0).")
    latency_ms: Optional[float] = Field(None, ge=0.0, description="Maximum allowable inference latency in milliseconds.")
    throughput_fps: Optional[float] = Field(None, ge=0.0, description="Minimum required inference throughput in frames per second (FPS).")
    other_constraints: Optional[List[str]] = Field(None, description="Other performance requirements (e.g., 'model size < 50MB').")


class AugmentationSpecModel(BaseModel):
    """
    Describes the data augmentation strategy.
    """
    model_config = ConfigDict(extra="forbid")

    strategy_name: Optional[str] = Field(None, description="High-level strategy (e.g., 'standard_transforms', 'autoaugment', 'mosaic_mixup').")
    transforms: Optional[List[str]] = Field(None, description="List of specific augmentation techniques to use (e.g., 'random_crop', 'horizontal_flip', 'color_jitter').")
    details: Optional[str] = Field(None, description="Detailed notes on augmentation intensity or sequence.")


# --- Main Structured Output Model ---

class InterpretationRequirements(BaseModel):
    """
    A flexible, high-level schema for defining Computer Vision project requirements.
    Only the 'task' field uses a Literal constraint.
    """
    model_config = ConfigDict(extra="forbid")

    # --- Computer Vision Task ---
    task: Literal["classification", "detection", "segmentation", "visual question answering"] = Field(
        ...,
        description="The primary Computer Vision task; must be one of the four specified options."
    )
    application_domain: Optional[str] = Field(None, min_length=1, description="Application domain (e.g., 'medical imaging', 'autonomous driving', 'satellite analysis').")
    description: Optional[str] = Field(None, min_length=1, description="Detailed problem description and objectives.")
    user_query: Optional[str] = Field(None, min_length=1, description="Original user prompt or query for context.")
    use_case_description: Optional[str] = Field(
        None, 
        description="General description of the overall use case or goal."
    )
    questions_list: Optional[List[str]] = Field(
        None, 
        description="Specific questions the user wants answered, if explicitly provided."
    )

    # --- Data ---
    classes: List[str] = Field(..., min_length=1, description="Class names in label order; list must be non-empty.")
    # Additional optional data fields for flexibility (you can add more if needed)
    dataset_name: Optional[str] = Field(None, description="Dataset name or identifier.")
    dataset_size: Optional[str] = Field(None, description="Approximate size of the dataset (e.g., '10,000 images').")

    available_data: Optional[Dict[str, Dict[str, int]]] = Field(
        None,
        description="Map of class names to dataset sources and their respective image counts."
    )

    # --- Performance Requirements ---
    performance_requirements: Optional[PerformanceSpecModel] = Field(
        None,
        description="Details on required performance metrics and target values."
    )

    # --- Available Hardware ---
    available_hardware: Optional[HardwareSpecModel] = Field(
        None,
        description="Specifications of the computing resources available."
    )

    # --- Model Requirements ---
    model_requirements: Optional[List[ModelSpecModel]] = Field(
        None,
        min_length=1,
        description="One or more candidate model architecture requirements."
    )

    # --- Augmentations ---
    augmentations: Optional[AugmentationSpecModel] = Field(
        None,
        description="The required data augmentation strategy."
    )

    @model_validator(mode="after")
    def _validate_data_vs_task(self) -> Self:
        # Example high-level validator: Ensure classes are present for any task
        if not self.classes:
            raise ValueError("The 'classes' list must not be empty for any computer vision task.")

        # You could add a check here for specific task requirements, e.g.:
        # if self.task == "visual question answering":
        #     # Ensure the description implies both images and questions are involved
        #     ...

        return self