from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing_extensions import Self


class ModelSpecModel(BaseModel):
    """
    Candidate model specification kept union-free with simple primitives and optional Literals.
    """
    model_config = ConfigDict(extra="forbid")

    # Selector-style fields (use Literals when enumerating known options; keep optional to stay flexible)
    model_architecture: Optional[
        Literal[
            "resnet50", "vgg16", "mobilenet_v2", "mobilenet_v3_large",
            "efficientnet_b0", "densenet121", "convnext_tiny", "vit_b_16",
            "swin_v2_t", "swin_v2_s", "swin_v2_b"
        ]
    ] = Field(
        None,
        description="Backbone architecture identifier; optional to allow other values downstream."
    )
    architecture_family: Optional[
        Literal["resnet", "vgg", "mobilenet", "efficientnet", "densenet", "convnext", "vit", "swin_v2"]
    ] = Field(
        None,
        description="High-level architecture family; if provided with model_architecture, must be consistent."
    )

    # Descriptive and loop controls (kept as simple primitives)
    description: str = Field(..., min_length=1, description="Short model rationale/notes.")
    num_epochs: Optional[int] = Field(None, ge=1, description="Max training epochs when this model is used; must be ≥ 1.")
    patience: Optional[int] = Field(None, ge=0, description="Early stopping patience (epochs without improvement); must be ≥ 0.")

    @model_validator(mode="after")
    def _check_family_consistency(self) -> Self:
        # Enforce family consistent with architecture when both are provided
        if self.model_architecture and self.architecture_family:
            family_map = {
                "resnet50": "resnet",
                "vgg16": "vgg",
                "mobilenet_v2": "mobilenet",
                "mobilenet_v3_large": "mobilenet",
                "efficientnet_b0": "efficientnet",
                "densenet121": "densenet",
                "convnext_tiny": "convnext",
                "vit_b_16": "vit",
                "swin_v2_t": "swin_v2",
                "swin_v2_s": "swin_v2",
                "swin_v2_b": "swin_v2",
            }
            inferred = family_map.get(self.model_architecture)
            if inferred and inferred != self.architecture_family:
                raise ValueError(
                    f"architecture_family='{self.architecture_family}' "
                    f"does not match model_architecture='{self.model_architecture}'"
                )
        return self


class ClassificationOutputModel(BaseModel):
    """
    Union-free schema designed for LLM structured outputs:
    - Forbids unknown keys.
    - Flattens nested content into simple primitives where possible.
    - Uses a post-parse validator for cross-field consistency.
    """
    model_config = ConfigDict(extra="forbid")

    # Problem
    task: str = Field(..., min_length=1, description="Primary ML task (e.g., classification, detection, segmentation, visual question answering).")
    application_domain: str = Field(..., min_length=1, description="Application domain (e.g., medical, retail).")
    description: str = Field(..., min_length=1, description="Problem description and objectives.")
    user_query: str = Field(..., min_length=1, description="Original user prompt or query for context.")

    # Dataset (flattened; no unions)
    dataset_name: str = Field(..., min_length=1, description="Dataset name or identifier.")
    classes: List[str] = Field(..., min_length=1, description="Class names in label order; list must be non-empty.")
    source: Optional[str] = Field(None, description="Dataset source (URL, paper, registry, or internal ID).")
    path_images: Optional[str] = Field(None, description="Local/remote path to images if applicable.")
    path_labels: Optional[str] = Field(None, description="Local/remote path to labels if applicable.")
    preprocessing: Optional[str] = Field(None, description="Text description of preprocessing steps.")
    augmentation: Optional[str] = Field(None, description="Text description of augmentation strategy.")
    available_data: Optional[Dict[str, Dict[str, int]]] = Field(None, description="Map of class names to dataset sources and their respective image counts. Format: {class: {dataset_name: image_count}}.")
    selected_data: Optional[Dict[str, Dict[str, int]]] = Field(None, description="The subset of available_data selected for training. Format: {class: {dataset_name: image_count}}.")

    # Model candidates (kept as a simple list of a non-union spec)
    model: List[ModelSpecModel] = Field(
        ..., min_length=1, description="One or more candidate model specs."
    )

    # Optional user change requests
    user_change_requests: Optional[List[str]] = Field(
        default=None,
        description="Optional list of user-provided change requests; use [] or null when not provided."
    )

    rationale: str = Field(
        ..., 
        description="A clear explanation of why you've chosen specific field values. Cite sources or dataset characteristics if possible."
    )

