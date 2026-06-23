from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing_extensions import Self
from cvmodellearning.models.registry import VQAModelId

# --- Helper Models for Strict JSON Schema ---
class DatasetSourceCount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str = Field(..., description="The name of the dataset")
    count: int = Field(..., description="Number of images selected from this dataset")

class ClassDataSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_name: str = Field(..., description="The name of the class or subset")
    sources: List[DatasetSourceCount] = Field(..., description="List of datasets and their counts")

class VQAModelSpecModel(BaseModel):
    """
    Candidate model specification for visual question answering using VLMs.
    """
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    # Architecture fields
    model_architecture: VQAModelId = Field(
        ..., description="Backbone architecture identifier for VQA."
    )
    architecture_family: Literal["qwen-vl"] = Field(
        ..., description="High-level architecture family."
    )

    # VLM/LLM Specific Hyperparameters
    precision: Optional[Literal["bf16", "fp16", "fp32", "fp8"]] = Field(
        None, description="Training precision (bf16 recommended for Qwen3-VL)."
    )
    optimizer: Optional[Literal["adamw", "adamw_torch", "paged_adamw_8bit", "rmsprop"]] = Field(
        None, description="Optimization algorithm."
    )
    learning_rate: Optional[float] = Field(
        None, description="Learning rate for fine-tuning."
    )
    max_seq_length: Optional[int] = Field(
        None, description="Maximum sequence length for the vision-language context."
    )
    
    # LoRA parameters for efficient fine-tuning
    use_lora: Optional[bool] = Field(
        True, description="Whether to use LoRA for fine-tuning."
    )
    lora_r: Optional[int] = Field(
        None, description="LoRA rank parameter (e.g., 8, 16, 64)."
    )
    lora_alpha: Optional[int] = Field(
        None, description="LoRA alpha parameter (usually 2x lora_r)."
    )
    lora_target_modules: Optional[str] = Field(
        None, description="Target modules for LoRA (e.g., 'all-linear')."
    )

    # Standard controls
    description: str = Field(..., min_length=1, description="Short model rationale/notes.")
    num_epochs: Optional[int] = Field(None, ge=1, description="Max training epochs.")
    patience: Optional[int] = Field(None, ge=0, description="Early stopping patience.")

    @model_validator(mode="after")
    def _check_family_consistency(self) -> Self:
        if self.architecture_family != "qwen-vl" and self.model_architecture == "Qwen3-VL-2B-Instruct":
            raise ValueError("architecture_family must be 'qwen-vl' for Qwen3-VL-2B-Instruct.")
        return self


class VQAOutputModel(BaseModel):
    """
    Structured schema for LLM output, designed for Visual Question Answering.
    """
    model_config = ConfigDict(extra="forbid")

    # Problem
    task: Literal["classification", "detection", "segmentation", "visual question answering"] = Field(...)
    application_domain: str = Field(..., min_length=1, description="Application domain.")
    description: str = Field(..., min_length=1, description="Problem description and objectives.")
    user_query: str = Field(..., min_length=1, description="Original user prompt or query.")

    # Dataset
    dataset_name: str = Field(..., min_length=1, description="Dataset name or identifier.")
    classes: List[str] = Field(..., description="Classes or answer categories. Can be empty for open-ended generation.")
    source: Optional[str] = Field(None, description="Dataset source.")
    path_images: Optional[str] = Field(None, description="Local/remote path to images.")
    path_labels: Optional[str] = Field(None, description="Local/remote path to labels (Q&A pairs).")
    preprocessing: Optional[str] = Field(None, description="Preprocessing steps.")
    augmentation: Optional[str] = Field(None, description="Augmentation strategy.")
    
    # NEW: Added num_qa_pairs as requested previously
    num_qa_pairs: Optional[int] = Field(
        None, ge=1, description="Number of question-answer pairs to generate per image."
    )

    # CHANGED: Replaced Dict with List of ClassDataSelection
    available_data: Optional[List[ClassDataSelection]] = Field(
        None, description="List mapping answer classes/types to image counts."
    )
    selected_data: Optional[List[ClassDataSelection]] = Field(
        None, description="The subset of available_data selected."
    )

    # Model candidates
    model: List[VQAModelSpecModel] = Field(
        ..., min_length=1, description="Candidate VQA model specs."
    )

    # Optional user change requests
    user_change_requests: Optional[List[str]] = Field(
        default=None, description="Optional list of user-provided change requests."
    )

    rationale: str = Field(
        ..., description="Explanation of why specific field values and hyperparameters (like LoRA configuration) were chosen."
    )
