import math
from typing import List, Literal, Optional, Self
from pydantic import BaseModel, Field, ConfigDict, model_validator
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

class VQAConfigModel(BaseModel):
    """
    Structured schema for Visual Question Answering (VQA) training configuration,
    specifically tailored for Vision-Language Models (VLMs) like Qwen-VL.
    Uses registry-backed enums for key selector fields.
    """
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    # --- Data/Task ---
    task_type: Literal["visual question answering"] = Field(
        "visual question answering",
        description="The task type must be 'visual question answering'."
    )
    classes: List[str] = Field(
        ..., 
        description="List of answer categories if treated as classification, or an empty list if purely open-ended generation."
    )
    
    # CHANGED: Now uses the strict-compatible list structure
    selected_data: List[ClassDataSelection] = Field(
        ..., 
        description="List of dataset sources and respective image/question counts."
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
        description="Set the maximum number of training epochs; must be ≥ 1. Often very low for VLMs (1-3)."
    )
    patience: int = Field(
        ..., ge=0,
        description="Set epochs without improvement to wait before early stopping; must be ≥ 0."
    )
    batch_size: int = Field(
        2, ge=1,
        description="Set mini-batch size in samples. Usually very small (1, 2, or 4) for VLMs due to memory constraints."
    )
    max_seq_length: int = Field(
        2048, ge=128,
        description="Maximum sequence length for the combined text and image tokens."
    )
    track_metric: Literal["val_loss", "exact_match", "f1", "meteor", "rouge", "cider"] = Field(
        "val_loss",
        description="Validation metric to monitor. 'val_loss' is standard for early stopping, while 'meteor', 'rouge', or 'cider' are better for evaluating semantic meaning."
    )

    # --- Model Selection (VLM) ---
    model_name: VQAModelId = Field(
        ...,
        description="Select the Vision-Language Model architecture."
    )
    precision: Literal["bf16", "fp16", "fp32", "fp8"] = Field(
        "bf16",
        description="Training precision. 'bf16' is highly recommended for modern VLMs to prevent overflow and save memory."
    )

    # --- PEFT / LoRA Configuration ---
    use_lora: bool = Field(
        True,
        description="Whether to use Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning."
    )
    lora_r: int = Field(
        16, ge=1,
        description="LoRA rank parameter. Usually 8, 16, or 64. Used only if use_lora is True."
    )
    lora_alpha: int = Field(
        32, ge=1,
        description="LoRA alpha scaling parameter. Typically 2x lora_r. Used only if use_lora is True."
    )
    lora_dropout: float = Field(
        0.05, ge=0.0, lt=1.0,
        description="Dropout probability for LoRA layers."
    )
    
    # --- Optimizer Selection ---
    optimizer_name: Literal["adamw", "paged_adamw_8bit", "rmsprop", "sgd"] = Field(
        "adamw",
        description="Select the optimizer algorithm. 'paged_adamw_8bit' is great for saving memory."
    )
    learning_rate: float = Field(
        2e-5, gt=0, 
        description="Base learning rate. VLMs require much lower learning rates than pure vision models (e.g., 1e-5 to 5e-5)."
    )
    weight_decay: float = Field(
        0.01, ge=0, 
        description="L2 regularization coefficient."
    )
    eps: float = Field(
        1e-8, gt=0, lt=0.01,
        description="Epsilon for AdamW/RMSprop numerical stability."
    )
    beta1: float = Field(0.9, gt=0, lt=1, description="AdamW β1 parameter.")
    beta2: float = Field(0.999, gt=0, lt=1, description="AdamW β2 parameter.")
    
    rationale: str = Field(
        ..., 
        description="Explain the choice of hyperparameters, especially memory considerations (batch size, LoRA, precision)."
    )

    @model_validator(mode="after")
    def _validate_combinations(self) -> Self:
        total_ratio = self.train_data_ratio + self.val_data_ratio + self.test_data_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
            raise ValueError(f"train_data_ratio, val_data_ratio, and test_data_ratio must sum to 1.0. Current sum: {total_ratio}")

        if self.learning_rate > 1e-3:
            raise ValueError("Learning rate is suspiciously high for a pre-trained VLM. It should typically be <= 1e-3 to avoid catastrophic forgetting.")

        return self
