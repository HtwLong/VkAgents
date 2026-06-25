from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator

# --- Structured Schema definitions to fix OpenAI "Any" errors ---
class DatasetSourceCount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str = Field(..., description="The name of the dataset")
    count: int = Field(..., description="Number of images from this dataset")

class ClassDataSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_name: str = Field(..., description="The name of the class or subset")
    sources: List[DatasetSourceCount] = Field(..., description="List of datasets and their counts")

class SynonymMatch(BaseModel):
    original_class: str
    found_match: bool
    dataset_classes: List[str] = Field(
        default_factory=list,
        description=(
            "Exact strings from the allowed list that match the user class. "
            "Return one class for synonym/subcategory matches, or up to ten "
            "non-overlapping valid classes for supercategory matches."
        ),
    )
    reason: str

class HardwareSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cpu_cores: Optional[int] = Field(None, ge=1, description="Number of available CPU cores.")
    gpu_type: Optional[str] = Field(None, description="Specific GPU model.")
    gpu_count: Optional[int] = Field(None, ge=0, description="Number of GPUs available.")
    ram_gb: Optional[float] = Field(None, ge=1.0, description="Available system RAM in GB.")
    storage_gb: Optional[float] = Field(None, ge=1.0, description="Available storage space in GB.")
    details: Optional[str] = Field(None, description="Any additional hardware details.")

class PerformanceSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_metric: Optional[str] = Field(..., min_length=1, description="The main metric to optimize.")
    priority: Optional[
        Literal["LatencyFirst", "AccuracyFirst", "Balanced", "ThroughputFirst"]
    ] = Field(
        None,
        description=(
            "The inferred optimization priority, normalized to ontology labels. "
            "Use LatencyFirst for real-time or low-latency needs, AccuracyFirst for best-quality "
            "or highest-score needs, ThroughputFirst for high FPS/batch processing needs, "
            "and Balanced when the user asks for a trade-off."
        ),
    )
    target_value: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target value for the primary metric.")
    latency_ms: Optional[float] = Field(None, ge=0.0, description="Max allowable latency.")
    throughput_fps: Optional[float] = Field(None, ge=0.0, description="Min required FPS.")
    other_constraints: Optional[List[str]] = Field(None, description="Other performance requirements.")

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        normalized = str(value).strip().replace("-", "_").replace(" ", "_").lower()
        priority_map = {
            "latency_first": "LatencyFirst",
            "latencyfirst": "LatencyFirst",
            "low_latency": "LatencyFirst",
            "realtime": "LatencyFirst",
            "real_time": "LatencyFirst",
            "accuracy_first": "AccuracyFirst",
            "accuracyfirst": "AccuracyFirst",
            "quality_first": "AccuracyFirst",
            "qualityfirst": "AccuracyFirst",
            "throughput_first": "ThroughputFirst",
            "throughputfirst": "ThroughputFirst",
            "high_throughput": "ThroughputFirst",
            "high_fps": "ThroughputFirst",
            "balanced": "Balanced",
            "balance": "Balanced",
            "tradeoff": "Balanced",
            "trade_off": "Balanced",
        }
        return priority_map.get(normalized, value)

class ModelSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str] = Field(..., description="Architecture name.")
    framework: Optional[str] = Field(None, description="ML framework.")
    backbone: Optional[str] = Field(None, description="Specific backbone.")
    hyperparameters: Optional[Dict[str, Union[str, float, int, bool]]] = Field(default=None)
    description: Optional[str] = Field(None, description="Rationale for choosing this model.")

class InterpretationRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Optional[Literal["classification", "detection", "segmentation", "visual question answering"]] = Field(None)
    application_domain: Optional[str] = None
    user_query: Optional[str] = None
    use_case_description: Optional[str] = None
    questions_list: Optional[List[str]] = None
    classes: List[str] = Field(default_factory=list)
    dataset_name: Optional[str] = None
    
    # Strictly enforce the list structure so OpenAI can generate valid JSON Schemas
    available_data: Optional[List[ClassDataSelection]] = None
    selected_data: Optional[List[ClassDataSelection]] = None
    
    performance_requirements: Optional[PerformanceSpecModel] = None
    available_hardware: Optional[HardwareSpecModel] = None
    model_requirements: Optional[List[ModelSpecModel]] = None
    
    # Native preprocessing fields (replaces the complex AugmentationSpecModel)
    augmentation: Optional[str] = None
    preprocessing: Optional[str] = None
    num_qa_pairs: Optional[int] = None

class PipelineState(InterpretationRequirements):
    """The unified state that grows over time."""
    model_config = ConfigDict(extra="allow")
    
    selected_model_info: Optional[Dict[str, Any]] = None
    hpo_config: Optional[Dict[str, Any]] = None
    hpo_decision: Optional[Dict[str, Any]] = None
    step_history: List[str] = Field(default_factory=list)
    last_updated: Optional[str] = None
