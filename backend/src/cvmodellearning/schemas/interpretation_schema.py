from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from cvmodellearning.schemas.dataset_assignment import (
    ClassDataAssignment,
    ClassDataSelection,
    DatasetSourceCount,
    DatasetSplitCounts,
)

HardwareCategory = Literal[
    "ConsumerCPU",
    "ConsumerGPU",
    "EdgeDevice",
    "DataCenterGPU",
    "ConsumerCPU | EdgeDevice",
]
TrainingAccelerator = Literal["cpu", "mps", "cuda"]
PerformanceCategory = Literal["VeryLow", "Low", "Medium", "MediumHigh", "High"]
DeploymentLimit = Literal[
    "max_runtime_memory_mb",
    "max_model_size_mb",
    "max_parameters_m",
    "max_cpu_latency_ms",
]

class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_selected_images: int = Field(
        ...,
        ge=0,
        description=(
            "Sum of per-class selected image counts; multi-label detection images may be "
            "represented in more than one class allocation."
        ),
    )
    minimum_images_per_class: int = Field(..., ge=0)
    maximum_images_per_class: int = Field(..., ge=0)
    class_balance_ratio: float = Field(..., ge=0.0, le=1.0)
    number_of_sources: int = Field(..., ge=0)
    domains: List[str] = Field(
        default_factory=list,
        description="Deterministic domain and dataset-property tags from the local registry.",
    )
    multi_domain: bool = Field(
        False,
        description="Whether selected sources span more than one primary registry domain.",
    )
    characteristics: List[str] = Field(
        default_factory=list,
        description="Active evidence-backed dataset properties when GraphRAG was used.",
    )
    characteristic_support: Dict[str, float] = Field(
        default_factory=dict,
        description="Selected-allocation support ratio for each active characteristic.",
    )
    planned_counts: DatasetSplitCounts = Field(default_factory=DatasetSplitCounts)
    official_counts: DatasetSplitCounts = Field(default_factory=DatasetSplitCounts)
    derived_counts: DatasetSplitCounts = Field(default_factory=DatasetSplitCounts)

class SynonymMatch(BaseModel):
    original_class: str
    found_match: bool
    dataset_classes: List[str] = Field(
        default_factory=list,
        description="Exact strings from the allowed list that match the meaning.",
    )
    reason: str

class HardwareSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hardware_category: Optional[HardwareCategory] = Field(
        None,
        description=(
            "Normalized hardware class. Use ConsumerCPU, ConsumerGPU, EdgeDevice, "
            "or DataCenterGPU when stated or inferable. Use ConsumerCPU | EdgeDevice "
            "only as the fallback when neither hardware_category nor VRAM is known."
        ),
    )
    cpu_cores: Optional[int] = Field(None, ge=1, description="Number of available CPU cores.")
    gpu_type: Optional[str] = Field(None, description="Specific GPU model.")
    gpu_count: Optional[int] = Field(None, ge=0, description="Number of GPUs available.")
    vram_gb: Optional[float] = Field(None, ge=0.0, description="Available GPU VRAM in GB.")
    ram_gb: Optional[float] = Field(None, ge=1.0, description="Available system RAM in GB.")
    storage_gb: Optional[float] = Field(None, ge=1.0, description="Available storage space in GB.")
    details: Optional[str] = Field(None, description="Any additional hardware details.")

    @field_validator("hardware_category", mode="before")
    @classmethod
    def normalize_hardware_category(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        normalized = str(value).strip().replace("-", "_").replace(" ", "_").lower()
        category_map = {
            "consumer_cpu": "ConsumerCPU",
            "consumercpu": "ConsumerCPU",
            "cpu": "ConsumerCPU",
            "cpu_only": "ConsumerCPU",
            "consumer_gpu": "ConsumerGPU",
            "consumergpu": "ConsumerGPU",
            "gpu": "ConsumerGPU",
            "edge_device": "EdgeDevice",
            "edgedevice": "EdgeDevice",
            "edge": "EdgeDevice",
            "mobile": "EdgeDevice",
            "embedded": "EdgeDevice",
            "data_center_gpu": "DataCenterGPU",
            "datacenter_gpu": "DataCenterGPU",
            "datacentergpu": "DataCenterGPU",
            "data_center": "DataCenterGPU",
            "datacenter": "DataCenterGPU",
            "server_gpu": "DataCenterGPU",
            "consumer_cpu_|_edge_device": "ConsumerCPU | EdgeDevice",
            "consumercpu|edgedevice": "ConsumerCPU | EdgeDevice",
            "consumer_cpu_edge_device": "ConsumerCPU | EdgeDevice",
        }
        return category_map.get(normalized, value)


class TrainingHardwareSpec(BaseModel):
    """Server-selected hardware used to execute training, not user deployment hardware."""

    model_config = ConfigDict(extra="forbid")
    profile_id: str = Field(..., min_length=1)
    accelerator: TrainingAccelerator
    hardware_category: HardwareCategory
    gpu_type: Optional[str] = None
    gpu_count: int = Field(0, ge=0)
    vram_gb: Optional[float] = Field(None, ge=0.0)
    ram_gb: Optional[float] = Field(None, ge=1.0)
    unified_memory: bool = False
    training_memory_budget_gb: float = Field(..., gt=0.0)
    max_batch_size: int = Field(..., ge=1)
    workers: int = Field(..., ge=0)
    supports_amp: bool = False


class PerformanceSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_metric: Optional[str] = Field(..., min_length=1, description="The main metric to optimize.")
    target_value: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target value for the primary metric.")
    target_is_hard: bool = Field(
        False,
        description="True only when the user states an explicit mandatory numeric target.",
    )
    latency_category: Optional[PerformanceCategory] = Field(
        None,
        description="Optional latency category matching the model ontology.",
    )
    accuracy_category: Optional[PerformanceCategory] = Field(
        None,
        description="Optional accuracy category matching the model ontology.",
    )
    other_constraints: Optional[List[str]] = Field(None, description="Other performance requirements.")

    @field_validator("latency_category", "accuracy_category", mode="before")
    @classmethod
    def normalize_performance_category(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        normalized = str(value).strip().replace("-", "").replace("_", "").replace(" ", "").lower()
        category_map = {
            "verylow": "VeryLow",
            "low": "Low",
            "medium": "Medium",
            "mediumhigh": "MediumHigh",
            "high": "High",
        }
        return category_map.get(normalized, value)


class DeploymentConstraints(BaseModel):
    """Structured inference limits; distinct from the hardware that is available."""

    model_config = ConfigDict(extra="forbid")

    memory_category: Optional[PerformanceCategory] = Field(
        None,
        description="Requested inference-memory footprint: VeryLow, Low, Medium, MediumHigh, or High.",
    )
    max_runtime_memory_mb: Optional[float] = Field(None, gt=0)
    max_model_size_mb: Optional[float] = Field(None, gt=0)
    max_parameters_m: Optional[float] = Field(None, gt=0)
    max_cpu_latency_ms: Optional[float] = Field(None, gt=0)
    hard_limits: List[DeploymentLimit] = Field(
        default_factory=list,
        description=(
            "Numeric limit fields stated as mandatory by the user. Approximate, desirable, "
            "or preferred targets must not be included."
        ),
    )

    @field_validator("memory_category", mode="before")
    @classmethod
    def normalize_memory_category(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().replace("-", "").replace("_", "").replace(" ", "").lower()
        return {
            "verylow": "VeryLow",
            "low": "Low",
            "medium": "Medium",
            "mediumhigh": "MediumHigh",
            "high": "High",
        }.get(normalized, value)

class ModelSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str] = Field(..., description="Architecture name.")
    framework: Optional[str] = Field(None, description="ML framework.")
    backbone: Optional[str] = Field(None, description="Specific backbone.")
    hyperparameters: Optional[Dict[str, Union[str, float, int, bool]]] = Field(default=None)
    description: Optional[str] = Field(None, description="Rationale for choosing this model.")

class InterpretationRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Optional[Literal["classification", "detection", "visual question answering"]] = Field(None)
    application_domain: Optional[str] = None
    user_query: Optional[str] = None
    use_case_description: Optional[str] = None
    questions_list: Optional[List[str]] = None
    classes: List[str] = Field(default_factory=list)
    
    # Strictly enforce the list structure so OpenAI can generate valid JSON Schemas
    available_data: Optional[List[ClassDataSelection]] = None
    selected_data: Optional[List[ClassDataAssignment]] = None
    
    performance_requirements: Optional[PerformanceSpecModel] = None
    deployment_constraints: Optional[DeploymentConstraints] = None
    available_hardware: Optional[HardwareSpecModel] = Field(
        None,
        description="User-provided hardware for inference and deployment, not server training.",
    )
    model_requirements: Optional[List[ModelSpecModel]] = None
    
    # Native preprocessing fields (replaces the complex AugmentationSpecModel)
    augmentation: Optional[str] = None
    preprocessing: Optional[str] = None
    num_qa_pairs: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_selected_data(cls, value):
        if isinstance(value, dict) and value.get("selected_data") is not None:
            from cvmodellearning.schemas.dataset_assignment import normalize_dataset_assignments

            value = dict(value)
            value["selected_data"] = [
                item.model_dump(mode="json")
                for item in normalize_dataset_assignments(value["selected_data"])
            ]
        return value

class PipelineState(InterpretationRequirements):
    """The unified state that grows over time."""
    model_config = ConfigDict(extra="allow")
    training_hardware: Optional[TrainingHardwareSpec] = None
    class_expansions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Dataset classes inferred by expanding one broader user class. This provenance "
            "allows later planning steps to distinguish optional inferred labels from classes "
            "that the user requested directly."
        ),
    )
    
    model_selection_graph_context: Optional[Dict[str, Any]] = None
    dataset_selection_graph_context: Optional[Dict[str, Any]] = None
    hyperparameter_graph_context: Optional[Dict[str, Any]] = None
    hyperparameter_policy_context: Optional[Dict[str, Any]] = None
    use_graphrag: bool = True
    use_policy_registry: bool = True
    selected_model_info: Optional[Dict[str, Any]] = None
    hpo_config: Optional[Dict[str, Any]] = None
    hpo_decision: Optional[Dict[str, Any]] = None
    model_selection_decision_evidence: Optional[Dict[str, Any]] = None
    dataset_selection_decision_evidence: Optional[Dict[str, Any]] = None
    hyperparameter_decision_evidence: Optional[Dict[str, Any]] = None
    dataset_profile: Optional[DatasetProfile] = None
    step_history: List[str] = Field(default_factory=list)
    last_updated: Optional[str] = None
