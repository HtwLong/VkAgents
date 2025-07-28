from pydantic import BaseModel
from typing import List, Optional

# Pydantic Models describing wanted requirements as JSON Schema

class ProblemModel(BaseModel):
    task: str
    application_domain: str
    description: str

class DatasetModel(BaseModel):
    name: str
    classes: List[str]
    description: str
    preprocessing: Optional[str] = None
    augmentation: Optional[str] = None
    source: Optional[str] = None
    path_images: Optional[str] = None
    number_images_per_class: Optional[int]
    number_of_classes: Optional[int]
    total_number_images: Optional[int]
    path_labels: Optional[str] = None

class ModelItemModel(BaseModel):
    model_architecture: Optional[str] = None
    architecture_family: Optional[str] = None
    height: Optional[float]
    width: Optional[float]
    color_channels: Optional[float]
    description: str

class MetricItemModel(BaseModel):
    name: Optional[str] = None
    value: Optional[float]

class StructuredOutputModel(BaseModel):
    problem: ProblemModel
    dataset: DatasetModel
    model: List[ModelItemModel]
    performance_metrics: List[MetricItemModel]
