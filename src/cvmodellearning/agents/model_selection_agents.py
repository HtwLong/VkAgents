from pydantic import BaseModel, Field
from typing import List
from agents import Agent

# Import ONLY the specific sub-schemas
from cvmodellearning.schemas.interpretation_schema import ModelSpecModel
from cvmodellearning.schemas.detection_model_requirements import ObjectDetectionModelSpecModel
from cvmodellearning.schemas.vqa_model_requirements import VQAModelSpecModel

# --- 1. Define Targeted 'Patch' Schemas ---
class ClassificationModelPatch(BaseModel):
    model: List[ModelSpecModel] = Field(..., description="The chosen model architectures.")
    rationale: str = Field(..., description="Explanation of why this model was chosen.")

class DetectionModelPatch(BaseModel):
    model: List[ObjectDetectionModelSpecModel] = Field(..., description="The chosen model architectures.")
    rationale: str = Field(..., description="Explanation of why this model was chosen.")

class VQAModelPatch(BaseModel):
    model: List[VQAModelSpecModel] = Field(..., description="The chosen VQA model architectures.")
    rationale: str = Field(..., description="Explanation for the VQA configuration and LoRA settings.")

# --- 2. Blueprint Constant ---
PIPELINE_STATE_BLUEPRINT = """
### PIPELINE STATE STRUCTURE (Input Context):
You will receive a JSON object with the following fields. Some may be null:
- `task`: The core CV task.
- `application_domain`: The real-world use case.
- `user_query`: The original prompt.
- `classes`: Target objects/classes.
- `available_data`: Datasets found containing these classes.
"""

# --- 3. Agents ---
classification_model_selector_agent = Agent(
    name="Model Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        "Review the 'task', 'application_domain', and 'use_case_description'. "
        "Choose the best architecture. "
        "Available: resnet50, vgg16, mobilenet_v2, mobilenet_v3_large, efficientnet_b0, densenet121, convnext_tiny, vit_b_16, swin_v2_t, swin_v2_s, swin_v2_b."
    ),
    output_type=ClassificationModelPatch,
    model="gpt-5-nano"
)

detection_model_selector_agent = Agent(
    name="Detection Model Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        "Observe the extracted 'classes' and 'application_domain'. "
        "Select a model architecture. "
        "Available: yolov8, yolov10, yolo11, yolo12, fasterrcnn, maskrcnn, ssd, retinanet, rt_detr."
    ),
    output_type=DetectionModelPatch,
    model="gpt-5-nano"
)

vqq_model_selector_agent = Agent(
    name="VQA Model Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        "Review the state. For Visual Question Answering, the only architecture is 'Qwen3-VL-2B-Instruct'. "
        "Fill the rationale based on the user's specific 'questions_list' if present in the state."
    ),
    output_type=VQAModelPatch,
    model="gpt-5-nano"
)