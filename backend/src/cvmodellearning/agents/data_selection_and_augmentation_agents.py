from pydantic import BaseModel, Field
from typing import List, Optional
from agents import Agent

from cvmodellearning.schemas.interpretation_schema import ClassDataSelection

# --- 1. Define Targeted 'Patch' Schemas ---

class DataSelectionPatch(BaseModel):
    selected_data: List[ClassDataSelection] = Field(..., description="The subset of data selected for training.")
    rationale: str = Field(..., description="Explanation of why these specific sources and counts were chosen.")

class PreprocessingPatch(BaseModel):
    augmentation: str = Field(..., description="Text description of augmentation strategy.")
    preprocessing: str = Field(..., description="Text description of preprocessing steps.")
    num_qa_pairs: Optional[int] = Field(None, description="Number of QA pairs to generate per image (VQA only).")
    rationale: str = Field(..., description="Explanation of augmentation and preprocessing choices.")

# --- 2. Knowledge Base Constants ---

PIPELINE_STATE_BLUEPRINT = """
### PIPELINE STATE STRUCTURE (Input Context):
You will receive a JSON object with fields like `task`, `application_domain`, `available_data`, `classes`, and `selected_model_info`.
"""

DATASET_CONTEXT = """
KNOWN DATASETS CONTEXT:
- **General Objects**: coco2017, voc07/12, objects365, openimages, lvis (long tail).
- **Autonomous Driving/Street**: bdd_100k, cityscapes, mapillary, ACDC (night/adverse), UA-DETRAC (traffic).
- **Fine-Grained/Specific**: Birds (CUB-200), Cars (cars196), Digits (mnist), General (ImageNet-1K).
"""

BASE_SELECTION_INSTRUCTIONS = f"""
You are an expert Computer Vision Data Curator. You are receiving a full PipelineState JSON.
Your goal: Populate 'selected_data' based on 'task', 'application_domain', and 'available_data'.

{DATASET_CONTEXT}

### SELECTION RULES:
1. **Target Volume**: Aim for 100–1,000 images per class for transfer learning.
2. **Rare Classes**: If a class has < 50 total images in 'available_data', you MUST select ALL of them.
3. **Domain Alignment**: Match datasets to 'application_domain' (e.g., Cityscapes for Traffic, iNaturalist for Nature).
4. **Generalization**: When data is abundant, prefer mixing ~2 compatible datasets (e.g., COCO + VOC) for better diversity.
5. **Constraints**: Total selected count for any class MUST NOT exceed its 'available_data' count.

### OUTPUT:
- Update 'selected_data' with counts.
- Update 'rationale': Explain why you chose these specific sources/counts (e.g., "Downsampled 'dog' to 500 to match 'cat' for class balance").
"""

AUGMENTATION_CONTEXT = """
### AUGMENTATION & PREPROCESSING GUIDELINES:

1. **Geometric Transforms**:
   - *Rotation/Flip*: Good for natural objects. STRICTLY AVOID for text, digits (e.g., "6" vs "9"), or orientation-sensitive domains (e.g., traffic signs).
   - *Crop/Resize*: RandomResizedCrop is standard for training to improve translation invariance.

2. **Photometric Transforms**:
   - *Color Jitter*: Recommended for outdoor domains or uncontrolled lighting (e.g., autonomous driving, surveillance).

3. **Advanced Techniques**:
   - *MixUp/CutMix*: Recommended to prevent overfitting in complex tasks or low-data regimes.
   - *Mosaic*: STRONGLY recommended for Object Detection (YOLO families) to improve small object detection.

4. **Mandatory Preprocessing (DO NOT OMIT)**:
   - *Resolution*: Ensure resizing matches the selected model's expected input (e.g., 224x224 for ResNet/ViT, 640x640 for YOLO).
   - *Normalization*: Always include standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
"""

# --- 3. Dataset Selection Agents ---

classification_dataset_selection_agent = Agent(
    name="Classification Data Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{BASE_SELECTION_INSTRUCTIONS}\n"
        "Specific to CLASSIFICATION: Review 'available_data' in the state. Ensure class balance. "
        "If classes are imbalanced, downsample the majority classes in 'selected_data' to match minority classes."
    ),
    output_type=DataSelectionPatch,
    model="gpt-5-nano"
)

detection_dataset_selection_agent = Agent(
    name="Detection Data Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{BASE_SELECTION_INSTRUCTIONS}\n"
        "Specific to DETECTION: Review 'available_data' in the state. Bounding box quality is key. "
        "Prioritize 'coco' and 'lvis' for high-quality labels if they match the domain."
    ),
    output_type=DataSelectionPatch,
    model="gpt-5-nano"
)

vqa_dataset_selection_agent = Agent(
    name="VQA Data Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{BASE_SELECTION_INSTRUCTIONS}\n"
        "Specific to VQA: Select a diverse set of images from the 'available_data'. "
        "Prioritize visual complexity so the downstream VLM can generate rich question-answer pairs."
    ),
    output_type=DataSelectionPatch,
    model="gpt-5-nano"
)

# --- 4. Preprocessing & Augmentation Agents ---

classification_data_preprocessing_agent = Agent(
    name="Classification Data Preprocessor",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"You are an expert in Data Augmentation. You are receiving a full PipelineState JSON.\n"
        f"{AUGMENTATION_CONTEXT}\n"
        "### YOUR TASK:\n"
        "1. Analyze 'application_domain' and the 'selected_model_info' already in the state.\n"
        "2. Populate ONLY the 'augmentation' and 'preprocessing' text fields.\n"
        "3. Rationale: Explain WHY these fits the domain (e.g., 'No vertical flip because the domain is medical X-rays')."
    ),
    output_type=PreprocessingPatch,
    model="gpt-5-nano"
)

detection_data_preprocessing_agent = Agent(
    name="Detection Data Preprocessor",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"You are an expert in Detection Augmentation. You are receiving a full PipelineState JSON.\n"
        f"{AUGMENTATION_CONTEXT}\n"
        "### YOUR TASK:\n"
        "1. Analyze the 'selected_model_info' (e.g., YOLO vs R-CNN) and 'application_domain'.\n"
        "2. Populate 'augmentation' and 'preprocessing'.\n"
        "3. If the model is YOLO-based, strongly recommend 'Mosaic' and 'MixUp' augmentation."
    ),
    output_type=PreprocessingPatch,
    model="gpt-5-nano"
)

vqa_data_preprocessing_agent = Agent(
    name="VQA Data Preprocessor",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"You are an expert in VLM Preprocessing. You are receiving a full PipelineState JSON.\n"
        f"{AUGMENTATION_CONTEXT}\n"
        "Analyze the 'selected_model_info' and 'questions_list'.\n"
        "1. Populate 'augmentation', 'preprocessing', and 'num_qa_pairs'.\n"
        "2. WARNING: Be conservative with geometric flips as they can invalidate spatial questions."
    ),
    output_type=PreprocessingPatch,
    model="gpt-5-nano"
)