from pydantic import BaseModel, Field
from typing import List, Optional
from agents import Agent

from cvmodellearning.schemas.interpretation_schema import ClassDataSelection

# --- 1. Define Targeted 'Patch' Schemas ---

class DataSelectionPatch(BaseModel):
    selected_data: List[ClassDataSelection] = Field(
        ...,
        description=(
            "Per-class source/count choices used locally to create train, validation, "
            "and test assignments. Every source must be listed under that exact class "
            "in allowed_sources_by_class."
        ),
    )
    rationale: str = Field(..., description="Explanation of why these specific sources and counts were chosen.")

class PreprocessingPatch(BaseModel):
    augmentation: str = Field(..., description="Text description of augmentation strategy.")
    preprocessing: str = Field(..., description="Text description of preprocessing steps.")
    num_qa_pairs: Optional[int] = Field(None, description="Number of QA pairs to generate per image (VQA only).")
    rationale: str = Field(..., description="Explanation of augmentation and preprocessing choices.")

# --- 2. Knowledge Base Constants ---

PIPELINE_STATE_BLUEPRINT = """
### SELECTION CONTEXT STRUCTURE:
You will receive a JSON object with fields such as `task`, `application_domain`,
`classes`, `performance_requirements`, `selected_model_info`, and
`allowed_sources_by_class`.
"""

BASE_SELECTION_INSTRUCTIONS = f"""
You are an expert Computer Vision Data Curator. You are receiving a focused selection context.
Your goal: Populate 'selected_data' based on 'task', 'application_domain',
'allowed_sources_by_class', and any performance categories.

`allowed_sources_by_class` is authoritative. Each entry contains the exact locally
eligible sources for one class, including availability and local registry metadata.
Optional `dataset_guidance` describes or ranks some of those sources using GraphRAG.
It can improve preference quality, but it never grants eligibility. If it is absent,
use the local role, family, domain, description, and availability metadata.

### SELECTION RULES:
1. **Target Volume**: For pretrained classification, target around 1,500 usable images per class by default. For detection, target around 2,000 positive images containing each class by default. These targets mean the complete per-class pool across train, validation, and test—not an amount for each split. Select larger coherent pools whenever compatible data is available and additional diversity is likely to improve performance. Scale the selection with visual variation, target accuracy, class difficulty, and the user's stated data or resource constraints. Image quality and distribution match take priority over reaching a numeric target.
2. **Rare Classes**: If a class has < 50 total images in 'allowed_sources_by_class', you MUST select ALL of them.
3. **Domain Alignment**: Match datasets to 'application_domain' (e.g., Cityscapes for Traffic, iNaturalist for Nature).
4. **Generalization**: Prefer one coherent dataset family. Mix datasets only when their domain, resolution, annotation conversion, and class meaning are compatible and the diversity benefit is explicit. Never mix merely to obtain an official holdout.
5. **Constraints**: For each class, select ONLY exact dataset IDs listed under that exact class in `allowed_sources_by_class`. Never copy a source from one class to another. Never invent a dataset, class, or count. Each source count MUST be positive and MUST NOT exceed its `available_count`.
6. **Performance Categories**: For MediumHigh or High accuracy requirements, prefer more diverse data when available. For VeryLow or Low latency requirements, keep selections practical enough for faster iteration unless the user gave explicit target counts.
7. **Official Splits**: Select training sources only for the primary pool. Local deterministic code owns final split sizing. For multi-family pools it derives representative validation and test holdouts proportionally from every selected training source. For single-family pools it may use compatible official holdouts and derives any missing holdout. Do not select official validation or test sources; those may be evaluated separately as external benchmarks.
8. **Evaluation Distribution**: Primary validation and test data must represent the training/target distribution. Unrelated data belongs only in a separate external robustness evaluation, outside `selected_data`.
9. **Count Meaning**: Counts selected from training sources describe the total source pool before derived validation/test splitting. Do not manually subtract expected holdouts. Keep class pools reasonably balanced, but do not discard useful data merely to obtain exact equality.

### OUTPUT:
- Update 'selected_data' with counts.
- Update 'rationale': Explain why you chose these specific training sources/counts, report the source-pool count per class and approximate total download count, and state whether the default download budget limited the selection. Do not predict validation/test counts; deterministic code reports the final split allocations.
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
        "Specific to CLASSIFICATION: Review 'allowed_sources_by_class'. Select only datasets "
        "whose native task is image classification; detection annotations are not valid "
        "image-level classification labels. Ensure class balance. "
        "Keep per-class source pools reasonably balanced, but do not automatically discard useful majority-class images. "
        "Target around 1,500 total images per class across train, validation, and test, and keep the recommendation within 10,000 total images per class and 50,000 images overall unless the user explicitly requests a smaller pool."
    ),
    output_type=DataSelectionPatch,
    model="gpt-5-nano"
)

detection_dataset_selection_agent = Agent(
    name="Detection Data Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{BASE_SELECTION_INSTRUCTIONS}\n"
        "Specific to DETECTION: Review 'allowed_sources_by_class'. Bounding box quality is key. "
        "Use 'training_class_coverage_by_family' to avoid class-dataset confounding. Prefer one "
        "compatible training family as a shared backbone across as many requested classes as "
        "possible. A shared backbone should contribute at least 25% of each class pool (normally "
        "at least 500 images when available). Do not make dataset identity a proxy for class: a "
        "major family should normally contribute to multiple classes, and each class should "
        "preferably occur in multiple compatible families. Mix sources only with overlapping "
        "class coverage and compatible annotation semantics. Dataset diversity is not class "
        "diversity. Prefer domain alignment and shared class coverage; COCO, LVIS, or another "
        "broad source can be a common backbone, with domain-specific data used as overlapping "
        "supplements. Aim for a min/max class-pool ratio of at least 0.67 unless availability or "
        "domain relevance justifies an exception, and state that counts are positive images rather "
        "than object-instance counts. "
        "Target around 2,000 positive image allocations per class across train, validation, and test. Keep selections within 10,000 total image allocations per class and 50,000 summed class-source image allocations overall. The overall allocation count is a conservative proxy because one detection image may contain multiple selected classes. Instance-count limits cannot be applied unless instance statistics are present in the input."
    ),
    output_type=DataSelectionPatch,
    model="gpt-5-nano"
)

vqa_dataset_selection_agent = Agent(
    name="VQA Data Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{BASE_SELECTION_INSTRUCTIONS}\n"
        "Specific to VQA: Select a diverse set of images from 'allowed_sources_by_class'. "
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
        "1. Analyze 'application_domain', 'performance_requirements', and the 'selected_model_info' already in the state.\n"
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
        "1. Analyze the 'selected_model_info' (e.g., YOLO vs R-CNN), 'performance_requirements', and 'application_domain'.\n"
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
        "Analyze the 'selected_model_info', 'performance_requirements', and 'questions_list'.\n"
        "1. Populate 'augmentation', 'preprocessing', and 'num_qa_pairs'.\n"
        "2. WARNING: Be conservative with geometric flips as they can invalidate spatial questions."
    ),
    output_type=PreprocessingPatch,
    model="gpt-5-nano"
)
