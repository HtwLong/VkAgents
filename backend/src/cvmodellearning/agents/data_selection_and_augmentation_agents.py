from pydantic import BaseModel, Field
from typing import List, Optional
from agents import Agent
from cvmodellearning.llm_config import PLANNING_MODEL

from cvmodellearning.schemas.interpretation_schema import ClassDataSelection
from cvmodellearning.skills import load_cv_skill

# --- 1. Define Targeted 'Patch' Schemas ---

class DataSelectionPatch(BaseModel):
    selected_data: List[ClassDataSelection] = Field(
        ...,
        description=(
            "Per-class source/count choices used locally to create train, validation, "
            "and test assignments. Every source must be listed under that exact class "
            "in allowed_sources_by_class. The source dataset_name field must contain the "
            "exact allowed dataset_id, never its display_name."
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
`classes`, `performance_requirements`, `selected_model_info`, `target_images_per_class`,
`data_selection_policy`, and
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
`data_selection_policy` is authoritative for pool targets, mixed-source sufficiency,
and original/derived lineage safety. Its target-domain share and maximum number of
generalization sources are quality recommendations: follow them when suitable, or
deviate with an explicit domain, availability, or transfer-evidence rationale.

### SELECTION RULES:
1. **Target Volume**: Treat `target_images_per_class` as the authoritative requested pool target. For detection it is calculated before source selection from pretrained-class coverage, domain shift, robustness requirements, and accuracy demand. Do not independently increase it because the request mentions high accuracy or difficult conditions. The target means the complete per-class pool across train, validation, and test—not an amount for each split. Image quality and distribution match take priority over reaching a numeric target.
2. **Scarce Classes**: If all compatible sources for a class are scarce, select the useful compatible samples and explicitly report that the class has insufficient coverage. Scarcity never overrides domain, label, annotation, or operational compatibility.
3. **Domain Alignment**: Match datasets to 'application_domain' using the supplied deterministic `domain_alignment`. The preferred primary-domain share (normally 80%) is a minimum preference, not a quota for adding generalization data. When one aligned family covers every requested class and can fill the target, prefer that family alone. Add at most one non-domain generalization source only to fill a real coverage or availability gap.
4. **Generalization**: Prefer one coherent dataset family. Mix datasets only when their domain, resolution, annotation conversion, and class meaning are compatible and the diversity benefit is explicit. Never mix merely to obtain an official holdout.
5. **Constraints**: For each class, select ONLY exact dataset IDs listed under that exact class in `allowed_sources_by_class`. In the output, `sources[].dataset_name` MUST equal the input's `dataset_id`; never put a human-readable `display_name` in that field. Never copy a source from one class to another. Never invent a dataset, class, or count. Each source count MUST be positive and MUST NOT exceed its `available_count`.
6. **Performance Categories**: For MediumHigh or High accuracy requirements, prefer more diverse data when available. For VeryLow or Low latency requirements, keep selections practical enough for faster iteration unless the user gave explicit target counts.
7. **Official Splits**: Select training sources only for the primary pool. Local deterministic code owns final split sizing. For multi-family pools it derives validation and test holdouts from the sufficiently represented training sources and keeps sources that would create statistically tiny source-level holdouts training-only. For single-family pools it may use compatible official holdouts and derives any missing holdout. Do not select official validation or test sources; those may be evaluated separately as external benchmarks.
8. **Evaluation Distribution**: Primary validation and test data must represent the training/target distribution. Unrelated data belongs only in a separate external robustness evaluation, outside `selected_data`.
9. **Count Meaning**: Counts selected from training sources describe the total source pool before derived validation/test splitting. Do not manually subtract expected holdouts. Keep class pools reasonably balanced, but do not discard useful data merely to obtain exact equality.
10. **Source Purpose and Sufficiency**: Select a source only when it provides a clear benefit: primary domain coverage, a shared multi-class backbone, or substantial coverage of an explicitly requested condition. Do not add a source merely to increase the number of dataset names. Until the output schema explicitly represents training-only auxiliary sources, every source in a mixed pool must provide at least `data_selection_policy.min_mixed_source_count` positive images per class and at least `data_selection_policy.min_mixed_source_share` of that class pool. If a useful source cannot meet both thresholds, omit it rather than emitting a plan that local validation must reject.
11. **Related and Synthetic Sources**: Do not count translated, synthetic, or otherwise derived variants as independent scene diversity when their counts mirror an original source. `mutually_exclusive_source_pairs` is a hard constraint: never select both IDs in one pair. Unless pair/group metadata guarantees leakage-safe splitting, prefer the original source and describe the derived variant as a possible training augmentation rather than selecting both as independent pools.

### OUTPUT:
- Update 'selected_data' with counts.
- Update 'rationale': Explain why you chose these specific training sources/counts, report the source-pool count per class and approximate total download count, and state whether the default download budget limited the selection. Do not predict validation/test counts; deterministic code reports the final split allocations.
"""

DATA_SELECTION_SKILLS = (
    f"{load_cv_skill('diagnose')}\n\n"
    f"{load_cv_skill('dataset-and-split')}\n\n"
    f"{load_cv_skill('data-problems')}"
)

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
        f"{DATA_SELECTION_SKILLS}\n"
        "Specific to CLASSIFICATION: Review 'allowed_sources_by_class'. Select only datasets "
        "whose native task is image classification; detection annotations are not valid "
        "image-level classification labels. Ensure class balance. "
        "Keep per-class source pools reasonably balanced, but do not automatically discard useful majority-class images. "
        "Target around 1,500 total images per class across train, validation, and test, and keep the recommendation within 10,000 total images per class and 50,000 images overall unless the user explicitly requests a smaller pool."
    ),
    output_type=DataSelectionPatch,
    model=PLANNING_MODEL
)

detection_dataset_selection_agent = Agent(
    name="Detection Data Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{BASE_SELECTION_INSTRUCTIONS}\n"
        f"{DATA_SELECTION_SKILLS}\n"
        "Return selected_data directly, with exact per-class dataset IDs and positive-image counts. "
        "Use only dataset IDs listed under that exact class in allowed_sources_by_class; copy IDs exactly. "
        "Bounding-box annotation compatibility and target-domain relevance are the main quality criteria. "
        "The deterministic split planner—not you—will choose compatible official holdouts and derive any missing "
        "validation/test split safely from training data. Do not make split-policy decisions. "
        "Use dataset_sizing.target_images_per_class as the preferred complete pool per class and "
        "dataset_sizing.minimum_images_per_class as a soft sufficiency threshold when availability permits. "
        "Counts may differ across classes and must never exceed available_count. "
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
        "Counts are positive images per class and can overlap across classes in detection datasets. "
        "Explain source choices, domain compromises, and any shortfall in rationale."
    ),
    output_type=DataSelectionPatch,
    model=PLANNING_MODEL
)

vqa_dataset_selection_agent = Agent(
    name="VQA Data Selector",
    instructions=(
        f"{PIPELINE_STATE_BLUEPRINT}\n"
        f"{BASE_SELECTION_INSTRUCTIONS}\n"
        f"{DATA_SELECTION_SKILLS}\n"
        "Specific to VQA: Select a diverse set of images from 'allowed_sources_by_class'. "
        "Prioritize visual complexity so the downstream VLM can generate rich question-answer pairs."
    ),
    output_type=DataSelectionPatch,
    model=PLANNING_MODEL
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
    model=PLANNING_MODEL
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
    model=PLANNING_MODEL
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
    model=PLANNING_MODEL
)
