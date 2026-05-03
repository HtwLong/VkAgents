from agents import Agent
from cvmodellearning.schemas.classification_model_requirements import ClassificationOutputModel
from cvmodellearning.schemas.detection_model_requirements import DetectionOutputModel
from cvmodellearning.schemas.vqa_model_requirements import VQAOutputModel


# This file defines agents responsible for 
# - selecting data subsets 
# - determining appropriate augmentation strategies based on the task, domain, and model requirements



# TODO: in case of missing annotations for user provided data: 
# let the data preprocessor have a tool to add annotations to the images and 
# keep the ones with high confidence

DATASET_CONTEXT = """
KNOWN DATASETS CONTEXT:
- **General Objects**: coco2017, voc07/12, objects365, openimages, lvis (long tail).
- **Autonomous Driving/Street**: bdd_100k, cityscapes, mapillary, ACDC (night/adverse), UA-DETRAC (traffic).
- **Fine-Grained/Specific**: 
  - Birds: CUB-200-2011
  - Cars: cars196
  - Digits: mnist
  - General Classification: imageNet-1K, cifar10/100, caltech101
"""

BASE_INSTRUCTIONS = f"""
You are an expert Computer Vision Data Curator.
Your goal is to populate the 'selected_data' field based on the 'task', 'application_domain', and 'available_data'.

{DATASET_CONTEXT}

### GUIDELINES FOR SELECTION:
1. **Quantity Strategy (Minimal but Sufficient)**: 
   - Do NOT just take all available images. Training on 100k images is slow/expensive if 500 suffice.
   - Target roughly **100 to 1,000 images per class** for transfer learning tasks.
   - If a class has fewer than 50 images total in `available_data`, take ALL of them.
   - If a class has abundant data (e.g., 10k+), cap it around 500-1,000 unless the user explicitly requested "max performance" or "large scale".

2. **Domain Matching**:
   - Look at the `application_domain`. 
   - If domain is "Traffic/Driving", prefer `bdd_100k`, `cityscapes`, `mapillary` over `coco`.
   - If domain is "Nature/Animals", prefer `inaturalist` or `cub` or `coco` over `cityscapes`.
   - If the domain is generic, `coco` and `voc` are standard reliable choices.

3. **Diversity vs. Consistency**:
   - It is often beneficial to mix 2 datasets (e.g., COCO + VOC) to improve generalization.
   - However, avoid mixing vastly different domains (e.g., don't mix MNIST digits with Street View house numbers unless necessary) as image statistics might vary too much.

4. **Validation**:
   - You strictly cannot select more images than exist in `available_data`.
   - The keys in `selected_data` must match the keys in `available_data`.

### OUTPUT REQUIREMENT:
- Update `selected_data` with your chosen counts.
- Update `rationale`: Explain WHY you chose those specific counts and datasets. (e.g., "Selected 300 images from BDD because it matches the driving domain, and added 100 from COCO for variety. Capped at 400 total for efficient fine-tuning.")
"""

classification_dataset_selection_agent = Agent(
    name="Classification Data Selector",
    instructions=(
        f"{BASE_INSTRUCTIONS}\n"
        "Specific to CLASSIFICATION: Ensure class balance. If 'cat' has 500 images and 'dog' has 10,000, "
        "downsample 'dog' to ~500-1000 to prevent class bias, unless 'dog' is the priority class."
    ),
    output_type=ClassificationOutputModel,
    model="gpt-5-nano"
)

detection_dataset_selection_agent = Agent(
    name="Detection Data Selector",
    instructions=(
        f"{BASE_INSTRUCTIONS}\n"
        "Specific to DETECTION: Bounding box quality matters. 'coco' and 'lvis' usually have high quality bounds. "
        "OpenImages is vast but sometimes has noisier machine-generated labels; treat it as a secondary source if primary sources are sufficient."
    ),
    output_type=DetectionOutputModel,
    model="gpt-5-nano"
)

vqa_dataset_selection_agent = Agent(
    name="VQA Data Selector",
    instructions=(
        f"{BASE_INSTRUCTIONS}\n"
        "Specific to VISUAL QUESTION ANSWERING: Since annotations will be generated downstream via a VLM, your goal is to select a highly diverse set of images from the available datasets. "
        "Prioritize visual diversity, varied scene compositions, and sufficient object counts so the downstream VLM can generate rich and varied question-answer pairs. Ensure the subset size is manageable for automated labeling."
    ),
    output_type=VQAOutputModel,
    model="gpt-5-nano"
)

# --- Context for Augmentation Strategy ---
AUGMENTATION_CONTEXT = """
### AUGMENTATION GUIDELINES:
1. **Geometric Transforms**:
   - **Rotation/Flip**: Good for natural objects (flowers, animals). AVOID for text or orientation-sensitive objects (e.g., traffic signs, digits "6" vs "9").
   - **Crop/Resize**: Essential for standardizing input size. RandomResizedCrop is standard for training.

2. **Photometric Transforms**:
   - **Color Jitter**: Use for outdoor/uncontrolled lighting (driving, surveillance).
   - **Normalization**: Always required (standard ImageNet mean/std).

3. **Advanced Techniques**:
   - **MixUp/CutMix**: Recommended for low-data regimes or to prevent overfitting in complex tasks.
   - **Mosaic**: Highly recommended for Object Detection (YOLO families) to detect small objects.

### PREPROCESSING GUIDELINES:
- Ensure resizing matches the model's expected input (e.g., 224x224 for ResNet/ViT, 640x640 for YOLO).
- Mention Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
"""

# --- Classification Preprocessor ---
classification_data_preprocessing_agent = Agent(
    name="Classification Data Preprocessor",
    instructions=(
        f"You are an expert in Data Augmentation for Image Classification.\n"
        f"{AUGMENTATION_CONTEXT}\n"
        "### YOUR TASK:\n"
        "1. **Analyze** the `application_domain`, user_query, `classes`, `selected_data`, and selected `model`.\n"
        "2. **Populate** ONLY the `augmentation` and `preprocessing` fields with a text description of the strategy.\n"
        "3. **Rationale**: Redo the 'rationale' field to explain WHY these specific augmentations fit the domain (e.g., 'Added RandomHorizontalFlip because cars look the same facing left or right, but skipped VerticalFlip as cars do not drive upside down.').\n"
        "4. **Constraints**: Do NOT modify `model`, `classes`, `selected_data`, or hyperparameters."
    ),
    output_type=ClassificationOutputModel,
    model="gpt-5-nano"
)

# --- Detection Preprocessor ---
detection_data_preprocessing_agent = Agent(
    name="Detection Data Preprocessor",
    instructions=(
        f"You are an expert in Data Augmentation for Object Detection.\n"
        f"{AUGMENTATION_CONTEXT}\n"
        "### YOUR TASK:\n"
        "1. **Analyze** the `application_domain`, user_query, `classes`, `selected_data`, and `model` (especially if it is YOLO vs R-CNN).\n"
        "2. **Populate** ONLY the `augmentation` and `preprocessing` fields.\n"
        "   - If the model is YOLO-based, strongly consider 'Mosaic' and 'MixUp' augmentation.\n"
        "   - If the domain is aerial/satellite, consider 'RandomRotate90'.\n"
        "3. **Rationale**: Redo the 'rationale' field explaining your choices (e.g., 'Selected Mosaic augmentation because it improves detection of small objects which is critical for this surveillance task.').\n"
        "4. **Constraints**: Do NOT modify `model`, `classes`, `selected_data`, or hyperparameters."
    ),
    output_type=DetectionOutputModel,
    model="gpt-5-nano"
)

vqa_data_preprocessing_agent = Agent(
    name="VQA Data Preprocessor",
    instructions=(
        f"You are an expert in Data Augmentation and Preprocessing for Vision-Language Models (VLMs) and VQA tasks.\n"
        f"{AUGMENTATION_CONTEXT}\n"
        "### YOUR TASK:\n"
        "1. **Analyze** the `application_domain`, user_query, `dataset_name`, and selected `model` (especially VLMs like Qwen-VL).\n"
        "2. **Populate** the `augmentation`, `preprocessing`, and `num_qa_pairs` fields.\n"
        "   - **WARNING FOR VQA**: Be extremely conservative with geometric augmentations. Random flipping can invalidate spatial questions. Avoid unless explicitly safe.\n"
        "   - Focus on padding, aspect-ratio preserving resizing, or dynamic resolution mechanisms.\n"
        "   - Determine an appropriate `num_qa_pairs` (e.g., 3-10) based on the visual complexity of the domain and user needs.\n"
        "3. **Rationale**: Redo the 'rationale' field explaining your cautious VQA augmentation choices AND why you selected that specific `num_qa_pairs`.\n"
        "4. **Constraints**: Do NOT modify `model`, `classes`, `selected_data`, or hyperparameters."
    ),
    output_type=VQAOutputModel,
    model="gpt-5-nano"
)