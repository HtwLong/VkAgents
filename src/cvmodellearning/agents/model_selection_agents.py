from agents import Agent
from cvmodellearning.schemas.classification_model_requirements import ClassificationOutputModel
from cvmodellearning.schemas.detection_model_requirements import DetectionOutputModel   
from cvmodellearning.schemas.vqa_model_requirements import VQAOutputModel

classification_model_selector_agent = Agent(
    name="Model Selector",
    instructions=(
        "Given the available information about task, data, model and evaluation metrics, "
        "fill in only the fields model_architecture, model_family, description of the model property."
        "Available classification architectures include: resnet50, vgg16, mobilenet_v2, mobilenet_v3_large, "
        "efficientnet_b0, densenet121, convnext_tiny, vit_b_16, swin_v2_t, swin_v2_s, swin_v2_b. \n"
        "IMPORTANT: Fill the 'rationale' field explaining concisely WHY you chose this architecture and description. Cite your logic."
    ),
    output_type=ClassificationOutputModel,
    model="gpt-5-nano"
)

detection_model_selector_agent = Agent(
    name="Detection Model Selector",
    instructions=(
        "Given the available information about task, data, model and evaluation metrics, "
        "fill in ONLY the fields model_architecture, model_family, description of the model property."
        "Do NOT modify any hyperparameter, data related fields, the img_per_class field, the available_data field, the selected_data field, and the total_images field."
        "Available classification architectures include: yolov8, yolov10, yolo11, yolo12, fasterrcnn, maskrcnn, ssd, retinanet, rt_detr.\n"
        "IMPORTANT: Fill the 'rationale' field explaining concisely WHY you chose this architecture and description. Cite your logic."
    ),
    output_type=DetectionOutputModel,
    model="gpt-5-nano"
)

vqq_model_selector_agent = Agent(
    name="VQA Model Selector",
    instructions=(
        "Given the available information about task, data, model and evaluation metrics, "
        "fill in the VQA model specification. For visual question answering, the only available "
        "model_architecture is 'Qwen3-VL-2B-Instruct' (family 'qwen-vl'). "
        "Do NOT modify data-related fields if provided. "
        "IMPORTANT: Fill the 'rationale' field explaining your choices for the model and hyperparameters "
        "(like LoRA configuration, precision, and max_seq_length)."
    ),
    output_type=VQAOutputModel,
    model="gpt-5-nano"
)