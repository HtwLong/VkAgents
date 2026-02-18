from agents import Agent
from cvmodellearning.schemas.classification_model_requirements import ClassificationOutputModel
from cvmodellearning.schemas.detection_model_requirements import DetectionOutputModel   

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
    model="gpt-4o-mini"
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
    model="gpt-4o-mini"
)