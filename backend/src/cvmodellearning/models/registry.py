from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


TaskName = Literal["classification", "detection", "visual question answering"]


class ClassificationModelId(str, Enum):
    RESNET50 = "resnet50"
    VGG16 = "vgg16"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    EFFICIENTNET_B0 = "efficientnet_b0"
    DENSENET121 = "densenet121"
    CONVNEXT_TINY = "convnext_tiny"
    VIT_B_16 = "vit_b_16"
    SWIN_V2_T = "swin_v2_t"
    SWIN_V2_S = "swin_v2_s"
    SWIN_V2_B = "swin_v2_b"


class ClassificationModelFamily(str, Enum):
    RESNET = "resnet"
    VGG = "vgg"
    MOBILENET = "mobilenet"
    EFFICIENTNET = "efficientnet"
    DENSENET = "densenet"
    CONVNEXT = "convnext"
    VIT = "vit"
    SWIN_V2 = "swin_v2"


class DetectionModelId(str, Enum):
    YOLOV8 = "yolov8"
    YOLOV10 = "yolov10"
    YOLOV11 = "yolov11"
    YOLOV12 = "yolov12"
    RETINANET_R50_FPN_1X_COCO = "retinanet_r50_fpn_1x_coco"
    FASTER_RCNN_R50_FPN_1X_COCO = "faster-rcnn_r50_fpn_1x_coco"
    MASK_RCNN_R50_FPN_1X_COCO = "mask-rcnn_r50_fpn_1x_coco"
    SSD300_COCO = "ssd300_coco"
    RT_DETR_R50_8XB2_100E_COCO = "rt-detr-r50_8xb2-100e_coco"


class DetectionModelFamily(str, Enum):
    YOLO = "yolo"
    RETINANET = "retinanet"
    FASTER_RCNN = "faster-rcnn"
    MASK_RCNN = "mask-rcnn"
    SSD = "ssd"
    DETR = "detr"


class DetectionHpoModelId(str, Enum):
    YOLOV8_N = "yolov8_n"
    YOLOV8_S = "yolov8_s"
    YOLOV8_M = "yolov8_m"
    YOLOV8_L = "yolov8_l"
    YOLOV8_X = "yolov8_x"
    YOLOV10_N = "yolov10_n"
    YOLOV10_S = "yolov10_s"
    YOLOV10_M = "yolov10_m"
    YOLOV10_L = "yolov10_l"
    YOLOV10_X = "yolov10_x"
    YOLOV11_N = "yolov11_n"
    YOLOV11_S = "yolov11_s"
    YOLOV11_M = "yolov11_m"
    YOLOV11_L = "yolov11_l"
    YOLOV11_X = "yolov11_x"
    YOLOV12_N = "yolov12_n"
    YOLOV12_S = "yolov12_s"
    YOLOV12_M = "yolov12_m"
    YOLOV12_L = "yolov12_l"
    YOLOV12_X = "yolov12_x"
    RETINANET_R50 = "retinanet_r50"
    FASTER_RCNN_R50 = "faster_rcnn_r50"
    MASK_RCNN_R50 = "mask_rcnn_r50"
    SSD300 = "ssd300"
    RT_DETR_R50 = "rt_detr_r50"


class VQAModelId(str, Enum):
    QWEN3_VL_2B_INSTRUCT = "Qwen3-VL-2B-Instruct"


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    task: TaskName
    family: str
    display_name: str
    trainer_key: Optional[str] = None
    hpo_id: Optional[str] = None
    aliases: tuple[str, ...] = ()
    enabled: bool = True
    notes: Optional[str] = None


MODEL_REGISTRY: tuple[ModelDefinition, ...] = (
    ModelDefinition("resnet50", "classification", "resnet", "ResNet-50", "resnet50"),
    ModelDefinition("vgg16", "classification", "vgg", "VGG-16", "vgg16"),
    ModelDefinition("mobilenet_v2", "classification", "mobilenet", "MobileNet V2", "mobilenet_v2"),
    ModelDefinition("mobilenet_v3_large", "classification", "mobilenet", "MobileNet V3 Large", "mobilenet_v3_large"),
    ModelDefinition("efficientnet_b0", "classification", "efficientnet", "EfficientNet-B0", "efficientnet_b0"),
    ModelDefinition("densenet121", "classification", "densenet", "DenseNet-121", "densenet121"),
    ModelDefinition("convnext_tiny", "classification", "convnext", "ConvNeXt Tiny", "convnext_tiny"),
    ModelDefinition("vit_b_16", "classification", "vit", "ViT-B/16", "vit_b_16"),
    ModelDefinition("swin_v2_t", "classification", "swin_v2", "Swin V2 Tiny", "swin_v2_t"),
    ModelDefinition("swin_v2_s", "classification", "swin_v2", "Swin V2 Small", "swin_v2_s"),
    ModelDefinition("swin_v2_b", "classification", "swin_v2", "Swin V2 Base", "swin_v2_b"),
    ModelDefinition("yolov8", "detection", "yolo", "YOLOv8", "yolo_v8", aliases=("yolo8",)),
    ModelDefinition("yolov10", "detection", "yolo", "YOLOv10", "yolo_v10", aliases=("yolo10",)),
    ModelDefinition("yolov11", "detection", "yolo", "YOLO11", "yolo_v11", aliases=("yolo11",)),
    ModelDefinition("yolov12", "detection", "yolo", "YOLO12", "yolo_v12", aliases=("yolo12",)),
    ModelDefinition(
        "retinanet_r50_fpn_1x_coco",
        "detection",
        "retinanet",
        "RetinaNet R50 FPN",
        "retinanet_r50",
        aliases=("retinanet", "retinanet_r50"),
    ),
    ModelDefinition(
        "faster-rcnn_r50_fpn_1x_coco",
        "detection",
        "faster-rcnn",
        "Faster R-CNN R50 FPN",
        "faster_rcnn_r50",
        aliases=("fasterrcnn", "faster_rcnn", "faster-rcnn"),
    ),
    ModelDefinition(
        "mask-rcnn_r50_fpn_1x_coco",
        "detection",
        "mask-rcnn",
        "Mask R-CNN R50 FPN",
        "mask_rcnn_r50",
        aliases=("maskrcnn", "mask_rcnn", "mask-rcnn"),
    ),
    ModelDefinition("ssd300_coco", "detection", "ssd", "SSD300", "ssd300", aliases=("ssd",)),
    ModelDefinition(
        "rt-detr-r50_8xb2-100e_coco",
        "detection",
        "detr",
        "RT-DETR R50",
        "rt_detr_r50",
        aliases=("rt_detr", "rtdetr", "rt-detr"),
    ),
    ModelDefinition("Qwen3-VL-2B-Instruct", "visual question answering", "qwen-vl", "Qwen3-VL 2B Instruct"),
)

DETECTION_HPO_MODEL_IDS: tuple[str, ...] = tuple(model.value for model in DetectionHpoModelId)


def enabled_models(task: TaskName) -> tuple[ModelDefinition, ...]:
    return tuple(model for model in MODEL_REGISTRY if model.enabled and model.task == task)


def model_ids(task: TaskName) -> tuple[str, ...]:
    return tuple(model.id for model in enabled_models(task))


def families(task: TaskName) -> tuple[str, ...]:
    return tuple(dict.fromkeys(model.family for model in enabled_models(task)))


def family_by_model_id(task: TaskName) -> dict[str, str]:
    return {model.id: model.family for model in enabled_models(task)}


def resolve_model_id(task: TaskName, value: str) -> Optional[str]:
    normalized = value.strip().lower()
    for model in enabled_models(task):
        if normalized == model.id.lower():
            return model.id
        if normalized in {alias.lower() for alias in model.aliases}:
            return model.id
    return None


def format_available_models(task: TaskName) -> str:
    return ", ".join(model.id for model in enabled_models(task))
