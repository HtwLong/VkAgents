from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


TaskName = Literal["classification", "detection", "visual question answering"]


class ClassificationModelId(str, Enum):
    RESNET50 = "resnet50"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B3 = "efficientnet_b3"
    EFFICIENTNET_B4 = "efficientnet_b4"
    EFFICIENTNET_B5 = "efficientnet_b5"
    EFFICIENTNET_B6 = "efficientnet_b6"
    EFFICIENTNET_B7 = "efficientnet_b7"
    DENSENET121 = "densenet121"
    CONVNEXT_TINY = "convnext_tiny"
    CLIP_VIT_B16 = "clip_vit_b16"
    DINOV2_VITS14 = "dinov2_vits14"
    DINOV2_VITB14 = "dinov2_vitb14"
    VIT_B_16 = "vit_b_16"
    SWIN_V2_T = "swin_v2_t"
    SWIN_V2_S = "swin_v2_s"


class ClassificationModelFamily(str, Enum):
    RESNET = "resnet"
    MOBILENET = "mobilenet"
    EFFICIENTNET = "efficientnet"
    DENSENET = "densenet"
    CONVNEXT = "convnext"
    CLIP = "clip"
    DINOV2 = "dinov2"
    VIT = "vit"
    SWIN_V2 = "swin_v2"


# Execution capabilities shared by schema validation and the training loop.
# Keep this deliberately small: only models with tested head/backbone handling
# belong here.
CLASSIFIER_HEAD_PATHS = {
    "resnet50": "fc",
    "mobilenet_v2": "classifier.1",
    "mobilenet_v3_large": "classifier.3",
    "mobilenet_v3_small": "classifier.3",
    "efficientnet_b0": "classifier.1",
    "efficientnet_b1": "classifier.1",
    "efficientnet_b2": "classifier.1",
    "efficientnet_b3": "classifier.1",
    "efficientnet_b4": "classifier.1",
    "efficientnet_b5": "classifier.1",
    "efficientnet_b6": "classifier.1",
    "efficientnet_b7": "classifier.1",
    "densenet121": "classifier",
    "convnext_tiny": "classifier.2",
    "clip_vit_b16": "head",
    "dinov2_vits14": "head",
    "dinov2_vitb14": "head",
    "vit_b_16": "heads.head",
    "swin_v2_t": "head",
    "swin_v2_s": "head",
}
FREEZABLE_CLASSIFICATION_MODEL_IDS = frozenset(CLASSIFIER_HEAD_PATHS)
HEAD_LR_MULTIPLIER_MODEL_IDS = frozenset(CLASSIFIER_HEAD_PATHS)

# PEFT target modules are execution details, not LLM-generated values. Keep
# them registry-backed so schema validation and the trainer cannot disagree.
CLASSIFICATION_LORA_TARGET_MODULES = {
    "clip_vit_b16": r"(?:^|.*\.)attn\.(qkv|proj)$",
    "dinov2_vits14": r"(?:^|.*\.)attn\.(qkv|proj)$",
    "dinov2_vitb14": r"(?:^|.*\.)attn\.(qkv|proj)$",
    # TorchVision ViT uses fused MultiheadAttention parameters rather than
    # Linear q/k/v modules, so adapt its transformer MLP projections.
    "vit_b_16": r"(?:^|.*\.)mlp\.(0|3)$",
    "swin_v2_t": r"(?:^|.*\.)attn\.(qkv|proj)$",
    "swin_v2_s": r"(?:^|.*\.)attn\.(qkv|proj)$",
}
LORA_CLASSIFICATION_MODEL_IDS = frozenset(CLASSIFICATION_LORA_TARGET_MODULES)


class DetectionModelId(str, Enum):
    YOLOV8 = "yolov8"
    YOLOV10 = "yolov10"
    YOLOV11 = "yolov11"
    YOLOV12 = "yolov12"
    RETINANET_R50_FPN_1X_COCO = "retinanet_r50_fpn_1x_coco"
    FASTER_RCNN_R50_FPN_1X_COCO = "faster-rcnn_r50_fpn_1x_coco"
    SSD300_COCO = "ssd300_coco"
    RTDETR_HGNETV2_L = "rtdetr_hgnetv2_l"


class DetectionModelFamily(str, Enum):
    YOLO = "yolo"
    RETINANET = "retinanet"
    FASTER_RCNN = "faster-rcnn"
    SSD = "ssd"
    RTDETR = "rtdetr"


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
    SSD300 = "ssd300"
    RTDETR_HGNETV2_L = "rtdetr_hgnetv2_l"


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
    lora_supported: bool = False


MODEL_REGISTRY: tuple[ModelDefinition, ...] = (
    ModelDefinition("resnet50", "classification", "resnet", "ResNet-50", "resnet50"),
    ModelDefinition("mobilenet_v2", "classification", "mobilenet", "MobileNet V2", "mobilenet_v2"),
    ModelDefinition("mobilenet_v3_large", "classification", "mobilenet", "MobileNet V3 Large", "mobilenet_v3_large"),
    ModelDefinition("mobilenet_v3_small", "classification", "mobilenet", "MobileNet V3 Small", "mobilenet_v3_small"),
    ModelDefinition("efficientnet_b0", "classification", "efficientnet", "EfficientNet-B0", "efficientnet_b0"),
    ModelDefinition("efficientnet_b1", "classification", "efficientnet", "EfficientNet-B1", "efficientnet_b1"),
    ModelDefinition("efficientnet_b2", "classification", "efficientnet", "EfficientNet-B2", "efficientnet_b2"),
    ModelDefinition("efficientnet_b3", "classification", "efficientnet", "EfficientNet-B3", "efficientnet_b3"),
    ModelDefinition("efficientnet_b4", "classification", "efficientnet", "EfficientNet-B4", "efficientnet_b4"),
    ModelDefinition("efficientnet_b5", "classification", "efficientnet", "EfficientNet-B5", "efficientnet_b5"),
    ModelDefinition("efficientnet_b6", "classification", "efficientnet", "EfficientNet-B6", "efficientnet_b6"),
    ModelDefinition("efficientnet_b7", "classification", "efficientnet", "EfficientNet-B7", "efficientnet_b7"),
    ModelDefinition("densenet121", "classification", "densenet", "DenseNet-121", "densenet121"),
    ModelDefinition("convnext_tiny", "classification", "convnext", "ConvNeXt Tiny", "convnext_tiny"),
    ModelDefinition("clip_vit_b16", "classification", "clip", "CLIP ViT-B/16", "clip_vit_b16", lora_supported=True),
    ModelDefinition("dinov2_vits14", "classification", "dinov2", "DINOv2 ViT-S/14", "dinov2_vits14", lora_supported=True),
    ModelDefinition("dinov2_vitb14", "classification", "dinov2", "DINOv2 ViT-B/14", "dinov2_vitb14", lora_supported=True),
    ModelDefinition("vit_b_16", "classification", "vit", "ViT-B/16", "vit_b_16", lora_supported=True),
    ModelDefinition("swin_v2_t", "classification", "swin_v2", "Swin V2 Tiny", "swin_v2_t", lora_supported=True),
    ModelDefinition("swin_v2_s", "classification", "swin_v2", "Swin V2 Small", "swin_v2_s", lora_supported=True),
    ModelDefinition("yolov8", "detection", "yolo", "YOLOv8", "yolo_v8", "yolov8_n", aliases=("yolo8",)),
    ModelDefinition("yolov10", "detection", "yolo", "YOLOv10", "yolo_v10", "yolov10_n", aliases=("yolo10",)),
    ModelDefinition("yolov11", "detection", "yolo", "YOLO11", "yolo_v11", "yolov11_n", aliases=("yolo11",)),
    ModelDefinition("yolov12", "detection", "yolo", "YOLO12", "yolo_v12", "yolov12_n", aliases=("yolo12",)),
    ModelDefinition(
        "retinanet_r50_fpn_1x_coco",
        "detection",
        "retinanet",
        "RetinaNet R50 FPN",
        "retinanet_r50",
        aliases=("retinanet", "retinanet_r50", "retinanet_resnet50_fpn"),
    ),
    ModelDefinition(
        "faster-rcnn_r50_fpn_1x_coco",
        "detection",
        "faster-rcnn",
        "Faster R-CNN R50 FPN",
        "faster_rcnn_r50",
        aliases=(
            "fasterrcnn",
            "faster_rcnn",
            "faster-rcnn",
            "fasterrcnn_resnet50_fpn",
        ),
    ),
    ModelDefinition(
        "ssd300_coco",
        "detection",
        "ssd",
        "SSD300 VGG16",
        "ssd300",
        aliases=("ssd", "ssd300_vgg16"),
    ),
    ModelDefinition(
        "rtdetr_hgnetv2_l",
        "detection",
        "rtdetr",
        "RT-DETR HGNetV2-L",
        "rtdetr_hgnetv2_l",
        aliases=("rtdetr-l", "rt_detr_l", "rt-detr-l", "rtdetr"),
    ),
    ModelDefinition("Qwen3-VL-2B-Instruct", "visual question answering", "qwen-vl", "Qwen3-VL 2B Instruct"),
)

DETECTION_HPO_MODEL_IDS: tuple[str, ...] = tuple(model.value for model in DetectionHpoModelId)


def canonical_model_id(value: str) -> str:
    """Normalize equivalent ontology and executable model identifiers."""
    normalized = "".join(character for character in value.lower() if character.isalnum())
    return normalized.replace("yolo11", "yolov11").replace("yolo12", "yolov12")


def model_ids_equivalent(left: str, right: str) -> bool:
    """Return whether two non-empty model references identify the same model."""
    return bool(left and right) and canonical_model_id(left) == canonical_model_id(right)


def enabled_models(task: TaskName) -> tuple[ModelDefinition, ...]:
    return tuple(model for model in MODEL_REGISTRY if model.enabled and model.task == task)


def model_ids(task: TaskName) -> tuple[str, ...]:
    return tuple(model.id for model in enabled_models(task))


def families(task: TaskName) -> tuple[str, ...]:
    return tuple(dict.fromkeys(model.family for model in enabled_models(task)))


def family_by_model_id(task: TaskName) -> dict[str, str]:
    return {model.id: model.family for model in enabled_models(task)}


def family_for_model_reference(task: TaskName, value: str) -> Optional[str]:
    """Resolve the family of a registered model or executable YOLO variant."""
    resolved = resolve_model_id(task, value)
    if resolved:
        return family_by_model_id(task)[resolved]

    if task == "detection":
        normalized = canonical_model_id(value)
        for model in enabled_models(task):
            base = canonical_model_id(model.id)
            if (
                model.family == "yolo"
                and normalized.startswith(base)
                and normalized[len(base):] in {"n", "s", "m", "l", "x"}
            ):
                return model.family

    return None


def resolve_model_id(task: TaskName, value: str) -> Optional[str]:
    normalized = value.strip().lower()
    for model in enabled_models(task):
        if normalized == model.id.lower():
            return model.id
        if normalized in {alias.lower() for alias in model.aliases}:
            return model.id
    return None


def is_executable_model_reference(task: TaskName, value: str) -> bool:
    """Return whether an ontology/model identifier resolves to executable code."""
    if resolve_model_id(task, value) is not None:
        return True
    if task != "detection":
        return False

    normalized = canonical_model_id(value)
    executable_hpo_ids = {
        canonical_model_id(model_id)
        for model_id in DETECTION_HPO_MODEL_IDS
    }
    return normalized in executable_hpo_ids


def format_available_models(task: TaskName) -> str:
    return ", ".join(model.id for model in enabled_models(task))
