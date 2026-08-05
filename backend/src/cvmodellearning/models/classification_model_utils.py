import torch.nn as nn
import timm
from torchvision.transforms import InterpolationMode
from torchvision.models import (
    resnet50, ResNet50_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
    efficientnet_b6, EfficientNet_B6_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
    densenet121, DenseNet121_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    swin_v2_s, Swin_V2_S_Weights,
)

from cvmodellearning.preprocessing.transformations import (
    CLIP_MEAN,
    CLIP_STD,
    ClassificationWeightsMetadata,
)


CLIP_VIT_B16_TIMM_ID = "vit_base_patch16_clip_quickgelu_224.openai"
CLIP_VIT_B16_WEIGHTS = ClassificationWeightsMetadata(
    crop_size=224,
    resize_size=248,
    interpolation=InterpolationMode.BICUBIC,
    mean=tuple(CLIP_MEAN),
    std=tuple(CLIP_STD),
)
DINO_V2_WEIGHTS = ClassificationWeightsMetadata(
    crop_size=224,
    resize_size=256,
    interpolation=InterpolationMode.BICUBIC,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

_TIMM_MODELS = {
    "clip_vit_b16": (CLIP_VIT_B16_TIMM_ID, CLIP_VIT_B16_WEIGHTS, {}),
    "dinov2_vits14": (
        "vit_small_patch14_dinov2.lvd142m",
        DINO_V2_WEIGHTS,
        {"img_size": 224},
    ),
    "dinov2_vitb14": (
        "vit_base_patch14_dinov2.lvd142m",
        DINO_V2_WEIGHTS,
        {"img_size": 224},
    ),
}

def _sel(weights_flag: str, enum_default):
    if weights_flag == "default":
        return enum_default
    if weights_flag == "none":
        return None
    raise ValueError(f"Unsupported weights selection: {weights_flag}")


_EFFICIENTNET_MODELS = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT),
    "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
    "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
    "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.DEFAULT),
    "efficientnet_b5": (efficientnet_b5, EfficientNet_B5_Weights.DEFAULT),
    "efficientnet_b6": (efficientnet_b6, EfficientNet_B6_Weights.DEFAULT),
    "efficientnet_b7": (efficientnet_b7, EfficientNet_B7_Weights.DEFAULT),
}


_DEFAULT_WEIGHTS = {
    "resnet50": ResNet50_Weights.DEFAULT,
    "mobilenet_v2": MobileNet_V2_Weights.DEFAULT,
    "mobilenet_v3_large": MobileNet_V3_Large_Weights.DEFAULT,
    "mobilenet_v3_small": MobileNet_V3_Small_Weights.DEFAULT,
    **{name: weights for name, (_, weights) in _EFFICIENTNET_MODELS.items()},
    "densenet121": DenseNet121_Weights.DEFAULT,
    "convnext_tiny": ConvNeXt_Tiny_Weights.DEFAULT,
    **{name: weights for name, (_, weights, _) in _TIMM_MODELS.items()},
    "vit_b_16": ViT_B_16_Weights.DEFAULT,
    "swin_v2_t": Swin_V2_T_Weights.DEFAULT,
    "swin_v2_s": Swin_V2_S_Weights.DEFAULT,
}


def get_model_weights(name: str, which_weights: str):
    """Resolve weight metadata without constructing or downloading a model."""
    if name not in _DEFAULT_WEIGHTS:
        raise ValueError(f"Unsupported model: {name}")
    if which_weights == "default":
        return _DEFAULT_WEIGHTS[name]
    if which_weights == "none":
        return None
    raise ValueError(f"Unsupported weights selection for {name}: {which_weights}")


def get_model_weights_id(name: str, which_weights: str) -> str:
    """Return the stable upstream identifier needed to reconstruct adapter bases."""
    weights = get_model_weights(name, which_weights)
    if weights is None:
        return "none"
    if name in _TIMM_MODELS:
        return f"timm:{_TIMM_MODELS[name][0]}"
    return f"torchvision:{type(weights).__name__}.{weights.name}"

def make_model(name:str, which_weights: str, num_classes: int):
    if name == "resnet50":
        w = _sel(which_weights, ResNet50_Weights.DEFAULT)
        m = resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, w
    if name == "mobilenet_v2":
        w = _sel(which_weights, MobileNet_V2_Weights.DEFAULT)
        m = mobilenet_v2(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m, w
    if name == "mobilenet_v3_large":
        w = _sel(which_weights, MobileNet_V3_Large_Weights.DEFAULT)
        m = mobilenet_v3_large(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m, w
    if name == "mobilenet_v3_small":
        w = _sel(which_weights, MobileNet_V3_Small_Weights.DEFAULT)
        m = mobilenet_v3_small(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m, w
    if name in _EFFICIENTNET_MODELS:
        constructor, default_weights = _EFFICIENTNET_MODELS[name]
        w = _sel(which_weights, default_weights)
        m = constructor(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m, w
    if name == "densenet121":
        w = _sel(which_weights, DenseNet121_Weights.DEFAULT)
        m = densenet121(weights=w)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m, w
    if name == "convnext_tiny":
        w = _sel(which_weights, ConvNeXt_Tiny_Weights.DEFAULT)
        m = convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m, w
    if name in _TIMM_MODELS:
        model_id, default_weights, model_kwargs = _TIMM_MODELS[name]
        w = _sel(which_weights, default_weights)
        m = timm.create_model(
            model_id,
            pretrained=w is not None,
            num_classes=num_classes,
            **model_kwargs,
        )
        return m, w
    if name == "vit_b_16":
        w = _sel(which_weights, ViT_B_16_Weights.DEFAULT)
        m = vit_b_16(weights=w)
        # heads is a ClassifierHead with a .head Linear layer
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m, w
    if name == "swin_v2_t":
        w = _sel(which_weights, Swin_V2_T_Weights.DEFAULT)
        m = swin_v2_t(weights=w)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m, w
    if name == "swin_v2_s":
        w = _sel(which_weights, Swin_V2_S_Weights.DEFAULT)
        m = swin_v2_s(weights=w)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m, w
    raise ValueError(f"Unsupported model: {name}")
