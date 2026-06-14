import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    swin_v2_s, Swin_V2_S_Weights,
    swin_v2_b, Swin_V2_B_Weights,
)

def _sel(weights_flag: str, enum_default):
    return enum_default if weights_flag == "default" else None

# TODO_ later add a function to allow usage of LoRA or not
# only for the models: ViT16, Swin Transformers, CLIP, ConvNeXt, Dino V2
def make_lora_model():
    pass

def make_model(name:str, which_weights: str, num_classes: int):
    if name == "resnet50":
        w = _sel(which_weights, ResNet50_Weights.DEFAULT)
        m = resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, w
    if name == "vgg16":
        w = _sel(which_weights, VGG16_Weights.DEFAULT)
        m = vgg16(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
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
    if name == "efficientnet_b0":
        w = _sel(which_weights, EfficientNet_B0_Weights.DEFAULT)
        m = efficientnet_b0(weights=w)
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
        m.classifier[47] = nn.Linear(m.classifier[47].in_features, num_classes)
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
    if name == "swin_v2_b":
        w = _sel(which_weights, Swin_V2_B_Weights.DEFAULT)
        m = swin_v2_b(weights=w)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m, w
    raise ValueError(f"Unsupported model: {name}")

