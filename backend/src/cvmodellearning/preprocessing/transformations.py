from torchvision import transforms

# This file contains predefined image transform pipelines

# ================================
# ResNet50

# Training
train_tfms_resnet50 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Evaluation
eval_tfms_resnet50 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ==============================
# EfficientNet-B0

from torchvision import transforms
from torchvision.transforms import InterpolationMode

train_tfms_efficientnet = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_tfms_efficientnet = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# =========================
# Inception v3

from torchvision import transforms

# Training
train_tfms_inceptionv3 = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Evaluation (often Resize to ~342 then CenterCrop to 299)
eval_tfms_inceptionv3 = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =============================
# ViT-B/16 (Vision Transformer)

from torchvision import transforms

train_tfms_vit = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_tfms_vit = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ======================
# MobileNetV3-Large

from torchvision import transforms

train_tfms_mobilenetv3 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_tfms_mobilenetv3 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ==========================
# VGG16

from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Training transforms
train_tfms_vgg16 = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    # Optional: transforms.ColorJitter(0.2, 0.2, 0.2)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Evaluation transforms
eval_tfms_vgg16 = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Training transforms for Swin V2
train_tfms_swin_v2 = transforms.Compose([
    transforms.RandomResizedCrop(256, interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==========================
# Swin V2
eval_tfms_swin_v2 = transforms.Compose([
    transforms.Resize(260, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def select_transforms(name):
    """
    Return (train_transform, eval_transform) picked to match the model architecture.
    Falls back to the standard 224 ImageNet pipeline when no specific pair is defined.
    """
    if name == "resnet50":
        return train_tfms_resnet50, eval_tfms_resnet50
    if name == "efficientnet_b0":
        return train_tfms_efficientnet, eval_tfms_efficientnet
    if name == "vit_b_16":
        return train_tfms_vit, eval_tfms_vit
    if name == "mobilenet_v3_large":
        return train_tfms_mobilenetv3, eval_tfms_mobilenetv3
    if name == "vgg16":
        return train_tfms_vgg16, eval_tfms_vgg16
    if name in ["swin_v2_t", "swin_v2_s", "swin_v2_b"]:
        return train_tfms_swin_v2, eval_tfms_swin_v2
    # Defaults for architectures that follow the standard 224×224 ImageNet recipe
    # (e.g., mobilenet_v2, densenet121, convnext_tiny)
    return train_tfms_resnet50, eval_tfms_resnet50
