from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass(frozen=True)
class ClassificationTransformProfile:
    native_crop_size: int
    native_resize_size: int
    interpolation: InterpolationMode
    configurable_image_size: bool = True


@dataclass(frozen=True)
class ClassificationWeightsMetadata:
    """Minimal weight metadata for non-TorchVision classification backbones."""

    crop_size: int
    resize_size: int
    interpolation: InterpolationMode
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def transforms(self):
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(self.resize_size, interpolation=self.interpolation, antialias=True),
            v2.CenterCrop(self.crop_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
        ])
        # Match the small public metadata surface consumed from TorchVision presets.
        transform.crop_size = [self.crop_size]
        transform.resize_size = [self.resize_size]
        transform.interpolation = self.interpolation
        transform.mean = list(self.mean)
        transform.std = list(self.std)
        return transform


# These profiles mirror the DEFAULT weight metadata in the TorchVision version
# installed by this project. When a concrete weight enum is available, its
# transform metadata remains authoritative over these no-weight fallbacks.
CLASSIFICATION_TRANSFORM_PROFILES = {
    "resnet50": ClassificationTransformProfile(224, 232, InterpolationMode.BILINEAR),
    "mobilenet_v2": ClassificationTransformProfile(224, 232, InterpolationMode.BILINEAR),
    "mobilenet_v3_large": ClassificationTransformProfile(224, 232, InterpolationMode.BILINEAR),
    "mobilenet_v3_small": ClassificationTransformProfile(224, 256, InterpolationMode.BILINEAR),
    "efficientnet_b0": ClassificationTransformProfile(224, 256, InterpolationMode.BICUBIC),
    "efficientnet_b1": ClassificationTransformProfile(240, 255, InterpolationMode.BILINEAR),
    "efficientnet_b2": ClassificationTransformProfile(288, 288, InterpolationMode.BICUBIC),
    "efficientnet_b3": ClassificationTransformProfile(300, 320, InterpolationMode.BICUBIC),
    "efficientnet_b4": ClassificationTransformProfile(380, 384, InterpolationMode.BICUBIC),
    "efficientnet_b5": ClassificationTransformProfile(456, 456, InterpolationMode.BICUBIC),
    "efficientnet_b6": ClassificationTransformProfile(528, 528, InterpolationMode.BICUBIC),
    "efficientnet_b7": ClassificationTransformProfile(600, 600, InterpolationMode.BICUBIC),
    "densenet121": ClassificationTransformProfile(224, 256, InterpolationMode.BILINEAR),
    "convnext_tiny": ClassificationTransformProfile(224, 236, InterpolationMode.BILINEAR),
    # OpenAI CLIP ViT-B/16 uses a fixed 224px input. The 248px evaluation
    # resize comes from timm's pretrained configuration (crop_pct=0.9).
    "clip_vit_b16": ClassificationTransformProfile(224, 248, InterpolationMode.BICUBIC, False),
    # The executable DINOv2 transfer-learning path uses the documented 224px
    # processor setup. timm resizes the 518px checkpoint positional embedding
    # when the model is constructed with img_size=224.
    "dinov2_vits14": ClassificationTransformProfile(224, 256, InterpolationMode.BICUBIC, False),
    "dinov2_vitb14": ClassificationTransformProfile(224, 256, InterpolationMode.BICUBIC, False),
    # The registered TorchVision ViT constructor and DEFAULT weights are fixed
    # at 224. Supporting another size requires model-construction and positional
    # embedding changes, not merely a different crop.
    "vit_b_16": ClassificationTransformProfile(224, 256, InterpolationMode.BILINEAR, False),
    "swin_v2_t": ClassificationTransformProfile(256, 260, InterpolationMode.BICUBIC),
    "swin_v2_s": ClassificationTransformProfile(256, 260, InterpolationMode.BICUBIC),
}


def _profile(name: str) -> ClassificationTransformProfile:
    try:
        return CLASSIFICATION_TRANSFORM_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported classification transform profile: {name}") from exc


def _weight_preprocessing(weights, profile: ClassificationTransformProfile):
    if weights is None:
        return (
            profile.native_crop_size,
            profile.native_resize_size,
            profile.interpolation,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

    preset = weights.transforms()
    crop_size = int(preset.crop_size[0])
    resize_size = int(preset.resize_size[0])
    return (
        crop_size,
        resize_size,
        preset.interpolation,
        list(preset.mean),
        list(preset.std),
    )


def select_transforms(
    name: str,
    image_size: int | None = None,
    *,
    weights=None,
    auto_augment_policy: str = "none",
    random_erasing: float = 0.0,
    random_resized_crop_scale_min: float = 0.6,
    horizontal_flip_probability: float = 0.5,
):
    """Build model-aware stochastic training and deterministic evaluation transforms.

    Training preprocessing inherits interpolation and normalization from the
    selected pretrained weights, while HPO controls crop size and regularizers.
    Evaluation uses the same metadata and configured image size.
    """
    profile = _profile(name)
    native_crop, native_resize, interpolation, mean, std = _weight_preprocessing(weights, profile)
    configured_size = int(image_size or native_crop)

    if not profile.configurable_image_size and configured_size != native_crop:
        raise ValueError(
            f"{name} supports only image_size={native_crop} with the registered constructor and weights."
        )
    if not 0.0 < random_resized_crop_scale_min <= 1.0:
        raise ValueError("random_resized_crop_scale_min must be in (0, 1].")
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must be in [0, 1].")
    if auto_augment_policy not in {"none", "ta_wide"}:
        raise ValueError(f"Unsupported auto_augment_policy: {auto_augment_policy}")
    if not 0.0 <= random_erasing <= 1.0:
        raise ValueError("random_erasing must be in [0, 1].")

    train_steps = [
        v2.RandomResizedCrop(
            configured_size,
            scale=(random_resized_crop_scale_min, 1.0),
            interpolation=interpolation,
            antialias=True,
        ),
    ]
    if horizontal_flip_probability > 0:
        train_steps.append(v2.RandomHorizontalFlip(p=horizontal_flip_probability))
    if auto_augment_policy == "ta_wide":
        train_steps.append(v2.TrivialAugmentWide(interpolation=interpolation))
    train_steps.extend([
        # Keep geometric/automatic augmentation on PIL because TorchVision's
        # tensor affine kernels do not support bicubic for every TA-Wide op.
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    if random_erasing > 0:
        train_steps.append(v2.RandomErasing(p=random_erasing))

    resize_size = max(configured_size, round(configured_size * native_resize / native_crop))
    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(resize_size, interpolation=interpolation, antialias=True),
        v2.CenterCrop(configured_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    return v2.Compose(train_steps), eval_transform


def select_evaluation_transform(
    name: str,
    *,
    image_size: int | None = None,
    weights=None,
):
    """Resolve identical deterministic preprocessing for validation/test/inference."""
    profile = _profile(name)
    native_crop, _, _, _, _ = _weight_preprocessing(weights, profile)
    configured_size = int(image_size or native_crop)

    # At the checkpoint's native resolution the official preset is exact and
    # remains the single source of truth for resize, interpolation and scaling.
    if weights is not None and configured_size == native_crop:
        return weights.transforms()

    _, eval_transform = select_transforms(
        name,
        image_size=configured_size,
        weights=weights,
    )
    return eval_transform
