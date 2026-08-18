from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionCapabilities:
    architecture_family: str
    supported_weights: tuple[str, ...]
    supported_training_modes: tuple[str, ...]
    lora_supported: bool


def detection_capabilities(model_name: str) -> DetectionCapabilities:
    """Return facts imposed by the registered detection executors."""
    if model_name.startswith(("yolov8_", "yolov10_", "yolov11_", "yolov12_")):
        return DetectionCapabilities("yolo", ("coco", "default"), ("full_finetune",), False)
    if model_name in {"retinanet_r50", "faster_rcnn_r50"}:
        return DetectionCapabilities("torchvision", ("coco", "default"), ("full_finetune",), False)
    if model_name == "ssd300":
        return DetectionCapabilities("torchvision", ("imagenet_backbone",), ("full_finetune",), False)
    if model_name == "rtdetr_hgnetv2_l":
        return DetectionCapabilities("rtdetr", ("coco", "default"), ("full_finetune", "lora"), True)
    raise ValueError(f"No execution capabilities registered for {model_name}.")


def detection_prompt_constraints(model_name: str) -> dict[str, Any]:
    """Serialize model capabilities for HPO reasoning without choosing tunable values."""
    return {"model_name": model_name, **asdict(detection_capabilities(model_name))}

