from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Tuple, Union

import numpy as np
import torch
from agents import function_tool
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import tv_tensors
from torchvision.models import VGG16_Weights
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

from cvmodellearning.paths import (
    best_model_path,
    data_dir,
    metrics_json_path,
    run_dir,
    test_json_path,
    tool_call_args_path,
    train_json_path,
    training_log_path,
    val_json_path,
)
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.jobs.run_control import raise_if_cancelled
from cvmodellearning.evaluation.detection_result import dataset_statistics


TVModel = Literal[
    "retinanet_r50_fpn",
    "faster_rcnn_r50_fpn",
    "ssd300_vgg16",
]
MonitorMetric = Literal["coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75"]

TV_MODEL_NAME_MAP: Dict[str, TVModel] = {
    "retinanet_r50": "retinanet_r50_fpn",
    "retinanet": "retinanet_r50_fpn",
    "retinanet_resnet50_fpn": "retinanet_r50_fpn",
    "retinanet_r50_fpn_1x_coco": "retinanet_r50_fpn",
    "faster_rcnn_r50": "faster_rcnn_r50_fpn",
    "faster-rcnn_r50_fpn_1x_coco": "faster_rcnn_r50_fpn",
    "fasterrcnn_resnet50_fpn": "faster_rcnn_r50_fpn",
    "ssd300": "ssd300_vgg16",
    "ssd": "ssd300_vgg16",
    "ssd300_coco": "ssd300_vgg16",
    "ssd300_vgg16": "ssd300_vgg16",
}


def _adapt_retinanet_classification_head(model: nn.Module, num_classes: int) -> None:
    """Replace only the COCO logits while preserving the pretrained conv tower."""
    head = model.head.classification_head
    num_anchors = head.num_anchors
    old_logits = head.cls_logits
    head.num_classes = num_classes
    head.cls_logits = nn.Conv2d(
        old_logits.in_channels,
        num_anchors * num_classes,
        old_logits.kernel_size,
        old_logits.stride,
        old_logits.padding,
    )
    nn.init.normal_(head.cls_logits.weight, std=0.01)
    nn.init.constant_(head.cls_logits.bias, -math.log((1 - 0.01) / 0.01))


def get_detection_model(
    model_name: TVModel,
    num_classes: int,
    *,
    pre_trained: bool = True,
    pretrained_backbone_only: bool = False,
    input_size: int = 800,
    max_size: int = 1333,
    trainable_backbone_layers: int = 3,
    confidence_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 300,
    topk_candidates: int = 400,
    positive_fraction: float = 0.25,
    matching_iou_threshold: float = 0.5,
) -> nn.Module:
    """Build a TorchVision detector with one consistent custom-class head."""
    common = {
        "min_size": input_size,
        "max_size": max_size,
    }
    if pre_trained:
        common["trainable_backbone_layers"] = trainable_backbone_layers
    weights = "DEFAULT" if pre_trained else None
    weights_backbone = None if not pre_trained else "DEFAULT"

    if model_name == "ssd300_vgg16":
        if input_size != 300 or max_size != 300:
            raise ValueError("SSD300 VGG16 requires input_size=max_size=300.")
        return ssd300_vgg16(
            weights=None,
            weights_backbone=(
                VGG16_Weights.IMAGENET1K_FEATURES if pretrained_backbone_only else None
            ),
            num_classes=num_classes,
            trainable_backbone_layers=(
                trainable_backbone_layers if pretrained_backbone_only else None
            ),
            score_thresh=confidence_threshold,
            nms_thresh=nms_iou_threshold,
            detections_per_img=max_detections,
            topk_candidates=topk_candidates,
            positive_fraction=positive_fraction,
            iou_thresh=matching_iou_threshold,
        )

    if model_name == "retinanet_r50_fpn":
        model = retinanet_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=None if pre_trained else num_classes,
            score_thresh=confidence_threshold,
            nms_thresh=nms_iou_threshold,
            detections_per_img=max_detections,
            **common,
        )
        if pre_trained:
            _adapt_retinanet_classification_head(model, num_classes)
        return model

    if model_name == "faster_rcnn_r50_fpn":
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            box_score_thresh=confidence_threshold,
            box_nms_thresh=nms_iou_threshold,
            box_detections_per_img=max_detections,
            **common,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model name: {model_name}")


def _ssd_target_fields(inputs):
    target = inputs[1]
    return target["labels"], target["iscrowd"]


def _ssd_training_transform(horizontal_flip_probability: float) -> T.Compose:
    """TorchVision's reference SSD augmentation policy for box targets."""
    return T.Compose([
        T.RandomPhotometricDistort(),
        T.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        T.RandomIoUCrop(),
        T.RandomHorizontalFlip(horizontal_flip_probability),
        T.SanitizeBoundingBoxes(labels_getter=_ssd_target_fields),
    ])


class DetectionCocoDataset(CocoDetection):
    """Convert COCO annotations into the tensor contract expected by detectors."""

    def __init__(
        self,
        root: str,
        ann_file: str,
        *,
        horizontal_flip_probability: float = 0.0,
        augmentation_policy: str = "basic",
    ):
        super().__init__(root=root, annFile=ann_file)
        self.horizontal_flip_probability = horizontal_flip_probability
        self.augmentation_policy = augmentation_policy
        self.training_transform = (
            _ssd_training_transform(horizontal_flip_probability)
            if augmentation_policy == "ssd"
            else None
        )
        category_ids = sorted(self.coco.getCatIds())
        self.category_to_label = {category_id: index + 1 for index, category_id in enumerate(category_ids)}
        self.label_to_category = {label: category for category, label in self.category_to_label.items()}

    def _load_image(self, image_id: int) -> Image.Image:
        image_record = self.coco.loadImgs(image_id)[0]
        relative_path = image_record.get("image_path") or image_record["file_name"]
        return Image.open(Path(self.root) / relative_path).convert("RGB")

    def __getitem__(self, index: int):
        image, annotations = super().__getitem__(index)
        width, height = image.size
        boxes: list[list[float]] = []
        labels: list[int] = []
        crowds: list[int] = []

        for annotation in annotations:
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2 = min(float(width), x + box_width)
            y2 = min(float(height), y + box_height)
            category_id = int(annotation["category_id"])
            if x2 <= x1 or y2 <= y1 or category_id not in self.category_to_label:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_to_label[category_id])
            crowds.append(int(annotation.get("iscrowd", 0)))

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        image_tensor = T.functional.to_image(image)
        crowds_tensor = torch.as_tensor(crowds, dtype=torch.int64)
        if self.training_transform is not None:
            transform_target = {
                "boxes": tv_tensors.BoundingBoxes(
                    boxes_tensor,
                    format="XYXY",
                    canvas_size=(height, width),
                ),
                "labels": labels_tensor,
                "iscrowd": crowds_tensor,
            }
            image_tensor, transform_target = self.training_transform(image_tensor, transform_target)
            boxes_tensor = torch.as_tensor(transform_target["boxes"], dtype=torch.float32)
            labels_tensor = transform_target["labels"]
            crowds_tensor = transform_target["iscrowd"]
        elif self.horizontal_flip_probability and torch.rand(()) < self.horizontal_flip_probability:
            image_tensor = T.functional.horizontal_flip(image_tensor)
            if boxes_tensor.numel():
                old_x1 = boxes_tensor[:, 0].clone()
                old_x2 = boxes_tensor[:, 2].clone()
                boxes_tensor[:, 0] = width - old_x2
                boxes_tensor[:, 2] = width - old_x1

        image_tensor = T.functional.to_dtype(image_tensor, torch.float32, scale=True)
        areas_tensor = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        )

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(int(self.ids[index]), dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": crowds_tensor,
        }
        return image_tensor, target


def _collate_detection_batch(batch):
    return tuple(zip(*batch))


def _data_loader(
    job_id: str,
    annotation_path: Path,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    horizontal_flip_probability: float = 0.0,
    augmentation_policy: str = "basic",
) -> tuple[DataLoader, DetectionCocoDataset]:
    dataset = DetectionCocoDataset(
        str(data_dir(job_id)),
        str(annotation_path),
        horizontal_flip_probability=horizontal_flip_probability,
        augmentation_policy=augmentation_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=_collate_detection_batch,
    )
    return loader, dataset


def get_data_loaders(
    job_id: str,
    batch_size: int,
    workers: int = 0,
    horizontal_flip_probability: float = 0.5,
    augmentation_policy: str = "basic",
) -> tuple[DataLoader, DataLoader, COCO, dict[int, int]]:
    train_loader, _ = _data_loader(
        job_id,
        train_json_path(job_id),
        batch_size,
        workers,
        shuffle=True,
        horizontal_flip_probability=horizontal_flip_probability,
        augmentation_policy=augmentation_policy,
    )
    val_loader, val_dataset = _data_loader(
        job_id, val_json_path(job_id), batch_size, workers, shuffle=False
    )
    return train_loader, val_loader, val_dataset.coco, val_dataset.label_to_category


def evaluate_coco_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    coco_gt: COCO,
    device: torch.device,
    label_to_category: Mapping[int, int],
) -> Dict[str, float]:
    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                boxes = output["boxes"].detach().cpu().clone()
                boxes[:, 2:] -= boxes[:, :2]
                for box, score, label in zip(boxes, output["scores"].cpu(), output["labels"].cpu()):
                    category_id = label_to_category.get(int(label))
                    if category_id is not None:
                        results.append({
                            "image_id": int(target["image_id"]),
                            "category_id": category_id,
                            "bbox": box.tolist(),
                            "score": float(score),
                        })

    if not results:
        return {"coco/bbox_mAP": 0.0, "coco/bbox_mAP_50": 0.0, "coco/bbox_mAP_75": 0.0,
                "per_class": [], "size_metrics": {}}

    coco_eval = COCOeval(coco_gt, coco_gt.loadRes(results), "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    summary: Dict[str, Any] = {
        "coco/bbox_mAP": np.float64(coco_eval.stats[0]).item(),
        "coco/bbox_mAP_50": np.float64(coco_eval.stats[1]).item(),
        "coco/bbox_mAP_75": np.float64(coco_eval.stats[2]).item(),
        "size_metrics": {
            "ap_small": float(coco_eval.stats[3]),
            "ap_medium": float(coco_eval.stats[4]),
            "ap_large": float(coco_eval.stats[5]),
            "ar_small": float(coco_eval.stats[9]),
            "ar_medium": float(coco_eval.stats[10]),
            "ar_large": float(coco_eval.stats[11]),
        },
    }
    category_to_label = {category: label for label, category in label_to_category.items()}
    categories = {int(row["id"]): str(row["name"]) for row in coco_gt.dataset.get("categories", [])}
    per_class = []
    for category_id in coco_eval.params.catIds:
        precision = coco_eval.eval["precision"][:, :, coco_eval.params.catIds.index(category_id), 0, -1]
        valid = precision[precision > -1]
        ap = float(valid.mean()) if valid.size else 0.0
        ap50_values = precision[0]
        ap50_valid = ap50_values[ap50_values > -1]
        per_class.append({
            "class_name": categories.get(category_id, str(category_to_label.get(category_id, category_id))),
            "ap": ap,
            "ap50": float(ap50_valid.mean()) if ap50_valid.size else 0.0,
        })
    summary["per_class"] = per_class
    return summary


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _optimizer(name: str, parameters, config: Mapping[str, Any]):
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay", 0.0001))
    if name in {"auto", "sgd"}:
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(float(config.get("beta1", 0.9)), 0.999),
            weight_decay=weight_decay,
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported TorchVision optimizer: {name}")


def flexible_torchvision_training(config: Mapping[str, Any], job_id: str) -> None:
    """Train one saved, schema-validated TorchVision detection configuration."""
    torch.manual_seed(int(config.get("seed", 0)))
    model_name = TV_MODEL_NAME_MAP[str(config["model_name"])]
    num_classes = len(config["classes"]) + 1
    device = _choose_device()
    model = get_detection_model(
        model_name,
        num_classes,
        pre_trained=config.get("model_weights") in {"default", "coco"},
        pretrained_backbone_only=config.get("model_weights") == "imagenet_backbone",
        input_size=int(config.get("input_size", 800)),
        max_size=int(config.get("max_size", 1333)),
        trainable_backbone_layers=int(config.get("trainable_backbone_layers", 3)),
        confidence_threshold=float(config.get("confidence_threshold", 0.05)),
        nms_iou_threshold=float(config.get("nms_iou_threshold", 0.5)),
        max_detections=int(config.get("max_detections", 300)),
        topk_candidates=int(config.get("topk_candidates", 400)),
        positive_fraction=float(config.get("positive_fraction", 0.25)),
        matching_iou_threshold=float(config.get("matching_iou_threshold", 0.5)),
    ).to(device)
    train_loader, val_loader, coco_gt, label_to_category = get_data_loaders(
        job_id,
        int(config["batch_size"]),
        int(config.get("workers", 0)),
        float(config.get("horizontal_flip_probability", 0.5)),
        str(config.get("augmentation_policy", "basic")),
    )
    if not train_loader:
        raise ValueError("The training split is empty.")

    optimizer = _optimizer(
        str(config.get("optimizer_name", "sgd")),
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        config,
    )
    scheduler = None
    if config.get("scheduler_name", "multistep") == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(config.get("lr_milestones", [16, 22])),
            gamma=float(config.get("scheduler_gamma", 0.1)),
        )

    metric_names = {
        "val_mAP": "coco/bbox_mAP",
        "val_mAP_50": "coco/bbox_mAP_50",
        "val_mAP_75": "coco/bbox_mAP_75",
    }
    monitor_metric = metric_names.get(str(config.get("track_metric")), "coco/bbox_mAP")
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_metric = -math.inf
    stale_epochs = 0
    start_time = time.time()
    checkpoint = best_model_path(job_id)
    log_path = training_log_path(job_id)

    with log_path.open("w", encoding="utf-8") as log_file:
        configured_epochs = int(config["num_epochs"])
        max_epochs = int(config.get("_benchmark_max_epochs", configured_epochs))
        run_epochs = min(configured_epochs, max_epochs)
        max_batches = config.get("_benchmark_max_batches")
        for epoch in range(run_epochs):
            raise_if_cancelled(job_id)
            model.train()
            total_loss = 0.0
            completed_batches = 0
            for images, targets in train_loader:
                raise_if_cancelled(job_id)
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    losses = sum(model(images, targets).values())
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(losses.detach())
                completed_batches += 1
                if max_batches is not None and completed_batches >= int(max_batches):
                    break
            if scheduler is not None:
                scheduler.step()

            metrics = evaluate_coco_metrics(model, val_loader, coco_gt, device, label_to_category)
            current_metric = metrics[monitor_metric]
            average_loss = total_loss / completed_batches
            message = (
                f"Epoch {epoch + 1}/{run_epochs} | Loss: {average_loss:.4f} | "
                f"{monitor_metric}: {current_metric:.4f} | Time: {time.time() - start_time:.2f}s"
            )
            print(message)
            log_file.write(message + "\n")
            (run_dir(job_id) / "progress.json").write_text(json.dumps({
                "status": "running",
                "current_epoch": epoch + 1,
                "total_epochs": run_epochs,
                "train_loss": average_loss,
                "val_mAP": metrics["coco/bbox_mAP"],
                "val_mAP50": metrics["coco/bbox_mAP_50"],
            }, indent=2), encoding="utf-8")

            if current_metric > best_metric:
                best_metric = current_metric
                stale_epochs = 0
                temporary_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
                torch.save(model.state_dict(), temporary_checkpoint)
                temporary_checkpoint.replace(checkpoint)
            else:
                stale_epochs += 1
            patience = int(config.get("patience", 0))
            if patience and stale_epochs >= patience:
                break


def train_torchvision_from_config(config: Mapping[str, Any], job_id: str) -> str:
    """Deterministic entry point used by the execution pipeline."""
    config = training_compatible_hpo_config(config)
    audit_path = tool_call_args_path(job_id)
    audit_path.write_text(json.dumps(dict(config), indent=2, default=str), encoding="utf-8")
    flexible_torchvision_training(config, job_id)
    return f"Successfully trained {config['model_name']}; checkpoint saved to {best_model_path(job_id)}."


@function_tool(strict_mode=True)
def train_torchvision_model(
    model_name: str,
    num_classes: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    monitor_metric: MonitorMetric,
    patience: int,
    save_best_model: bool,
    job_id: str,
    config_override_json: str = "{}",
) -> str:
    """Compatibility wrapper for older agent callers."""
    del save_best_model
    try:
        overrides = json.loads(config_override_json)
        classes = overrides.pop(
            "classes",
            [f"class_{index}" for index in range(1, max(1, num_classes))],
        )
        config = {
            "model_name": model_name,
            "classes": classes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": epochs,
            "patience": patience,
            "track_metric": {
                "coco/bbox_mAP": "val_mAP",
                "coco/bbox_mAP_50": "val_mAP_50",
                "coco/bbox_mAP_75": "val_mAP_75",
            }[monitor_metric],
            "model_weights": (
                "imagenet_backbone" if "ssd" in model_name.lower() else "coco"
            ),
            **overrides,
        }
        return train_torchvision_from_config(config, job_id)
    except Exception as exc:
        return f"Training failed: {exc}"


def evaluate_torchvision_model(
    model_name: str,
    num_classes: int,
    job_id: str,
    batch_size: int = 1,
    *,
    input_size: int = 800,
    max_size: int = 1333,
    workers: int = 0,
    confidence_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 300,
    topk_candidates: int = 400,
    positive_fraction: float = 0.25,
    matching_iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    if model_name not in TV_MODEL_NAME_MAP:
        return {"error": f"Unknown model name: {model_name}"}
    checkpoint = best_model_path(job_id)
    if not checkpoint.exists():
        return {"error": f"Model checkpoint not found at: {checkpoint}"}

    device = _choose_device()
    model = get_detection_model(
        TV_MODEL_NAME_MAP[model_name],
        num_classes,
        pre_trained=False,
        input_size=input_size,
        max_size=max_size,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        topk_candidates=topk_candidates,
        positive_fraction=positive_fraction,
        matching_iou_threshold=matching_iou_threshold,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    test_loader, test_dataset = _data_loader(
        job_id, test_json_path(job_id), batch_size, workers, shuffle=False
    )
    metrics = evaluate_coco_metrics(
        model, test_loader, test_dataset.coco, device, test_dataset.label_to_category
    )
    classes = [
        name for _, name in sorted(
            ((int(row["id"]), str(row["name"])) for row in test_dataset.coco.dataset.get("categories", []))
        )
    ]
    statistics = dataset_statistics(job_id, classes)
    metrics["dataset_statistics"] = statistics
    support = {row["class_name"]: row["instances"] for row in statistics.get("per_class", [])}
    for row in metrics.get("per_class", []):
        row["support"] = support.get(row["class_name"], 0)
    metrics_json_path(job_id).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_torchvision_model_for_inference(
    model_name: str,
    model_path: Path,
    num_classes: int,
    device: torch.device,
    *,
    input_size: int = 800,
    max_size: int = 1333,
    confidence_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 300,
    topk_candidates: int = 400,
    positive_fraction: float = 0.25,
    matching_iou_threshold: float = 0.5,
) -> Tuple[nn.Module, T.Compose]:
    if model_name not in TV_MODEL_NAME_MAP:
        raise ValueError(f"Unsupported torchvision model for loading: {model_name}")
    model = get_detection_model(
        TV_MODEL_NAME_MAP[model_name],
        num_classes,
        pre_trained=False,
        input_size=input_size,
        max_size=max_size,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        topk_candidates=topk_candidates,
        positive_fraction=positive_fraction,
        matching_iou_threshold=matching_iou_threshold,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


def run_torchvision_inference(
    model: nn.Module,
    image: Image.Image,
    transform: T.Compose,
    device: torch.device,
    *,
    confidence_threshold: float = 0.05,
) -> List[List[Union[float, int]]]:
    with torch.no_grad():
        output = model([transform(image).to(device)])[0]
    return [
        [*box, score, float(int(label) - 1)]
        for box, score, label in zip(
            output["boxes"].cpu().tolist(),
            output["scores"].cpu().tolist(),
            output["labels"].cpu().tolist(),
        )
        if score >= confidence_threshold and int(label) > 0
    ]
