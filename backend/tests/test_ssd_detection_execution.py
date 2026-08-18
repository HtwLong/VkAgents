import json
import os

import pytest
import torch
from PIL import Image

from cvmodellearning import paths
from cvmodellearning.graphrag.hyperparameter_context import (
    build_hyperparameter_context,
    validate_detection_graph_grounded_config,
)
from cvmodellearning.models.detection_models import torchvision_trainer
from cvmodellearning.models.registry import resolve_model_id
from cvmodellearning.schemas.detection_hpo import (
    DetectionConfigModel,
    expand_detection_config_for_validation,
)
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.schemas.interpretation_schema import PipelineState


RECIPE_ID = "torchvision_ssd300_vgg16_imagenet_backbone_custom_training"


def _config(**overrides):
    values = {
        "task_type": "detection",
        "classes": ["car"],
        "selected_data": [
            {"class_name": "car", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
        "train_data_ratio": 0.8,
        "val_data_ratio": 0.1,
        "test_data_ratio": 0.1,
        "num_epochs": 120,
        "patience": 0,
        "batch_size": 4,
        "input_size": 300,
        "max_size": 300,
        "aspect_ratio_range": None,
        "model_name": "ssd300",
        "model_weights": "imagenet_backbone",
        "training_recipe_id": RECIPE_ID,
        "optimizer_name": "sgd",
        "learning_rate": 0.002,
        "weight_decay": 0.0005,
        "scheduler_name": "multistep",
        "lr_milestones": [80, 110],
        "scheduler_gamma": 0.1,
        "warmup_epochs": 0.0,
        "loss_box": "smooth_l1",
        "loss_cls": "cross_entropy",
        "lambda_dfl": 0.0,
        "augmentation_policy": "ssd",
        "mosaic": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0,
        "confidence_threshold": 0.01,
        "nms_iou_threshold": 0.45,
        "max_detections": 200,
        "topk_candidates": 400,
        "positive_fraction": 0.25,
        "matching_iou_threshold": 0.5,
        "trainable_backbone_layers": 4,
        "rationale": "Grounded SSD300 VGG16 test configuration.",
    }
    values.update(overrides)
    return DetectionConfigModel.model_validate(values)


def _write_coco(path, image_name: str, image_id: int, size: int = 300):
    path.write_text(json.dumps({
        "info": {},
        "images": [{"id": image_id, "file_name": image_name, "width": size, "height": size}],
        "annotations": [{
            "id": image_id,
            "image_id": image_id,
            "category_id": 7,
            "bbox": [size // 4, size // 4, size // 2, size // 2],
            "area": (size // 2) ** 2,
            "iscrowd": 0,
        }],
        "categories": [{"id": 7, "name": "car"}],
    }), encoding="utf-8")


def test_ssd_runtime_contains_only_torchvision_fields():
    runtime = _config().runtime_config()

    assert runtime["augmentation_policy"] == "ssd"
    assert "max_size" in runtime
    assert "mosaic" not in runtime
    assert "lambda_dfl" not in runtime
    assert "warmup_momentum" not in runtime
    assert "rationale" not in runtime

    expanded = training_compatible_hpo_config(runtime)
    expanded["rationale"] = "Execution validation."
    validated = DetectionConfigModel.model_validate(
        expand_detection_config_for_validation(expanded)
    )
    assert validated.runtime_config() == runtime


def test_ssd_expansion_does_not_hide_explicit_incompatible_fields():
    runtime = training_compatible_hpo_config(_config().runtime_config())
    runtime.update({
        "rationale": "Execution validation.",
        "mosaic": 0.5,
    })

    with pytest.raises(ValueError, match="YOLO-specific augmentation fields"):
        DetectionConfigModel.model_validate(
            expand_detection_config_for_validation(runtime)
        )


def test_ssd_registry_id_retrieves_and_validates_executable_recipe():
    assert resolve_model_id("detection", "ssd300_vgg16") == "ssd300_coco"
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_model_info={"model": {"model_architecture": "ssd300_coco"}},
        selected_data=[
            {"class_name": "car", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
    )
    context = build_hyperparameter_context(state)

    assert context["selected_model_id"] == "ssd300_vgg16"
    assert context["base_recipe"]["id"] == RECIPE_ID
    assert context["reference_configuration"]["model_weights"] == "imagenet_backbone"
    assert context["reference_configuration"]["input_size"] == 300
    assert context["reference_configuration"]["max_size"] == 300
    assert context["reference_configuration"]["lr_milestones"] == [80, 110]
    assert context["reference_configuration"]["augmentation_policy"] == "ssd"
    assert context["reference_configuration"]["max_detections"] == 200

    candidate = _config(**{
        key: value
        for key, value in context["reference_configuration"].items()
        if key != "training_recipe_id"
    }).model_dump(mode="json")
    validate_detection_graph_grounded_config(candidate, context)


def test_ssd_schema_rejects_incompatible_weights_size_loss_and_augmentation():
    with pytest.raises(ValueError, match="ImageNet VGG16 backbone"):
        _config(model_weights="coco")
    with pytest.raises(ValueError, match="input_size=max_size=300"):
        _config(input_size=640, max_size=640)
    with pytest.raises(ValueError, match="Smooth L1"):
        _config(loss_box="ciou")
    with pytest.raises(ValueError, match="augmentation_policy='ssd'"):
        _config(augmentation_policy="basic")


def test_ssd_model_uses_custom_classes_fixed_transform_and_postprocessing():
    model = torchvision_trainer.get_detection_model(
        "ssd300_vgg16",
        3,
        pre_trained=False,
        input_size=300,
        max_size=300,
        confidence_threshold=0.01,
        nms_iou_threshold=0.45,
        max_detections=200,
        topk_candidates=400,
        positive_fraction=0.25,
        matching_iou_threshold=0.5,
    )
    assert model.head.classification_head.num_columns == 3
    assert model.transform.fixed_size == (300, 300)
    assert model.score_thresh == 0.01
    assert model.nms_thresh == 0.45
    assert model.detections_per_img == 200
    assert model.topk_candidates == 400
    assert model.neg_to_pos_ratio == 3.0


def test_ssd_training_augmentation_keeps_valid_box_targets(tmp_path):
    Image.new("RGB", (320, 240), "white").save(tmp_path / "image.jpg")
    annotations = tmp_path / "annotations.json"
    _write_coco(annotations, "image.jpg", 1, size=240)
    data = json.loads(annotations.read_text(encoding="utf-8"))
    data["images"][0].update({"width": 320, "height": 240})
    annotations.write_text(json.dumps(data), encoding="utf-8")

    torch.manual_seed(0)
    dataset = torchvision_trainer.DetectionCocoDataset(
        str(tmp_path),
        str(annotations),
        horizontal_flip_probability=0.5,
        augmentation_policy="ssd",
    )
    image, target = dataset[0]

    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert target["boxes"].shape[1:] == (4,)
    assert len(target["boxes"]) == len(target["labels"]) == len(target["area"])
    assert torch.all(target["boxes"][:, 2:] > target["boxes"][:, :2])
    assert torch.all(target["boxes"] >= 0)


@pytest.mark.skipif(
    os.getenv("RUN_SSD_SMOKE_TEST") != "1",
    reason="Set RUN_SSD_SMOKE_TEST=1 to run pretrained-backbone SSD train/evaluate/infer.",
)
def test_pretrained_ssd_one_epoch_evaluation_and_inference(tmp_path, monkeypatch):
    job_id = "ssd-smoke"
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(torchvision_trainer, "_choose_device", lambda: torch.device("cpu"))

    root = paths.data_dir(job_id)
    for split, image_id in (("train", 1), ("val", 2), ("test", 3)):
        image_name = f"{split}.jpg"
        Image.new("RGB", (300, 300), "white").save(root / image_name)
        _write_coco(root / f"{split}_annotations.json", image_name, image_id)

    config = _config(
        num_epochs=1,
        batch_size=1,
        workers=0,
        learning_rate=1e-5,
        lr_milestones=[1],
        trainable_backbone_layers=0,
        horizontal_flip_probability=0.0,
        max_detections=10,
        topk_candidates=20,
        amp=False,
    ).runtime_config()
    torchvision_trainer.train_torchvision_from_config(config, job_id)
    assert paths.best_model_path(job_id).exists()

    metrics = torchvision_trainer.evaluate_torchvision_model(
        "ssd300",
        2,
        job_id,
        batch_size=1,
        input_size=300,
        max_size=300,
        workers=0,
        confidence_threshold=0.01,
        nms_iou_threshold=0.45,
        max_detections=10,
        topk_candidates=20,
    )
    assert "coco/bbox_mAP" in metrics

    model, transform = torchvision_trainer.load_torchvision_model_for_inference(
        "ssd300",
        paths.best_model_path(job_id),
        2,
        torch.device("cpu"),
        input_size=300,
        max_size=300,
        confidence_threshold=0.01,
        nms_iou_threshold=0.45,
        max_detections=10,
        topk_candidates=20,
    )
    detections = torchvision_trainer.run_torchvision_inference(
        model,
        Image.new("RGB", (300, 300), "white"),
        transform,
        torch.device("cpu"),
        confidence_threshold=0.01,
    )
    assert isinstance(detections, list)

