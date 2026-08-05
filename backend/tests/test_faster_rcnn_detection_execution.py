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
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel
from cvmodellearning.schemas.interpretation_schema import PipelineState


RECIPE_ID = "torchvision_fasterrcnn_resnet50_fpn_coco_pretrained_custom_finetune"


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
        "num_epochs": 2,
        "patience": 0,
        "batch_size": 2,
        "input_size": 800,
        "max_size": 1333,
        "aspect_ratio_range": None,
        "model_name": "faster_rcnn_r50",
        "model_weights": "coco",
        "training_recipe_id": RECIPE_ID,
        "optimizer_name": "sgd",
        "scheduler_name": "multistep",
        "lr_milestones": [16, 22],
        "warmup_epochs": 0.0,
        "loss_box": "smooth_l1",
        "loss_cls": "cross_entropy",
        "lambda_dfl": 0.0,
        "mosaic": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0,
        "rationale": "Grounded Faster R-CNN test configuration.",
    }
    values.update(overrides)
    return DetectionConfigModel.model_validate(values)


def _write_coco(path, image_name: str, image_id: int):
    path.write_text(json.dumps({
        "info": {},
        "images": [{"id": image_id, "file_name": image_name, "width": 64, "height": 64}],
        "annotations": [{
            "id": image_id,
            "image_id": image_id,
            "category_id": 7,
            "bbox": [16, 16, 32, 32],
            "area": 1024,
            "iscrowd": 0,
        }],
        "categories": [{"id": 7, "name": "car"}],
    }), encoding="utf-8")


def test_faster_rcnn_registry_id_retrieves_and_validates_executable_recipe():
    assert resolve_model_id("detection", "fasterrcnn_resnet50_fpn") == (
        "faster-rcnn_r50_fpn_1x_coco"
    )
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_model_info={"model": {"model_architecture": "faster-rcnn_r50_fpn_1x_coco"}},
        selected_data=[
            {"class_name": "car", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
    )
    context = build_hyperparameter_context(state)

    assert context["selected_model_id"] == "fasterrcnn_resnet50_fpn"
    assert context["base_recipe"]["id"] == RECIPE_ID
    assert context["base_configuration"]["loss_box"] == "smooth_l1"
    assert context["base_configuration"]["loss_cls"] == "cross_entropy"
    assert context["base_configuration"]["input_size"] == 800
    assert context["base_configuration"]["max_detections"] == 100
    assert context["base_configuration"]["amp"] is False
    assert not context["applicable_rules"]

    candidate = _config(**{
        key: value
        for key, value in context["base_configuration"].items()
        if key != "training_recipe_id"
    }).model_dump(mode="json")
    validate_detection_graph_grounded_config(candidate, context)


def test_faster_rcnn_schema_rejects_incompatible_recipe_weights_and_losses():
    with pytest.raises(ValueError, match="pretrained COCO"):
        _config(model_weights="none")
    with pytest.raises(ValueError, match="Smooth L1"):
        _config(loss_box="ciou")
    with pytest.raises(ValueError, match="executable TorchVision"):
        _config(training_recipe_id="detectron2_fasterrcnn_r50_fpn_coco_pretrained_finetune")


def test_faster_rcnn_model_uses_custom_predictor_and_configured_postprocessing():
    model = torchvision_trainer.get_detection_model(
        "faster_rcnn_r50_fpn",
        3,
        pre_trained=False,
        input_size=64,
        max_size=96,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections=100,
    )
    assert type(model.roi_heads.box_predictor).__name__ == "FastRCNNPredictor"
    assert model.roi_heads.box_predictor.cls_score.out_features == 3
    assert model.transform.min_size == (64,)
    assert model.transform.max_size == 96
    assert model.roi_heads.score_thresh == 0.05
    assert model.roi_heads.nms_thresh == 0.5
    assert model.roi_heads.detections_per_img == 100


@pytest.mark.skipif(
    os.getenv("RUN_FASTER_RCNN_SMOKE_TEST") != "1",
    reason="Set RUN_FASTER_RCNN_SMOKE_TEST=1 to run pretrained Faster R-CNN train/evaluate/infer.",
)
def test_pretrained_faster_rcnn_one_epoch_evaluation_and_inference(tmp_path, monkeypatch):
    job_id = "faster-rcnn-smoke"
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(torchvision_trainer, "_choose_device", lambda: torch.device("cpu"))

    root = paths.data_dir(job_id)
    for split, image_id in (("train", 1), ("val", 2), ("test", 3)):
        image_name = f"{split}.jpg"
        Image.new("RGB", (64, 64), "white").save(root / image_name)
        _write_coco(root / f"{split}_annotations.json", image_name, image_id)

    config = _config(
        num_epochs=1,
        batch_size=1,
        input_size=64,
        max_size=64,
        workers=0,
        learning_rate=1e-5,
        lr_milestones=[1],
        trainable_backbone_layers=0,
        horizontal_flip_probability=0.0,
        max_detections=10,
        amp=False,
    ).runtime_config()
    torchvision_trainer.train_torchvision_from_config(config, job_id)
    assert paths.best_model_path(job_id).exists()

    metrics = torchvision_trainer.evaluate_torchvision_model(
        "faster_rcnn_r50",
        2,
        job_id,
        batch_size=1,
        input_size=64,
        max_size=64,
        workers=0,
        max_detections=10,
    )
    assert "coco/bbox_mAP" in metrics

    model, transform = torchvision_trainer.load_torchvision_model_for_inference(
        "faster_rcnn_r50",
        paths.best_model_path(job_id),
        2,
        torch.device("cpu"),
        input_size=64,
        max_size=64,
        max_detections=10,
    )
    detections = torchvision_trainer.run_torchvision_inference(
        model,
        Image.new("RGB", (64, 64), "white"),
        transform,
        torch.device("cpu"),
    )
    assert isinstance(detections, list)
