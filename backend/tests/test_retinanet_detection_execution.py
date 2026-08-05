import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from cvmodellearning import paths
from cvmodellearning.graphrag.hyperparameter_context import (
    build_hyperparameter_context,
    validate_detection_graph_grounded_config,
)
from cvmodellearning.models.detection_models import torchvision_trainer
from cvmodellearning.pipelines.detection_pipe import DetectionPipeline
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel
from cvmodellearning.schemas.interpretation_schema import PipelineState
import routers.execution as execution


RECIPE_ID = "torchvision_retinanet_resnet50_fpn_coco_pretrained_custom_finetune"


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
        "patience": 1,
        "batch_size": 2,
        "input_size": 800,
        "max_size": 1333,
        "aspect_ratio_range": None,
        "model_name": "retinanet_r50",
        "model_weights": "coco",
        "training_recipe_id": RECIPE_ID,
        "optimizer_name": "sgd",
        "scheduler_name": "multistep",
        "lr_milestones": [16, 22],
        "warmup_epochs": 0.0,
        "loss_box": "l1",
        "loss_cls": "focal",
        "lambda_dfl": 0.0,
        "mosaic": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "close_mosaic": 0,
        "rationale": "Grounded RetinaNet test configuration.",
    }
    values.update(overrides)
    return DetectionConfigModel.model_validate(values)


def _write_coco(
    path: Path,
    image_name: str,
    image_id: int,
    *,
    width: int = 64,
    height: int = 64,
    image_path: str | None = None,
):
    image_record = {"id": image_id, "file_name": image_name, "width": width, "height": height}
    if image_path:
        image_record["image_path"] = image_path
    path.write_text(json.dumps({
        "info": {},
        "images": [image_record],
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


def test_retinanet_registry_id_retrieves_and_validates_executable_recipe():
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_model_info={"model": {"model_architecture": "retinanet_r50_fpn_1x_coco"}},
        selected_data=[
            {"class_name": "car", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
    )
    context = build_hyperparameter_context(state)

    assert context["selected_model_id"] == "retinanet_resnet50_fpn"
    assert context["base_recipe"]["id"] == RECIPE_ID
    assert context["base_configuration"]["loss_cls"] == "focal"
    assert context["base_configuration"]["lr_milestones"] == [16, 22]
    assert context["base_configuration"]["amp"] is False
    assert not context["applicable_rules"]

    candidate = _config(**{
        key: value
        for key, value in context["base_configuration"].items()
        if key not in {"training_recipe_id"}
    }).model_dump(mode="json")
    validate_detection_graph_grounded_config(candidate, context)


def test_detection_execution_revalidates_recipe_and_omits_inactive_fields():
    config = _config(
        freeze=4,
        single_cls=True,
        rect=True,
        lambda_box=9.0,
        lambda_cls=4.0,
        lambda_giou=2.0,
    )

    validated = execution._validate_config(
        DetectionPipeline(),
        config.runtime_config(),
    )

    assert {
        "freeze",
        "single_cls",
        "rect",
        "lambda_box",
        "lambda_cls",
        "lambda_giou",
    }.isdisjoint(validated)


def test_retinanet_schema_rejects_incompatible_loss_or_uninitialized_weights():
    with pytest.raises(ValueError, match="pretrained COCO"):
        _config(model_weights="none")
    with pytest.raises(ValueError, match="sigmoid focal"):
        _config(loss_cls="bce")


def test_coco_dataset_builds_contiguous_labels_and_target_aware_flip(tmp_path):
    (tmp_path / "demo").mkdir()
    Image.new("RGB", (100, 50), "white").save(tmp_path / "demo" / "image.jpg")
    annotations = tmp_path / "annotations.json"
    _write_coco(
        annotations,
        "image.jpg",
        1,
        width=100,
        height=50,
        image_path="demo/image.jpg",
    )

    dataset = torchvision_trainer.DetectionCocoDataset(
        str(tmp_path), str(annotations), horizontal_flip_probability=1.0
    )
    image, target = dataset[0]

    assert image.dtype == torch.float32
    assert image.shape == (3, 50, 100)
    assert target["labels"].tolist() == [1]
    assert target["boxes"].tolist() == [[52.0, 16.0, 84.0, 48.0]]
    assert dataset.label_to_category == {1: 7}


def test_retinanet_model_has_a_real_classification_head_and_configured_resize():
    model = torchvision_trainer.get_detection_model(
        "retinanet_r50_fpn",
        3,
        pre_trained=False,
        input_size=64,
        max_size=96,
    )
    assert type(model.head.classification_head).__name__ == "RetinaNetClassificationHead"
    assert model.head.classification_head.num_classes == 3
    assert model.transform.min_size == (64,)
    assert model.transform.max_size == 96


def test_deterministic_retinanet_adapter_forwards_saved_config(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        torchvision_trainer,
        "flexible_torchvision_training",
        lambda config, job_id: captured.update(config=config, job_id=job_id),
    )
    monkeypatch.setattr(torchvision_trainer, "tool_call_args_path", lambda job_id: tmp_path / "args.json")
    monkeypatch.setattr(torchvision_trainer, "best_model_path", lambda job_id: tmp_path / "best.pt")

    result = torchvision_trainer.train_torchvision_from_config(_config().runtime_config(), "job")

    assert result.startswith("Successfully trained")
    assert captured["job_id"] == "job"
    assert captured["config"]["model_name"] == "retinanet_r50"
    assert captured["config"]["scheduler_name"] == "multistep"


@pytest.mark.skipif(
    os.getenv("RUN_RETINANET_SMOKE_TEST") != "1",
    reason="Set RUN_RETINANET_SMOKE_TEST=1 to run pretrained RetinaNet train/evaluate/infer.",
)
def test_pretrained_retinanet_one_epoch_evaluation_and_inference(tmp_path, monkeypatch):
    job_id = "retinanet-smoke"
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(torchvision_trainer, "_choose_device", lambda: torch.device("cpu"))

    root = paths.data_dir(job_id)
    for split, image_id in (("train", 1), ("val", 2), ("test", 3)):
        image_name = f"{split}.jpg"
        Image.new("RGB", (64, 64), "white").save(root / image_name)
        _write_coco(root / f"{split}_annotations.json", image_name, image_id)

    config = _config(
        num_epochs=1,
        patience=0,
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
        "retinanet_r50",
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
        "retinanet_r50",
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
