import csv
import json
import os
from pathlib import Path

import pytest
import yaml
from PIL import Image

from cvmodellearning.graphrag.hyperparameter_context import (
    build_hyperparameter_context,
    llm_controlled_fields,
    validate_detection_graph_grounded_config,
)
from cvmodellearning.download.visionkg_utils import visionkg2cocoDet
from cvmodellearning.models.detection_models import yolo_trainer
from cvmodellearning import paths
from cvmodellearning.models.registry import DetectionHpoModelId, MODEL_REGISTRY
from cvmodellearning.pipelines import detection_pipe
from cvmodellearning.schemas.detection_hpo import (
    DetectionConfigDraft,
    DetectionConfigModel,
)
from cvmodellearning.schemas.detection_hpo_completion import complete_detection_config
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.hardware_profiles import get_training_hardware_profile


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
        "input_size": 640,
        "model_name": "yolov8_n",
        "training_recipe_id": "ultralytics_yolo_detection_finetune_balanced",
        "rationale": "Grounded test configuration.",
    }
    values.update(overrides)
    return DetectionConfigModel.model_validate(values)


def test_all_twenty_yolo_hpo_variants_map_to_an_executable_checkpoint():
    model_ids = [
        item.value
        for item in DetectionHpoModelId
        if item.value.startswith(("yolov8_", "yolov10_", "yolov11_", "yolov12_"))
    ]

    assert len(model_ids) == 20
    checkpoints = set()
    for model_id in model_ids:
        version, size = yolo_trainer._yolo_version_and_size(model_id)
        checkpoints.add(f"{yolo_trainer.MODEL_BASE_MAP[version]}{size}.pt")
    assert len(checkpoints) == 20


def test_yolo_ontology_variants_and_executable_registry_stay_aligned():
    models_csv = Path(__file__).parents[1] / "ontology_data" / "nodes" / "models.csv"
    with models_csv.open(newline="", encoding="utf-8") as handle:
        ontology_ids = {
            row["id"]
            for row in csv.DictReader(handle)
            if row["task"] == "object_detection"
            and row["model_family"] in {"YOLOv8", "YOLOv10", "YOLO11", "YOLO12"}
        }

    executable_ids = {
        item.value
        for item in DetectionHpoModelId
        if item.value.startswith(("yolov8_", "yolov10_", "yolov11_", "yolov12_"))
    }
    expected_executable_ids = {
        f"{model_id[:-1].replace('yolo11', 'yolov11').replace('yolo12', 'yolov12')}_{model_id[-1]}"
        for model_id in ontology_ids
    }
    executable_families = {
        model.id
        for model in MODEL_REGISTRY
        if model.task == "detection" and model.family == "yolo" and model.enabled
    }

    assert len(ontology_ids) == 20
    assert executable_ids == expected_executable_ids
    assert executable_families == {"yolov8", "yolov10", "yolov11", "yolov12"}


def test_visionkg_center_boxes_are_converted_to_coco_top_left_boxes():
    result = visionkg2cocoDet(
        [
            {
                "imageName": "car.jpg",
                "datasetName": "demo",
                "labelName": "car",
                "imageWidth": "100",
                "imageHeight": "80",
                "bbCentreX": "50",
                "bbCentreY": "30",
                "bbWidth": "20",
                "bbHeight": "10",
            }
        ]
    )
    assert result["annotations"][0]["bbox"] == [40.0, 25.0, 20.0, 10.0]


def test_yolo_schema_requires_pretrained_weights_and_normalizes_short_close_mosaic():
    config = _config(num_epochs=1)
    assert config.close_mosaic == 0

    with pytest.raises(ValueError, match="requires pretrained"):
        _config(model_weights="none")
    with pytest.raises(ValueError, match="linear learning-rate schedule"):
        _config(scheduler_name="multistep")
    with pytest.raises(ValueError, match="derives learning rate and momentum"):
        _config(optimizer_name="auto", learning_rate=0.005)


def test_yolo_runtime_adapter_forwards_the_saved_configuration(monkeypatch, tmp_path):
    captured = {}

    def fake_training(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(yolo_trainer, "_run_yolo_training", fake_training)
    monkeypatch.setattr(yolo_trainer, "tool_call_args_path", lambda job_id: tmp_path / "args.json")

    result = yolo_trainer.train_yolo_from_config(
        _config(
            model_name="yolov11_s",
            optimizer_name="adamw",
            beta1=0.85,
            mosaic=0.6,
            cutmix=0.2,
            confidence_threshold=0.3,
        ).runtime_config(),
        "test-job",
    )

    assert result.startswith("✅")
    assert captured["model_version"] == "yolo_v11"
    assert captured["model_size"] == "s"
    assert captured["optimizer"] == "AdamW"
    assert captured["momentum"] == 0.85
    assert captured["mosaic"] == 0.6
    assert captured["cutmix"] == 0.2


def test_yolo_training_preserves_the_original_exception(monkeypatch, tmp_path):
    def fail_training(**_kwargs):
        raise RuntimeError("inner tensor failure")

    monkeypatch.setattr(yolo_trainer, "_run_yolo_training", fail_training)
    monkeypatch.setattr(yolo_trainer, "tool_call_args_path", lambda job_id: tmp_path / "args.json")

    with pytest.raises(RuntimeError, match="YOLO training failed") as error:
        yolo_trainer.train_yolo_from_config(_config().runtime_config(), "test-job")

    assert str(error.value.__cause__) == "inner tensor failure"


@pytest.mark.parametrize("model_version", ["yolo_v11", "yolo_v12"])
def test_mps_unsafe_yolo_versions_use_cpu(monkeypatch, model_version):
    monkeypatch.setattr(yolo_trainer, "select_ultralytics_device_string", lambda: "mps")

    assert yolo_trainer.select_yolo_training_device(model_version) == "cpu"


@pytest.mark.parametrize("model_version", ["yolo_v8", "yolo_v10"])
def test_unconfirmed_yolo_versions_keep_mps(monkeypatch, model_version):
    monkeypatch.setattr(yolo_trainer, "select_ultralytics_device_string", lambda: "mps")

    assert yolo_trainer.select_yolo_training_device(model_version) == "mps"


def test_yolo_checkpoint_relocation_fails_when_training_produced_no_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="produced no checkpoint"):
        yolo_trainer._move_best_checkpoint(
            tmp_path / "empty-run",
            tmp_path / "best.pt",
        )


def _write_coco_split(path, image_name, image_id):
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": image_id, "file_name": image_name, "width": 20, "height": 10}
                ],
                "annotations": [
                    {
                        "id": image_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [2, 1, 4, 2],
                    }
                ],
                "categories": [{"id": 1, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )


def test_yolo_dataset_keeps_train_validation_and_test_isolated(tmp_path, monkeypatch):
    for split, image_id in (("train", 1), ("val", 2), ("test", 3)):
        image_name = f"source/{split}.jpg"
        image_path = tmp_path / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10)).save(image_path)
        _write_coco_split(tmp_path / f"{split}_annotations.json", image_name, image_id)

    yaml_path = tmp_path / "data.yaml"
    monkeypatch.setattr(yolo_trainer, "yolo_data_yaml_path", lambda job_id: yaml_path)
    yolo_trainer.create_yolo_data_yaml(
        "job",
        tmp_path,
        [{"id": 1, "name": "car"}],
    )

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["test"] == "images/test"
    for split in ("train", "val", "test"):
        images = list((tmp_path / "yolo_dataset" / "images" / split).glob("*.jpg"))
        labels = list((tmp_path / "yolo_dataset" / "labels" / split).glob("*.txt"))
        assert [path.name for path in images] == [f"source__{split}.jpg"]
        assert len(labels) == 1
        values = labels[0].read_text(encoding="utf-8").split()
        assert values == ["0", "0.200000", "0.200000", "0.200000", "0.200000"]


def test_yolo_graphrag_materializes_recipe_and_enforces_selected_family():
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_model_info={"model": {"model_architecture": "yolov11"}},
        selected_data=[
            {"class_name": "car", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
    )
    context = build_hyperparameter_context(state)
    assert context["base_recipe"]["id"] == "ultralytics_yolo_detection_finetune_balanced"
    assert {item["id"] for item in context["model_variants"]} == {
        "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
    }
    assert context["variant_benchmarks"]
    assert context["base_configuration"]["mosaic"] == 1.0
    assert context["base_configuration"]["confidence_threshold"] == 0.25
    assert context["base_configuration"]["scheduler_name"] == "linear"
    assert context["base_configuration"]["patience"] == 20
    assert "learning_rate" not in context["base_configuration"]
    assert "optimizer_name" in context["fields_available_for_policy_guidance"]
    assert "learning_rate" in context["fields_available_for_policy_guidance"]
    assert "optimizer_name" in context["allowed_adjustment_fields"]
    assert "learning_rate" in context["allowed_adjustment_fields"]
    assert context["base_configuration"]["loss_box"] == "ciou"
    assert context["base_configuration"]["loss_cls"] == "bce"
    assert not context["materialization_warnings"]

    candidate = _config(
        model_name="yolov11_n",
        **{
            key: value
            for key, value in context["base_configuration"].items()
            if key not in {"training_recipe_id", "model_name"}
        },
    ).model_dump(mode="json")
    validate_detection_graph_grounded_config(candidate, context)

    explicit = dict(candidate)
    explicit.update({"optimizer_name": "adamw", "learning_rate": 0.001})
    validate_detection_graph_grounded_config(explicit, context)
    assert {"optimizer_name", "learning_rate"} <= llm_controlled_fields(
        explicit,
        context,
        DetectionConfigModel,
    )

    candidate["model_name"] = "yolov8_n"
    with pytest.raises(ValueError, match="incompatible"):
        validate_detection_graph_grounded_config(candidate, context)


def test_detection_draft_completion_repairs_authoritative_yolo_fields():
    draft_data = _config().model_dump(mode="json")
    draft_data.update({
        "train_data_ratio": 0.0,
        "val_data_ratio": 0.0,
        "test_data_ratio": 0.0,
        "optimizer_name": "auto",
        "learning_rate": 0.2,
        "momentum": 0.0,
        "single_cls": False,
        "loss_box": "ciou",
        "loss_cls": "cross_entropy",
    })
    draft = DetectionConfigDraft.model_validate(draft_data)
    state = {
        "classes": ["furniture"],
        "selected_data": draft_data["selected_data"],
        "dataset_profile": {
            "planned_counts": {"train": 1600, "validation": 200, "test": 200},
        },
        "use_graphrag": True,
    }

    completed, adjustments = complete_detection_config(
        draft,
        state,
        "yolov10_n",
    )

    assert isinstance(completed, DetectionConfigModel)
    assert (
        completed.train_data_ratio,
        completed.val_data_ratio,
        completed.test_data_ratio,
    ) == (0.8, 0.1, 0.1)
    assert completed.learning_rate == 0.01
    assert completed.momentum == 0.9
    assert completed.single_cls is True
    assert completed.loss_box == "ciou"
    assert completed.loss_cls == "bce"
    assert {item["field"] for item in adjustments} >= {
        "train_data_ratio",
        "val_data_ratio",
        "test_data_ratio",
        "learning_rate",
        "momentum",
        "single_cls",
        "loss_cls",
    }


def test_detection_completion_disables_single_cls_for_multiple_classes():
    draft_data = _config(single_cls=True).model_dump(mode="json")
    draft = DetectionConfigDraft.model_validate(draft_data)
    state = {
        "classes": ["chair", "table"],
        "selected_data": draft_data["selected_data"],
        "dataset_profile": {
            "planned_counts": {"train": 16, "validation": 2, "test": 2},
        },
        "use_graphrag": True,
    }

    completed, adjustments = complete_detection_config(draft, state, "yolov10_n")

    assert completed.single_cls is False
    assert any(item["field"] == "single_cls" for item in adjustments)


def test_yolo_low_vram_rule_is_materialized_and_enforced():
    state = PipelineState(
        task="detection",
        classes=["car"],
        available_hardware={
            "hardware_category": "ConsumerGPU",
            "vram_gb": 8,
        },
        training_hardware=get_training_hardware_profile("macbook_air_m4_16gb"),
        selected_model_info={"model": {"model_architecture": "yolov8"}},
        selected_data=[
            {"class_name": "car", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
    )
    context = build_hyperparameter_context(state)

    assert context["required_adjustments"] == {
        "batch_size": 4,
        "workers": 4,
        "amp": False,
    }
    assert context["adjustment_rule_provenance"]["batch_size"] == (
        "rule_yolo_low_vram_batch"
    )

    candidate = _config(
        model_name="yolov8_n",
        **{
            key: value
            for key, value in context["recommended_configuration"].items()
            if key not in {"training_recipe_id", "model_name"}
        },
    ).model_dump(mode="json")
    validate_detection_graph_grounded_config(candidate, context)

    candidate["batch_size"] = 16
    with pytest.raises(ValueError, match="requires 'batch_size' to be 4"):
        validate_detection_graph_grounded_config(candidate, context)


def test_non_default_unmaterialized_yolo_field_requires_llm_rationale():
    state = PipelineState(
        task="detection",
        classes=["car"],
        selected_model_info={"model": {"model_architecture": "yolov8"}},
        selected_data=[
            {"class_name": "car", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
    )
    context = build_hyperparameter_context(state)
    candidate = _config(model_name="yolov8_n", cutmix=0.25).model_dump(mode="json")

    controlled = llm_controlled_fields(candidate, context, DetectionConfigModel)

    assert {"model_name", "cutmix"} <= controlled


def test_yolo_inference_category_map_is_zero_based(tmp_path, monkeypatch):
    annotations = tmp_path / "train.json"
    annotations.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 2, "name": "truck"},
                    {"id": 1, "name": "car"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(detection_pipe, "train_json_path", lambda job_id: annotations)
    assert detection_pipe._get_category_map("job") == {0: "car", 1: "truck"}


@pytest.mark.skipif(
    os.getenv("RUN_YOLO_SMOKE_TEST") != "1",
    reason="Set RUN_YOLO_SMOKE_TEST=1 to run the local one-epoch pretrained-model smoke test.",
)
def test_pretrained_yolov8_one_epoch_evaluation_and_inference(tmp_path, monkeypatch):
    job_id = "yolo-smoke"
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    local_prefix = Path(__file__).parents[1] / "src" / "yolov8"
    monkeypatch.setitem(yolo_trainer.MODEL_BASE_MAP, "yolo_v8", str(local_prefix))
    monkeypatch.setattr(yolo_trainer, "select_ultralytics_device_string", lambda: "cpu")

    data_root = paths.data_dir(job_id)
    for split, image_id in (("train", 1), ("val", 2), ("test", 3)):
        image_name = f"source/{split}.jpg"
        image_path = data_root / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "white").save(image_path)
        split_path = data_root / f"{split}_annotations.json"
        split_path.write_text(
            json.dumps(
                {
                    "images": [{"id": image_id, "file_name": image_name, "width": 64, "height": 64}],
                    "annotations": [
                        {"id": image_id, "image_id": image_id, "category_id": 1, "bbox": [16, 16, 32, 32]}
                    ],
                    "categories": [{"id": 1, "name": "car"}],
                }
            ),
            encoding="utf-8",
        )
    yolo_trainer.create_yolo_data_yaml(job_id, data_root, [{"id": 1, "name": "car"}])

    result = yolo_trainer.train_yolo_from_config(
        _config(
            num_epochs=1,
            patience=0,
            batch_size=1,
            input_size=64,
            workers=0,
            amp=False,
            mosaic=0.0,
        ).runtime_config(),
        job_id,
    )
    assert result.startswith("✅")
    assert paths.best_yolo_model_path(job_id).exists()

    metrics = yolo_trainer.evaluate_yolo_model(batch_size=1, image_size=64, job_id=job_id)
    assert "error" not in metrics

    model = yolo_trainer.YOLO(str(paths.best_yolo_model_path(job_id)))
    predictions = model.predict(Image.new("RGB", (64, 64), "white"), imgsz=64, device="cpu", verbose=False)
    assert len(predictions) == 1
