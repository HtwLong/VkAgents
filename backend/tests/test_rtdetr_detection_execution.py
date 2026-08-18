import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from ultralytics.nn.tasks import RTDETRDetectionModel

from cvmodellearning import paths
from cvmodellearning.graphrag.hyperparameter_context import (
    build_hyperparameter_context,
    validate_detection_graph_grounded_config,
)
from cvmodellearning.models.detection_models import rtdetr_trainer, yolo_trainer
from cvmodellearning.models.detection_models.rtdetr_lora import (
    LoRALinear,
    apply_rtdetr_lora,
    merge_rtdetr_lora_,
    set_rtdetr_lora_trainability,
)
from cvmodellearning.graphrag.model_selection_context import build_model_selection_context
from cvmodellearning.models.registry import (
    DetectionHpoModelId,
    model_ids,
    resolve_model_id,
)
from cvmodellearning.pipelines import detection_pipe
from cvmodellearning.pipelines.detection_pipe import _get_trainer_type
from cvmodellearning.schemas.detection_hpo import (
    DetectionConfigModel,
    expand_detection_config_for_validation,
)
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.schemas.revision import initial_hpo_override_values
from cvmodellearning.training.hardware_profiles import get_training_hardware_profile


RECIPE_ID = "ultralytics_rtdetr_l_coco_pretrained_custom_finetune"
HIGH_THROUGHPUT_RECIPE_ID = (
    "ultralytics_rtdetr_l_coco_pretrained_custom_finetune_high_throughput"
)


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
        "num_epochs": 100,
        "patience": 20,
        "batch_size": 1,
        "input_size": 640,
        "max_size": 640,
        "aspect_ratio_range": None,
        "model_name": "rtdetr_hgnetv2_l",
        "model_weights": "coco",
        "training_recipe_id": RECIPE_ID,
        "optimizer_name": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "beta1": 0.9,
        "scheduler_name": "linear",
        "final_learning_rate_factor": 0.01,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "amp": False,
        "loss_box": "l1_giou",
        "loss_cls": "varifocal",
        "lambda_box": 5.0,
        "lambda_giou": 2.0,
        "lambda_cls": 1.0,
        "lambda_dfl": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "close_mosaic": 10,
        "rect": False,
        "multi_scale": 0.0,
        "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.0,
        "max_detections": 300,
        "freeze": 0,
        "rationale": "Grounded Ultralytics RT-DETR-L test configuration.",
    }
    values.update(overrides)
    return DetectionConfigModel.model_validate(values)


def _state():
    return PipelineState(
        task="detection",
        classes=["car"],
        selected_model_info={"model": {"model_architecture": "rtdetr_hgnetv2_l"}},
        selected_data=[
            {"class_name": "car", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
    )


def test_only_rtdetr_l_is_executable_and_routes_to_its_trainer():
    assert resolve_model_id("detection", "rtdetr-l") == "rtdetr_hgnetv2_l"
    assert "rt-detr-r50_8xb2-100e_coco" not in model_ids("detection")
    assert DetectionHpoModelId.RTDETR_HGNETV2_L.value == "rtdetr_hgnetv2_l"
    assert _get_trainer_type("rtdetr_hgnetv2_l") == "rtdetr"
    with pytest.raises(ValueError, match="Unknown model architecture"):
        _get_trainer_type("rtdetr_r50")


def test_rtdetr_runtime_contains_only_active_backend_fields():
    runtime = _config().runtime_config()

    assert "mosaic" in runtime
    assert "nms_iou_threshold" not in runtime
    assert "lambda_dfl" not in runtime
    assert "trainable_backbone_layers" not in runtime
    assert "topk_candidates" not in runtime
    assert "rationale" not in runtime

    expanded = training_compatible_hpo_config(runtime)
    expanded["rationale"] = "Execution validation."
    validated = DetectionConfigModel.model_validate(
        expand_detection_config_for_validation(expanded)
    )
    assert validated.runtime_config() == runtime


def test_rtdetr_graphrag_selects_high_throughput_recipe_on_rtx6000():
    state = _state()
    state.training_hardware = get_training_hardware_profile("rtx6000_48gb")
    context = build_hyperparameter_context(state)

    assert context["selected_model_id"] == "rtdetr_hgnetv2_l"
    assert context["base_recipe"]["id"] == HIGH_THROUGHPUT_RECIPE_ID
    assert context["reference_configuration"] == {
        "training_recipe_id": HIGH_THROUGHPUT_RECIPE_ID,
        "model_weights": "coco",
        "optimizer_name": "adamw",
        "scheduler_name": "linear",
        "learning_rate": 0.001,
        "batch_size": 16,
        "num_epochs": 100,
        "weight_decay": 0.0005,
        "input_size": 640,
        "patience": 20,
        "warmup_epochs": 3.0,
        "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "close_mosaic": 10,
        "amp": False,
        "beta1": 0.9,
        "final_learning_rate_factor": 0.01,
        "warmup_momentum": 0.8,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "loss_box": "l1_giou",
        "loss_cls": "varifocal",
        "lambda_box": 5.0,
        "lambda_giou": 2.0,
        "lambda_cls": 1.0,
        "lambda_dfl": 0.0,
        "max_size": 640,
        "aspect_ratio_range": None,
        "rect": False,
        "multi_scale": 0.0,
        "max_detections": 300,
        "freeze": 0,
        "lr_milestones": [],
        "scheduler_gamma": 1.0,
        "augmentation_policy": "basic",
        "trainable_backbone_layers": 0,
        "horizontal_flip_probability": 0.0,
        "topk_candidates": 400,
        "positive_fraction": 0.25,
        "matching_iou_threshold": 0.5,
    }
    assert not context["critical_materialization_errors"]
    assert {
        rule["id"] for rule in context["applicable_rules"]
    } == {
        "rule_rtdetr_l_low_memory_batch",
        "rule_rtdetr_l_high_memory_batch",
    }
    assert {
        rule["id"] for rule in context["matched_adjustment_rules"]
    } == {"rule_rtdetr_l_high_memory_batch"}

    candidate = _config(
        batch_size=16,
        training_recipe_id=HIGH_THROUGHPUT_RECIPE_ID,
    ).model_dump(mode="json")
    validate_detection_graph_grounded_config(candidate, context)


def test_rtdetr_high_throughput_recipe_requires_batch_8_to_16():
    _config(batch_size=8, training_recipe_id=HIGH_THROUGHPUT_RECIPE_ID)
    _config(batch_size=16, training_recipe_id=HIGH_THROUGHPUT_RECIPE_ID)

    state = _state()
    state.training_hardware = get_training_hardware_profile("rtx6000_48gb")
    context = build_hyperparameter_context(state)
    candidate = _config(
        batch_size=1,
        training_recipe_id=HIGH_THROUGHPUT_RECIPE_ID,
    ).model_dump(mode="json")
    with pytest.raises(ValueError, match="requires batch_size >= 8"):
        validate_detection_graph_grounded_config(candidate, context)


def test_detection_lora_request_shortlists_only_executable_rtdetr():
    state_data = _state().model_dump(mode="json")
    state_data.update({
        "user_query": "Train an object detector with LoRA rank 4.",
        "model_requirements": [
            {
                "name": None,
                "training_mode": "lora",
                "lora_rank": 4,
                "requirement_strength": "required",
            }
        ],
    })
    state = PipelineState.model_validate(state_data)

    assert initial_hpo_override_values(state) == {
        "training_mode": "lora",
        "model_weights": "default",
        "lora_rank": 4,
    }
    selection = build_model_selection_context(state)
    assert selection["filters"]["requires_lora"] is True
    assert {
        candidate["model"]["id"] for candidate in selection["candidate_models"]
    } == {"rtdetr_hgnetv2_l"}
    context = build_hyperparameter_context(
        state.model_copy(update={
            "selected_model_info": {"model": {"model_architecture": "rtdetr_hgnetv2_l"}}
        })
    )
    assert context["selected_model_id"] == "rtdetr_hgnetv2_l"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("amp", True, "AMP is disabled"),
        ("nms_iou_threshold", 0.5, "NMS-free"),
        ("max_detections", 301, "300 object queries"),
        ("loss_box", "giou", "L1 plus GIoU"),
        ("input_size", 800, "640px square"),
        ("batch_size", -1, "explicit batch_size"),
    ],
)
def test_rtdetr_schema_rejects_incompatible_execution_fields(field, value, message):
    overrides = {field: value}
    if field == "input_size":
        overrides["max_size"] = value
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_rtdetr_schema_matches_the_installed_varifocal_loss_contract():
    criterion = RTDETRDetectionModel.init_criterion(SimpleNamespace(nc=3))

    assert criterion.vfl is not None
    assert criterion.fl is not None
    assert criterion.loss_gain["class"] == 1
    assert criterion.loss_gain["bbox"] == 5
    assert criterion.loss_gain["giou"] == 2


def test_rtdetr_lora_schema_and_runtime_contract():
    config = _config(training_mode="lora", lora_rank=4, lora_alpha=8)
    runtime = config.runtime_config()

    assert runtime["training_mode"] == "lora"
    assert runtime["lora_rank"] == 4
    assert runtime["lora_alpha"] == 8
    assert runtime["lora_target_profile"] == "decoder_attention"
    assert runtime["train_detection_head"] is True


def test_rtdetr_lora_injection_is_zero_initialized_and_adapter_head_only():
    torch.manual_seed(0)
    model = RTDETRDetectionModel("rtdetr-l.yaml", nc=3, verbose=False)
    target_name = "model.28.decoder.layers.0.cross_attn.value_proj"
    inputs = torch.randn(2, 5, 256)
    before = model.get_submodule(target_name)(inputs).detach()

    summary = apply_rtdetr_lora(
        model,
        {
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "lora_target_profile": "decoder_attention",
        },
    )
    wrapped = model.get_submodule(target_name)
    after = wrapped(inputs).detach()

    assert isinstance(wrapped, LoRALinear)
    assert len(summary.target_modules) == 12
    assert torch.equal(before, after)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any("lora_A" in name for name in trainable)
    assert any("lora_B" in name for name in trainable)
    assert any("dec_score_head" in name for name in trainable)
    assert not any(name.startswith("model.0.") for name in trainable)


def test_rtdetr_lora_trainability_is_restored_and_merge_preserves_output():
    model = RTDETRDetectionModel("rtdetr-l.yaml", nc=3, verbose=False)
    apply_rtdetr_lora(
        model,
        {
            "lora_rank": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "lora_target_profile": "decoder_attention",
        },
    )
    target_name = "model.28.decoder.layers.0.cross_attn.output_proj"
    wrapped = model.get_submodule(target_name)
    wrapped.lora_B.weight.data.normal_(std=0.01)
    inputs = torch.randn(2, 5, 256)
    expected = wrapped(inputs).detach()

    model.requires_grad_(True)
    set_rtdetr_lora_trainability(model)
    assert not model.get_parameter("model.0.stem1.conv.weight").requires_grad
    merge_rtdetr_lora_(model)
    merged = model.get_submodule(target_name)

    assert isinstance(merged, torch.nn.Linear)
    assert torch.allclose(expected, merged(inputs), atol=1e-6, rtol=1e-5)


def test_rtdetr_trainer_forwards_validated_fields_and_moves_artifacts(tmp_path, monkeypatch):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    run_root = tmp_path / "run"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    captured = {}

    class FakeRTDETR:
        def __init__(self, checkpoint):
            captured["checkpoint"] = checkpoint
            self.trainer = None

        def train(self, **kwargs):
            captured.update(kwargs)
            save_dir = run_root / "temp_run"
            (save_dir / "weights").mkdir(parents=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"checkpoint")
            (save_dir / "results.csv").write_text("epoch,map\n0,0\n", encoding="utf-8")
            self.trainer = SimpleNamespace(save_dir=save_dir)

    monkeypatch.setattr(rtdetr_trainer, "RTDETR", FakeRTDETR)
    monkeypatch.setattr(rtdetr_trainer, "_checkpoint_path", lambda _: "rtdetr-l.pt")
    monkeypatch.setattr(rtdetr_trainer, "select_ultralytics_device_string", lambda: "cpu")
    monkeypatch.setattr(rtdetr_trainer, "yolo_data_yaml_path", lambda _: data_yaml)
    monkeypatch.setattr(rtdetr_trainer, "run_dir", lambda _: run_root)
    monkeypatch.setattr(rtdetr_trainer, "best_yolo_model_path", lambda _: artifact_root / "best.pt")
    monkeypatch.setattr(rtdetr_trainer, "training_log_path", lambda _: artifact_root / "results.csv")
    monkeypatch.setattr(rtdetr_trainer, "plots_dir", lambda _: artifact_root)
    monkeypatch.setattr(rtdetr_trainer, "tool_call_args_path", lambda _: artifact_root / "args.json")

    result = rtdetr_trainer.train_rtdetr_from_config(
        _config(
            num_epochs=1,
            patience=0,
            batch_size=1,
            close_mosaic=0,
            warmup_epochs=0.0,
        ).runtime_config(),
        "job",
    )

    assert result.startswith("✅")
    assert captured["checkpoint"] == "rtdetr-l.pt"
    assert captured["optimizer"] == "AdamW"
    assert captured["imgsz"] == 640
    assert captured["batch"] == 1
    assert captured["close_mosaic"] == 0
    assert captured["amp"] is False
    assert captured["deterministic"] is False
    assert captured["cos_lr"] is False
    assert (artifact_root / "best.pt").exists()
    assert (artifact_root / "results.csv").exists()


def test_rtdetr_lora_training_selects_custom_trainer(tmp_path, monkeypatch):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    run_root = tmp_path / "run"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    captured = {}

    class FakeRTDETR:
        def __init__(self, _checkpoint):
            self.trainer = None

        def train(self, **kwargs):
            captured.update(kwargs)
            save_dir = run_root / "temp_run"
            (save_dir / "weights").mkdir(parents=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"checkpoint")
            self.trainer = SimpleNamespace(save_dir=save_dir)

    monkeypatch.setattr(rtdetr_trainer, "RTDETR", FakeRTDETR)
    monkeypatch.setattr(rtdetr_trainer, "_checkpoint_path", lambda _: "rtdetr-l.pt")
    monkeypatch.setattr(rtdetr_trainer, "select_ultralytics_device_string", lambda: "cpu")
    monkeypatch.setattr(rtdetr_trainer, "yolo_data_yaml_path", lambda _: data_yaml)
    monkeypatch.setattr(rtdetr_trainer, "run_dir", lambda _: run_root)
    monkeypatch.setattr(rtdetr_trainer, "best_yolo_model_path", lambda _: artifact_root / "best.pt")
    monkeypatch.setattr(rtdetr_trainer, "training_log_path", lambda _: artifact_root / "results.csv")
    monkeypatch.setattr(rtdetr_trainer, "plots_dir", lambda _: artifact_root)
    monkeypatch.setattr(rtdetr_trainer, "tool_call_args_path", lambda _: artifact_root / "args.json")

    result = rtdetr_trainer.train_rtdetr_from_config(
        _config(
            training_mode="lora",
            lora_rank=4,
            lora_alpha=8,
            num_epochs=1,
            patience=0,
            close_mosaic=0,
            warmup_epochs=0.0,
        ).runtime_config(),
        "job",
    )

    assert result.startswith("✅")
    assert issubclass(captured["trainer"], rtdetr_trainer.RTDETRTrainer)
    execution = json.loads((artifact_root / "args.json").read_text(encoding="utf-8"))
    assert execution["lora"] == {
        "rank": 4,
        "alpha": 8,
        "dropout": 0.05,
        "target_profile": "decoder_attention",
        "train_detection_head": True,
    }


def test_rtdetr_evaluation_and_pipeline_inference_use_dedicated_api(tmp_path, monkeypatch):
    captured = {}

    class FakeRTDETR:
        def __init__(self, checkpoint):
            captured.setdefault("checkpoints", []).append(str(checkpoint))

        def val(self, **kwargs):
            captured["val"] = kwargs
            return SimpleNamespace(
                box=SimpleNamespace(map=0.4, map50=0.6, map75=0.3, mp=0.5, mr=0.45),
                save_dir=tmp_path / "evaluation",
            )

        def predict(self, **kwargs):
            captured["predict"] = kwargs
            return [SimpleNamespace(boxes=SimpleNamespace(
                data=torch.tensor([[1.0, 1.0, 8.0, 8.0, 0.9, 0.0]])
            ))]

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"checkpoint")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\n", encoding="utf-8")
    monkeypatch.setattr(rtdetr_trainer, "RTDETR", FakeRTDETR)
    monkeypatch.setattr(rtdetr_trainer, "best_yolo_model_path", lambda _: model_path)
    monkeypatch.setattr(rtdetr_trainer, "yolo_data_yaml_path", lambda _: data_yaml)
    monkeypatch.setattr(rtdetr_trainer, "run_dir", lambda _: tmp_path)

    metrics = rtdetr_trainer.evaluate_rtdetr_model(
        batch_size=1,
        image_size=640,
        job_id="job",
    )
    assert metrics["mAP@.50:.95"] == 0.4
    assert captured["val"]["split"] == "test"

    config_path = tmp_path / "hpo.json"
    config_path.write_text(json.dumps(_config().runtime_config()), encoding="utf-8")
    annotations = tmp_path / "train.json"
    annotations.write_text(json.dumps({"categories": [{"id": 1, "name": "car"}]}), encoding="utf-8")
    inference_dir = tmp_path / "inference"
    monkeypatch.setattr(detection_pipe, "RTDETR", FakeRTDETR)
    monkeypatch.setattr(detection_pipe, "hpo_config_path", lambda _: config_path)
    monkeypatch.setattr(detection_pipe, "best_yolo_model_path", lambda _: model_path)
    monkeypatch.setattr(detection_pipe, "train_json_path", lambda _: annotations)
    monkeypatch.setattr(detection_pipe, "_get_inference_save_dir", lambda _: inference_dir)
    inference_dir.mkdir()

    pipeline = detection_pipe.DetectionPipeline()
    detection_pipe.MODEL_CACHE_MANAGER.unload_model("job")
    assert pipeline.load_model_step("job")["trainer_type"] == "rtdetr"
    result = pipeline.infer_step("job", Image.new("RGB", (32, 32), "white"))
    pipeline.unload_model("job")

    assert result["status"] == "success"
    assert result["detections_count"] == 1
    assert result["image_width"] == 32
    assert result["image_height"] == 32
    assert result["detections"][0]["box"] == [1.0, 1.0, 8.0, 8.0]
    assert result["detections"][0]["confidence"] == pytest.approx(0.9)
    assert result["detections"][0]["class_id"] == 0
    assert result["detections"][0]["label"] == "car"
    assert captured["predict"]["imgsz"] == 640
    assert captured["predict"]["max_det"] == 300
    assert "iou" not in captured["predict"]


def _write_coco_split(path: Path, image_name: str, image_id: int) -> None:
    path.write_text(json.dumps({
        "images": [{"id": image_id, "file_name": image_name, "width": 640, "height": 640}],
        "annotations": [{
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [160, 160, 320, 320],
        }],
        "categories": [{"id": 1, "name": "car"}],
    }), encoding="utf-8")


@pytest.mark.skipif(
    os.getenv("RUN_RTDETR_SMOKE_TEST") != "1",
    reason="Set RUN_RTDETR_SMOKE_TEST=1 to run one-epoch pretrained RT-DETR-L train/val/predict.",
)
def test_pretrained_rtdetr_one_epoch_evaluation_and_inference(tmp_path, monkeypatch):
    job_id = "rtdetr-smoke"
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(rtdetr_trainer, "select_ultralytics_device_string", lambda: "cpu")

    root = paths.data_dir(job_id)
    for split, image_id in (("train", 1), ("val", 2), ("test", 3)):
        image_name = f"{split}.jpg"
        Image.new("RGB", (640, 640), "white").save(root / image_name)
        _write_coco_split(root / f"{split}_annotations.json", image_name, image_id)
    yolo_trainer.create_yolo_data_yaml(job_id, root, [{"id": 1, "name": "car"}])

    result = rtdetr_trainer.train_rtdetr_from_config(
        _config(
            num_epochs=1,
            patience=0,
            batch_size=1,
            workers=0,
            close_mosaic=0,
        ).runtime_config(),
        job_id,
    )
    assert result.startswith("✅")
    assert paths.best_yolo_model_path(job_id).exists()

    metrics = rtdetr_trainer.evaluate_rtdetr_model(
        batch_size=1,
        image_size=640,
        job_id=job_id,
    )
    assert "error" not in metrics

    model = rtdetr_trainer.RTDETR(str(paths.best_yolo_model_path(job_id)))
    predictions = model.predict(
        Image.new("RGB", (640, 640), "white"),
        imgsz=640,
        device="cpu",
        verbose=False,
    )
    assert len(predictions) == 1
