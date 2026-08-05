from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

import cvmodellearning.pipelines.classification_pipe as classification_pipe
from cvmodellearning.models.classification_model_utils import get_model_weights
from cvmodellearning.models.registry import model_ids
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from cvmodellearning.preprocessing.transformations import (
    CLASSIFICATION_TRANSFORM_PROFILES,
    select_evaluation_transform,
    select_transforms,
)


@pytest.mark.parametrize("model_name", model_ids("classification"))
def test_registered_model_transform_profile_matches_default_weight_metadata(model_name):
    profile = CLASSIFICATION_TRANSFORM_PROFILES[model_name]
    preset = get_model_weights(model_name, "default").transforms()

    assert profile.native_crop_size == preset.crop_size[0]
    assert profile.native_resize_size == preset.resize_size[0]
    assert profile.interpolation == preset.interpolation


@pytest.mark.parametrize("model_name", model_ids("classification"))
def test_all_registered_models_execute_hpo_training_transforms_and_native_evaluation(model_name):
    weights = get_model_weights(model_name, "default")
    profile = CLASSIFICATION_TRANSFORM_PROFILES[model_name]
    train_transform, configured_eval = select_transforms(
        model_name,
        image_size=profile.native_crop_size,
        weights=weights,
        auto_augment_policy="ta_wide",
        random_erasing=0.1,
        random_resized_crop_scale_min=0.7,
        horizontal_flip_probability=0.25,
    )
    native_eval = select_evaluation_transform(
        model_name,
        image_size=profile.native_crop_size,
        weights=weights,
    )
    image = Image.new("RGB", (512, 480), color=(127, 63, 31))

    assert train_transform(image).shape == (3, profile.native_crop_size, profile.native_crop_size)
    assert configured_eval(image).shape == (3, profile.native_crop_size, profile.native_crop_size)
    assert native_eval(image).shape == (3, profile.native_crop_size, profile.native_crop_size)
    crop = next(step for step in train_transform.transforms if step.__class__.__name__ == "RandomResizedCrop")
    assert crop.scale == (0.7, 1.0)
    assert crop.interpolation == weights.transforms().interpolation
    assert any(step.__class__.__name__ == "TrivialAugmentWide" for step in train_transform.transforms)
    assert any(step.__class__.__name__ == "RandomErasing" for step in train_transform.transforms)


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet50",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "densenet121",
        "convnext_tiny",
        "swin_v2_t",
        "swin_v2_s",
    ],
)
def test_configurable_models_use_custom_image_size_for_train_validation_and_inference(model_name):
    weights = get_model_weights(model_name, "default")
    custom_size = 320
    train_transform, validation_transform = select_transforms(
        model_name,
        image_size=custom_size,
        weights=weights,
    )
    inference_transform = select_evaluation_transform(
        model_name,
        image_size=custom_size,
        weights=weights,
    )
    image = Image.new("RGB", (500, 420), color="white")

    assert train_transform(image).shape[-2:] == (custom_size, custom_size)
    assert validation_transform(image).shape[-2:] == (custom_size, custom_size)
    assert inference_transform(image).shape[-2:] == (custom_size, custom_size)


def test_vit_rejects_transform_size_not_supported_by_registered_constructor():
    weights = get_model_weights("vit_b_16", "default")

    with pytest.raises(ValueError, match="supports only image_size=224"):
        select_transforms("vit_b_16", image_size=256, weights=weights)


def test_pipeline_rejects_non_model_ready_transformed_samples():
    pipeline = ClassificationPipeline()
    pipeline._validate_transformed_sample(
        TensorDataset(torch.zeros(1, 3, 224, 224), torch.zeros(1, dtype=torch.long)),
        224,
        "training",
    )

    with pytest.raises(ValueError, match="must produce a tensor shaped"):
        pipeline._validate_transformed_sample(
            TensorDataset(torch.zeros(1, 3, 128, 128), torch.zeros(1, dtype=torch.long)),
            224,
            "validation",
        )


def test_final_test_evaluation_uses_checkpoint_weight_transform(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "best_model.pt"
    stub_model = torch.nn.Linear(1, 2)
    config = {
        "model_name": "convnext_tiny",
        "model_weights": "default",
        "image_size": 224,
        "criterion_name": "cross_entropy",
        "label_smoothing": 0.0,
    }
    torch.save(
        {
            "classes": ["cat", "dog"],
            "config": config,
            "model_state_dict": stub_model.state_dict(),
        },
        checkpoint_path,
    )
    captured = {}

    class StubDataset(TensorDataset):
        def __init__(self, *, transform, **_kwargs):
            captured["transform"] = transform
            super().__init__(torch.ones(2, 3, 224, 224), torch.tensor([0, 1]))

    monkeypatch.setattr(classification_pipe, "best_model_path", lambda _job_id: checkpoint_path)
    monkeypatch.setattr(classification_pipe, "test_csv_path", lambda _job_id: tmp_path / "test.csv")
    monkeypatch.setattr(classification_pipe, "data_dir", lambda _job_id: tmp_path)
    monkeypatch.setattr(classification_pipe, "test_report_json_path", lambda _job_id: tmp_path / "report.json")
    monkeypatch.setattr(classification_pipe, "test_cm_path", lambda _job_id: tmp_path / "cm.csv")
    monkeypatch.setattr(classification_pipe, "CocoImageDataset", StubDataset)
    monkeypatch.setattr(
        classification_pipe,
        "make_model",
        lambda _name, _weights, num_classes: (torch.nn.Linear(1, num_classes), None),
    )
    monkeypatch.setattr(
        classification_pipe,
        "evaluate",
        lambda *_args, **_kwargs: (
            0.5,
            0.5,
            {
                "macro_f1": 0.5,
                "micro_f1": 0.5,
                "classification_report_dict": {"accuracy": 0.5},
                "confusion_matrix": [[1, 0], [1, 0]],
            },
        ),
    )
    monkeypatch.setattr(classification_pipe, "save_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ClassificationPipeline,
        "_require_prepared_data",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        classification_pipe,
        "record_split_access",
        lambda *_args, **_kwargs: tmp_path / "provenance.json",
    )
    monkeypatch.setattr(
        classification_pipe,
        "save_classification_report",
        lambda _job_id, _config, _metrics: Path(tmp_path / "report.json"),
    )

    ClassificationPipeline().evaluate_model_step(config, "example")

    transform = captured["transform"]
    assert transform.crop_size == [224]
    assert transform.resize_size == [236]
