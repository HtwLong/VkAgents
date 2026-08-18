from __future__ import annotations

import os

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cvmodellearning.graphrag.hyperparameter_context import (
    build_hyperparameter_context,
    get_hyperparameter_graph,
    validate_executable_recipe_config,
    validate_graph_grounded_config,
)
from cvmodellearning.models import classification_model_utils
from cvmodellearning.models.classification_model_utils import get_model_weights, make_model
from cvmodellearning.models.registry import CLASSIFIER_HEAD_PATHS, model_ids
from cvmodellearning.preprocessing.transformations import (
    CLASSIFICATION_TRANSFORM_PROFILES,
    CLIP_MEAN,
    CLIP_STD,
    select_evaluation_transform,
    select_transforms,
)
from cvmodellearning.training.hardware_profiles import get_training_hardware_profile
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.training_utils import (
    classification_parameter_groups,
    classifier_head,
    set_backbone_trainable,
    train_one_epoch,
)


SMALL_DATA = [
    {"class_name": name, "sources": [{"dataset_name": "example", "count": 20}]}
    for name in ("cat", "dog")
]


class _TinyClipClassifier(nn.Module):
    """Cheap stand-in used to test the pretrained factory/training path."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.ReLU())
        self.head = nn.Linear(8, num_classes)

    def forward(self, images):
        return self.head(self.backbone(images))


def test_clip_registry_factory_and_preprocessing_are_consistent():
    assert "clip_vit_b16" in model_ids("classification")
    assert CLASSIFIER_HEAD_PATHS["clip_vit_b16"] == "head"

    model, weights = make_model("clip_vit_b16", "none", num_classes=3)
    assert weights is None
    assert classifier_head(model, "clip_vit_b16").out_features == 3

    weights = get_model_weights("clip_vit_b16", "default")
    preset = weights.transforms()
    profile = CLASSIFICATION_TRANSFORM_PROFILES["clip_vit_b16"]
    assert profile.native_crop_size == preset.crop_size[0] == 224
    assert profile.native_resize_size == preset.resize_size[0] == 248
    assert preset.mean == CLIP_MEAN
    assert preset.std == CLIP_STD


def test_clip_train_validation_test_and_inference_transforms_match():
    weights = get_model_weights("clip_vit_b16", "default")
    train_transform, validation_transform = select_transforms(
        "clip_vit_b16", image_size=224, weights=weights
    )
    inference_transform = select_evaluation_transform(
        "clip_vit_b16", image_size=224, weights=weights
    )
    image = Image.new("RGB", (320, 280), color=(80, 120, 200))

    assert train_transform(image).shape == (3, 224, 224)
    assert validation_transform(image).shape == (3, 224, 224)
    assert inference_transform(image).shape == (3, 224, 224)
    assert inference_transform.resize_size == [248]


def test_pretrained_clip_factory_path_runs_one_short_head_only_epoch(monkeypatch):
    called = {}

    def fake_create_model(model_id, *, pretrained, num_classes):
        called.update(model_id=model_id, pretrained=pretrained, num_classes=num_classes)
        return _TinyClipClassifier(num_classes)

    monkeypatch.setattr(classification_model_utils.timm, "create_model", fake_create_model)
    model, weights = make_model("clip_vit_b16", "default", num_classes=2)
    set_backbone_trainable(model, "clip_vit_b16", False)
    optimizer = torch.optim.AdamW(
        classification_parameter_groups(
            model,
            "clip_vit_b16",
            {"learning_rate": 1e-4, "head_learning_rate_multiplier": 1.0},
        ),
        lr=1e-4,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 2, 2), torch.tensor([0, 1])),
        batch_size=2,
    )

    loss, _ = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        frozen_backbone=model,
        trainable_head=classifier_head(model, "clip_vit_b16"),
    )

    assert loss > 0
    assert weights is get_model_weights("clip_vit_b16", "default")
    assert called == {
        "model_id": classification_model_utils.CLIP_VIT_B16_TIMM_ID,
        "pretrained": True,
        "num_classes": 2,
    }


@pytest.mark.skipif(
    os.getenv("RUN_PRETRAINED_CLIP_TEST") != "1",
    reason="Set RUN_PRETRAINED_CLIP_TEST=1 for the real pretrained-weight smoke test.",
)
def test_real_pretrained_clip_runs_one_head_only_training_batch():
    """Optional network/cache test: one sample, one batch, one epoch."""
    model, _ = make_model("clip_vit_b16", "default", num_classes=2)
    set_backbone_trainable(model, "clip_vit_b16", False)
    optimizer = torch.optim.AdamW(
        classifier_head(model, "clip_vit_b16").parameters(), lr=1e-4
    )
    loader = DataLoader(
        TensorDataset(torch.randn(1, 3, 224, 224), torch.tensor([1])),
        batch_size=1,
    )

    loss, _ = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        frozen_backbone=model,
        trainable_head=classifier_head(model, "clip_vit_b16"),
    )

    assert loss > 0
    assert model.eval()(torch.randn(1, 3, 224, 224)).shape == (1, 2)


def test_clip_graphrag_materializes_an_executable_config():
    get_hyperparameter_graph.cache_clear()
    context = build_hyperparameter_context(
        PipelineState(
            task="classification",
            classes=["cat", "dog"],
            selected_data=SMALL_DATA,
            selected_model_info={"model": [{"model_architecture": "clip_vit_b16"}]},
        )
    )

    assert context["base_recipe"]["id"] == (
        "timm_clip_vit_b16_openai_adapted_custom_finetune"
    )
    assert context["reference_configuration"]["image_size"] == 224
    assert context["critical_materialization_errors"] == []

    config = ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=SMALL_DATA,
        track_metric="val_acc",
        rationale="Graph-grounded CLIP visual-encoder fine-tuning configuration.",
        **context["reference_configuration"],
    )
    serialized = config.model_dump(mode="json")
    validate_executable_recipe_config(serialized)
    validate_graph_grounded_config(serialized, context)


def obsolete_clip_low_vram_rule_only_changes_executable_fields():
    context = build_hyperparameter_context(
        PipelineState(
            task="classification",
            classes=["cat", "dog"],
            selected_data=SMALL_DATA,
            training_hardware=get_training_hardware_profile("macbook_air_m4_16gb"),
            selected_model_info={"model": [{"model_architecture": "clip_vit_b16"}]},
        )
    )

    assert {rule["id"] for rule in context["matched_adjustment_rules"]} == {
        "rule_clip_low_vram_reduce_batch_or_resolution",
        "rule_transformer_classifier_low_vram_use_lora",
    }
    assert context["reference_configuration"]["batch_size"] == 2
    assert context["reference_configuration"]["gradient_accumulation_steps"] == 4
    assert context["reference_configuration"]["image_size"] == 224
    assert {
        rule["id"] for rule in context["applicable_rules"]
    }.isdisjoint(
        {
            "rule_clip_finetune_with_layer_decay",
            "rule_clip_finetune_effective_batch_2048",
            "rule_clip_disable_mixup_cutmix_drop_path",
            "rule_clip_use_randaug_label_smoothing_ema",
        }
    )
    validate_graph_grounded_config(context["reference_configuration"], context)


def test_clip_schema_rejects_non_native_image_size():
    context = build_hyperparameter_context(
        PipelineState(
            task="classification",
            classes=["cat", "dog"],
            selected_data=SMALL_DATA,
            selected_model_info={"model": [{"model_architecture": "clip_vit_b16"}]},
        )
    )
    candidate = {**context["reference_configuration"], "image_size": 256}

    with pytest.raises(ValueError, match="supports only image_size=224"):
        ClassificationConfigModel(
            classes=["cat", "dog"],
            selected_data=SMALL_DATA,
            track_metric="val_acc",
            rationale="Invalid CLIP resolution test.",
            **candidate,
        )

