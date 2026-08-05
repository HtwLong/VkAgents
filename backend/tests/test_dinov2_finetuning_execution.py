from __future__ import annotations

import os

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cvmodellearning.evaluation.evaluation_utils import evaluate
from cvmodellearning.graphrag.hyperparameter_context import (
    build_hyperparameter_context,
    get_hyperparameter_graph,
    validate_executable_recipe_config,
    validate_graph_grounded_config,
)
from cvmodellearning.graphrag.inference_memory import (
    calculate_inference_memory,
    estimate_cnn_activation_workspace,
)
from cvmodellearning.graphrag.model_selection_context import (
    build_model_selection_context,
    get_model_selection_graph,
)
from cvmodellearning.models import classification_model_utils
from cvmodellearning.models.classification_model_utils import get_model_weights, make_model
from cvmodellearning.models.model_manager import MODEL_CACHE_MANAGER
from cvmodellearning.models.registry import CLASSIFIER_HEAD_PATHS, model_ids
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from cvmodellearning.preprocessing.transformations import (
    CLASSIFICATION_TRANSFORM_PROFILES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    select_evaluation_transform,
    select_transforms,
)
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.hardware_profiles import get_training_hardware_profile
from cvmodellearning.training.training_utils import (
    classification_parameter_groups,
    classifier_head,
    set_backbone_trainable,
    train_one_epoch,
)


DINOV2_MODELS = ("dinov2_vits14", "dinov2_vitb14")
TIMM_IDS = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
}


def _selected_data(count: int):
    return [
        {"class_name": name, "sources": [{"dataset_name": "example", "count": count}]}
        for name in ("cat", "dog")
    ]


def _context(model_name: str, *, count: int = 1200, vram_gb: int | None = None):
    training_hardware = None
    if vram_gb is not None:
        training_hardware = get_training_hardware_profile("macbook_air_m4_16gb").model_copy(
            update={"training_memory_budget_gb": vram_gb}
        )
    return build_hyperparameter_context(
        PipelineState(
            task="classification",
            classes=["cat", "dog"],
            selected_data=_selected_data(count),
            training_hardware=training_hardware,
            selected_model_info={"model": [{"model_architecture": model_name}]},
        )
    )


class _TinyDinoClassifier(nn.Module):
    """Cheap logits-compatible stand-in for mandatory pipeline tests."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(12, 8),
            nn.GELU(),
        )
        self.head = nn.Linear(8, num_classes)

    def forward(self, images):
        return self.head(self.backbone(images))


@pytest.mark.parametrize("model_name", DINOV2_MODELS)
def test_dinov2_registry_factory_and_preprocessing_are_consistent(model_name):
    assert model_name in model_ids("classification")
    assert CLASSIFIER_HEAD_PATHS[model_name] == "head"

    model, weights = make_model(model_name, "none", num_classes=3)
    assert weights is None
    assert classifier_head(model, model_name).out_features == 3
    assert model.patch_embed.img_size == (224, 224)

    weights = get_model_weights(model_name, "default")
    preset = weights.transforms()
    profile = CLASSIFICATION_TRANSFORM_PROFILES[model_name]
    assert profile.native_crop_size == preset.crop_size[0] == 224
    assert profile.native_resize_size == preset.resize_size[0] == 256
    assert preset.mean == IMAGENET_MEAN
    assert preset.std == IMAGENET_STD


@pytest.mark.parametrize("model_name", DINOV2_MODELS)
def test_dinov2_transforms_match_training_validation_testing_and_inference(model_name):
    weights = get_model_weights(model_name, "default")
    train_transform, validation_transform = select_transforms(
        model_name, image_size=224, weights=weights
    )
    inference_transform = select_evaluation_transform(
        model_name, image_size=224, weights=weights
    )
    image = Image.new("RGB", (320, 280), color=(80, 120, 200))

    assert train_transform(image).shape == (3, 224, 224)
    assert validation_transform(image).shape == (3, 224, 224)
    assert inference_transform(image).shape == (3, 224, 224)
    assert inference_transform.resize_size == [256]


@pytest.mark.parametrize("model_name", DINOV2_MODELS)
def test_pretrained_dinov2_path_trains_evaluates_and_runs_pipeline_inference(
    monkeypatch, model_name
):
    called = {}

    def fake_create_model(model_id, *, pretrained, num_classes, img_size):
        called.update(
            model_id=model_id,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )
        return _TinyDinoClassifier(num_classes)

    monkeypatch.setattr(classification_model_utils.timm, "create_model", fake_create_model)
    model, weights = make_model(model_name, "default", num_classes=2)
    set_backbone_trainable(model, model_name, False)
    optimizer = torch.optim.AdamW(
        classification_parameter_groups(
            model,
            model_name,
            {"learning_rate": 5e-5, "head_learning_rate_multiplier": 1.0},
        ),
        lr=5e-5,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 8, 8), torch.tensor([0, 1])),
        batch_size=2,
    )

    loss, _ = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        frozen_backbone=model,
        trainable_head=classifier_head(model, model_name),
    )
    eval_loss, _, metrics = evaluate(
        ["cat", "dog"], model, loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )

    job_id = f"test-{model_name}"
    MODEL_CACHE_MANAGER.set_model_bundle(
        job_id,
        {
            "model": model,
            "device": torch.device("cpu"),
            "classes": ["cat", "dog"],
            "transform": select_evaluation_transform(
                model_name, image_size=224, weights=weights
            ),
        },
    )
    try:
        prediction = ClassificationPipeline().infer_step(
            job_id, Image.new("RGB", (300, 260), color=(120, 80, 40))
        )
    finally:
        MODEL_CACHE_MANAGER.unload_model(job_id)

    assert loss > 0 and eval_loss > 0
    assert set(prediction["probabilities"]) == {"cat", "dog"}
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    assert "accuracy" in metrics and "macro_f1" in metrics
    assert called == {
        "model_id": TIMM_IDS[model_name],
        "pretrained": True,
        "num_classes": 2,
        "img_size": 224,
    }


def test_dinov2_small_checkpoint_rebuild_uses_the_same_architecture():
    model, _ = make_model("dinov2_vits14", "none", num_classes=3)
    checkpoint = model.state_dict()
    rebuilt, _ = make_model("dinov2_vits14", "none", num_classes=3)

    rebuilt.load_state_dict(checkpoint)

    assert rebuilt.patch_embed.img_size == (224, 224)
    assert classifier_head(rebuilt, "dinov2_vits14").out_features == 3


@pytest.mark.skipif(
    os.getenv("RUN_PRETRAINED_DINOV2_TEST") != "1",
    reason="Set RUN_PRETRAINED_DINOV2_TEST=1 for real pretrained-weight smoke tests.",
)
@pytest.mark.parametrize("model_name", DINOV2_MODELS)
def test_real_pretrained_dinov2_trains_evaluates_and_runs_inference(model_name):
    """Optional cache/network test: one image and one head-only optimizer step."""
    model, weights = make_model(model_name, "default", num_classes=2)
    set_backbone_trainable(model, model_name, False)
    optimizer = torch.optim.AdamW(classifier_head(model, model_name).parameters(), lr=5e-5)
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
        trainable_head=classifier_head(model, model_name),
    )
    eval_loss, _, metrics = evaluate(
        ["cat", "dog"], model, loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )

    job_id = f"real-{model_name}"
    MODEL_CACHE_MANAGER.set_model_bundle(
        job_id,
        {
            "model": model,
            "device": torch.device("cpu"),
            "classes": ["cat", "dog"],
            "transform": select_evaluation_transform(
                model_name, image_size=224, weights=weights
            ),
        },
    )
    try:
        prediction = ClassificationPipeline().infer_step(
            job_id, Image.new("RGB", (300, 260), color=(120, 80, 40))
        )
    finally:
        MODEL_CACHE_MANAGER.unload_model(job_id)

    assert loss > 0 and eval_loss > 0
    assert "accuracy" in metrics
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("model_name", DINOV2_MODELS)
def test_dinov2_graphrag_materializes_a_valid_executable_config(model_name):
    get_hyperparameter_graph.cache_clear()
    context = _context(model_name)

    assert context["base_recipe"]["id"] == "timm_dinov2_s_b14_adapted_custom_finetune"
    assert context["base_configuration"]["image_size"] == 224
    assert context["critical_materialization_errors"] == []
    assert context["fields_requiring_llm_completion"] == [
        "optimizer_name",
        "patience",
        "precision",
        "scheduler_name",
        "track_metric",
    ]

    config = ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=_selected_data(1200),
        optimizer_name="adamw",
        scheduler_name="none",
        precision="fp32",
        patience=1,
        track_metric="val_acc",
        rationale="Graph-grounded DINOv2 custom classification fine-tuning.",
        **context["recommended_configuration"],
    )
    serialized = config.model_dump(mode="json")
    validate_executable_recipe_config(serialized)
    validate_graph_grounded_config(serialized, context)


@pytest.mark.parametrize("model_name", DINOV2_MODELS)
def test_dinov2_small_dataset_and_low_vram_rules_are_executable(model_name):
    context = _context(model_name, count=20, vram_gb=8)
    matched = {rule["id"] for rule in context["matched_adjustment_rules"]}

    assert matched == {
        "rule_dinov2_freeze_backbone_small_dataset",
        "rule_dinov2_low_vram_use_smaller_variant_or_accumulation",
    }
    assert context["recommended_configuration"]["training_mode"] == "head_only"
    assert context["recommended_configuration"]["freeze_backbone_epochs"] == 3
    assert context["recommended_configuration"]["batch_size"] == 2
    assert context["recommended_configuration"]["gradient_accumulation_steps"] == 4
    assert context["recommended_configuration"]["image_size"] == 224


@pytest.mark.parametrize("model_name", DINOV2_MODELS)
def test_dinov2_schema_rejects_unregistered_image_size(model_name):
    context = _context(model_name)
    candidate = {
        **context["recommended_configuration"],
        "image_size": 518,
        "optimizer_name": "adamw",
        "scheduler_name": "none",
        "precision": "fp32",
        "patience": 1,
        "track_metric": "val_acc",
    }

    with pytest.raises(ValueError, match="supports only image_size=224"):
        ClassificationConfigModel(
            classes=["cat", "dog"],
            selected_data=_selected_data(1200),
            rationale="Invalid DINOv2 resolution test.",
            **candidate,
        )


def test_large_and_giant_dinov2_remain_ontology_only():
    assert "dinov2_vitl14" not in model_ids("classification")
    assert "dinov2_vitg14" not in model_ids("classification")
    assert _context("dinov2_vitl14")["base_recipe"] is None
    assert _context("dinov2_vitg14")["base_recipe"] is None


@pytest.mark.parametrize(
    ("model_name", "params_m", "flops_b", "expected_total"),
    (
        ("dinov2_vits14", 22.1, 8.8, 0.137),
        ("dinov2_vitb14", 86.6, 28.4, 0.477),
    ),
)
def test_dinov2_224px_memory_rows_are_reproducible(
    model_name, params_m, flops_b, expected_total
):
    get_model_selection_graph.cache_clear()
    context = build_model_selection_context(PipelineState(task="classification"), top_k=50)
    candidates = {
        candidate["model"]["id"]: candidate for candidate in context["candidate_models"]
    }
    row = candidates[model_name]["model_inference_memory_estimate"]
    activation = estimate_cnn_activation_workspace(
        flops_b=flops_b, task="classification", precision_mode="FP16"
    )
    estimate = calculate_inference_memory(
        params_m=params_m,
        precision_mode="FP16",
        activation_workspace_gb=activation,
    )

    assert float(row["params_m"]) == params_m
    assert float(row["flops_b"]) == flops_b
    assert estimate.total_estimated_vram_gb == float(row["total_estimated_vram_gb"])
    assert estimate.total_estimated_vram_gb == expected_total


@pytest.mark.parametrize(
    "recipe_id",
    ("meta_dinov2_imagenet1k_linear_eval", "hf_dinov2_image_classification_finetune"),
)
def test_non_timm_dinov2_recipes_cannot_be_claimed_by_execution(recipe_id):
    with pytest.raises(ValueError, match="not executable"):
        validate_executable_recipe_config(
            {
                "training_recipe_id": recipe_id,
                "model_name": "dinov2_vits14",
                "training_mode": "fine_tune_pretrained",
                "model_weights": "default",
                "learning_rate": 5e-5,
                "batch_size": 16,
                "num_epochs": 3,
                "image_size": 224,
            }
        )
