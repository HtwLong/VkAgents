import json

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import routers.execution as execution
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
from cvmodellearning.models.classification_model_utils import get_model_weights, make_model
from cvmodellearning.optimization.optimization_utils import make_optimizer
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.preprocessing.transformations import select_evaluation_transform
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.training_utils import (
    classification_parameter_groups,
    make_epoch_scheduler,
    set_backbone_trainable,
    train_one_epoch,
)


def _state() -> PipelineState:
    selected_data = [
        {
            "class_name": class_name,
            "sources": [{"dataset_name": "coco", "count": 250}],
        }
        for class_name in ("cat", "dog")
    ]
    return PipelineState(
        task="classification",
        classes=["cat", "dog"],
        selected_data=selected_data,
        selected_model_info={
            "model": [
                {
                    "model_architecture": "convnext_tiny",
                    "architecture_family": "convnext",
                }
            ]
        },
    )


def _candidate(context: dict) -> ClassificationConfigModel:
    state = _state()
    return ClassificationConfigModel(
        classes=state.classes,
        selected_data=[selection.model_dump() for selection in state.selected_data],
        train_data_ratio=0.8,
        val_data_ratio=0.1,
        test_data_ratio=0.1,
        rationale="Evidence-grounded ConvNeXt Tiny fine-tuning configuration.",
        **context["recommended_configuration"],
    )


def test_convnext_graphrag_materializes_complete_executable_recipe():
    get_hyperparameter_graph.cache_clear()
    context = build_hyperparameter_context(_state())
    recommended = context["recommended_configuration"]

    assert recommended == {
        "model_name": "convnext_tiny",
        "training_recipe_id": "torchvision_convnext_tiny_imagenet_v1_adapted_custom_finetune",
        "criterion_name": "cross_entropy",
        "training_mode": "fine_tune_pretrained",
        "model_weights": "default",
        "optimizer_name": "sgd",
        "scheduler_name": "step",
        "min_learning_rate": 0.0,
        "precision": "fp32",
        "learning_rate": 0.001,
        "batch_size": 4,
        "num_epochs": 25,
        "weight_decay": 0.0,
        "image_size": 224,
        "patience": 0,
        "warmup_epochs": 0,
        "momentum": 0.9,
        "gradient_accumulation_steps": 1,
        "freeze_backbone_epochs": 0,
        "label_smoothing": 0.0,
        "scheduler_step_size": 7,
        "scheduler_gamma": 0.1,
        "track_metric": "val_acc",
        "nesterov": False,
        "head_learning_rate_multiplier": 1.0,
    }
    assert context["materialization_warnings"] == []
    assert context["critical_materialization_errors"] == []
    assert context["recipe_details"][0]["feature_extraction_supported"] == "true"

    candidate = _candidate(context)
    validate_graph_grounded_config(candidate.model_dump(mode="json"), context)


def test_convnext_memory_estimate_is_reproducible_and_hardware_selectable():
    activation = estimate_cnn_activation_workspace(
        flops_b=4.456,
        task="classification",
        precision_mode="FP32",
    )
    estimate = calculate_inference_memory(
        params_m=28.589128,
        precision_mode="FP32",
        activation_workspace_gb=activation,
    )

    assert estimate.weight_memory_gb == 0.107
    assert estimate.activation_workspace_gb == 0.100
    assert estimate.runtime_overhead_gb == 0.021
    assert estimate.total_estimated_vram_gb == 0.228

    get_model_selection_graph.cache_clear()
    state = PipelineState.model_validate({
        **_state().model_dump(),
        "available_hardware": {"hardware_category": "ConsumerGPU", "vram_gb": 8},
    })
    context = build_model_selection_context(state, top_k=50)
    candidates = {item["model"]["id"]: item for item in context["candidate_models"]}

    assert candidates["convnext_tiny"]["model_inference_memory_estimate"][
        "total_estimated_vram_gb"
    ] == 0.228


def test_convnext_saved_runtime_config_is_accepted_at_execution(monkeypatch, tmp_path):
    get_hyperparameter_graph.cache_clear()
    context = build_hyperparameter_context(_state())
    candidate = _candidate(context)
    saved_path = tmp_path / "RESULT_HYPERPARAMETERS.json"
    saved_path.write_text(json.dumps(candidate.runtime_config()), encoding="utf-8")
    monkeypatch.setattr(execution, "hpo_config_path", lambda _job_id: saved_path)

    validated = execution._validate_config(
        ClassificationPipeline(),
        candidate.model_dump(mode="json"),
        job_id="planned-convnext",
        require_saved_config=True,
    )

    assert validated == training_compatible_hpo_config(candidate.runtime_config())


def test_convnext_recipe_allows_full_head_only_and_staged_finetuning():
    get_hyperparameter_graph.cache_clear()
    context = build_hyperparameter_context(_state())
    base = _candidate(context).model_dump(mode="json")

    for training_mode, freeze_epochs in (
        ("fine_tune_pretrained", 0),
        ("staged_fine_tune", 3),
        ("head_only", 25),
    ):
        config = {**base, "training_mode": training_mode, "freeze_backbone_epochs": freeze_epochs}
        parsed = ClassificationConfigModel.model_validate(config)
        validate_executable_recipe_config(parsed.model_dump(mode="json"))


def test_convnext_full_finetuning_updates_backbone_and_classifier():
    model, _ = make_model("convnext_tiny", "none", num_classes=2)
    backbone_before = model.features[0][0].weight.detach().clone()
    head_before = model.classifier[2].weight.detach().clone()
    config = {
        "optimizer_name": "sgd",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "nesterov": False,
        "head_learning_rate_multiplier": 1.0,
    }
    optimizer = make_optimizer(
        classification_parameter_groups(model, "convnext_tiny", config),
        config,
    )
    loader = DataLoader(
        TensorDataset(torch.rand(2, 3, 64, 64), torch.tensor([0, 1])),
        batch_size=2,
    )

    loss, _ = train_one_epoch(
        model,
        loader,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(backbone_before, model.features[0][0].weight)
    assert not torch.equal(head_before, model.classifier[2].weight)


def test_convnext_head_only_freezes_backbone_and_can_be_unfrozen():
    model, _ = make_model("convnext_tiny", "none", num_classes=2)
    set_backbone_trainable(model, "convnext_tiny", False)

    assert model.classifier[2].weight.requires_grad
    assert not model.features[0][0].weight.requires_grad

    set_backbone_trainable(model, "convnext_tiny", True)

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_convnext_head_only_step_updates_only_classifier():
    class TinyConvNextLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
                torch.nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Identity(),
                torch.nn.Flatten(1),
                torch.nn.Linear(4, 2),
            )

        def forward(self, images):
            return self.classifier(self.features(images))

    model = TinyConvNextLike()
    set_backbone_trainable(model, "convnext_tiny", False)
    backbone_before = model.features[0].weight.detach().clone()
    head_before = model.classifier[2].weight.detach().clone()
    config = {
        "optimizer_name": "sgd",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "nesterov": False,
        "head_learning_rate_multiplier": 1.0,
    }
    optimizer = make_optimizer(
        classification_parameter_groups(model, "convnext_tiny", config),
        config,
    )
    loader = DataLoader(
        TensorDataset(torch.rand(2, 3, 16, 16), torch.tensor([0, 1])),
        batch_size=2,
    )

    train_one_epoch(
        model,
        loader,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        frozen_backbone=model,
        trainable_head=model.classifier[2],
    )

    assert torch.equal(backbone_before, model.features[0].weight)
    assert not torch.equal(head_before, model.classifier[2].weight)
    assert model.training is False
    assert model.classifier[2].training is True


def test_convnext_recipe_scheduler_and_native_evaluation_transform_are_executable():
    get_hyperparameter_graph.cache_clear()
    config = _candidate(build_hyperparameter_context(_state())).model_dump()
    model, _ = make_model("convnext_tiny", "none", num_classes=2)
    optimizer = make_optimizer(
        classification_parameter_groups(model, "convnext_tiny", config),
        config,
    )
    scheduler = make_epoch_scheduler(optimizer, config)

    for _ in range(7):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.0001, 0.0001])

    weights = get_model_weights("convnext_tiny", "default")
    transform = select_evaluation_transform(
        "convnext_tiny",
        image_size=224,
        weights=weights,
    )
    transformed = transform(Image.new("RGB", (400, 300), color="white"))

    assert transform.resize_size == [236]
    assert transform.crop_size == [224]
    assert transformed.shape == (3, 224, 224)
