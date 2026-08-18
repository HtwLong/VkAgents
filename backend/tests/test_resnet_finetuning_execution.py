import json

import pytest
import torch
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError
from torch.utils.data import DataLoader, TensorDataset

from cvmodellearning.evaluation.evaluation_utils import evaluate
from cvmodellearning.graphrag.hyperparameter_context import (
    build_field_provenance,
    build_hyperparameter_context,
    validate_graph_grounded_config,
)
from cvmodellearning.models.classification_model_utils import get_model_weights, make_model
from cvmodellearning.optimization.optimization_utils import make_optimizer
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from cvmodellearning.preprocessing.transformations import select_evaluation_transform
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.training_utils import (
    classification_parameter_groups,
    make_epoch_scheduler,
    set_backbone_trainable,
    train_one_epoch,
)
import routers.execution as execution


def _state(images_per_class: int = 250) -> PipelineState:
    selected_data = [
        {
            "class_name": class_name,
            "sources": [{"dataset_name": "coco", "count": images_per_class}],
        }
        for class_name in ("cat", "dog")
    ]
    return PipelineState(
        task="classification",
        classes=["cat", "dog"],
        selected_data=selected_data,
        selected_model_info={"model": [{"model_architecture": "resnet50"}]},
    )


def _candidate(context: dict, images_per_class: int = 250) -> ClassificationConfigModel:
    selected_data = [
        {
            "class_name": class_name,
            "sources": [{"dataset_name": "coco", "count": images_per_class}],
        }
        for class_name in ("cat", "dog")
    ]
    return ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=selected_data,
        train_data_ratio=0.8,
        val_data_ratio=0.1,
        test_data_ratio=0.1,
        rationale="Grounded ResNet-50 fine-tuning configuration.",
        **context["reference_configuration"],
    )


@pytest.mark.parametrize(
    ("images_per_class", "training_mode", "freeze_epochs"),
    [
        (40, "head_only", 25),
        (250, "staged_fine_tune", 3),
        (1000, "fine_tune_pretrained", 0),
    ],
)
def obsolete_resnet_graphrag_materializes_valid_executable_modes(
    images_per_class,
    training_mode,
    freeze_epochs,
):
    context = build_hyperparameter_context(_state(images_per_class))
    base = context["reference_configuration"]

    assert base["scheduler_name"] == "step"
    assert base["scheduler_step_size"] == 7
    assert base["scheduler_gamma"] == 0.1
    assert base["patience"] == 5
    assert base["track_metric"] == "val_acc"
    assert context["critical_materialization_errors"] == []
    assert context["materialization_warnings"] == []
    assert "rule_resnet50_low_vram_batch_lr_scaling" not in {
        rule["id"] for rule in context["applicable_rules"]
    }

    candidate = _candidate(context, images_per_class)
    assert candidate.training_mode == training_mode
    assert candidate.freeze_backbone_epochs == freeze_epochs
    validate_graph_grounded_config(candidate.model_dump(mode="json"), context)


def test_execution_accepts_flat_equivalent_of_saved_runtime_config(monkeypatch, tmp_path):
    context = build_hyperparameter_context(_state(40))
    candidate = _candidate(context, 40)
    saved_path = tmp_path / "RESULT_HYPERPARAMETERS.json"
    saved_path.write_text(json.dumps(candidate.runtime_config()), encoding="utf-8")
    monkeypatch.setattr(execution, "hpo_config_path", lambda _job_id: saved_path)

    validated = execution._validate_config(
        ClassificationPipeline(),
        candidate.model_dump(mode="json"),
        job_id="planned-resnet",
    )

    assert validated == training_compatible_hpo_config(candidate.runtime_config())


def test_execution_ignores_saved_planning_provenance(monkeypatch, tmp_path):
    context = build_hyperparameter_context(_state(40))
    candidate = _candidate(context, 40)
    runtime = candidate.runtime_config()
    runtime["field_provenance"] = build_field_provenance(
        candidate.model_dump(mode="json"),
        context,
    )
    saved_path = tmp_path / "RESULT_HYPERPARAMETERS.json"
    saved_path.write_text(json.dumps(runtime), encoding="utf-8")
    monkeypatch.setattr(execution, "hpo_config_path", lambda _job_id: saved_path)

    validated = execution._validate_config(
        ClassificationPipeline(),
        runtime,
        job_id="planned-resnet-with-provenance",
        require_saved_config=True,
    )

    assert validated == training_compatible_hpo_config(candidate.runtime_config())


def obsolete_execution_rejects_changes_to_saved_graph_validated_config(monkeypatch, tmp_path):
    context = build_hyperparameter_context(_state(40))
    candidate = _candidate(context, 40)
    saved_path = tmp_path / "RESULT_HYPERPARAMETERS.json"
    saved_path.write_text(json.dumps(candidate.runtime_config()), encoding="utf-8")
    monkeypatch.setattr(execution, "hpo_config_path", lambda _job_id: saved_path)
    tampered = candidate.model_dump(mode="json")
    tampered.update({
        "training_mode": "fine_tune_pretrained",
        "freeze_backbone_epochs": 0,
        "scheduler_name": "none",
    })

    with pytest.raises(HTTPException) as error:
        execution._validate_config(
            ClassificationPipeline(),
            tampered,
            job_id="planned-resnet",
        )

    assert error.value.status_code == 409
    assert error.value.detail["changed_fields"] == [
        "freeze_backbone_epochs",
        "scheduler_name",
        "training_mode",
    ]


def test_planned_classification_execution_requires_saved_graph_config(monkeypatch, tmp_path):
    context = build_hyperparameter_context(_state(40))
    candidate = _candidate(context, 40)
    monkeypatch.setattr(
        execution,
        "hpo_config_path",
        lambda _job_id: tmp_path / "missing-hyperparameters.json",
    )

    with pytest.raises(HTTPException) as error:
        execution._validate_config(
            ClassificationPipeline(),
            candidate.model_dump(mode="json"),
            job_id="incomplete-planning-job",
            require_saved_config=True,
        )

    assert error.value.status_code == 409
    assert "Complete choose-hyperparameters" in error.value.detail


def test_training_request_requires_planning_job_id():
    with pytest.raises(ValidationError):
        execution.TrainStartRequest(chosen_parameters={})


@pytest.mark.parametrize(
    ("config", "epoch", "expected"),
    [
        ({"patience": 0, "training_mode": "fine_tune_pretrained"}, 2, False),
        (
            {
                "patience": 1,
                "training_mode": "staged_fine_tune",
                "freeze_backbone_epochs": 3,
            },
            3,
            False,
        ),
        (
            {
                "patience": 1,
                "training_mode": "staged_fine_tune",
                "freeze_backbone_epochs": 3,
            },
            4,
            True,
        ),
        ({"patience": 1, "training_mode": "head_only"}, 1, True),
    ],
)
def test_early_stopping_semantics(config, epoch, expected):
    assert ClassificationPipeline._early_stopping_active(config, epoch) is expected


def test_resnet_step_scheduler_and_head_learning_rate_are_executable():
    model, _ = make_model("resnet50", "none", num_classes=2)
    config = {
        "optimizer_name": "sgd",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "nesterov": False,
        "head_learning_rate_multiplier": 2.0,
        "scheduler_name": "step",
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.1,
        "num_epochs": 4,
        "warmup_epochs": 0,
    }
    optimizer = make_optimizer(
        classification_parameter_groups(model, "resnet50", config),
        config,
    )
    scheduler = make_epoch_scheduler(optimizer, config)

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.01, 0.02])
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.001, 0.002])


def test_step_scheduler_rejects_a_decay_boundary_after_training_ends():
    context = build_hyperparameter_context(_state(1000))
    data = _candidate(context, 1000).model_dump()
    data.update({"num_epochs": 5, "patience": 0, "scheduler_step_size": 7})

    with pytest.raises(ValueError, match="post-warmup training epochs"):
        ClassificationConfigModel.model_validate(data)


def test_resnet_head_only_training_updates_only_the_classifier():
    model, _ = make_model("resnet50", "none", num_classes=2)
    set_backbone_trainable(model, "resnet50", False)
    backbone_before = model.conv1.weight.detach().clone()
    head_before = model.fc.weight.detach().clone()
    config = {
        "optimizer_name": "sgd",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "nesterov": False,
        "head_learning_rate_multiplier": 1.0,
    }
    optimizer = make_optimizer(
        classification_parameter_groups(model, "resnet50", config),
        config,
    )
    loader = DataLoader(
        TensorDataset(torch.rand(2, 3, 64, 64), torch.tensor([0, 1])),
        batch_size=2,
    )

    train_one_epoch(
        model,
        loader,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        frozen_backbone=model,
        trainable_head=model.fc,
    )

    assert torch.equal(backbone_before, model.conv1.weight)
    assert not torch.equal(head_before, model.fc.weight)
    assert model.bn1.training is False
    assert model.fc.training is True


def test_resnet_backbone_can_be_unfrozen_for_staged_finetuning():
    model, _ = make_model("resnet50", "none", num_classes=2)
    set_backbone_trainable(model, "resnet50", False)
    assert model.fc.weight.requires_grad
    assert not model.conv1.weight.requires_grad

    set_backbone_trainable(model, "resnet50", True)

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_resnet_default_evaluation_transform_matches_v2_weights():
    weights = get_model_weights("resnet50", "default")
    transform = select_evaluation_transform("resnet50", image_size=224, weights=weights)
    transformed = transform(Image.new("RGB", (400, 300), color="white"))

    assert transform.resize_size == [232]
    assert transform.crop_size == [224]
    assert transformed.shape == (3, 224, 224)


def test_evaluation_reports_top5_only_when_it_is_defined():
    class FixedModel(torch.nn.Module):
        def forward(self, images):
            return torch.tensor(
                [[9.0, 8.0, 7.0, 6.0, 5.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
                device=images.device,
            )

    loader = DataLoader(
        TensorDataset(torch.zeros(2, 1), torch.tensor([0, 0])),
        batch_size=2,
    )
    _, _, metrics = evaluate(
        ["a", "b", "c", "d", "e", "f"],
        FixedModel(),
        loader,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )

    assert metrics["top5_acc"] == pytest.approx(0.5)

