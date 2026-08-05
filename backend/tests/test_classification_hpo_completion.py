import pytest

from cvmodellearning.models.classification_capabilities import (
    classification_capabilities,
    selected_classification_model_id,
)
from cvmodellearning.models.registry import ClassificationModelId
from cvmodellearning.schemas.classification_hpo import (
    ClassificationConfigDraft,
    ClassificationConfigModel,
)
from cvmodellearning.schemas.classification_hpo_completion import complete_classification_config


def _draft(**updates) -> ClassificationConfigDraft:
    values = {
        "classes": ["wrong"],
        "selected_data": [
            {"class_name": "wrong", "sources": [{"dataset_name": "wrong", "count": 1}]}
        ],
        "train_data_ratio": 0.8,
        "val_data_ratio": 0.1,
        "test_data_ratio": 0.1,
        "num_epochs": 40,
        "patience": 5,
        "batch_size": 16,
        "image_size": 384,
        "precision": "mixed",
        "scheduler_name": "cosine",
        "min_learning_rate": 1.0,
        "scheduler_step_size": 1,
        "scheduler_gamma": 0.1,
        "warmup_epochs": 3,
        "warmup_start_factor": 0.001,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
        "freeze_backbone_epochs": 5,
        "head_learning_rate_multiplier": 1.0,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 0.0,
        "random_erasing": 0.25,
        "auto_augment_policy": "ta_wide",
        "random_resized_crop_scale_min": 0.6,
        "horizontal_flip_probability": 0.5,
        "use_model_ema": True,
        "model_ema_decay": 0.9998,
        "model_ema_steps": 1000,
        "repeated_augmentation_repetitions": 1,
        "use_activation_checkpointing": True,
        "track_metric": "val_acc",
        "model_name": "resnet50",
        "model_weights": "default",
        "training_mode": "staged_fine_tune",
        "training_recipe_id": "",
        "optimizer_name": "adamw",
        "learning_rate": 0.0003,
        "weight_decay": 0.05,
        "eps": 1e-8,
        "beta1": 0.9,
        "beta2": 0.999,
        "nesterov": True,
        "momentum": 0.9,
        "alpha": 0.004,
        "centered": True,
        "criterion_name": "bce_with_logit",
        "label_smoothing": 0.1,
        "pos_weight": 2.0,
        "rationale": "LLM proposal.",
        "llm_field_rationales": [],
    }
    values.update(updates)
    return ClassificationConfigDraft.model_validate(values)


def test_dinov2_failure_from_log_is_completed_into_valid_config():
    state = {
        "classes": ["car", "truck", "bus", "motorcycle"],
        "selected_data": [
            {"class_name": "car", "sources": [{"dataset_name": "bdd", "count": 10}]}
        ],
    }

    completed, adjustments = complete_classification_config(
        _draft(), state, "dinov2_vitb14"
    )
    validated = ClassificationConfigModel.model_validate(completed)

    assert validated.model_name == "dinov2_vitb14"
    assert validated.image_size == 224
    assert validated.min_learning_rate < validated.learning_rate
    assert validated.classes == state["classes"]
    assert [item.model_dump() for item in validated.selected_data] == [{
        "class_name": "car",
        "sources": [{
            "dataset_name": "bdd",
            "allocations": [{
                "split": "train",
                "count": 10,
                "assignment_type": "official_split",
            }],
        }],
    }]
    assert validated.criterion_name == "cross_entropy"
    assert validated.pos_weight == 1.0
    assert validated.use_activation_checkpointing is False
    assert validated.eps == 1e-8  # active, schema-safe AdamW choice is preserved
    assert validated.nesterov is False
    assert validated.momentum == 0.0
    assert validated.alpha == 0.99
    assert validated.centered is False
    assert {item["field"] for item in adjustments} >= {
        "model_name",
        "classes",
        "selected_data",
        "image_size",
        "min_learning_rate",
    }
    assert "Executable constraints applied:" in validated.rationale


def test_every_registered_classifier_has_execution_capabilities():
    for model in ClassificationModelId:
        capabilities = classification_capabilities(model.value)
        assert capabilities.native_image_size >= 32


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ({"model": [{"model_architecture": "dinov2_vitb14"}]}, "dinov2_vitb14"),
        ({"model": {"model_name": "resnet50"}}, "resnet50"),
        ({"model_id": "vit_b_16"}, "vit_b_16"),
        ({"model_name": "not-registered"}, None),
    ],
)
def test_selected_model_resolution(selected, expected):
    assert selected_classification_model_id(selected) == expected
