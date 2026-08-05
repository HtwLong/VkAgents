import pytest

from cvmodellearning.agents.hyperparameter_agents import (
    apply_owned_pipeline_fields,
    selected_detection_model_id,
)
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.schemas.vqa_hpo import VQAConfigModel


SELECTED_DATA = [
    {
        "class_name": "person",
        "sources": [{"dataset_name": "coco", "count": 10}],
    }
]


def classification_config(**overrides):
    data = {
        "classes": ["person"],
        "selected_data": SELECTED_DATA,
        "train_data_ratio": 0.8,
        "val_data_ratio": 0.1,
        "test_data_ratio": 0.1,
        "num_epochs": 1,
        "patience": 0,
        "batch_size": 2,
        "image_size": 224,
        "track_metric": "val_acc",
        "model_name": "resnet50",
        "model_weights": "default",
        "optimizer_name": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "beta1": 0.9,
        "beta2": 0.999,
        "nesterov": False,
        "momentum": 0.9,
        "alpha": 0.99,
        "centered": False,
        "criterion_name": "cross_entropy",
        "label_smoothing": 0.0,
        "pos_weight": 1.0,
        "rationale": "test",
    }
    data.update(overrides)
    return ClassificationConfigModel(**data)


def test_classification_config_does_not_require_img_per_class():
    config = classification_config()
    assert "img_per_class" not in config.model_dump(exclude_none=True)


def test_rejects_unstable_tiny_input_full_finetuning_combination():
    with pytest.raises(ValueError, match="image_size <= 32"):
        classification_config(
            model_name="mobilenet_v3_small",
            training_mode="fine_tune_pretrained",
            image_size=32,
            batch_size=4,
            learning_rate=1e-3,
        )


def test_allows_tiny_input_with_safer_learning_rate():
    config = classification_config(
        model_name="mobilenet_v3_small",
        training_mode="fine_tune_pretrained",
        image_size=32,
        batch_size=4,
        learning_rate=5e-4,
    )

    assert config.image_size == 32
    assert config.learning_rate == 5e-4


def test_runtime_normalization_removes_non_executable_provenance():
    runtime = classification_config().runtime_config()
    runtime["field_provenance"] = {
        "learning_rate": {"source": "recipe", "source_id": "example"}
    }

    normalized = training_compatible_hpo_config(runtime)

    assert "field_provenance" not in normalized
    assert "rationale" not in normalized


def detection_config(**overrides):
    data = {
        "task_type": "detection",
        "classes": ["person"],
        "selected_data": SELECTED_DATA,
        "train_data_ratio": 0.8,
        "val_data_ratio": 0.1,
        "test_data_ratio": 0.1,
        "num_epochs": 1,
        "patience": 0,
        "batch_size": 2,
        "input_size": 640,
        "track_metric": "val_mAP",
        "model_name": "yolov8_n",
        "model_weights": "coco",
        "training_recipe_id": "ultralytics_yolo_detection_finetune_balanced",
        "optimizer_name": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "momentum": 0.9,
        "loss_box": "ciou",
        "loss_cls": "bce",
        "lambda_box": 1.0,
        "lambda_cls": 1.0,
        "rationale": "test",
    }
    data.update(overrides)
    return DetectionConfigModel(**data)


@pytest.mark.parametrize(
    ("use_graphrag", "use_policy_registry"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_classification_authoritative_fields_are_consistent_in_all_modes(
    use_graphrag,
    use_policy_registry,
):
    recipe_id = "torchvision_resnet50_imagenet_pretrained_custom_finetune"
    proposal = classification_config(
        batch_size=16,
        precision="mixed",
        training_recipe_id=recipe_id,
    )
    context = {
        "task": "classification",
        "classes": ["cat", "dog"],
        "selected_data": SELECTED_DATA,
        "selected_model_info": {
            "model": [{"model_architecture": "resnet50"}],
        },
        "use_graphrag": use_graphrag,
        "use_policy_registry": use_policy_registry,
        "training_hardware": {
            "max_batch_size": 4,
            "workers": 3,
            "supports_amp": False,
        },
    }

    normalized = apply_owned_pipeline_fields(proposal, context)

    assert normalized.model_name == "resnet50"
    assert normalized.classes == ["cat", "dog"]
    assert normalized.batch_size == 4
    assert normalized.precision == "fp32"
    assert normalized.training_recipe_id == (recipe_id if use_graphrag else "")


@pytest.mark.parametrize(
    ("use_graphrag", "use_policy_registry"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_detection_authoritative_fields_are_consistent_in_all_modes(
    use_graphrag,
    use_policy_registry,
):
    recipe_id = "ultralytics_yolo_detection_finetune_balanced"
    proposal = detection_config(
        model_name="yolov8_s",
        batch_size=16,
        amp=True,
        workers=8,
        training_recipe_id=recipe_id,
    )
    context = {
        "task": "detection",
        "classes": ["person"],
        "selected_data": SELECTED_DATA,
        "selected_model_info": {
            "model": [{"model_architecture": "yolov8_n"}],
        },
        "use_graphrag": use_graphrag,
        "use_policy_registry": use_policy_registry,
        "training_hardware": {
            "max_batch_size": 4,
            "workers": 3,
            "supports_amp": False,
        },
    }

    normalized = apply_owned_pipeline_fields(proposal, context)

    assert normalized.model_name == "yolov8_n"
    assert normalized.batch_size == 4
    assert normalized.amp is False
    assert normalized.workers == 3
    assert normalized.training_recipe_id == (recipe_id if use_graphrag else "")


def test_adamw_epsilon_rejects_values_that_are_schema_valid_but_unsafe():
    with pytest.raises(ValueError, match="less than or equal to 0.01"):
        classification_config(eps=1.0)


@pytest.mark.parametrize(
    ("selected_model_id", "expected"),
    [
        ("fasterrcnn_resnet50_fpn", "faster_rcnn_r50"),
        ("retinanet_resnet50_fpn", "retinanet_r50"),
        ("ssd300_vgg16", "ssd300"),
        ("rtdetr_hgnetv2_l", "rtdetr_hgnetv2_l"),
        ("yolov10", "yolov10_n"),
        ("yolov10n", "yolov10_n"),
        ("yolov11_n", "yolov11_n"),
    ],
)
def test_detection_model_selection_aliases_resolve_to_hpo_ids(
    selected_model_id,
    expected,
):
    context = {
        "hyperparameter_graph_context": {
            "selected_model_id": selected_model_id,
        },
    }

    assert selected_detection_model_id(context) == expected


def vqa_config(**overrides):
    data = {
        "task_type": "visual question answering",
        "classes": [],
        "selected_data": SELECTED_DATA,
        "train_data_ratio": 0.8,
        "val_data_ratio": 0.1,
        "test_data_ratio": 0.1,
        "num_epochs": 1,
        "patience": 0,
        "batch_size": 1,
        "max_seq_length": 2048,
        "track_metric": "val_loss",
        "model_name": "Qwen3-VL-2B-Instruct",
        "precision": "bf16",
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "optimizer_name": "adamw",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "beta1": 0.9,
        "beta2": 0.999,
        "nesterov": False,
        "momentum": 0.9,
        "alpha": 0.99,
        "centered": False,
        "rationale": "test",
    }
    data.update(overrides)
    return VQAConfigModel(**data)


def assert_optimizer_params(model, expected_name, expected_params):
    runtime = model.runtime_config()

    assert runtime["optimizer"]["name"] == expected_name
    assert set(runtime["optimizer"]["params"]) == set(expected_params)

    flat_optimizer_fields = {
        "optimizer_name",
        "learning_rate",
        "weight_decay",
        "eps",
        "beta1",
        "beta2",
        "nesterov",
        "momentum",
        "alpha",
        "centered",
    }
    assert flat_optimizer_fields.isdisjoint(runtime)


def test_classification_runtime_config_keeps_only_selected_optimizer_params():
    expected = {
        "adamw": {"learning_rate", "weight_decay", "eps", "beta1", "beta2"},
        "sgd": {"learning_rate", "weight_decay", "momentum", "nesterov"},
        "rmsprop": {"learning_rate", "weight_decay", "eps", "momentum", "alpha", "centered"},
    }

    for optimizer_name, expected_params in expected.items():
        assert_optimizer_params(
            classification_config(optimizer_name=optimizer_name),
            optimizer_name,
            expected_params,
        )


def test_classification_runtime_preserves_executable_transform_controls():
    runtime = classification_config(
        random_resized_crop_scale_min=0.75,
        horizontal_flip_probability=0.0,
        auto_augment_policy="ta_wide",
        random_erasing=0.1,
    ).runtime_config()

    assert runtime["random_resized_crop_scale_min"] == 0.75
    assert runtime["horizontal_flip_probability"] == 0.0
    assert runtime["auto_augment_policy"] == "ta_wide"
    assert runtime["random_erasing"] == 0.1


def test_classification_runtime_omits_inactive_dependent_fields():
    runtime = classification_config(
        training_mode="fine_tune_pretrained",
        use_model_ema=False,
        scheduler_name="none",
        warmup_epochs=0,
    ).runtime_config()

    assert {"lora_rank", "lora_alpha", "lora_dropout"}.isdisjoint(runtime)
    assert {"model_ema_decay", "model_ema_steps"}.isdisjoint(runtime)
    assert {"min_learning_rate", "scheduler_step_size", "scheduler_gamma"}.isdisjoint(runtime)
    assert "warmup_start_factor" not in runtime
    assert "rationale" not in runtime
    assert "llm_field_rationales" not in runtime


def test_classification_runtime_keeps_lora_fields_only_for_lora_mode():
    runtime = classification_config(
        model_name="vit_b_16",
        training_mode="lora",
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.1,
    ).runtime_config()

    assert runtime["lora_rank"] == 4
    assert runtime["lora_alpha"] == 8
    assert runtime["lora_dropout"] == 0.1


def test_yolo_runtime_omits_other_detector_family_fields():
    runtime = detection_config().runtime_config()

    assert "mosaic" in runtime
    assert "max_size" not in runtime
    assert "trainable_backbone_layers" not in runtime
    assert "topk_candidates" not in runtime
    assert "rationale" not in runtime
    assert "llm_field_rationales" not in runtime


def test_vit_schema_rejects_resolution_not_supported_by_model_constructor():
    import pytest

    with pytest.raises(ValueError, match="supports only image_size=224"):
        classification_config(model_name="vit_b_16", image_size=256)


def test_detection_runtime_config_keeps_only_selected_optimizer_params():
    expected = {
        "auto": set(),
        "adamw": {"learning_rate", "weight_decay", "beta1"},
        "sgd": {"learning_rate", "weight_decay", "momentum"},
        "rmsprop": {"learning_rate", "weight_decay", "momentum"},
    }

    for optimizer_name, expected_params in expected.items():
        overrides = {"optimizer_name": optimizer_name}
        if optimizer_name == "auto":
            overrides.update(learning_rate=0.01, momentum=0.9)
        assert_optimizer_params(
            detection_config(**overrides),
            optimizer_name,
            expected_params,
        )


def test_vqa_runtime_config_keeps_only_selected_optimizer_params():
    expected = {
        "adamw": {"learning_rate", "weight_decay", "eps", "beta1", "beta2"},
        "paged_adamw_8bit": {"learning_rate", "weight_decay", "eps", "beta1", "beta2"},
        "sgd": {"learning_rate", "weight_decay", "momentum", "nesterov"},
        "rmsprop": {"learning_rate", "weight_decay", "eps", "momentum", "alpha", "centered"},
    }

    for optimizer_name, expected_params in expected.items():
        assert_optimizer_params(
            vqa_config(optimizer_name=optimizer_name),
            optimizer_name,
            expected_params,
        )


def test_classification_runtime_config_keeps_only_selected_criterion_params():
    cross_entropy = classification_config(criterion_name="cross_entropy").runtime_config()
    assert cross_entropy["criterion"] == {
        "name": "cross_entropy",
        "params": {"label_smoothing": 0.0},
    }
    assert "criterion_name" not in cross_entropy
    assert "label_smoothing" not in cross_entropy
    assert "pos_weight" not in cross_entropy

    import pytest

    with pytest.raises(ValueError, match="supports only cross_entropy"):
        classification_config(
            criterion_name="bce_with_logit",
            label_smoothing=0.0,
            pos_weight=2.0,
        )


def test_training_compatible_hpo_config_flattens_runtime_optimizer_and_criterion():
    runtime = classification_config(
        optimizer_name="sgd",
        criterion_name="cross_entropy",
        label_smoothing=0.1,
        pos_weight=1.0,
    ).runtime_config()

    compatible = training_compatible_hpo_config(runtime)

    assert compatible["optimizer_name"] == "sgd"
    assert compatible["learning_rate"] == 1e-4
    assert compatible["momentum"] == 0.9
    assert compatible["nesterov"] is False
    assert compatible["criterion_name"] == "cross_entropy"
    assert compatible["label_smoothing"] == 0.1
    assert "optimizer" not in compatible
    assert "criterion" not in compatible
