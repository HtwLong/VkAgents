import pytest
import torch
from PIL import Image

from cvmodellearning.graphrag.hyperparameter_context import (
    build_hyperparameter_context,
    get_hyperparameter_graph,
    validate_executable_recipe_config,
)
from cvmodellearning.models.classification_model_utils import get_model_weights, make_model
from cvmodellearning.models.registry import CLASSIFIER_HEAD_PATHS, model_ids
from cvmodellearning.preprocessing.transformations import (
    CLASSIFICATION_TRANSFORM_PROFILES,
    select_evaluation_transform,
    select_transforms,
)
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.training_utils import classifier_head, set_backbone_trainable
from cvmodellearning.training.training_utils import classifier_training_module


SELECTED_DATA = [
    {"class_name": "cat", "sources": [{"dataset_name": "example", "count": 250}]},
    {"class_name": "dog", "sources": [{"dataset_name": "example", "count": 250}]},
]


MOBILENET_CASES = (
    ("mobilenet_v2", "classifier.1", 232),
    ("mobilenet_v3_large", "classifier.3", 232),
    ("mobilenet_v3_small", "classifier.3", 256),
)


def _context(model_name: str, selected_data=SELECTED_DATA):
    state = PipelineState(
        task="classification",
        classes=["cat", "dog"],
        selected_data=selected_data,
        selected_model_info={"model": [{"model_architecture": model_name}]},
    )
    return build_hyperparameter_context(state)


def _complete_config(context, selected_data=SELECTED_DATA) -> ClassificationConfigModel:
    completion = {
        "patience": 5,
        "track_metric": "val_acc",
    }
    if "precision" in context["fields_requiring_llm_completion"]:
        completion["precision"] = "fp32"
    return ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=selected_data,
        train_data_ratio=0.8,
        val_data_ratio=0.1,
        test_data_ratio=0.1,
        rationale="Graph-grounded MobileNet fine-tuning configuration.",
        **completion,
        **context["recommended_configuration"],
    )


@pytest.mark.parametrize("model_name,head_path,resize_size", MOBILENET_CASES)
def test_registered_mobilenet_factory_head_and_weight_transform(model_name, head_path, resize_size):
    assert model_name in model_ids("classification")
    assert CLASSIFIER_HEAD_PATHS[model_name] == head_path

    model, _ = make_model(model_name, "none", num_classes=3)
    assert classifier_head(model, model_name).out_features == 3

    weights = get_model_weights(model_name, "default")
    preset = weights.transforms()
    profile = CLASSIFICATION_TRANSFORM_PROFILES[model_name]
    assert profile.native_crop_size == preset.crop_size[0] == 224
    assert profile.native_resize_size == preset.resize_size[0] == resize_size
    assert profile.interpolation == preset.interpolation


@pytest.mark.parametrize("model_name", ("mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"))
def test_mobilenet_has_one_fp32_inference_memory_estimate_in_graph(model_name):
    graph = get_hyperparameter_graph()
    estimates = [
        graph.nodes[target]
        for _, target, edge in graph.out_edges(model_name, data=True)
        if edge.get("relation") == "has_inference_memory_estimate"
    ]

    assert len(estimates) == 1
    assert estimates[0]["precision_mode"] == "FP32"


@pytest.mark.parametrize("model_name,head_path,resize_size", MOBILENET_CASES)
def test_mobilenet_freezing_keeps_only_registered_final_classifier_trainable(
    model_name, head_path, resize_size
):
    del head_path, resize_size
    model, _ = make_model(model_name, "none", num_classes=3)

    set_backbone_trainable(model, model_name, False)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected_prefix = CLASSIFIER_HEAD_PATHS[model_name]
    assert trainable == {f"{expected_prefix}.weight", f"{expected_prefix}.bias"}


@pytest.mark.parametrize("model_name", ("mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"))
def test_frozen_mobilenet_keeps_classifier_dropout_in_training_mode(model_name):
    model, _ = make_model(model_name, "none", num_classes=3)
    model.eval()

    classifier_training_module(model, model_name).train()

    dropout_modules = [module for module in model.classifier.modules() if isinstance(module, torch.nn.Dropout)]
    assert dropout_modules
    assert all(module.training for module in dropout_modules)
    assert model.features.training is False


@pytest.mark.parametrize("model_name,head_path,resize_size", MOBILENET_CASES)
def test_mobilenet_optimizer_step_checkpoint_reload_and_transforms(model_name, head_path, resize_size):
    del head_path
    model, _ = make_model(model_name, "none", num_classes=3)
    model.train()
    images = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([0, 2])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()(model(images), targets)
    loss.backward()
    optimizer.step()

    fresh_model, _ = make_model(model_name, "none", num_classes=3)
    fresh_model.load_state_dict(model.state_dict())
    assert fresh_model.eval()(images[:1]).shape == (1, 3)

    weights = get_model_weights(model_name, "default")
    train_transform, _ = select_transforms(model_name, image_size=224, weights=weights)
    eval_transform = select_evaluation_transform(model_name, image_size=224, weights=weights)
    image = Image.new("RGB", (320, 280), color=(80, 120, 200))
    assert train_transform(image).shape == (3, 224, 224)
    assert eval_transform(image).shape == (3, 224, 224)
    assert eval_transform.resize_size == [resize_size]


def test_mobilenet_v2_recipe_uses_executable_scheduler_fields_and_excludes_v1_alias():
    context = _context("mobilenet_v2")
    config = _complete_config(context)

    assert context["base_recipe"]["id"] == "torchvision_mobilenetv2_imagenet_v2_custom_finetune"
    assert context["base_configuration"]["scheduler_step_size"] == 7
    assert context["base_configuration"]["scheduler_gamma"] == 0.1
    assert context["base_field_provenance"]["scheduler_step_size"]["source"] == "recipe_parameter"
    assert "torchvision_mobilenetv2_imagenet_v1_custom_finetune" in (
        context["excluded_non_executable_recipe_ids"]
    )
    assert "torchvision_mobilenetv2_imagenet_qnnpack_qat" in (
        context["excluded_non_executable_recipe_ids"]
    )
    validate_executable_recipe_config(config.model_dump(mode="json"))


@pytest.mark.parametrize("model_name", ("mobilenet_v3_large", "mobilenet_v3_small"))
def test_small_dataset_rule_materializes_valid_head_only_mobilenet_v3(model_name):
    context = _context(model_name)
    config = _complete_config(context)

    assert context["base_recipe"]["id"] == (
        "torchvision_mobilenetv3_imagenet_pretrained_custom_finetune"
    )
    assert context["recommended_configuration"]["training_mode"] == "head_only"
    assert context["recommended_configuration"]["freeze_backbone_epochs"] == 25
    assert context["adjustment_rule_provenance"]["training_mode"] == (
        "rule_mobilenetv3_freeze_features_small_dataset"
    )
    validate_executable_recipe_config(config.model_dump(mode="json"))


def test_mobilenet_v3_small_dataset_rule_does_not_match_larger_dataset():
    selected_data = [
        {"class_name": name, "sources": [{"dataset_name": "example", "count": 3000}]}
        for name in ("cat", "dog")
    ]
    context = _context("mobilenet_v3_small", selected_data=selected_data)

    assert context["recommended_configuration"]["training_mode"] == "fine_tune_pretrained"
    assert context["recommended_configuration"]["freeze_backbone_epochs"] == 0


@pytest.mark.parametrize("model_name", ("mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"))
def test_mobilenet_small_dataset_rule_materializes_head_only(model_name):
    context = _context(model_name)

    assert context["recommended_configuration"]["training_mode"] == "head_only"
    assert context["recommended_configuration"]["freeze_backbone_epochs"] == 25


@pytest.mark.parametrize("model_name", ("mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"))
def test_mobilenet_intermediate_dataset_rule_materializes_staged_finetuning(model_name):
    selected_data = [
        {"class_name": name, "sources": [{"dataset_name": "example", "count": 1500}]}
        for name in ("cat", "dog")
    ]
    context = _context(model_name, selected_data=selected_data)
    config = _complete_config(context, selected_data=selected_data)

    assert context["recommended_configuration"]["training_mode"] == "staged_fine_tune"
    assert context["recommended_configuration"]["freeze_backbone_epochs"] == 3
    validate_executable_recipe_config(config.model_dump(mode="json"))


@pytest.mark.parametrize("model_name", ("mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"))
def test_mobilenet_large_dataset_keeps_full_finetuning(model_name):
    selected_data = [
        {"class_name": name, "sources": [{"dataset_name": "example", "count": 3000}]}
        for name in ("cat", "dog")
    ]
    context = _context(model_name, selected_data=selected_data)

    assert context["recommended_configuration"]["training_mode"] == "fine_tune_pretrained"
    assert context["recommended_configuration"]["freeze_backbone_epochs"] == 0
