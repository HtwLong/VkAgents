import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

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
from cvmodellearning.models.registry import CLASSIFIER_HEAD_PATHS, model_ids
from cvmodellearning.preprocessing.transformations import (
    CLASSIFICATION_TRANSFORM_PROFILES,
    select_evaluation_transform,
    select_transforms,
)
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.interpretation_schema import PipelineState
from cvmodellearning.training.training_utils import (
    classification_parameter_groups,
    classifier_head,
    classifier_training_module,
    set_backbone_trainable,
    train_one_epoch,
)


SMALL_DATA = [
    {"class_name": name, "sources": [{"dataset_name": "example", "count": 250}]}
    for name in ("cat", "dog")
]

EFFICIENTNET_CASES = (
    ("efficientnet_b0", 224, 256),
    ("efficientnet_b1", 240, 255),
    ("efficientnet_b2", 288, 288),
    ("efficientnet_b3", 300, 320),
    ("efficientnet_b4", 380, 384),
    ("efficientnet_b5", 456, 456),
    ("efficientnet_b6", 528, 528),
    ("efficientnet_b7", 600, 600),
)
EFFICIENTNET_IDS = tuple(case[0] for case in EFFICIENTNET_CASES)


def _context(model_name="efficientnet_b0", selected_data=SMALL_DATA):
    state = PipelineState(
        task="classification",
        classes=["cat", "dog"],
        selected_data=selected_data,
        selected_model_info={"model": [{"model_architecture": model_name}]},
    )
    return build_hyperparameter_context(state)


def _complete_config(context, selected_data=SMALL_DATA) -> ClassificationConfigModel:
    return ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=selected_data,
        train_data_ratio=0.8,
        val_data_ratio=0.1,
        test_data_ratio=0.1,
        patience=5,
        rationale="Graph-grounded EfficientNet fine-tuning configuration.",
        **context["reference_configuration"],
    )


@pytest.mark.parametrize("model_name", EFFICIENTNET_IDS)
def test_all_efficientnet_variants_are_executable_with_registered_head(model_name):
    assert model_name in model_ids("classification")
    assert CLASSIFIER_HEAD_PATHS[model_name] == "classifier.1"

    model, _ = make_model(model_name, "none", num_classes=3)
    assert classifier_head(model, model_name).out_features == 3


def test_all_efficientnet_variants_are_available_to_model_selection():
    get_model_selection_graph.cache_clear()
    context = build_model_selection_context(
        PipelineState(task="classification", classes=["cat", "dog"]),
        top_k=50,
    )
    candidates = {candidate["model"]["id"] for candidate in context["candidate_models"]}

    assert set(EFFICIENTNET_IDS).issubset(candidates)
    assert "not proof that full fine-tuning fits" in context["instructions_for_selector"]


@pytest.mark.parametrize("model_name,crop_size,resize_size", EFFICIENTNET_CASES)
def test_efficientnet_weight_transform_matches_registered_profile(
    model_name, crop_size, resize_size
):
    weights = get_model_weights(model_name, "default")
    preset = weights.transforms()
    profile = CLASSIFICATION_TRANSFORM_PROFILES[model_name]

    assert profile.native_crop_size == preset.crop_size[0] == crop_size
    assert profile.native_resize_size == preset.resize_size[0] == resize_size
    assert profile.interpolation == preset.interpolation

    train_transform, _ = select_transforms(
        model_name, image_size=crop_size, weights=weights
    )
    eval_transform = select_evaluation_transform(
        model_name, image_size=crop_size, weights=weights
    )
    image = Image.new("RGB", (700, 680), color=(80, 120, 200))

    assert train_transform(image).shape == (3, crop_size, crop_size)
    assert eval_transform(image).shape == (3, crop_size, crop_size)
    assert eval_transform.resize_size == [resize_size]


@pytest.mark.parametrize("model_name", EFFICIENTNET_IDS)
def test_efficientnet_freezing_keeps_only_final_classifier_trainable(model_name):
    model, _ = make_model(model_name, "none", num_classes=3)

    set_backbone_trainable(model, model_name, False)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"classifier.1.weight", "classifier.1.bias"}


@pytest.mark.parametrize("model_name", ("efficientnet_b0", "efficientnet_b7"))
def test_frozen_efficientnet_keeps_classifier_dropout_in_training_mode(model_name):
    model, _ = make_model(model_name, "none", num_classes=3)
    model.eval()

    classifier_training_module(model, model_name).train()

    dropout_modules = [
        module for module in model.classifier.modules() if isinstance(module, torch.nn.Dropout)
    ]
    assert dropout_modules
    assert all(module.training for module in dropout_modules)
    assert model.features.training is False


@pytest.mark.parametrize("model_name,crop_size,resize_size", EFFICIENTNET_CASES)
def obsolete_small_dataset_rule_materializes_native_valid_head_only_config(
    model_name, crop_size, resize_size
):
    del resize_size
    context = _context(model_name)
    config = _complete_config(context)

    assert context["reference_configuration"]["model_name"] == model_name
    assert context["reference_configuration"]["image_size"] == crop_size
    assert context["base_field_provenance"]["image_size"] == {
        "source": "pretrained_weight_metadata",
        "source_id": model_name,
    }
    assert context["reference_configuration"]["training_mode"] == "head_only"
    assert context["reference_configuration"]["freeze_backbone_epochs"] == 25
    assert context["adjustment_rule_provenance"]["training_mode"] == (
        "rule_efficientnet_freeze_features_small_dataset"
    )
    assert context["fields_requiring_llm_completion"] == ["patience"]
    assert context["materialization_warnings"] == []
    assert context["critical_materialization_errors"] == []

    validate_executable_recipe_config(config.model_dump(mode="json"))
    validate_graph_grounded_config(config.model_dump(mode="json"), context)


@pytest.mark.parametrize("model_name,crop_size,resize_size", EFFICIENTNET_CASES)
def test_larger_dataset_keeps_full_efficientnet_finetuning(
    model_name, crop_size, resize_size
):
    del resize_size
    selected_data = [
        {"class_name": name, "sources": [{"dataset_name": "example", "count": 3000}]}
        for name in ("cat", "dog")
    ]
    context = _context(model_name, selected_data)
    config = _complete_config(context, selected_data)

    assert context["reference_configuration"]["image_size"] == crop_size
    assert context["reference_configuration"]["training_mode"] == "fine_tune_pretrained"
    assert context["reference_configuration"]["freeze_backbone_epochs"] == 0
    validate_executable_recipe_config(config.model_dump(mode="json"))


def test_efficientnet_scheduler_and_metric_have_recipe_provenance():
    context = _context()

    assert context["reference_configuration"]["scheduler_step_size"] == 7
    assert context["reference_configuration"]["scheduler_gamma"] == 0.1
    assert context["reference_configuration"]["track_metric"] == "val_acc"
    for field in ("scheduler_step_size", "scheduler_gamma", "track_metric"):
        assert context["base_field_provenance"][field]["source"] == "recipe_parameter"


@pytest.mark.parametrize("model_name", EFFICIENTNET_IDS)
def test_efficientnet_fp32_memory_row_matches_deterministic_formula(model_name):
    graph = get_hyperparameter_graph()
    estimates = [
        graph.nodes[target]
        for _, target, edge in graph.out_edges(model_name, data=True)
        if edge.get("relation") == "has_inference_memory_estimate"
    ]

    assert len(estimates) == 1
    row = estimates[0]
    assert row["precision_mode"] == "FP32"
    activation = estimate_cnn_activation_workspace(
        flops_b=float(row["flops_b"]),
        task="classification",
        precision_mode="FP32",
    )
    estimate = calculate_inference_memory(
        params_m=float(row["params_m"]),
        precision_mode=row["precision_mode"],
        activation_workspace_gb=activation,
    )
    assert estimate.total_estimated_vram_gb == float(row["total_estimated_vram_gb"])


@pytest.mark.parametrize("model_name,image_size", (("efficientnet_b0", 224), ("efficientnet_b1", 240)))
def test_head_only_efficientnet_optimizer_step_and_checkpoint_reload(model_name, image_size):
    model, _ = make_model(model_name, "none", num_classes=3)
    set_backbone_trainable(model, model_name, False)
    config = {"learning_rate": 1e-3, "head_learning_rate_multiplier": 1.0}
    optimizer = torch.optim.SGD(
        classification_parameter_groups(model, model_name, config),
        lr=1e-3,
        momentum=0.9,
    )
    images = torch.randn(2, 3, image_size, image_size)
    targets = torch.tensor([0, 2])
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)

    loss, _ = train_one_epoch(
        model,
        loader,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        frozen_backbone=model,
        trainable_head=classifier_training_module(model, model_name),
    )

    assert loss > 0
    fresh_model, _ = make_model(model_name, "none", num_classes=3)
    fresh_model.load_state_dict(model.state_dict())
    assert fresh_model.eval()(images[:1]).shape == (1, 3)


@pytest.mark.parametrize(
    "updates",
    (
        {"training_mode": "staged_fine_tune", "freeze_backbone_epochs": 3},
        {"head_learning_rate_multiplier": 2.0},
    ),
)
@pytest.mark.parametrize("model_name", ("efficientnet_b0", "efficientnet_b7"))
def test_efficientnet_schema_accepts_registered_finetuning_capabilities(updates, model_name):
    context = _context(model_name)
    candidate = {
        **context["reference_configuration"],
        **updates,
    }
    if updates.get("training_mode") == "staged_fine_tune":
        candidate["freeze_backbone_epochs"] = 3

    config = ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=SMALL_DATA,
        patience=5,
        rationale="Capability validation.",
        **candidate,
    )
    assert config.model_name == model_name
    validate_executable_recipe_config(config.model_dump(mode="json"))

