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
    set_backbone_trainable,
    train_one_epoch,
)


SMALL_DATA = [
    {"class_name": name, "sources": [{"dataset_name": "example", "count": 250}]}
    for name in ("cat", "dog")
]


def _context(selected_data=SMALL_DATA):
    state = PipelineState(
        task="classification",
        classes=["cat", "dog"],
        selected_data=selected_data,
        selected_model_info={"model": [{"model_architecture": "densenet121"}]},
    )
    return build_hyperparameter_context(state)


def _complete_config(context, selected_data=SMALL_DATA) -> ClassificationConfigModel:
    return ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=selected_data,
        patience=5,
        precision="fp32",
        rationale="Graph-grounded DenseNet-121 fine-tuning configuration.",
        **context["recommended_configuration"],
    )


def test_densenet121_factory_registry_and_weight_transform_are_consistent():
    assert "densenet121" in model_ids("classification")
    assert CLASSIFIER_HEAD_PATHS["densenet121"] == "classifier"

    model, _ = make_model("densenet121", "none", num_classes=3)
    assert classifier_head(model, "densenet121").out_features == 3

    weights = get_model_weights("densenet121", "default")
    preset = weights.transforms()
    profile = CLASSIFICATION_TRANSFORM_PROFILES["densenet121"]
    assert profile.native_crop_size == preset.crop_size[0] == 224
    assert profile.native_resize_size == preset.resize_size[0] == 256
    assert profile.interpolation == preset.interpolation


def test_densenet121_train_validation_test_and_inference_transforms_match():
    weights = get_model_weights("densenet121", "default")
    train_transform, validation_transform = select_transforms(
        "densenet121", image_size=224, weights=weights
    )
    test_or_inference_transform = select_evaluation_transform(
        "densenet121", image_size=224, weights=weights
    )
    image = Image.new("RGB", (320, 280), color=(80, 120, 200))

    assert train_transform(image).shape == (3, 224, 224)
    assert validation_transform(image).shape == (3, 224, 224)
    assert test_or_inference_transform(image).shape == (3, 224, 224)
    assert test_or_inference_transform.resize_size == [256]


def test_densenet121_freezing_keeps_only_classifier_trainable():
    model, _ = make_model("densenet121", "none", num_classes=3)

    set_backbone_trainable(model, "densenet121", False)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"classifier.weight", "classifier.bias"}


def test_small_dataset_rule_materializes_valid_head_only_densenet_config():
    context = _context()
    config = _complete_config(context)

    assert context["recommended_configuration"]["training_mode"] == "head_only"
    assert context["recommended_configuration"]["freeze_backbone_epochs"] == 25
    assert context["adjustment_rule_provenance"]["training_mode"] == (
        "rule_densenet121_freeze_features_small_dataset"
    )
    assert context["recipe_details"][0]["feature_extraction_supported"] == "true"
    assert context["fields_requiring_llm_completion"] == ["patience", "precision"]
    assert context["materialization_warnings"] == []
    assert context["critical_materialization_errors"] == []

    validate_executable_recipe_config(config.model_dump(mode="json"))
    validate_graph_grounded_config(config.model_dump(mode="json"), context)


def test_larger_dataset_keeps_full_densenet_finetuning():
    selected_data = [
        {"class_name": name, "sources": [{"dataset_name": "example", "count": 3000}]}
        for name in ("cat", "dog")
    ]
    context = _context(selected_data)
    config = _complete_config(context, selected_data)

    assert context["recommended_configuration"]["training_mode"] == "fine_tune_pretrained"
    assert context["recommended_configuration"]["freeze_backbone_epochs"] == 0
    validate_executable_recipe_config(config.model_dump(mode="json"))


def test_densenet_scheduler_and_metric_have_recipe_provenance():
    context = _context()

    assert context["base_configuration"]["scheduler_step_size"] == 7
    assert context["base_configuration"]["scheduler_gamma"] == 0.1
    assert context["base_configuration"]["track_metric"] == "val_acc"
    for field in ("scheduler_step_size", "scheduler_gamma", "track_metric"):
        assert context["base_field_provenance"][field]["source"] == "recipe_parameter"


def test_densenet_fp32_memory_row_enables_hardware_filtered_selection():
    get_hyperparameter_graph.cache_clear()
    graph = get_hyperparameter_graph()
    estimates = [
        graph.nodes[target]
        for _, target, edge in graph.out_edges("densenet121", data=True)
        if edge.get("relation") == "has_inference_memory_estimate"
    ]
    assert len(estimates) == 1
    row = estimates[0]
    assert row["precision_mode"] == "FP32"
    activation = estimate_cnn_activation_workspace(
        flops_b=float(row["flops_b"]), task="classification", precision_mode="FP32"
    )
    estimate = calculate_inference_memory(
        params_m=float(row["params_m"]),
        precision_mode="FP32",
        activation_workspace_gb=activation,
    )
    assert estimate.total_estimated_vram_gb == float(row["total_estimated_vram_gb"])

    get_model_selection_graph.cache_clear()
    context = build_model_selection_context(
        PipelineState(
            task="classification",
            classes=["cat", "dog"],
            available_hardware={"hardware_category": "ConsumerGPU", "vram_gb": 8},
        ),
        top_k=50,
    )
    assert "densenet121" in {
        candidate["model"]["id"] for candidate in context["candidate_models"]
    }


def test_head_only_densenet_optimizer_step_and_checkpoint_reload():
    model, _ = make_model("densenet121", "none", num_classes=3)
    set_backbone_trainable(model, "densenet121", False)
    optimizer = torch.optim.SGD(
        classification_parameter_groups(
            model,
            "densenet121",
            {"learning_rate": 1e-3, "head_learning_rate_multiplier": 1.0},
        ),
        lr=1e-3,
        momentum=0.9,
    )
    images = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([0, 2])
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)

    loss, _ = train_one_epoch(
        model,
        loader,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        frozen_backbone=model,
        trainable_head=classifier_head(model, "densenet121"),
    )

    assert loss > 0
    fresh_model, _ = make_model("densenet121", "none", num_classes=3)
    fresh_model.load_state_dict(model.state_dict())
    assert fresh_model.eval()(images[:1]).shape == (1, 3)


def test_densenet_schema_accepts_staged_and_head_lr_capabilities():
    context = _context()
    candidate = {
        **context["recommended_configuration"],
        "training_mode": "staged_fine_tune",
        "freeze_backbone_epochs": 3,
        "head_learning_rate_multiplier": 2.0,
    }
    config = ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=SMALL_DATA,
        patience=5,
        precision="fp32",
        rationale="DenseNet capability validation.",
        **candidate,
    )

    validate_executable_recipe_config(config.model_dump(mode="json"))
