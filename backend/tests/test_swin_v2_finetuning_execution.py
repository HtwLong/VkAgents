import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2
from fastapi import HTTPException

import cvmodellearning.models.classification_model_utils as classification_models
from cvmodellearning.models.classification_model_utils import make_model
from cvmodellearning.optimization.optimization_utils import make_criterion, make_optimizer
from cvmodellearning.preprocessing.transformations import (
    select_evaluation_transform,
    select_transforms,
)
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.hpo_runtime import training_compatible_hpo_config
from cvmodellearning.training.training_utils import (
    RepeatedAugmentationSampler,
    apply_swin_activation_checkpointing,
    make_epoch_scheduler,
    make_model_ema,
    set_backbone_trainable,
    swin_parameter_groups,
    train_one_epoch,
)
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from routers.execution import _validate_config


def _endpoint_shaped_swin_config() -> ClassificationConfigModel:
    """A structured output that /choose-hyperparameters is allowed to return."""
    return ClassificationConfigModel(
        classes=["cat", "dog"],
        selected_data=[
            {"class_name": "cat", "sources": [{"dataset_name": "test", "count": 2}]},
            {"class_name": "dog", "sources": [{"dataset_name": "test", "count": 2}]},
        ],
        train_data_ratio=0.8,
        val_data_ratio=0.1,
        test_data_ratio=0.1,
        num_epochs=30,
        patience=1,
        batch_size=2,
        image_size=256,
        precision="fp32",
        scheduler_name="cosine",
        min_learning_rate=1e-6,
        warmup_epochs=1,
        warmup_start_factor=0.01,
        gradient_accumulation_steps=2,
        gradient_clip_norm=5.0,
        freeze_backbone_epochs=0,
        head_learning_rate_multiplier=2.0,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        random_erasing=0.25,
        auto_augment_policy="ta_wide",
        use_model_ema=False,
        repeated_augmentation_repetitions=1,
        use_activation_checkpointing=False,
        track_metric="val_acc",
        model_name="swin_v2_t",
        model_weights="default",
        training_mode="fine_tune_pretrained",
        training_recipe_id="torchvision_swin_v2_t_imagenet_v1_adapted_custom_finetune",
        optimizer_name="adamw",
        learning_rate=5e-5,
        weight_decay=0.05,
        criterion_name="cross_entropy",
        label_smoothing=0.1,
        rationale="Executable Swin V2 Tiny GraphRAG-style configuration.",
    )


@pytest.mark.parametrize(
    ("model_name", "recipe_id", "learning_rate"),
    [
        ("swin_v2_t", "torchvision_swin_v2_t_imagenet_v1_adapted_custom_finetune", 5e-5),
        ("swin_v2_s", "torchvision_swin_v2_s_imagenet_v1_adapted_custom_finetune", 3e-5),
    ],
)
def test_endpoint_config_executes_a_real_swin_v2_optimizer_update(
    monkeypatch, model_name, recipe_id, learning_rate
):
    endpoint_config = _endpoint_shaped_swin_config().model_copy(
        update={
            "model_name": model_name,
            "training_recipe_id": recipe_id,
            "learning_rate": learning_rate,
        }
    )
    runtime = endpoint_config.runtime_config()
    config = training_compatible_hpo_config(runtime)

    assert runtime["scheduler_name"] == "cosine"
    assert runtime["gradient_clip_norm"] == 5.0
    assert runtime["optimizer"]["name"] == "adamw"

    # Keep the test offline while preserving the endpoint's pretrained-weight
    # selection path. Weight downloading itself is TorchVision's responsibility.
    original_builder = getattr(classification_models, model_name)
    monkeypatch.setattr(
        classification_models,
        model_name,
        lambda *, weights: original_builder(weights=None),
    )
    model, selected_weights = make_model(
        config["model_name"], config["model_weights"], len(config["classes"])
    )
    assert selected_weights is not None
    before = model.features[0][0].weight.detach().clone()
    optimizer = make_optimizer(swin_parameter_groups(model, config), config)
    criterion = make_criterion(config)
    scheduler = make_epoch_scheduler(optimizer, config)
    loader = DataLoader(
        TensorDataset(torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 0, 1])),
        batch_size=2,
    )
    batch_augmentation = v2.RandomChoice([
        v2.MixUp(alpha=config["mixup_alpha"], num_classes=2),
        v2.CutMix(alpha=config["cutmix_alpha"], num_classes=2),
    ])

    loss, accuracy = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        torch.device("cpu"),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_clip_norm=config["gradient_clip_norm"],
        batch_augmentation=batch_augmentation,
    )
    scheduler.step()

    assert loss > 0
    assert 0 <= accuracy <= 1
    assert not torch.equal(before, model.features[0][0].weight)
    assert optimizer.param_groups[0]["lr"] > 0


def test_swin_image_size_and_augmentation_fields_control_transforms():
    config = _endpoint_shaped_swin_config()
    train_transform, eval_transform = select_transforms(
        config.model_name,
        config.image_size,
        auto_augment_policy=config.auto_augment_policy,
        random_erasing=config.random_erasing,
    )

    assert any(step.__class__.__name__ == "TrivialAugmentWide" for step in train_transform.transforms)
    crop = next(step for step in train_transform.transforms if step.__class__.__name__ == "RandomResizedCrop")
    center_crop = next(step for step in eval_transform.transforms if step.__class__.__name__ == "CenterCrop")
    assert crop.size == (256, 256)
    assert center_crop.size == (256, 256)


def test_scheduler_uses_linear_warmup_then_cosine_decay():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = make_epoch_scheduler(
        optimizer,
        {
            "scheduler_name": "cosine",
            "num_epochs": 6,
            "warmup_epochs": 2,
            "warmup_start_factor": 0.1,
            "min_learning_rate": 0.01,
        },
    )

    used_learning_rates = []
    for _ in range(6):
        used_learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert used_learning_rates == pytest.approx(
        [0.1, 0.55, 1.0, 0.8550178567, 0.505, 0.1549821433]
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_model_ema_adjusts_decay_for_recipe_update_frequency():
    model = torch.nn.Linear(2, 2)
    ema_model, effective_decay, update_steps = make_model_ema(
        model,
        {"num_epochs": 10, "model_ema_decay": 0.99998, "model_ema_steps": 2},
        effective_batch_size=4,
        device=torch.device("cpu"),
    )

    assert update_steps == 2
    assert effective_decay == pytest.approx(0.999984)
    assert ema_model.use_buffers is True


def test_activation_checkpointing_preserves_strict_checkpoint_keys():
    model, _ = make_model("swin_v2_t", "none", num_classes=2)
    original_keys = set(model.state_dict())
    apply_swin_activation_checkpointing(model)

    assert set(model.state_dict()) == original_keys
    fresh_model, _ = make_model("swin_v2_t", "none", num_classes=2)
    fresh_model.load_state_dict(model.state_dict(), strict=True)


def test_head_only_training_updates_head_but_not_backbone():
    model, _ = make_model("swin_v2_t", "none", num_classes=2)
    set_backbone_trainable(model, "swin_v2_t", False)
    backbone_before = model.features[0][0].weight.detach().clone()
    head_before = model.head.weight.detach().clone()
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.1,
    )
    loader = DataLoader(
        TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1])),
        batch_size=2,
    )

    train_one_epoch(
        model,
        loader,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        frozen_backbone=model.features,
    )

    assert torch.equal(backbone_before, model.features[0][0].weight)
    assert not torch.equal(head_before, model.head.weight)
    assert model.features.training is False
    assert model.head.training is True


def test_swin_evaluation_transform_uses_configured_size_for_supported_variants():
    tiny_eval = select_evaluation_transform("swin_v2_t", image_size=384)
    small_eval = select_evaluation_transform("swin_v2_s", image_size=384)

    tiny_resize = next(step for step in tiny_eval.transforms if step.__class__.__name__ == "Resize")
    tiny_crop = next(step for step in tiny_eval.transforms if step.__class__.__name__ == "CenterCrop")
    small_resize = next(step for step in small_eval.transforms if step.__class__.__name__ == "Resize")
    small_crop = next(step for step in small_eval.transforms if step.__class__.__name__ == "CenterCrop")
    assert tiny_resize.size == [390]
    assert tiny_crop.size == (384, 384)
    assert small_resize.size == [390]
    assert small_crop.size == (384, 384)


def test_inference_loading_uses_checkpoint_image_size(monkeypatch, tmp_path):
    import cvmodellearning.pipelines.classification_pipe as classification_pipe

    checkpoint_path = tmp_path / "best_model.pt"
    stub_model = torch.nn.Linear(1, 2)
    torch.save(
        {
            "classes": ["cat", "dog"],
            "config": {
                "model_name": "swin_v2_t",
                "model_weights": "default",
                "image_size": 384,
            },
            "model_state_dict": stub_model.state_dict(),
        },
        checkpoint_path,
    )
    captured = {}
    monkeypatch.setattr(classification_pipe, "best_model_path", lambda _job_id: checkpoint_path)
    monkeypatch.setattr(
        classification_pipe,
        "make_model",
        lambda _name, _weights, num_classes: (torch.nn.Linear(1, num_classes), None),
    )
    monkeypatch.setattr(
        classification_pipe.MODEL_CACHE_MANAGER,
        "get_model_bundle",
        lambda _key: None,
    )
    monkeypatch.setattr(
        classification_pipe.MODEL_CACHE_MANAGER,
        "set_model_bundle",
        lambda _key, bundle: captured.update(bundle),
    )

    ClassificationPipeline().load_model_step("example")

    center_crop = next(
        step for step in captured["transform"].transforms if step.__class__.__name__ == "CenterCrop"
    )
    assert center_crop.size == (384, 384)


def test_inference_cache_is_invalidated_when_checkpoint_changes(monkeypatch, tmp_path):
    import cvmodellearning.pipelines.classification_pipe as classification_pipe

    checkpoint_path = tmp_path / "best_model.pt"
    stub_model = torch.nn.Linear(1, 2)
    torch.save(
        {
            "classes": ["cat", "dog"],
            "config": {
                "model_name": "swin_v2_t",
                "model_weights": "none",
                "image_size": 256,
            },
            "model_state_dict": stub_model.state_dict(),
        },
        checkpoint_path,
    )
    captured = {}
    monkeypatch.setattr(classification_pipe, "best_model_path", lambda _job_id: checkpoint_path)
    monkeypatch.setattr(
        classification_pipe,
        "make_model",
        lambda _name, _weights, num_classes: (torch.nn.Linear(1, num_classes), None),
    )
    monkeypatch.setattr(
        classification_pipe.MODEL_CACHE_MANAGER,
        "get_model_bundle",
        lambda _key: {"checkpoint_fingerprint": {"path": "stale", "mtime_ns": 0, "size": 0}},
    )
    monkeypatch.setattr(
        classification_pipe.MODEL_CACHE_MANAGER,
        "set_model_bundle",
        lambda _key, bundle: captured.update(bundle),
    )

    result = ClassificationPipeline().load_model_step("example")

    assert result["status"] == "loaded from disk"
    assert captured["checkpoint_fingerprint"]["path"] == str(checkpoint_path.resolve())


def test_final_partial_gradient_accumulation_group_is_not_underweighted():
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loader = DataLoader(
        TensorDataset(torch.ones(3, 1), torch.ones(3, 1)),
        batch_size=1,
        shuffle=False,
    )

    train_one_epoch(
        model,
        loader,
        optimizer,
        torch.nn.MSELoss(),
        torch.device("cpu"),
        gradient_accumulation_steps=2,
    )

    # The first two batches average to an update from 0 -> 2. The final
    # one-batch group must use its full gradient, producing 2 -> 0.
    assert model.weight.item() == pytest.approx(0.0, abs=1e-7)


def test_repeated_augmentation_preserves_epoch_length_and_reseeds_by_epoch():
    dataset = TensorDataset(torch.arange(8))
    sampler = RepeatedAugmentationSampler(dataset, repetitions=3, seed=17)

    epoch_zero = list(sampler)
    assert len(sampler) == len(dataset)
    assert len(epoch_zero) == len(dataset)
    assert epoch_zero == list(sampler)
    assert all(0 <= index < len(dataset) for index in epoch_zero)

    sampler.set_epoch(1)
    assert list(sampler) != epoch_zero


@pytest.mark.parametrize(
    "updates",
    [
        {"training_mode": "fine_tune_pretrained", "model_weights": "none"},
        {"training_mode": "train_from_scratch", "model_weights": "default"},
        {
            "training_mode": "train_from_scratch",
            "model_weights": "none",
            "freeze_backbone_epochs": 1,
        },
        {"training_mode": "staged_fine_tune", "freeze_backbone_epochs": 0},
        {"training_mode": "head_only", "freeze_backbone_epochs": 1},
    ],
)
def test_training_mode_rejects_incompatible_initialization_or_freezing(updates):
    data = _endpoint_shaped_swin_config().model_dump()
    data.update(updates)

    with pytest.raises(ValueError):
        ClassificationConfigModel.model_validate(data)


@pytest.mark.parametrize(
    "updates",
    [
        {"training_mode": "train_from_scratch", "model_weights": "none"},
        {"training_mode": "staged_fine_tune", "freeze_backbone_epochs": 1},
        {"training_mode": "head_only", "freeze_backbone_epochs": 30},
    ],
)
def test_training_mode_accepts_executable_swin_modes(updates):
    data = _endpoint_shaped_swin_config().model_dump()
    data.update(updates)

    assert ClassificationConfigModel.model_validate(data).training_mode == updates["training_mode"]


def test_execution_validation_accepts_planning_runtime_representation():
    runtime = _endpoint_shaped_swin_config().runtime_config()

    validated = _validate_config(ClassificationPipeline(), runtime)

    assert validated["optimizer_name"] == "adamw"
    assert validated["criterion_name"] == "cross_entropy"
    assert validated["learning_rate"] == 5e-5
    assert "optimizer" not in validated
    assert "criterion" not in validated


def test_execution_validation_accepts_unprovenanced_pretrained_swin_config():
    data = _endpoint_shaped_swin_config().model_dump()
    data["training_recipe_id"] = ""

    validated = _validate_config(ClassificationPipeline(), data)

    assert validated["training_recipe_id"] == ""


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "criterion_name": "bce_with_logit",
                "label_smoothing": 0.0,
                "mixup_alpha": 0.0,
                "cutmix_alpha": 0.0,
            },
            "supports only cross_entropy",
        ),
        ({"image_size": 250}, "divisible by 32"),
        ({"image_size": 224}, "requires image_size >= 256"),
        ({"freeze_backbone_epochs": 31}, "cannot exceed num_epochs"),
        (
            {"freeze_backbone_epochs": 1, "model_weights": "none"},
            "requires model_weights='default'",
        ),
        ({"scheduler_name": "none", "min_learning_rate": 1e-6, "warmup_epochs": 0}, "must be 0"),
        ({"warmup_epochs": 30}, "lower than num_epochs"),
        ({"batch_size": 1}, "batch_size >= 2"),
        (
            {
                "training_mode": "head_only",
                "freeze_backbone_epochs": 30,
                "use_activation_checkpointing": True,
            },
            "no benefit during head-only",
        ),
    ],
)
def test_swin_schema_rejects_incompatible_combinations(updates, message):
    data = _endpoint_shaped_swin_config().model_dump()
    data.update(updates)

    with pytest.raises(ValueError, match=message):
        ClassificationConfigModel.model_validate(data)


def test_classification_schema_rejects_zero_sized_data_splits():
    data = _endpoint_shaped_swin_config().model_dump()
    data.update({"train_data_ratio": 0.9, "val_data_ratio": 0.0})

    with pytest.raises(ValueError, match="greater than 0"):
        ClassificationConfigModel.model_validate(data)
